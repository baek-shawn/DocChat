"""이미지 크기 제한(§5.2)과 "VLM에 보낼 이미지 목록 조립".

두 역할을 일부러 분리했다.
  1) 이미지 준비      : `cap_scale`, `prepare_uploaded_image` (PDF 페이지는 `pdf.render_page`)
  2) 전송 목록 조립    : `assemble_model_images`
지금은 2)가 "전체 이미지 한 장"만 돌려주지만, 나중에 타일링을 넣을 때는 이 함수만 바꾸면 된다.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass

from PIL import Image, ImageOps

from .. import config

# CAD 스캔본은 매우 클 수 있다. Pillow 기본 한도(약 8,900만 픽셀)보다 넉넉히 잡되 상한은 둔다.
Image.MAX_IMAGE_PIXELS = 250_000_000

# 로컬 VLM 서버 대부분이 PNG/JPEG만 안정적으로 받는다.
_PASSTHROUGH_MIMES = {"image/png", "image/jpeg"}
_LOSSLESS_SOURCES = {"image/png", "image/gif", "image/bmp", "image/tiff"}


class ImageError(ValueError):
    """사용자에게 보여 줄 수 있는 이미지 처리 오류."""


@dataclass
class PreparedImage:
    data: bytes
    mime: str
    width: int
    height: int
    source_width: int
    source_height: int
    resized: bool


@dataclass
class ModelImage:
    """모델 요청에 실리는 이미지 한 장.

    source_box는 이 이미지가 원본에서 차지하는 영역(정규화 x0, y0, x1, y1)이다.
    전체 이미지는 (0, 0, 1, 1). 타일링을 도입하면 타일마다 다른 값이 들어가고,
    타일 좌표 → 전체 좌표 역변환에 쓰인다.
    """
    name: str
    mime: str
    data: bytes
    source_box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)


def cap_scale(width: float, height: float) -> float:
    """비율을 유지한 채 긴 변 ≤ MAX_VISION_IMAGE_EDGE, 픽셀 수 ≤ MAX_VISION_IMAGE_PIXELS가 되는 배율(≤ 1)."""
    if width <= 0 or height <= 0:
        return 1.0
    edge_scale = config.MAX_VISION_IMAGE_EDGE / max(width, height)
    pixel_scale = math.sqrt(config.MAX_VISION_IMAGE_PIXELS / (width * height))
    return min(1.0, edge_scale, pixel_scale)


def capped_size(width: int, height: int) -> tuple[int, int, float]:
    scale = cap_scale(width, height)
    if scale >= 1.0:
        return width, height, 1.0
    # 5000 * (3072/5000) = 3071.9999…처럼 부동소수 오차로 1px를 잃지 않도록 아주 작은 여유를 준다.
    new_width = max(1, int(width * scale + 1e-6))
    new_height = max(1, int(height * scale + 1e-6))
    while new_width * new_height > config.MAX_VISION_IMAGE_PIXELS or max(new_width, new_height) > config.MAX_VISION_IMAGE_EDGE:
        if new_width >= new_height:
            new_width -= 1
        else:
            new_height -= 1
    return new_width, new_height, scale


def prepare_uploaded_image(data: bytes, mime: str) -> PreparedImage:
    """업로드 이미지를 한도 이하로 줄이고 PNG/JPEG로 정규화한다(동기 함수 — `asyncio.to_thread`로 호출)."""
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.load()
            source_mime = Image.MIME.get(opened.format or "", mime)
            # exif_transpose는 회전이 없어도 복사본을 돌려주므로, 방향 태그(0x0112)를 직접 확인한다.
            rotated = opened.getexif().get(0x0112, 1) not in (0, 1)
            image = (ImageOps.exif_transpose(opened) or opened) if rotated else opened
            source_width, source_height = image.size
            if not source_width or not source_height:
                raise ImageError("이미지 크기를 읽을 수 없습니다.")
            width, height, scale = capped_size(source_width, source_height)
            has_alpha = image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info)

            if scale >= 1.0 and not rotated and not has_alpha and source_mime in _PASSTHROUGH_MIMES:
                # 이미 한도 이내의 PNG/JPEG → 재인코딩으로 화질을 깎지 않는다.
                return PreparedImage(data, source_mime, source_width, source_height, source_width, source_height, False)

            if has_alpha:
                # 투명 배경은 흰색으로 깐다(도면·문서 스크린샷의 기본 배경).
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.split()[-1])
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")
            if scale < 1.0:
                image = image.resize((width, height), Image.LANCZOS)

            buffer = io.BytesIO()
            if source_mime in _LOSSLESS_SOURCES:
                # 무손실 PNG가 도면의 날카로운 선과 작은 글자를 지켜 준다.
                image.save(buffer, format="PNG", optimize=False)
                out_mime = "image/png"
            else:
                image.save(buffer, format="JPEG", quality=95)
                out_mime = "image/jpeg"
            return PreparedImage(buffer.getvalue(), out_mime, width, height, source_width, source_height, scale < 1.0)
    except ImageError:
        raise
    except Image.DecompressionBombError as error:
        raise ImageError("이미지가 너무 큽니다(2억 5천만 픽셀 초과). 해상도를 낮춰 다시 올려 주세요.") from error
    except Exception as error:  # Pillow는 손상 파일에 다양한 예외를 던진다.
        raise ImageError(f"이미지를 열 수 없습니다: {error}") from error


def fit_image_bytes(data: bytes, mime: str, max_bytes: int) -> tuple[bytes, str]:
    """바이트 상한이 있는 API(예: Anthropic 이미지당 5MB)를 위해 JPEG 품질·크기를 단계적으로 낮춘다."""
    if len(data) <= max_bytes:
        return data, mime
    with Image.open(io.BytesIO(data)) as opened:
        image = opened.convert("RGB")
    for quality, shrink in ((90, 1.0), (80, 1.0), (80, 0.8), (70, 0.65), (60, 0.5)):
        candidate = image if shrink == 1.0 else image.resize(
            (max(1, int(image.width * shrink)), max(1, int(image.height * shrink))), Image.LANCZOS)
        buffer = io.BytesIO()
        candidate.save(buffer, format="JPEG", quality=quality)
        if buffer.tell() <= max_bytes:
            return buffer.getvalue(), "image/jpeg"
    return buffer.getvalue(), "image/jpeg"


def assemble_model_images(name: str, mime: str, data: bytes, *, purpose: str = "analysis") -> list[ModelImage]:
    """VLM에 보낼 이미지 목록을 만든다.

    purpose: "ocr" | "grounding" | "analysis". 현재는 목적과 무관하게 전체 이미지 한 장.
    (추후 타일링: 큰 페이지를 겹침 타일로 나눠 여러 ModelImage를 반환하고 source_box를 채운다.)
    """
    return [ModelImage(name=name, mime=mime, data=data)]
