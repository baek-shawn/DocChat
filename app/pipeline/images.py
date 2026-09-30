"""이미지 크기 제한(§5.2)과 "VLM에 보낼 이미지 목록 조립".

두 역할을 일부러 분리했다.
  1) 이미지 준비      : `prepare_uploaded_image`, `crop_image_tiles` (PDF 쪽은 `pdf.render_page`, `pdf.render_page_tiles`)
  2) 전송 목록 조립    : `assemble_model_images`
전체/타일 모드가 갈리는 곳은 2) 한 곳뿐이다. 1)의 함수들은 모드를 모른다.
"""
from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass

from PIL import Image, ImageOps

from .geometry import (TileSet, cap_scale, capped_size, is_blank_histogram, needs_tiling,  # noqa: F401 (재수출)
                       plan_tiles, RenderedTile)
from .pdf import render_page_tiles, run_pdf

# CAD 스캔본은 매우 클 수 있다. Pillow 기본 한도(약 8,900만 픽셀)보다 넉넉히 잡되 상한은 둔다.
Image.MAX_IMAGE_PIXELS = 250_000_000

# 로컬 VLM 서버 대부분이 PNG/JPEG만 안정적으로 받는다.
_PASSTHROUGH_MIMES = {"image/png", "image/jpeg"}
_LOSSLESS_SOURCES = {"image/png", "image/gif", "image/bmp", "image/tiff"}
# 타일로 나누는 호출. 답변(analysis) 호출의 이미지는 Step 8에서 따로 다룬다.
TILED_PURPOSES = ("ocr", "grounding")


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
    source_mime: str = ""      # 파일 내용으로 확인한 원본 형식(선언된 MIME과 다를 수 있다)


@dataclass
class TileGrid:
    """한 장을 나눈 타일 격자. 같은 장에서 나온 타일들이 함께 가리킨다."""
    rows: int
    cols: int
    blank: int = 0                 # 내용이 없어 보내지 않은 타일 수
    width: int = 0                 # 타일을 나눈 기준 해상도(px)
    height: int = 0
    dpi: float | None = None


@dataclass
class ModelImage:
    """모델 요청에 실리는 이미지 한 장.

    source_box는 이 이미지가 원본에서 차지하는 영역(정규화 x0, y0, x1, y1)이다.
    전체 이미지는 (0, 0, 1, 1), 타일은 타일마다 다른 값이고 타일 좌표 → 전체 좌표 역변환에 쓰인다.
    """
    name: str
    mime: str
    data: bytes
    source_box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    width: int | None = None       # 이 이미지 자체의 픽셀 크기(타일일 때만 채운다)
    height: int | None = None
    tile: tuple[int, int] | None = None     # (행, 열), 1부터. 전체 이미지면 None
    grid: TileGrid | None = None


@dataclass
class TileSource:
    """타일을 잘라 낼 고해상도 원본. PDF면 (원본 PDF, 쪽 번호), 이미지면 보관해 둔 업로드 원본."""
    kind: str                      # "pdf" | "image"
    data: bytes
    page_number: int = 1
    mime: str = ""


@dataclass
class VisionUsage:
    """한 턴 동안 비전 호출이 어떻게 나갔는지 — 답변 메타데이터와 비교 스크립트가 읽는다."""
    ocr_calls: int = 0
    grounding_calls: int = 0
    answer_calls: int = 0
    tiled_images: int = 0          # 타일로 나눠 보낸 쪽·이미지 수
    tiles: int = 0                 # 모델에 보낸 타일 수
    blank_tiles: int = 0           # 내용이 없어 건너뛴 타일 수
    # 출력 상한에 닿아 끊긴 호출 수(Step 6-0). 이런 호출은 다시 보내지 않는다.
    ocr_length_stops: int = 0
    grounding_length_stops: int = 0

    def count_images(self, images: list[ModelImage]) -> None:
        grid = images[0].grid if images else None
        if grid is not None:
            self.tiled_images += 1
            self.tiles += len(images)
            self.blank_tiles += grid.blank

    def to_public(self) -> dict[str, int]:
        return {"ocrCalls": self.ocr_calls, "groundingCalls": self.grounding_calls, "answerCalls": self.answer_calls,
                "tiledImages": self.tiled_images, "tiles": self.tiles, "blankTiles": self.blank_tiles,
                "ocrLengthStops": self.ocr_length_stops, "groundingLengthStops": self.grounding_length_stops}


# --------------------------------------------------------------------------- 업로드 이미지 준비
def _upright(opened: Image.Image, mime: str) -> tuple[Image.Image, str, bool, bool]:
    """(바로 세운 이미지, 실제 형식, 회전했는지, 투명 채널이 있는지)"""
    opened.load()
    source_mime = Image.MIME.get(opened.format or "", mime)
    # exif_transpose는 회전이 없어도 복사본을 돌려주므로, 방향 태그(0x0112)를 직접 확인한다.
    rotated = opened.getexif().get(0x0112, 1) not in (0, 1)
    image = (ImageOps.exif_transpose(opened) or opened) if rotated else opened
    if not image.width or not image.height:
        raise ImageError("이미지 크기를 읽을 수 없습니다.")
    has_alpha = image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info)
    return image, source_mime, rotated, has_alpha


def _flatten(image: Image.Image, has_alpha: bool) -> Image.Image:
    if has_alpha:
        # 투명 배경은 흰색으로 깐다(도면·문서 스크린샷의 기본 배경).
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return image if image.mode == "RGB" else image.convert("RGB")


def _encode(image: Image.Image, source_mime: str) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    if source_mime in _LOSSLESS_SOURCES:
        # 무손실 PNG가 도면의 날카로운 선과 작은 글자를 지켜 준다.
        image.save(buffer, format="PNG", optimize=False)
        return buffer.getvalue(), "image/png"
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue(), "image/jpeg"


def _explain(error: Exception) -> ImageError:
    if isinstance(error, Image.DecompressionBombError):
        return ImageError("이미지가 너무 큽니다(2억 5천만 픽셀 초과). 해상도를 낮춰 다시 올려 주세요.")
    return ImageError(f"이미지를 열 수 없습니다: {error}")     # Pillow는 손상 파일에 다양한 예외를 던진다.


def prepare_uploaded_image(data: bytes, mime: str) -> PreparedImage:
    """업로드 이미지를 한도 이하로 줄이고 PNG/JPEG로 정규화한다(동기 함수 — `asyncio.to_thread`로 호출).

    돌려주는 것은 모델 전송·뷰어용 사본이다. 원본 바이트는 호출부가 따로 보관한다(타일링 재료).
    """
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image, source_mime, rotated, has_alpha = _upright(opened, mime)
            source_width, source_height = image.size
            width, height, scale = capped_size(source_width, source_height)

            if scale >= 1.0 and not rotated and not has_alpha and source_mime in _PASSTHROUGH_MIMES:
                # 이미 한도 이내의 PNG/JPEG → 재인코딩으로 화질을 깎지 않는다.
                return PreparedImage(data, source_mime, source_width, source_height, source_width, source_height,
                                     False, source_mime)

            image = _flatten(image, has_alpha)
            if scale < 1.0:
                image = image.resize((width, height), Image.LANCZOS)
            encoded, out_mime = _encode(image, source_mime)
            return PreparedImage(encoded, out_mime, width, height, source_width, source_height, scale < 1.0, source_mime)
    except ImageError:
        raise
    except Exception as error:
        raise _explain(error) from error


def crop_image_tiles(data: bytes, mime: str) -> TileSet | None:
    """업로드 원본을 겹치는 타일로 자른다(동기 함수 — `asyncio.to_thread`로 호출).

    나눌 만큼 크지 않으면 None → 호출부는 전체 모드와 똑같이 한 장을 보낸다.
    """
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image, source_mime, _rotated, has_alpha = _upright(opened, mime)
            source_width, source_height = image.size
            if not needs_tiling(source_width, source_height):
                return None
            image = _flatten(image, has_alpha)
            plan = plan_tiles(source_width, source_height)
            tiles: list[RenderedTile] = []
            blank = 0
            for tile in plan.tiles:
                if plan.scale < 1.0:
                    # 타일 수 상한 때문에 기준 해상도를 낮췄다 → 원본의 더 넓은 영역을 타일 크기로 줄인다.
                    area = (tile.left / plan.scale, tile.top / plan.scale,
                            min(source_width, tile.right / plan.scale), min(source_height, tile.bottom / plan.scale))
                    piece = image.resize((tile.width, tile.height), Image.LANCZOS, box=area)
                else:
                    piece = image.crop((tile.left, tile.top, tile.right, tile.bottom))
                if is_blank_histogram(piece.convert("L").histogram()):
                    blank += 1
                    continue
                encoded, out_mime = _encode(piece, source_mime)
                tiles.append(RenderedTile(row=tile.row, col=tile.col, box=plan.source_box(tile),
                                          width=piece.width, height=piece.height, mime=out_mime, data=encoded))
            return TileSet(rows=plan.rows, cols=plan.cols, width=plan.width, height=plan.height, tiles=tiles, blank=blank)
    except ImageError:
        raise
    except Exception as error:
        raise _explain(error) from error


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


# --------------------------------------------------------------------------- 전송 목록 조립
async def assemble_model_images(name: str, mime: str, data: bytes, *, purpose: str = "analysis",
                                mode: str = "whole", source: TileSource | None = None) -> list[ModelImage]:
    """VLM에 보낼 이미지 목록을 만든다. **전체/타일 모드가 갈리는 곳은 여기 한 곳이다.**

    purpose: "ocr" | "grounding" | "analysis"
      - 전체 모드이거나 답변(analysis) 호출이면 받은 이미지 한 장을 그대로 돌려준다(Step 4까지와 동일).
      - 타일 모드의 전사·bbox 호출이면 고해상도 원본(source)을 겹치는 타일로 나눈다.
        PDF는 타일 영역만 바로 렌더하고, 업로드 이미지는 보관해 둔 원본에서 자른다.
    원본이 타일 기준 크기보다 작거나, 원본을 구할 수 없거나, 타일이 전부 비어 있으면 전체 모드와 똑같이 한 장이다.
    """
    whole = [ModelImage(name=name, mime=mime, data=data)]
    if mode != "tile" or purpose not in TILED_PURPOSES or source is None or not source.data:
        return whole
    if source.kind == "pdf":
        tiles = await run_pdf(render_page_tiles, source.data, page_number=source.page_number)
    else:
        tiles = await asyncio.to_thread(crop_image_tiles, source.data, source.mime or mime)
    if tiles is None or not tiles.tiles:
        return whole
    grid = TileGrid(rows=tiles.rows, cols=tiles.cols, blank=tiles.blank, width=tiles.width, height=tiles.height,
                    dpi=tiles.dpi)
    return [
        ModelImage(name=f"{name} · tile r{tile.row}c{tile.col}", mime=tile.mime, data=tile.data, source_box=tile.box,
                   width=tile.width, height=tile.height, tile=(tile.row, tile.col), grid=grid)
        for tile in tiles.tiles
    ]
