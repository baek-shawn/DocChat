"""첨부 파일의 내부 표현과 업로드 정제.

하나의 업로드는 여러 Attachment로 펼쳐진다.
  - PDF 원본            kind="pdf"      (네이티브 텍스트 + 원본 바이트)
  - PDF 페이지 이미지    kind="image"    (이름: "<pdf> · page N", 전사 대상)
  - 시각 OCR 증거        kind="document" (이름: "<pdf> · visual OCR", 전사 텍스트)
  - 업로드 이미지        kind="image"    (send_to_model=True → 메인 요청에 이미지로 직접 전달)
"""
from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from typing import Any

from . import config


class UploadError(ValueError):
    """사용자에게 그대로 보여 줄 수 있는 업로드 오류."""


@dataclass
class Attachment:
    name: str
    mime: str
    kind: str
    size: int = 0
    text: str = ""
    data: bytes | None = None
    id: int | None = None
    has_data: bool = False
    width: int | None = None
    height: int | None = None
    source_width: int | None = None
    source_height: int | None = None
    page_number: int | None = None
    page_classification: str | None = None
    # 메인 답변 전에 VLM 전사가 필요한 이미지(= needs_vlm 페이지)
    ocr_required: bool = False
    # 메인 요청에 이미지 바이트로 직접 실어 보낼지(= 사용자가 올린 일반 이미지)
    send_to_model: bool = False
    visual_pages: int = 0
    total_pages: int = 0

    @property
    def is_image(self) -> bool:
        return self.mime.startswith("image/")

    @property
    def is_pdf(self) -> bool:
        return self.mime == config.PDF_MIME or self.name.lower().endswith(".pdf")

    def metadata(self) -> dict[str, Any]:
        """DB의 metadata_json 컬럼에 들어가는 값."""
        pairs = {
            "width": self.width,
            "height": self.height,
            "sourceWidth": self.source_width,
            "sourceHeight": self.source_height,
            "pageNumber": self.page_number,
            "pageClassification": self.page_classification,
            "ocrRequired": self.ocr_required or None,
            "sendToModel": self.send_to_model or None,
            "visualPages": self.visual_pages or None,
            "totalPages": self.total_pages or None,
        }
        return {key: value for key, value in pairs.items() if value is not None}

    def apply_metadata(self, meta: dict[str, Any]) -> "Attachment":
        self.width = meta.get("width")
        self.height = meta.get("height")
        self.source_width = meta.get("sourceWidth")
        self.source_height = meta.get("sourceHeight")
        self.page_number = meta.get("pageNumber")
        self.page_classification = meta.get("pageClassification")
        self.ocr_required = bool(meta.get("ocrRequired"))
        self.send_to_model = bool(meta.get("sendToModel"))
        self.visual_pages = int(meta.get("visualPages") or 0)
        self.total_pages = int(meta.get("totalPages") or 0)
        return self

    def to_public(self) -> dict[str, Any]:
        """프론트로 내려보내는 요약. 바이트(base64)는 절대 포함하지 않는다."""
        out: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "mime": self.mime,
            "kind": self.kind,
            "size": self.size,
            "parsedCharacters": len(self.text or ""),
            "hasData": bool(self.has_data or self.data),
        }
        if self.width and self.height:
            out["width"], out["height"] = self.width, self.height
        if self.page_number:
            out["pageNumber"] = self.page_number
        if self.page_classification:
            out["pageClassification"] = self.page_classification
        if self.visual_pages:
            out["visualPages"] = self.visual_pages
        if self.total_pages:
            out["totalPages"] = self.total_pages
        if self.id is not None and out["hasData"]:
            out["url"] = f"/api/attachments/{self.id}/content"
        return out


# --------------------------------------------------------------------------- 업로드 정제
_EXTENSION_MIMES = {
    ".pdf": config.PDF_MIME,
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def mime_from_name(name: str) -> str:
    lower = name.lower()
    for extension, mime in _EXTENSION_MIMES.items():
        if lower.endswith(extension):
            return mime
    return "application/octet-stream"


def clean_upload_name(name: Any) -> str:
    text = str(name or "attachment").replace("\\", "/").split("/")[-1]
    # " · "는 파생 첨부(페이지/시각 OCR)를 구분하는 예약 구분자다.
    text = text.replace(" · ", " - ")
    text = re.sub(r"[\x00-\x1f]", "", text).strip()
    return (text or "attachment")[:240]


def _decode_base64(value: Any, name: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise UploadError(f"'{name}' 파일의 내용이 비어 있습니다.")
    if len(value) > config.MAX_ATTACHMENT_BASE64_CHARS:
        raise UploadError(f"'{name}' 파일이 너무 큽니다(최대 약 64MB).")
    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]
    try:
        return base64.b64decode(value, validate=False)
    except (binascii.Error, ValueError) as error:
        raise UploadError(f"'{name}' 파일의 base64 인코딩이 올바르지 않습니다.") from error


def sanitize_uploads(items: Any) -> list[Attachment]:
    """요청 본문의 attachments 배열을 검증해 원본 Attachment 목록으로 만든다(PDF·이미지만 허용)."""
    if not isinstance(items, list):
        return []
    if len(items) > config.MAX_ATTACHMENTS_PER_MESSAGE:
        raise UploadError(f"한 번에 첨부할 수 있는 파일은 최대 {config.MAX_ATTACHMENTS_PER_MESSAGE}개입니다.")
    seen: set[str] = set()
    result: list[Attachment] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = clean_upload_name(item.get("name"))
        mime = str(item.get("mime") or "").strip().lower()[:120]
        if mime in ("", "application/octet-stream"):
            mime = mime_from_name(name)
        if mime == "image/jpg":
            mime = "image/jpeg"
        if mime != config.PDF_MIME and mime not in config.ALLOWED_IMAGE_MIMES:
            raise UploadError(f"'{name}': PDF와 이미지(PNG/JPEG/WebP/GIF/BMP/TIFF)만 첨부할 수 있습니다.")
        name = _unique_name(name, seen)
        data = _decode_base64(item.get("base64"), name)
        if mime == config.PDF_MIME and not data.lstrip()[:5].startswith(b"%PDF"):
            raise UploadError(f"'{name}'은(는) 올바른 PDF 파일이 아닙니다.")
        result.append(Attachment(
            name=name, mime=mime, kind="pdf" if mime == config.PDF_MIME else "image",
            size=len(data), data=data, has_data=True,
        ))
    return result


def _unique_name(name: str, seen: set[str]) -> str:
    candidate, counter = name, 2
    stem, dot, extension = name.rpartition(".")
    while candidate.lower() in seen:
        candidate = f"{stem} ({counter}).{extension}" if dot else f"{name} ({counter})"
        counter += 1
    seen.add(candidate.lower())
    return candidate
