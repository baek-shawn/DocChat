"""판별 규칙을 검증하기 위한 합성 PDF·이미지 생성기."""
from __future__ import annotations

import io

import pymupdf
from PIL import Image, ImageDraw

LONG_TEXT = [
    "DRAWING NO A-1024 REVISION C SHEET 1 OF 3",
    "PART LIST ITEM 001 BRACKET STEEL QTY 4 MASS 12.5 KG",
    "ITEM 002 BOLT M12x40 QTY 16 ITEM 003 WASHER 12 QTY 16",
    "GENERAL TOLERANCE ISO 2768 mK SURFACE FINISH Ra 3.2",
]


def png_bytes(width: int = 320, height: int = 200, text: str = "SCANNED PAGE 0042", mode: str = "RGB") -> bytes:
    image = Image.new(mode, (width, height), (255, 255, 255) if mode == "RGB" else (255, 255, 255, 0))
    ImageDraw.Draw(image).text((12, 12), text, fill=(0, 0, 0) if mode == "RGB" else (0, 0, 0, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes(width: int = 320, height: int = 200) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (240, 240, 240)).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _add_text(page: pymupdf.Page, lines: list[str], top: float = 72) -> None:
    for index, line in enumerate(lines):
        page.insert_text((72, top + index * 18), line, fontsize=11)


def _add_image(page: pymupdf.Page) -> None:
    page.insert_image(pymupdf.Rect(36, 300, 560, 760), stream=png_bytes())


def _add_vectors(page: pymupdf.Page) -> None:
    for offset in range(0, 200, 20):
        page.draw_line((72, 100 + offset), (520, 100 + offset), color=(0, 0, 0), width=0.7)
    page.draw_rect(pymupdf.Rect(72, 320, 300, 480), color=(0, 0, 0))


def build_pdf(*kinds: str, page_size: tuple[float, float] = (595, 842)) -> bytes:
    """kinds: native | scanned | vector | mixed_sparse | mixed_native | empty"""
    document = pymupdf.open()
    for kind in kinds:
        page = document.new_page(width=page_size[0], height=page_size[1])
        if kind == "native":
            _add_text(page, LONG_TEXT)
        elif kind == "scanned":
            _add_image(page)
        elif kind == "vector":
            _add_vectors(page)
        elif kind == "mixed_sparse":      # 스캔 이미지 + 글자 조금(24자 이상 120자 미만)
            _add_image(page)
            _add_text(page, ["SCAN OVERLAY PAGE NUMBER 0042 REVISION B"])
        elif kind == "mixed_native":      # 이미지가 있어도 글자가 충분(120자 이상)
            _add_image(page)
            _add_text(page, LONG_TEXT)
        elif kind == "empty":
            pass
        else:
            raise ValueError(kind)
    data = document.tobytes()
    document.close()
    return data
