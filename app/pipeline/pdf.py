"""PDF 페이지 판별(§5.1)과 선택적 렌더링(§5.2).

판별 규칙과 임계값은 참고 구현(vectra-web `pdf-renderer.mjs`의 페이지 검사)과 동일하고,
라이브러리만 pdf.js → PyMuPDF로 바꿨다.

주의: PyMuPDF는 스레드 안전하지 않다. 이 모듈의 동기 함수는 반드시 `run_pdf()`를 통해
전용 단일 워커 스레드에서만 실행한다.
"""
from __future__ import annotations

import asyncio
import functools
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

import pymupdf

from .. import config
from .images import cap_scale

T = TypeVar("T")

_PDF_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf-worker")


async def run_pdf(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """모든 PyMuPDF 작업을 단일 워커 스레드로 직렬화한다."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_PDF_WORKER, functools.partial(func, *args, **kwargs))


class PdfError(ValueError):
    """사용자에게 보여 줄 수 있는 PDF 처리 오류."""


@dataclass
class PageAnalysis:
    page_number: int
    classification: str
    native_characters: int
    raster_images: int
    vector_operations: int
    needs_vlm: bool
    native_text: str = ""


@dataclass
class RenderedPage:
    page_number: int
    page_classification: str
    width: int
    height: int
    width_points: float
    height_points: float
    mime: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)


@dataclass
class PdfInspection:
    metadata: dict[str, str]
    total_pages: int
    processed_pages: int
    visual_pages: int
    truncated: bool
    native_text: str
    page_analysis: list[PageAnalysis] = field(default_factory=list)
    pages: list[RenderedPage] = field(default_factory=list)


# --------------------------------------------------------------------------- 판별
def classify_page(native_characters: int, raster_images: int, vector_operations: int) -> tuple[str, bool]:
    """(분류, needs_vlm)을 돌려준다. 순수 함수라 PDF 없이도 테스트할 수 있다.

    - 텍스트 객체에서 뽑힌 글자가 충분하면(usable_native) 네이티브 텍스트만 쓴다.
    - 단, 스캔 이미지 위에 글자가 조금 얹힌 경우(sparse_overlay)는 그것만으로 부족하다고 본다.
    """
    usable_native = native_characters >= config.NATIVE_MIN_CHARS
    sparse_overlay = raster_images > 0 and native_characters < config.SPARSE_OVERLAY_CHARS
    needs_vlm = (not usable_native) or sparse_overlay
    if usable_native:
        if raster_images > 0:
            classification = "mixed-needs-vision" if sparse_overlay else "mixed-native"
        else:
            classification = "native-vector"
    elif raster_images > 0:
        classification = "scanned-raster"
    elif vector_operations > 0:
        classification = "vector-outlines"
    else:
        classification = "unknown"
    return classification, needs_vlm


def count_alnum(text: str) -> int:
    """유니코드 글자(L*)와 숫자(N*)만 센다 — 한글·한자도 포함된다."""
    return sum(1 for character in text if character.isalnum())


def _normalize_native_text(text: str) -> str:
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.replace("\r", "").split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# 경로를 그리는 연산자: S s f F f* B B* b b* sh  (공백으로 구분된 독립 토큰만)
_PAINT_OPERATORS = re.compile(rb"(?:^|(?<=\s))(?:B\*?|b\*?|f\*?|F|S|s|sh)(?=\s|$)")
_STRING_LITERAL = re.compile(rb"\((?:\\.|[^\\()])*\)")
_MAX_SCANNED_STREAM_BYTES = 64 * 1024 * 1024


def _count_vector_operations(document: pymupdf.Document, page: pymupdf.Page) -> int:
    """콘텐츠 스트림(+폼 XObject)에서 경로 페인팅 연산자 수를 센다.

    `page.get_drawings()`는 경로마다 파이썬 객체를 만들어 수십만 개의 선으로 이뤄진 CAD 도면에서
    매우 느리다. 분류에는 "0보다 큰가"만 쓰이므로 스트림을 정규식으로 훑는 근사치로 충분하다.
    """
    streams: list[bytes] = []
    try:
        streams.append(page.read_contents() or b"")
    except Exception:
        pass
    try:
        for xref, *_ in document.get_page_xobjects(page.number):
            stream = document.xref_stream(xref)
            if stream:
                streams.append(stream)
    except Exception:
        pass
    total, scanned = 0, 0
    for stream in streams:
        if scanned >= _MAX_SCANNED_STREAM_BYTES:
            break
        chunk = stream[: _MAX_SCANNED_STREAM_BYTES - scanned]
        scanned += len(chunk)
        # 문자열 리터럴 안의 " f " 같은 조각을 연산자로 오인하지 않도록 먼저 지운다.
        total += len(_PAINT_OPERATORS.findall(_STRING_LITERAL.sub(b"()", chunk)))
    return total


def inspect_page(document: pymupdf.Document, page: pymupdf.Page, page_number: int) -> PageAnalysis:
    native_text = _normalize_native_text(page.get_text("text", sort=True) or "")
    native_characters = count_alnum(native_text)
    try:
        raster_images = len(page.get_image_info())
    except Exception:
        raster_images = len(page.get_images(full=True))
    vector_operations = _count_vector_operations(document, page)
    classification, needs_vlm = classify_page(native_characters, raster_images, vector_operations)
    return PageAnalysis(page_number, classification, native_characters, raster_images,
                        vector_operations, needs_vlm, native_text)


# --------------------------------------------------------------------------- 렌더링
def render_page(page: pymupdf.Page, analysis: PageAnalysis, dpi: int) -> RenderedPage:
    """페이지 전체를 한 장으로 렌더한다. 긴 변과 픽셀 수를 모두 한도 이하로 맞춘다."""
    rect = page.rect
    base_zoom = dpi / 72.0
    zoom = base_zoom * cap_scale(rect.width * base_zoom, rect.height * base_zoom)
    pixmap = None
    for _ in range(4):
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False, colorspace=pymupdf.csRGB)
        within_edge = max(pixmap.width, pixmap.height) <= config.MAX_VISION_IMAGE_EDGE
        within_pixels = pixmap.width * pixmap.height <= config.MAX_VISION_IMAGE_PIXELS
        if within_edge and within_pixels:
            break
        zoom *= 0.995  # 반올림으로 1px 넘친 경우만 해당
    assert pixmap is not None
    return RenderedPage(
        page_number=analysis.page_number,
        page_classification=analysis.classification,
        width=pixmap.width,
        height=pixmap.height,
        width_points=float(rect.width),
        height_points=float(rect.height),
        mime="image/png",
        data=pixmap.tobytes("png"),
    )


_METADATA_KEYS = (
    ("title", "Title"), ("author", "Author"), ("subject", "Subject"), ("keywords", "Keywords"),
    ("creator", "Creator"), ("producer", "Producer"), ("creationDate", "CreationDate"), ("modDate", "ModDate"),
)


def _open(data: bytes) -> pymupdf.Document:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as error:
        raise PdfError(f"PDF를 열 수 없습니다: {error}") from error
    if document.needs_pass:
        document.close()
        raise PdfError("암호로 보호된 PDF는 열 수 없습니다. 암호를 해제한 뒤 다시 올려 주세요.")
    return document


def _read_metadata(document: pymupdf.Document) -> dict[str, str]:
    info = document.metadata or {}
    return {label: str(info[key]) for key, label in _METADATA_KEYS if info.get(key)}


def render_pdf_for_vision(data: bytes, *, dpi: int | None = None, max_pages: int | None = None,
                          render_images: bool = True) -> PdfInspection:
    """허용된 모든 페이지를 검사하되, 네이티브 텍스트로 읽을 수 없는 페이지만 렌더한다(동기 — `run_pdf`로 호출)."""
    dpi = dpi or config.PDF_RENDER_DPI
    max_pages = max_pages or config.pdf_visual_page_limit()
    document = _open(data)
    try:
        page_limit = min(document.page_count, max(1, max_pages))
        analyses: list[PageAnalysis] = []
        rendered: list[RenderedPage] = []
        native_pages: list[str] = []
        # 페이지를 하나씩 순차 처리해 긴 문서·무거운 도면에서도 메모리를 일정하게 유지한다.
        for index in range(page_limit):
            page = document.load_page(index)
            try:
                analysis = inspect_page(document, page, index + 1)
            except Exception:
                # 손상된 페이지 하나 때문에 문서 전체를 버리지 않는다 → 비전으로 넘긴다.
                analysis = PageAnalysis(index + 1, "unknown", 0, 0, 0, True, "")
            analyses.append(analysis)
            if analysis.native_text:
                native_pages.append(f"[PAGE {index + 1}]\n{analysis.native_text}")
            # 네이티브 텍스트가 더 싸고 정확하다. 비전은 래스터/윤곽선 글자에만 쓴다.
            if analysis.needs_vlm and render_images:
                try:
                    rendered.append(render_page(page, analysis, dpi))
                except Exception:
                    pass
            del page
        return PdfInspection(
            metadata=_read_metadata(document),
            total_pages=document.page_count,
            processed_pages=page_limit,
            visual_pages=sum(1 for item in analyses if item.needs_vlm),
            truncated=page_limit < document.page_count,
            native_text="\n\n".join(native_pages),
            page_analysis=analyses,
            pages=rendered,
        )
    finally:
        document.close()


def render_pdf_page_image(data: bytes, *, page_number: int = 1, dpi: int | None = None) -> RenderedPage:
    """분류와 무관하게 특정 페이지(1부터)를 렌더한다 — `inspect_visual`이 임의 페이지를 볼 때 쓴다."""
    document = _open(data)
    try:
        if page_number < 1 or page_number > document.page_count:
            raise PdfError(f"{page_number}쪽은 범위를 벗어났습니다. 이 PDF는 {document.page_count}쪽입니다.")
        page = document.load_page(page_number - 1)
        analysis = inspect_page(document, page, page_number)
        return render_page(page, analysis, dpi or config.PDF_RENDER_DPI)
    finally:
        document.close()
