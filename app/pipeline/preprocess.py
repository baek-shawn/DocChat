"""업로드 → 분석 가능한 첨부 묶음으로 펼치기.

- PDF  : 네이티브 텍스트를 우선 쓰고, 검사에서 탈락한 페이지만 VLM용 이미지가 된다(§5.1, §5.2).
- 이미지: 전사 없이 그대로 VLM에 전달한다. 크기 한도만 적용한다(§5.4).
          한도에 맞춘 사본과 별개로 업로드 원본을 보관한다 — 타일 모드는 원본에서 자른다.
"""
from __future__ import annotations

import asyncio

from .. import config, trace
from ..attachments import Attachment
from .evidence import page_image_name
from .images import prepare_uploaded_image
from .pdf import PdfInspection, RenderedPage, render_pdf_for_vision, render_pdf_page_image, run_pdf

_PREPROCESS_CONCURRENCY = 2


async def preprocess_attachments(uploads: list[Attachment], *, include_visual_assets: bool = True) -> list[Attachment]:
    """소량의 업로드를 동시에 처리하되 사용자가 올린 순서를 유지한다."""
    results: list[list[Attachment]] = [[] for _ in uploads]
    semaphore = asyncio.Semaphore(_PREPROCESS_CONCURRENCY)

    async def work(index: int, upload: Attachment) -> None:
        async with semaphore:
            results[index] = await _preprocess_one(upload, include_visual_assets)

    await asyncio.gather(*(work(index, upload) for index, upload in enumerate(uploads)))
    return [item for group in results for item in group]


async def _preprocess_one(upload: Attachment, include_visual_assets: bool) -> list[Attachment]:
    async with trace.scope("preprocess", f"전처리 · {upload.name}", kind=upload.kind, mime=upload.mime,
                           size=upload.size) as span:
        if upload.is_pdf and upload.data:
            inspection = await run_pdf(render_pdf_for_vision, upload.data, render_images=include_visual_assets)
            span.set(**_describe_inspection(inspection))
            return [_describe_pdf(upload, inspection), *_page_attachments(upload.name, inspection)]
        if upload.is_image and upload.data:
            original = upload.data
            prepared = await asyncio.to_thread(prepare_uploaded_image, original, upload.mime)
            upload.kind = "image"
            # 원본은 덮어쓰지 않고 따로 보관한다(타일링 재료). 사본이 원본과 같은 바이트면 한 벌만 둔다.
            upload.source_mime = prepared.source_mime or upload.mime
            upload.source_data = original if prepared.data != original else None
            upload.data, upload.mime, upload.size = prepared.data, prepared.mime, len(prepared.data)
            upload.width, upload.height = prepared.width, prepared.height
            upload.source_width, upload.source_height = prepared.source_width, prepared.source_height
            upload.ocr_required = False
            upload.send_to_model = True
            upload.text = (f"Original image: {prepared.source_width} x {prepared.source_height} px. "
                           f"Whole-image vision input: {prepared.width} x {prepared.height} px.")
            span.set(sourceWidth=prepared.source_width, sourceHeight=prepared.source_height, width=prepared.width,
                     height=prepared.height, resized=prepared.resized, sourceMime=prepared.source_mime,
                     limits={"maxEdge": config.MAX_VISION_IMAGE_EDGE, "maxPixels": config.MAX_VISION_IMAGE_PIXELS})
            return [upload]
        span.set(skipped="PDF도 이미지도 아니거나 내용이 비어 있음")
        return [upload]


def _describe_inspection(inspection: PdfInspection) -> dict:
    """쪽마다 어떤 판별을 내렸는지 — 임계값과 함께 적어야 "왜 전사로 갔나"를 나중에 알 수 있다."""
    rendered = {page.page_number: page for page in inspection.pages}
    return {
        "totalPages": inspection.total_pages, "processedPages": inspection.processed_pages,
        "visualPages": inspection.visual_pages, "truncated": inspection.truncated,
        "nativeChars": len(inspection.native_text), "metadata": inspection.metadata, "renderDpi": config.PDF_RENDER_DPI,
        "thresholds": {"nativeMinChars": config.NATIVE_MIN_CHARS, "sparseOverlayChars": config.SPARSE_OVERLAY_CHARS},
        "pages": [{
            "page": page.page_number, "classification": page.classification, "nativeCharacters": page.native_characters,
            "rasterImages": page.raster_images, "vectorOperations": page.vector_operations, "needsVlm": page.needs_vlm,
            "rendered": ({"width": rendered[page.page_number].width, "height": rendered[page.page_number].height}
                         if page.page_number in rendered else None),
        } for page in inspection.page_analysis],
    }


def _describe_pdf(upload: Attachment, inspection: PdfInspection) -> Attachment:
    page_summary = "\n".join(
        f"Page {page.page_number}: {page.classification}; native characters={page.native_characters}; "
        f"raster images={page.raster_images}; vector operations={page.vector_operations}; "
        f"vision OCR={'required' if page.needs_vlm else 'skipped'}"
        for page in inspection.page_analysis
    )
    metadata = [f"{key}: {value}" for key, value in inspection.metadata.items()]
    metadata += [
        f"Pages: {inspection.total_pages}",
        f"Inspected pages: {inspection.processed_pages}/{inspection.total_pages}",
        f"Pages requiring vision OCR: {inspection.visual_pages}",
    ]
    if inspection.truncated:
        metadata.append(f"Warning: inspection limited to the first {inspection.processed_pages} pages.")
    body = inspection.native_text[: config.MAX_DOCUMENT_TEXT_CHARS]
    upload.kind = "pdf"
    upload.text = (f"[PDF DOCUMENT METADATA]\n{chr(10).join(metadata)}\n\n"
                   f"[PAGE ANALYSIS]\n{page_summary}\n\n{body}").strip()
    upload.visual_pages = inspection.visual_pages
    upload.total_pages = inspection.total_pages
    return upload


def _page_attachments(root: str, inspection: PdfInspection) -> list[Attachment]:
    return [_page_attachment(root, page, ocr_required=True) for page in inspection.pages]


def _page_attachment(root: str, page: RenderedPage, *, ocr_required: bool) -> Attachment:
    # 쪽 이미지는 업로드 이미지가 아니므로 send_to_model=False — 답변 호출에 실을지는 요청의 답변 이미지 모드가 정한다.
    return Attachment(
        name=page_image_name(root, page.page_number),
        mime=page.mime,
        kind="image",
        size=page.size,
        data=page.data,
        has_data=True,
        width=page.width,
        height=page.height,
        page_number=page.page_number,
        page_classification=page.page_classification,
        ocr_required=ocr_required,
        send_to_model=False,
        text=(f"Page classification: {page.page_classification}. "
              f"Page dimensions: {page.width_points:.2f} x {page.height_points:.2f} points; "
              f"normalized whole-page image: {page.width} x {page.height} px."),
    )


async def render_page_attachment(pdf: Attachment, page_number: int, *, why: str, trace_kind: str = "preprocess") -> Attachment:
    """전처리 때 렌더하지 않은 쪽(네이티브 글자가 충분한 쪽)을 원본 PDF에서 지금 그려 첨부로 만든다.

    bbox 도구가 임의 쪽을 볼 때(`inspect_visual`)와 답변 호출에 쪽 이미지를 실을 때(Step 8 전체 모드)가 같이 쓴다.
    전사 대상은 아니다(ocr_required=False) — 네이티브 글자가 충분해서 전사를 건너뛴 쪽이다. `pdf.data`가 있어야 한다.
    """
    rendered = await run_pdf(render_pdf_page_image, pdf.data, page_number=page_number, dpi=config.PDF_RENDER_DPI)
    trace.note(trace_kind, f"{page_image_name(pdf.name, page_number)}을(를) 지금 렌더 · {why}", width=rendered.width,
               height=rendered.height, dpi=config.PDF_RENDER_DPI, classification=rendered.page_classification)
    return _page_attachment(pdf.name, rendered, ocr_required=False)
