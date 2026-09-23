"""업로드 → 분석 가능한 첨부 묶음으로 펼치기.

- PDF  : 네이티브 텍스트를 우선 쓰고, 검사에서 탈락한 페이지만 VLM용 이미지가 된다(§5.1, §5.2).
- 이미지: 전사 없이 그대로 VLM에 전달한다. 크기 한도만 적용한다(§5.4).
"""
from __future__ import annotations

import asyncio

from .. import config
from ..attachments import Attachment
from .evidence import page_image_name
from .images import prepare_uploaded_image
from .pdf import PdfInspection, render_pdf_for_vision, run_pdf

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
    if upload.is_pdf and upload.data:
        inspection = await run_pdf(render_pdf_for_vision, upload.data, render_images=include_visual_assets)
        return [_describe_pdf(upload, inspection), *_page_attachments(upload.name, inspection)]
    if upload.is_image and upload.data:
        prepared = await asyncio.to_thread(prepare_uploaded_image, upload.data, upload.mime)
        upload.kind = "image"
        upload.data, upload.mime, upload.size = prepared.data, prepared.mime, len(prepared.data)
        upload.width, upload.height = prepared.width, prepared.height
        upload.source_width, upload.source_height = prepared.source_width, prepared.source_height
        upload.ocr_required = False
        upload.send_to_model = True
        upload.text = (f"Original image: {prepared.source_width} x {prepared.source_height} px. "
                       f"Whole-image vision input: {prepared.width} x {prepared.height} px.")
        return [upload]
    return [upload]


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
    return [
        Attachment(
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
            ocr_required=True,
            send_to_model=False,
            text=(f"Page classification: {page.page_classification}. "
                  f"Page dimensions: {page.width_points:.2f} x {page.height_points:.2f} points; "
                  f"normalized whole-page image: {page.width} x {page.height} px."),
        )
        for page in inspection.pages
    ]
