"""업로드 직후 미리 검사 — 첨부 칩에 "N쪽 비전 OCR 필요 / 네이티브 텍스트 n자"를 보여 주기 위한 용도."""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..attachments import UploadError, sanitize_uploads
from ..pipeline.images import ImageError, prepare_uploaded_image
from ..pipeline.pdf import PdfError, render_pdf_for_vision, run_pdf

router = APIRouter(prefix="/api", tags=["files"])


class InspectBody(BaseModel):
    attachment: dict[str, Any]


@router.post("/attachments/inspect")
async def inspect_attachment(body: InspectBody) -> Any:
    try:
        uploads = sanitize_uploads([body.attachment])
        if not uploads:
            raise UploadError("검사할 첨부가 없습니다.")
        upload = uploads[0]
        if upload.is_pdf:
            # 판별만 한다. 이미지는 렌더하지 않는다(render_images=False).
            inspection = await run_pdf(render_pdf_for_vision, upload.data, render_images=False)
            return {
                "name": upload.name, "kind": "pdf", "mime": upload.mime,
                "parsedCharacters": len(inspection.native_text),
                "totalPages": inspection.total_pages,
                "processedPages": inspection.processed_pages,
                "visualPages": inspection.visual_pages,
                "truncated": inspection.truncated,
                "pages": [
                    {"pageNumber": page.page_number, "classification": page.classification,
                     "needsVlm": page.needs_vlm, "nativeCharacters": page.native_characters,
                     "rasterImages": page.raster_images, "vectorOperations": page.vector_operations}
                    for page in inspection.page_analysis
                ],
            }
        prepared = await asyncio.to_thread(prepare_uploaded_image, upload.data, upload.mime)
        return {
            "name": upload.name, "kind": "image", "mime": prepared.mime, "parsedCharacters": 0,
            "width": prepared.width, "height": prepared.height,
            "sourceWidth": prepared.source_width, "sourceHeight": prepared.source_height,
            "resized": prepared.resized, "visualPages": 0,
        }
    except (UploadError, PdfError, ImageError) as error:
        return JSONResponse({"error": str(error)}, status_code=400)
