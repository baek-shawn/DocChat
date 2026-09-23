"""온디맨드 도구.

- inspect_visual     : bbox 전용 **별도 호출**. 이미지 한 장만 떼어 grounding 프롬프트로 다시 묻는다(§6).
- read_attachment    : 프롬프트 예산 때문에 잘린 문서의 전체 텍스트를 구간별로 읽는다.
- search_attachments : 긴 문서에서 값의 위치를 찾는다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import config
from ..attachments import Attachment
from ..db import ChatStore
from ..pipeline.evidence import attachment_root_name, page_image_name
from ..pipeline.images import assemble_model_images
from ..pipeline.pdf import PdfError, render_pdf_page_image, run_pdf
from ..providers.base import Provider, ToolCall, ToolSpec
from .grounding import VisualInspection, map_box_to_source, parse_visual_inspection, valid_box
from .prompts import GROUNDING_RETRY_NOTE, GROUNDING_SYSTEM_PROMPT, grounding_instruction


class ToolError(Exception):
    """모델에게 되돌려 줄 도구 오류(모델이 인자를 고쳐 다시 부를 수 있게 한다)."""


INSPECT_VISUAL = ToolSpec(
    name="inspect_visual",
    description=(
        "Inspect an uploaded image, or one page of an uploaded PDF, and measure where things are. Use it to mark a "
        "location, detect objects, or verify tables, dimensions, stamps, signatures or diagrams. The tool measures the "
        "bounding boxes itself and shows them in the viewer; never estimate boxes yourself."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Exact attachment name from the attachment manifest."},
            "page": {"type": "integer", "minimum": 1, "description": "1-based page number when the attachment is a PDF."},
            "task": {"type": "string", "description": "What to detect, transcribe or verify visually."},
        },
        "required": ["name", "task"],
    },
)

READ_ATTACHMENT = ToolSpec(
    name="read_attachment",
    description="Read a bounded chunk of the parsed text of an uploaded file by its exact name.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Exact attachment name from the attachment manifest."},
            "start": {"type": "integer", "minimum": 0, "description": "Character offset, default 0."},
            "maxChars": {"type": "integer", "minimum": 1000, "maximum": 50000, "description": "Maximum characters to return."},
        },
        "required": ["name"],
    },
)

SEARCH_ATTACHMENTS = ToolSpec(
    name="search_attachments",
    description="Search all uploaded documents at once and return short excerpts with source names and offsets.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Text, item number, drawing number, heading or phrase to find."},
            "names": {"type": "array", "items": {"type": "string"}, "description": "Optional attachment names to limit the search."},
            "maxResults": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "required": ["query"],
    },
)


@dataclass
class ToolContext:
    provider: Provider
    attachments: list[Attachment]
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    store: ChatStore | None = None
    conversation_id: str = ""
    default_read_chars: int = 16_000
    on_progress: Callable[[str], None] = lambda _message: None


def available_tools(attachments: list[Attachment]) -> list[ToolSpec]:
    """첨부가 없으면 도구도 없다(순수 대화). 볼 수 있는 면이 있을 때만 inspect_visual을 내놓는다."""
    tools: list[ToolSpec] = []
    if any(item.is_image or item.is_pdf for item in attachments):
        tools.append(INSPECT_VISUAL)
    if any(item.text and not item.is_image for item in attachments):
        tools += [READ_ATTACHMENT, SEARCH_ATTACHMENTS]
    return tools


def describe_tool_call(call: ToolCall) -> str:
    if call.name == "inspect_visual":
        page = f" {call.arguments.get('page')}쪽" if call.arguments.get("page") else ""
        return f"이미지에서 위치를 확인하는 중… ({call.arguments.get('name', '')}{page})"
    if call.name == "read_attachment":
        return f"문서 본문을 읽는 중… ({call.arguments.get('name', '')})"
    if call.name == "search_attachments":
        return f"문서에서 '{call.arguments.get('query', '')}' 검색 중…"
    return f"도구 실행 중… ({call.name})"


async def execute_tool(context: ToolContext, call: ToolCall) -> str:
    """도구 하나를 실행해 모델에게 돌려줄 문자열을 만든다. 오류도 문자열로 돌려준다."""
    handlers = {
        "inspect_visual": _inspect_visual,
        "read_attachment": _read_attachment,
        "search_attachments": _search_attachments,
    }
    handler = handlers.get(call.name)
    if handler is None:
        return f"ERROR: unknown tool \"{call.name}\". Available tools: {', '.join(handlers)}."
    try:
        return await handler(context, call.arguments or {})
    except (ToolError, PdfError) as error:
        return f"ERROR: {error}"


# --------------------------------------------------------------------------- 공통
def _find(attachments: list[Attachment], name: str) -> Attachment | None:
    """정확 일치 → 대소문자 무시 → 확장자를 뺀 이름(유일할 때만).

    gemma3 실측: 매니페스트에 "sheet_with_stamp.png"라고 적혀 있어도 `"name": "sheet_with_stamp"`로 넘긴다.
    후보가 둘 이상이면(a.pdf와 a.png) 추측하지 않고 실패시켜 모델이 정확한 이름을 다시 고르게 한다.
    """
    wanted = str(name or "").strip().strip("\"'")
    exact = (next((item for item in attachments if item.name == wanted), None)
             or next((item for item in attachments if item.name.lower() == wanted.lower()), None))
    if exact is not None or not wanted:
        return exact
    stem_matches = [item for item in attachments if item.name.rpartition(".")[0].lower() == wanted.lower()]
    return stem_matches[0] if len(stem_matches) == 1 else None


def _known_names(attachments: list[Attachment]) -> str:
    return ", ".join(f'"{item.name}"' for item in attachments) or "(none)"


async def _bytes_of(context: ToolContext, attachment: Attachment) -> bytes | None:
    if attachment.data is None and attachment.id is not None and context.store is not None:
        attachment.data = await context.store.load_attachment_data(attachment.id)
    return attachment.data


def _upsert_artifact(artifacts: list[dict[str, Any]], artifact: dict[str, Any]) -> None:
    for index, existing in enumerate(artifacts):
        if existing.get("name") == artifact["name"]:
            artifacts[index] = artifact
            return
    artifacts.append(artifact)


# --------------------------------------------------------------------------- inspect_visual
async def _resolve_visual_surface(context: ToolContext, record: Attachment, page: int | None) -> Attachment:
    """대상 이미지 **한 장**을 떼어 낸다. OCR 단계에서 보관해 둔 표시용 바이트를 재사용한다."""
    if record.is_image and await _bytes_of(context, record):
        return record
    root = attachment_root_name(record.name)
    pdf = next((item for item in context.attachments if item.name == root and item.is_pdf), None)
    if pdf is None or not await _bytes_of(context, pdf):
        raise ToolError(f'"{record.name}" has no renderable visual surface. Choose an image or a PDF page.')
    page_number = page or record.page_number or 1
    existing = _find(context.attachments, page_image_name(pdf.name, page_number))
    if existing is not None and await _bytes_of(context, existing):
        return existing
    # 네이티브 텍스트로 충분해 미리 렌더하지 않았던 페이지도 요청이 오면 원본 PDF에서 바로 그린다.
    rendered = await run_pdf(render_pdf_page_image, pdf.data, page_number=page_number, dpi=config.PDF_RENDER_DPI)
    surface = Attachment(
        name=page_image_name(pdf.name, page_number), mime=rendered.mime, kind="image", size=rendered.size,
        data=rendered.data, has_data=True, width=rendered.width, height=rendered.height,
        page_number=page_number, page_classification=rendered.page_classification,
        ocr_required=False, send_to_model=False,
        text=f"Rendered on demand for visual inspection: {rendered.width} x {rendered.height} px.",
    )
    context.attachments.append(surface)
    if context.store is not None and context.conversation_id:
        await context.store.save_attachments(context.conversation_id, context.attachments)
    return surface


async def _inspect_visual(context: ToolContext, arguments: dict[str, Any]) -> str:
    name, task = str(arguments.get("name") or ""), str(arguments.get("task") or "").strip()[:500]
    if not task:
        raise ToolError("inspect_visual needs a non-empty \"task\".")
    record = _find(context.attachments, name)
    if record is None:
        raise ToolError(f'No attachment named "{name}". Exact names: {_known_names(context.attachments)}.')
    page = arguments.get("page")
    page_number = int(page) if isinstance(page, (int, float)) and page >= 1 else None
    surface = await _resolve_visual_surface(context, record, page_number)

    inspection = VisualInspection()
    boxes: list[dict[str, Any]] = []
    attempts = 1 + config.GROUNDING_RETRY_COUNT
    for attempt in range(attempts):
        instruction = grounding_instruction(task, surface.name)
        if attempt:
            instruction = f"{instruction}\n{GROUNDING_RETRY_NOTE}"
        boxes, texts, structured = [], [], True
        for image in assemble_model_images(surface.name, surface.mime, surface.data or b"", purpose="grounding"):
            response = await context.provider.analyze(
                [{"role": "system", "content": GROUNDING_SYSTEM_PROMPT}, {"role": "user", "content": instruction}],
                images=[image], temperature=0.0,
            )
            part = parse_visual_inspection(response.text, image_width=surface.width, image_height=surface.height)
            structured = structured and part.structured
            texts.append(part.text)
            boxes += [map_box_to_source(box, image.source_box) for box in part.boxes]
        inspection = VisualInspection(text="\n".join(text for text in texts if text), boxes=boxes, structured=structured)
        if inspection.structured:
            break

    regions = [box for box in inspection.boxes if valid_box(box)]
    artifact: dict[str, Any] = {
        "name": surface.name, "mime": surface.mime, "view": "image",
        "title": f"시각 검사 · {surface.name}", "task": task, "text": inspection.text[:20_000],
    }
    if surface.id is not None:
        artifact["attachmentId"] = surface.id
    if regions:
        artifact["boxes"] = regions
    _upsert_artifact(context.artifacts, artifact)
    result: dict[str, Any] = {"source": surface.name, "text": inspection.text, "regions": regions}
    if not inspection.structured:
        result["warning"] = "The vision model did not return structured regions; no boxes could be measured."
    return json.dumps(result, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- 텍스트 도구
async def _read_attachment(context: ToolContext, arguments: dict[str, Any]) -> str:
    record = _find(context.attachments, str(arguments.get("name") or ""))
    if record is None:
        raise ToolError(f'Attachment not found. Exact names: {_known_names(context.attachments)}.')
    text = record.text or ""
    if not text:
        return f"No extracted text is available for {record.name}. Use inspect_visual to look at it."
    start = arguments.get("start")
    offset = min(int(start), len(text)) if isinstance(start, (int, float)) and start >= 0 else 0
    limit = arguments.get("maxChars")
    limit = max(1000, min(50_000, int(limit))) if isinstance(limit, (int, float)) else context.default_read_chars
    content = text[offset: offset + limit]
    return json.dumps({
        "name": record.name, "start": offset, "end": offset + len(content), "totalCharacters": len(text),
        "hasMore": offset + len(content) < len(text), "content": content,
    }, ensure_ascii=False)


async def _search_attachments(context: ToolContext, arguments: dict[str, Any]) -> str:
    query = str(arguments.get("query") or "").strip()
    if len(query) < 2:
        raise ToolError("search_attachments needs a query of at least 2 characters.")
    names = arguments.get("names")
    allowed = {str(item) for item in names} if isinstance(names, list) else set()
    limit = arguments.get("maxResults")
    limit = max(1, min(50, int(limit))) if isinstance(limit, (int, float)) else 12
    needle = query.lower()
    results: list[dict[str, Any]] = []
    for record in context.attachments:
        if record.is_image or (allowed and record.name not in allowed):
            continue
        text = record.text or ""
        lowered, cursor = text.lower(), 0
        while len(results) < limit:
            offset = lowered.find(needle, cursor)
            if offset < 0:
                break
            excerpt = " ".join(text[max(0, offset - 240): offset + len(needle) + 520].split())
            results.append({"name": record.name, "offset": offset, "excerpt": excerpt})
            cursor = offset + max(len(needle), 1)
        if len(results) >= limit:
            break
    return json.dumps({"query": query, "results": results, "searchedFiles": len(context.attachments)}, ensure_ascii=False)
