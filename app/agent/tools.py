"""온디맨드 도구.

- inspect_visual     : bbox 전용 **별도 호출**. 이미지 한 장만 떼어 grounding 프롬프트로 다시 묻는다(§6).
- read_attachment    : 프롬프트 예산 때문에 잘린 문서의 전체 텍스트를 구간별로 읽는다.
- search_attachments : 긴 문서에서 값의 위치를 찾는다.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .. import config, trace
from ..attachments import Attachment
from ..db import ChatStore
from ..pipeline.evidence import attachment_root_name, page_image_name
from ..pipeline.images import ImageError, ModelImage, TileSource, VisionUsage, assemble_model_images
from ..pipeline.pdf import PdfError
from ..pipeline.preprocess import render_page_attachment
from ..providers.base import Provider, ToolCall, ToolSpec, is_output_length_stop
from .grounding import (VisualInspection, map_box_to_source, merge_tile_boxes, parse_visual_inspection,
                        valid_box)
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
    # 이미지 처리 방식(요청마다 고른다). "tile"이면 bbox 호출을 타일마다 따로 보낸다.
    image_mode: str = "whole"
    usage: VisionUsage = field(default_factory=VisionUsage)
    # 트레이스를 켰을 때 모델에 보낸 타일을 파일로 남기는 자리
    on_tiles: Callable[[str, str, list[ModelImage]], Awaitable[None]] | None = None
    # bbox 호출의 추론을 끈다(요청마다 고른다, Step 6-0). 출력 상한은 `config.VISION_MAX_TOKENS`.
    disable_thinking: bool = False


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
    except (ToolError, PdfError, ImageError) as error:
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
    surface = await render_page_attachment(pdf, page_number, why="inspect_visual 요청", trace_kind="tool")
    context.attachments.append(surface)
    if context.store is not None and context.conversation_id:
        await context.store.save_attachments(context.conversation_id, context.attachments)
    turn = trace.current()
    if turn is not None:        # 지금 렌더한 쪽도 트레이스가 첨부 ID·크기로 가리킬 수 있게
        turn.register_attachments(context.attachments)
    return surface


async def _tile_source(context: ToolContext, surface: Attachment) -> TileSource | None:
    """타일을 잘라 낼 고해상도 원본. PDF 쪽이면 원본 PDF, 업로드 이미지면 보관해 둔 원본(Step 5-0)."""
    root = attachment_root_name(surface.name)
    if root != surface.name:
        pdf = next((item for item in context.attachments if item.name == root and item.is_pdf), None)
        if pdf is not None and await _bytes_of(context, pdf):
            return TileSource(kind="pdf", data=pdf.data or b"", page_number=surface.page_number or 1)
        return None
    if surface.source_data is not None:
        return TileSource(kind="image", data=surface.source_data, mime=surface.source_mime or surface.mime)
    if surface.id is not None and context.store is not None:
        original = await context.store.load_attachment_source(surface.id)
        if original is not None:
            return TileSource(kind="image", data=original[0], mime=original[1])
    return TileSource(kind="image", data=surface.data, mime=surface.mime) if surface.data else None


def _limit_words() -> str:
    limit = config.vision_max_tokens()
    return f"the output limit of {limit} tokens" if limit else "its output limit"


async def _ground(context: ToolContext, surface: Attachment, image: ModelImage, task: str,
                  limiter: asyncio.Semaphore) -> VisualInspection:
    """이미지 한 장(전체 또는 타일)에 대한 grounding 호출. 구조화 JSON이 아니면 정해진 횟수만큼 다시 묻는다.

    출력 상한에 닿아 끊긴 호출은 다시 묻지 않는다 — 다시 물으면 상한만큼의 시간이 또 든다(예전에는 타일마다 3번).
    끊긴 글은 추론이거나 미완성 JSON이라 결과로도 쓰지 않는다.
    어느 타일이 끊기는지는 실행마다 달랐다(Qwen3.5 실측) — 다시 물으면 될 수도 있지만 시간을 보장할 수 없다.
    """
    part = VisualInspection()
    for attempt in range(1 + config.GROUNDING_RETRY_COUNT):
        instruction = grounding_instruction(task, surface.name, tile=image.tile is not None)
        if attempt:
            instruction = f"{instruction}\n{GROUNDING_RETRY_NOTE}"
        async with limiter:
            context.usage.grounding_calls += 1
            response = await context.provider.analyze(
                [{"role": "system", "content": GROUNDING_SYSTEM_PROMPT}, {"role": "user", "content": instruction}],
                images=[image], temperature=0.0, disable_thinking=context.disable_thinking,
                max_tokens=config.vision_max_tokens(),
            )
        if is_output_length_stop(response.finish_reason):
            context.usage.grounding_length_stops += 1
            trace.note("tool", f"출력 상한에서 끊김 → 다시 묻지 않음 · {image.name}", finishReason=response.finish_reason)
            return VisualInspection(cut_off=True)
        # 픽셀 좌표로 답하는 모델을 위해 **모델이 실제로 본 이미지**의 크기를 넘긴다(타일이면 타일 크기).
        part = parse_visual_inspection(response.text, image_width=image.width or surface.width,
                                       image_height=image.height or surface.height)
        if part.structured:
            break
        if attempt < config.GROUNDING_RETRY_COUNT:
            trace.note("tool", f"구조화 JSON이 아니어서 다시 묻는 중 ({attempt + 2}/{1 + config.GROUNDING_RETRY_COUNT}) "
                               f"· {image.name}", text=trace.clip(response.text))
    trace.note("tool", f"위치 확인 결과 · {image.name}", structured=part.structured, boxes=len(part.boxes),
               text=trace.clip(part.text), regions=[dict(box) for box in part.boxes[:50]])
    # 타일 기준 좌표 → 전체 기준 좌표. 전체 이미지(0,0,1,1)면 그대로다.
    return VisualInspection(text=part.text, structured=part.structured,
                            boxes=[map_box_to_source(box, image.source_box) for box in part.boxes])


async def _inspect_tiles(context: ToolContext, surface: Attachment, images: list[ModelImage],
                         task: str) -> tuple[VisualInspection, str]:
    """타일마다 따로 물은 뒤 전체 좌표에서 합친다. (합친 결과, 일부 실패 시 모델에게 알릴 경고)"""
    context.usage.count_images(images)
    if context.on_tiles is not None:
        await context.on_tiles(surface.name, "grounding", images)
    limiter = asyncio.Semaphore(config.OCR_CONCURRENCY)
    finished = 0

    async def one(image: ModelImage) -> VisualInspection:
        nonlocal finished
        part = await _ground(context, surface, image, task, limiter)
        finished += 1
        context.on_progress(f"타일에서 위치를 확인하는 중… {surface.name} ({finished}/{len(images)})")
        return part

    outcomes = await asyncio.gather(*(one(image) for image in images), return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, asyncio.CancelledError):
            raise outcome
    errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    parts = [outcome for outcome in outcomes if isinstance(outcome, VisualInspection)]
    if not parts:
        raise errors[0]      # 한 타일도 성공하지 못했다 → 전체 모드에서 호출이 실패한 것과 같게 처리한다
    structured = [part for part in parts if part.structured]
    boxes = merge_tile_boxes([part.boxes for part in structured])[: config.MAX_GROUNDING_REGIONS]
    # 소견은 대상을 찾은 타일의 것만 싣는다. "이 타일에는 없다"가 타일 수만큼 쌓이면 잡음이다.
    notes = list(dict.fromkeys(part.text.strip() for part in structured if part.boxes and part.text.strip()))
    if not structured:
        notes = list(dict.fromkeys(part.text.strip() for part in parts if part.text.strip()))[:3]
    unreadable = len(images) - len(structured)
    cut = sum(1 for part in parts if part.cut_off)
    warning = ""
    if structured and unreadable:
        warning = (f"{unreadable} of {len(images)} tiles did not return structured regions, "
                   "so targets inside those tiles may be missing.")
    elif cut:
        warning = "The vision model did not return structured regions for any tile; no boxes could be measured."
    if cut:
        warning += (f" {cut} of {len(images)} tiles stopped at {_limit_words()} before answering and were not retried.")
    return VisualInspection(text="\n".join(notes), boxes=boxes, structured=bool(structured),
                            cut_off=bool(cut) and not structured), warning


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

    source = await _tile_source(context, surface) if context.image_mode == "tile" else None
    images = await assemble_model_images(surface.name, surface.mime, surface.data or b"", purpose="grounding",
                                         mode=context.image_mode, source=source)
    grid = images[0].grid
    trace.note("tool", (f"{surface.name}을(를) 타일 {len(images)}장으로 나눠 확인" if grid is not None
                        else f"{surface.name} 전체 한 장으로 확인"),
               surface=surface.name, attachmentId=surface.id, width=surface.width, height=surface.height,
               imageMode=context.image_mode, task=task,
               grid={"rows": grid.rows, "cols": grid.cols, "blank": grid.blank, "width": grid.width,
                     "height": grid.height, "dpi": grid.dpi} if grid is not None else None)
    warning = ""
    if grid is None:
        inspection = await _ground(context, surface, images[0], task, asyncio.Semaphore(1))
        if inspection.cut_off:
            warning = (f"The vision model stopped at {_limit_words()} before answering, so no boxes could be measured. "
                       "The call was not retried.")
    else:
        inspection, warning = await _inspect_tiles(context, surface, images, task)

    regions = [box for box in inspection.boxes if valid_box(box)]
    trace.note("tool", f"inspect_visual 결과 · 영역 {len(regions)}개" + (" · 경고 있음" if warning else ""),
               structured=inspection.structured, cutOff=inspection.cut_off, boxes=len(regions), warning=warning or None,
               text=trace.clip(inspection.text))
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
        result["warning"] = warning or "The vision model did not return structured regions; no boxes could be measured."
    elif warning:
        result["warning"] = warning
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
