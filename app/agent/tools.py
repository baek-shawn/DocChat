"""온디맨드 도구.

- inspect_visual     : bbox 전용 **별도 호출**. 이미지 한 장만 떼어 grounding 프롬프트로 다시 묻는다(§6).
                       사용자가 보여 달라고 할 때만(Step 10의 역할 분담).
- view_page          : 보기 도구(Step 10). 모델이 답을 내려면 그림을 봐야 할 때 쪽(또는 업로드 이미지) **한 장**을 골라
                       **답변 모델 자신의 다음 호출부터** 붙인다(같이 보기). 턴 안에서 쌓이고 상한(`config.MAX_VIEWED_PAGES`)을
                       넘으면 붙이지 않고 알려만 준다. 답변 이미지 모드 자동에서만 제공.
- analyze_pages      : 따로 보기 도구(Step 10 2차). 쪽(하나 또는 범위)마다 **별도 VLM 호출**로 모델이 넘긴 질문에 답하게 하고
                       그 **글**을 돌려준다 — 답변 모델은 이미지를 보지 않는다. 쪽마다 독립인 질문과 후보가 많아 훑어야 하는
                       질문용. 한 호출의 쪽 수(`config.ANALYZE_PAGES_PER_CALL`)와 한 턴의 총 쪽 수(`config.MAX_ANALYZED_PAGES`,
                       시간 상한)를 따로 둔다. 자동 모드에서 요청 옵션으로 끄고 켠다. 같이 보기 / 따로 보기는 모델이 고른다.
- read_attachment    : 프롬프트 예산 때문에 잘린 문서의 전체 텍스트를 구간별로 읽는다.
- search_attachments : 긴 문서에서 값의 위치를 찾는다.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .. import config, trace
from ..attachments import Attachment
from ..db import ChatStore
from ..pipeline.evidence import attachment_root_name, page_image_name, page_ranges
from ..pipeline.images import ImageError, ModelImage, TileSource, VisionUsage, assemble_model_images
from ..pipeline.ocr import clean_transcription, cut_off_answer
from ..pipeline.pdf import PdfError
from ..pipeline.preprocess import render_page_attachment
from ..providers.base import (Provider, ReasoningEffortError, ToolCall, ToolSpec, is_output_length_stop,
                              is_reasoning_runaway)
from ..providers.reasoning import describe_reasoning_progress
from .grounding import (VisualInspection, map_box_to_source, merge_tile_boxes, parse_visual_inspection,
                        valid_box)
from .prompts import (GROUNDING_RETRY_NOTE, GROUNDING_SYSTEM_PROMPT, PAGE_ANALYSIS_SYSTEM_PROMPT, analyze_pages_limit_reached,
                      analyze_pages_result, grounding_instruction, page_analysis_instruction, view_page_already_attached,
                      view_page_limit_reached, view_page_result)


class ToolError(Exception):
    """모델에게 되돌려 줄 도구 오류(모델이 인자를 고쳐 다시 부를 수 있게 한다)."""


# Step 10: 설명을 "보여 달라는 요청"으로 좁혔다(표·치수·도장·다이어그램 확인 용도는 뺐다 — 그 용도는 view_page).
INSPECT_VISUAL = ToolSpec(
    name="inspect_visual",
    description=(
        "Measure bounding boxes on an uploaded image, or on one page of an uploaded PDF, and show them in the viewer. "
        "Use it only when the user asks to show, mark, highlight, box or visualize where something is. It does not "
        "describe or analyze the picture; never estimate boxes yourself."
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

VIEW_PAGE = ToolSpec(
    name="view_page",
    description=(
        "Attach the image of one page of an uploaded PDF (or an uploaded image) to your own next call so that you can "
        "look at it. Use it when the answer depends on what is drawn - shapes, positions, counts, which feature a "
        "dimension belongs to, or comparing pages - and the text cannot tell you. One page per call; the pages you "
        "request stay attached for this turn, so call it again (or several times in one reply) to see more pages."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Exact attachment name from the attachment manifest."},
            "page": {"type": "integer", "minimum": 1, "description": "1-based page number when the attachment is a PDF."},
        },
        "required": ["name"],
    },
)

ANALYZE_PAGES = ToolSpec(
    name="analyze_pages",
    description=(
        "Analyze pages of an uploaded PDF (or an uploaded image) one by one in separate vision calls and get a text answer "
        "per page; the pages are not attached to your own call. Pass a concrete question - what to look for on each page "
        "and what to report. Use it for questions each page can answer on its own, or to go through many pages: pass a "
        f'range such as "1-10" (up to {config.ANALYZE_PAGES_PER_CALL} pages per call) and call it again for the next range '
        "until every page is covered. To compare pages side by side, use view_page instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Exact attachment name from the attachment manifest."},
            "pages": {"type": "string", "description": '1-based page number, range or list: "7", "1-10", "2,5,7-9". '
                                                       "Omit for an uploaded image or a one-page PDF."},
            "question": {"type": "string", "description": "What to look for on each page and what to report, written concretely."},
        },
        "required": ["name", "question"],
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
    # 생성 중의 추론 진행("추론 중… n토큰")처럼 같은 줄을 갱신해 보여 줄 문구(Step 6). 단계 알림(on_progress)과 다르다.
    on_live: Callable[[str], None] | None = None
    # 추론을 켠 bbox 호출에 실을 추론 수준(Step 6 2차, 요청마다 고른다). 비어 있으면 보내지 않는다.
    reasoning_effort: str = ""
    # 보기 도구(Step 10): 이 턴에서 모델이 요청해 모은 쪽 이미지(순서대로). 루프가 다음 답변 호출부터 모두 싣는다.
    viewed: list[ModelImage] = field(default_factory=list)
    view_refusals: int = 0                    # 상한에 걸려 붙이지 못한 요청 수(답변 메타에 적는다)
    # 처음부터 답변 호출에 실려 있는 이미지 이름(업로드 이미지). 같은 것을 다시 요청하면 "이미 붙어 있다"고 알려 준다.
    base_image_names: list[str] = field(default_factory=list)
    # 따로 보기 도구(Step 10 2차): 이 턴에서 쪽마다 별도 호출로 본 쪽 이름(요청 순서, 중복 없음), 도구 호출 수, 턴 상한에 센
    # 쪽 호출 수(같은 쪽을 다른 질문으로 다시 보면 또 센다 — 시간 상한이다), 상한에 걸려 보지 못한 도구 호출 수.
    analyzed: list[str] = field(default_factory=list)
    analysis_calls: int = 0
    analyzed_pages: int = 0
    analysis_refusals: int = 0
    # (쪽 이름, 질문) → 답. 같은 턴 안에서 같은 쪽을 같은 질문으로 다시 부르면 호출하지 않고 이것을 돌려준다.
    analysis_cache: dict[tuple[str, str], str] = field(default_factory=dict)

    def viewed_names(self) -> list[str]:
        return [image.name for image in self.viewed]

    def attached_image_names(self) -> list[str]:
        """지금 답변 호출에 실리는 이미지 이름 — 처음부터 실린 것 + 보기 도구로 모은 것(이 순서로 붙는다)."""
        return [*self.base_image_names, *self.viewed_names()]


def available_tools(attachments: list[Attachment], *, view_tool: bool = False, analyze_tool: bool = False) -> list[ToolSpec]:
    """첨부가 없으면 도구도 없다(순수 대화). 볼 수 있는 면이 있을 때만 inspect_visual(과, 자동 모드면 view_page,
    따로 보기를 켰으면 analyze_pages)을 내놓는다."""
    tools: list[ToolSpec] = []
    if any(item.is_image or item.is_pdf for item in attachments):
        if view_tool:
            tools.append(VIEW_PAGE)
        if analyze_tool:
            tools.append(ANALYZE_PAGES)
        tools.append(INSPECT_VISUAL)
    if any(item.text and not item.is_image for item in attachments):
        tools += [READ_ATTACHMENT, SEARCH_ATTACHMENTS]
    return tools


def describe_tool_call(call: ToolCall) -> str:
    if call.name == "inspect_visual":
        page = f" {call.arguments.get('page')}쪽" if call.arguments.get("page") else ""
        return f"이미지에서 위치를 확인하는 중… ({call.arguments.get('name', '')}{page})"
    if call.name == "view_page":
        page = f" {call.arguments.get('page')}쪽" if call.arguments.get("page") else ""
        return f"그림을 보는 중… ({call.arguments.get('name', '')}{page})"
    if call.name == "analyze_pages":
        pages = f" {call.arguments.get('pages')}쪽" if call.arguments.get("pages") else ""
        return f"쪽을 따로 보는 중… ({call.arguments.get('name', '')}{pages})"
    if call.name == "read_attachment":
        return f"문서 본문을 읽는 중… ({call.arguments.get('name', '')})"
    if call.name == "search_attachments":
        return f"문서에서 '{call.arguments.get('query', '')}' 검색 중…"
    return f"도구 실행 중… ({call.name})"


async def execute_tool(context: ToolContext, call: ToolCall) -> str:
    """도구 하나를 실행해 모델에게 돌려줄 문자열을 만든다. 오류도 문자열로 돌려준다."""
    handlers = {
        "inspect_visual": _inspect_visual,
        "view_page": _view_page,
        "analyze_pages": _analyze_pages,
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
async def _resolve_visual_surface(context: ToolContext, record: Attachment, page: int | None, *,
                                  why: str = "inspect_visual 요청", save: bool = True) -> Attachment:
    """대상 이미지 **한 장**을 떼어 낸다. OCR 단계에서 보관해 둔 표시용 바이트를 재사용한다.

    save=False면 지금 그린 쪽을 저장하지 않는다 — 여러 쪽을 잇달아 그리는 호출부(따로 보기)가 `_save_rendered_pages`로 한 번에 저장한다.
    """
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
    surface = await render_page_attachment(pdf, page_number, why=why, trace_kind="tool")
    context.attachments.append(surface)
    if save:
        await _save_rendered_pages(context)
    return surface


async def _save_rendered_pages(context: ToolContext) -> None:
    """지금 그린 쪽을 첨부로 저장하고, 트레이스가 첨부 ID·크기로 가리킬 수 있게 등록한다."""
    if context.store is not None and context.conversation_id:
        await context.store.save_attachments(context.conversation_id, context.attachments)
    turn = trace.current()
    if turn is not None:
        turn.register_attachments(context.attachments)


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
    # 추론을 켠 bbox 호출의 추론 예산(Step 6). 넘거나 반복하면 provider가 추론을 끊고 답만 이어 쓰게 한다.
    budget = config.reasoning_budget("grounding") if not context.disable_thinking else None
    effort = (context.reasoning_effort or None) if not context.disable_thinking else None      # 추론 수준(Step 6 2차)
    watch = (lambda info: context.on_live(describe_reasoning_progress(info))) if context.on_live is not None else None
    for attempt in range(1 + config.GROUNDING_RETRY_COUNT):
        instruction = grounding_instruction(task, surface.name, tile=image.tile is not None)
        if attempt:
            instruction = f"{instruction}\n{GROUNDING_RETRY_NOTE}"
        async with limiter:
            context.usage.grounding_calls += 1
            response = await context.provider.analyze(
                [{"role": "system", "content": GROUNDING_SYSTEM_PROMPT}, {"role": "user", "content": instruction}],
                images=[image], temperature=0.0, disable_thinking=context.disable_thinking,
                max_tokens=config.vision_max_tokens(), reasoning_budget=budget, on_reasoning=watch,
                reasoning_effort=effort,
            )
        context.usage.count_reasoning("grounding", response, image.name)
        if is_reasoning_runaway(response.finish_reason):
            # 추론이 끝나지 않아 이어 쓰기로도 답을 받지 못했다(Step 6) → 상한 도달과 같이 다시 묻지 않는다.
            trace.note("tool", f"추론이 끝나지 않아 중단 → 다시 묻지 않음 · {image.name}", reason=response.runaway)
            return VisualInspection(runaway=True)
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
        # 취소, 그리고 서버가 추론 수준을 받지 않은 경우(설정 오류 — 모든 타일이 같은 값이다)는 "일부 타일 실패"가 아니다.
        if isinstance(outcome, (asyncio.CancelledError, ReasoningEffortError)):
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
    looped = sum(1 for part in parts if part.runaway)
    warning = ""
    if structured and unreadable:
        warning = (f"{unreadable} of {len(images)} tiles did not return structured regions, "
                   "so targets inside those tiles may be missing.")
    elif cut or looped:
        warning = "The vision model did not return structured regions for any tile; no boxes could be measured."
    if cut:
        warning += (f" {cut} of {len(images)} tiles stopped at {_limit_words()} before answering and were not retried.")
    if looped:
        warning += (f" {looped} of {len(images)} tiles were stopped because the model's reasoning did not finish "
                    "(it repeated itself or exceeded its budget) and were not retried.")
    return VisualInspection(text="\n".join(notes), boxes=boxes, structured=bool(structured),
                            cut_off=bool(cut) and not structured, runaway=bool(looped) and not structured), warning


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
        elif inspection.runaway:
            warning = ("The vision model's reasoning did not finish (it repeated itself or exceeded its budget) and was "
                       "stopped, so no boxes could be measured. The call was not retried.")
    else:
        inspection, warning = await _inspect_tiles(context, surface, images, task)

    regions = [box for box in inspection.boxes if valid_box(box)]
    trace.note("tool", f"inspect_visual 결과 · 영역 {len(regions)}개" + (" · 경고 있음" if warning else ""),
               structured=inspection.structured, cutOff=inspection.cut_off, runaway=inspection.runaway or None,
               boxes=len(regions), warning=warning or None, text=trace.clip(inspection.text))
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


# --------------------------------------------------------------------------- view_page (Step 10)
async def _view_page(context: ToolContext, arguments: dict[str, Any]) -> str:
    """쪽(또는 업로드 이미지) 한 장을 골라 이 턴의 "보고 있는 쪽" 목록에 더한다. 이미지 조립은 답변 호출과 같은 규칙(전체 한 장).

    모델 호출은 여기서 일어나지 않는다 — 다음 답변 호출부터 루프가 `context.viewed`를 `images`에 더해 싣는다.
    """
    record = _find(context.attachments, str(arguments.get("name") or ""))
    if record is None:
        raise ToolError(f'No attachment named "{arguments.get("name")}". Exact names: {_known_names(context.attachments)}.')
    page = arguments.get("page")
    page_number = int(page) if isinstance(page, (int, float)) and page >= 1 else None
    if record.is_pdf and page_number is None and (record.total_pages or 0) > 1:
        raise ToolError(f'"{record.name}" has {record.total_pages} pages: pass "page" (1-based) to choose one page per call.')
    wanted = page_image_name(record.name, page_number) if record.is_pdf and page_number else record.name
    if wanted in context.attached_image_names():
        # 이미 실려 있다(처음부터 실린 업로드 이미지이거나 이 턴에서 이미 요청한 쪽) → 더하지 않고 자리만 알려 준다.
        trace.note("tool", f"{wanted}은(는) 이미 답변 호출에 실려 있음 → 더하지 않음", attached=context.attached_image_names())
        return view_page_already_attached(wanted, context.attached_image_names())
    if len(context.viewed) >= config.MAX_VIEWED_PAGES:
        context.view_refusals += 1
        trace.note("tool", f"보기 상한({config.MAX_VIEWED_PAGES}장)에 닿아 {wanted}을(를) 붙이지 않음",
                   limit=config.MAX_VIEWED_PAGES, viewed=context.viewed_names())
        return view_page_limit_reached(wanted, config.MAX_VIEWED_PAGES)
    surface = await _resolve_visual_surface(context, record, page_number, why="view_page 요청")
    # 답변 호출의 이미지는 모드와 무관하게 전체 한 장이다(Step 8과 같은 규칙). 타일은 전사·bbox 호출에만.
    images = await assemble_model_images(surface.name, surface.mime, surface.data or b"", purpose="analysis",
                                         mode=context.image_mode)
    context.viewed += images
    names = context.attached_image_names()
    remaining = config.MAX_VIEWED_PAGES - len(context.viewed)
    trace.note("tool", f"{surface.name}을(를) 다음 답변 호출부터 실음 ({len(context.viewed)}/{config.MAX_VIEWED_PAGES}장)",
               surface=surface.name, attachmentId=surface.id, width=surface.width, height=surface.height,
               attached=names, remaining=remaining)
    context.on_progress(f"그림을 보는 중… {surface.name} ({len(context.viewed)}/{config.MAX_VIEWED_PAGES}장)")
    return view_page_result(surface.name, names, remaining)


# --------------------------------------------------------------------------- analyze_pages (Step 10 2차)
_PAGE_SPEC_PART = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+))?\s*$")
_MAX_PAGE_SPAN = 2000


def parse_page_spec(spec: Any) -> list[int]:
    """`"7"` / `7` / `"1-10"` / `"2, 5, 7-9"` / `[2, 5]` → 쪽 번호 목록(오름차순, 중복 없음). 모양이 틀리면 ToolError."""
    if isinstance(spec, bool) or spec is None:
        raise ToolError('"pages" must be a page number, a range like "1-10" or a list like "2,5,7-9".')
    if isinstance(spec, (int, float)):
        parts = [str(int(spec))]
    elif isinstance(spec, list):
        parts = [str(item) for item in spec]
    else:
        parts = str(spec).replace("–", "-").replace("~", "-").split(",")
    pages: set[int] = set()
    for part in parts:
        if not part.strip():
            continue
        match = _PAGE_SPEC_PART.match(part)
        if match is None:
            raise ToolError(f'"pages" must be a page number, a range like "1-10" or a list like "2,5,7-9" (got {spec!r}).')
        first, last = int(match.group(1)), int(match.group(2) or match.group(1))
        if first < 1 or last < first:
            raise ToolError(f'Invalid page range "{part.strip()}": pages are 1-based and a range runs from low to high.')
        if last - first >= _MAX_PAGE_SPAN:
            raise ToolError(f'Page range "{part.strip()}" is too wide; request at most {_MAX_PAGE_SPAN} pages at a time.')
        pages.update(range(first, last + 1))
    if not pages:
        raise ToolError('"pages" is empty: pass a page number, a range like "1-10" or a list like "2,5,7-9".')
    return sorted(pages)


def _limit_note(context: ToolContext) -> str:
    return f"{context.analyzed_pages}/{config.MAX_ANALYZED_PAGES}쪽"


async def _analyze_one(context: ToolContext, surface: Attachment, image: ModelImage, question: str) -> str:
    """쪽 한 장에 대한 따로 보기 호출(전용 시스템 프롬프트, 추론·출력 상한·추론 수준은 bbox 호출과 같은 축). 답 글을 돌려준다.

    재시도는 없다 — 전사(비전사 응답)·bbox(비구조화 JSON)와 달리 "답의 모양"을 검사할 기준이 없다. 출력 상한에 닿은 호출도
    다시 보내지 않고(Step 6-0) 읽은 데까지만 남기며, 끊긴 글이 추론일 수 있으면 버린다.
    """
    budget = config.reasoning_budget("grounding") if not context.disable_thinking else None
    effort = (context.reasoning_effort or None) if not context.disable_thinking else None
    watch = (lambda info: context.on_live(describe_reasoning_progress(info))) if context.on_live is not None else None
    context.usage.analysis_calls += 1
    response = await context.provider.analyze(
        [{"role": "system", "content": PAGE_ANALYSIS_SYSTEM_PROMPT},
         {"role": "user", "content": page_analysis_instruction(question, surface.name)}],
        images=[image], temperature=0.0, disable_thinking=context.disable_thinking,
        max_tokens=config.vision_max_tokens(), reasoning_budget=budget, on_reasoning=watch, reasoning_effort=effort,
    )
    context.usage.count_reasoning("analysis", response, image.name)
    if is_reasoning_runaway(response.finish_reason):
        trace.note("tool", f"추론이 끝나지 않아 중단 → 다시 묻지 않음 · {image.name}", reason=response.runaway)
        return "(not analyzed: the model's reasoning did not finish and was stopped; the call was not retried)"
    if is_output_length_stop(response.finish_reason):
        context.usage.analysis_length_stops += 1
        partial = clean_transcription(cut_off_answer(context.provider, response, thinking_disabled=context.disable_thinking))
        trace.note("tool", "출력 상한에서 끊김 → 다시 묻지 않음" + (" · 읽은 데까지 남김" if partial else " · 버림") + f" · {image.name}",
                   keptChars=len(partial), finishReason=response.finish_reason)
        if partial:
            return f"{partial}\n[cut off at {_limit_words()}; the rest of this page's answer is missing]"
        return f"(not analyzed: the model reached {_limit_words()} before answering; the call was not retried)"
    text = clean_transcription(response.text)
    return text or "(the vision model returned no answer for this page)"


async def _analyze_pages(context: ToolContext, arguments: dict[str, Any]) -> str:
    """쪽(하나 또는 범위)마다 별도 호출로 질문에 답하게 하고 `[page n] 답`으로 모아 돌려준다. 답변 호출에는 글만 간다.

    상한 둘: 한 호출의 쪽 수(`ANALYZE_PAGES_PER_CALL`, 넘치면 앞에서부터 보고 "다음 범위로 다시 불러라")와 한 턴의 총 쪽 수
    (`MAX_ANALYZED_PAGES`, 시간 상한 — 넘치면 남은 만큼만 보고 알려 준다). 어느 쪽을 볼지는 전부 모델이 정한다(앱이 훑지 않는다).
    """
    context.analysis_calls += 1
    name, question = str(arguments.get("name") or ""), str(arguments.get("question") or "").strip()[:500]
    if not question:
        raise ToolError('analyze_pages needs a non-empty "question" - say what to look for on each page and what to report.')
    record = _find(context.attachments, name)
    if record is None:
        raise ToolError(f'No attachment named "{name}". Exact names: {_known_names(context.attachments)}.')

    # 1) 대상 쪽: PDF는 쪽 번호·범위(없는 쪽은 무시), 업로드 이미지(또는 이미 그려 둔 쪽)는 그 한 장
    beyond = 0
    if record.is_pdf:
        total = int(record.total_pages or 0)
        spec = arguments.get("pages", arguments.get("page"))
        if spec in (None, ""):
            if total > 1:
                raise ToolError(f'"{record.name}" has {total} pages: pass "pages" (a page number, a range like "1-10" or a list).')
            wanted = [1]
        else:
            wanted = parse_page_spec(spec)
        if total:
            beyond = sum(1 for page in wanted if page > total)
            wanted = [page for page in wanted if page <= total]
        if not wanted:
            raise ToolError(f'"{record.name}" has only {total} pages; none of the requested pages exist.')
        targets: list[tuple[str, int | None, str]] = [(page_image_name(record.name, page), page, f"page {page}") for page in wanted]
    elif record.is_image:
        targets = [(record.name, record.page_number, f"page {record.page_number}" if record.page_number else record.name)]
    else:
        raise ToolError(f'"{record.name}" has no visual surface to analyze. Choose an image or a PDF.')

    # 2) 상한: 한 호출의 쪽 수 → 턴의 총 쪽 수(같은 턴에 같은 쪽·같은 질문은 다시 묻지 않고 상한에도 세지 않는다)
    per_call = config.ANALYZE_PAGES_PER_CALL
    deferred = [page for _, page, _ in targets[per_call:] if page]
    targets = targets[:per_call]
    fresh = [target for target in targets if (target[0], question) not in context.analysis_cache]
    remaining = config.MAX_ANALYZED_PAGES - context.analyzed_pages
    if fresh and remaining <= 0:
        context.analysis_refusals += 1
        trace.note("tool", f"따로 보기 상한({config.MAX_ANALYZED_PAGES}쪽)에 닿아 {record.name}을(를) 보지 않음",
                   limit=config.MAX_ANALYZED_PAGES, requested=[marker for _, _, marker in targets])
        return analyze_pages_limit_reached(record.name, page_ranges([page for _, page, _ in targets if page]), config.MAX_ANALYZED_PAGES)
    allowed = {target[0] for target in fresh[:max(0, remaining)]}
    not_analyzed = [page for surface_name, page, _ in fresh if surface_name not in allowed and page]
    targets = [target for target in targets if (target[0], question) in context.analysis_cache or target[0] in allowed]
    trace.note("tool", f"{record.name}의 {len(targets)}쪽을 쪽마다 따로 보기 (이 턴 {_limit_note(context)} 전)",
               question=question, pages=[marker for _, _, marker in targets], cached=[m for n, _, m in targets if (n, question) in context.analysis_cache],
               deferred=page_ranges(deferred) or None, notAnalyzed=page_ranges(not_analyzed) or None, beyond=beyond or None)

    # 3) 쪽 이미지 준비(그리지 않은 쪽은 지금 그려 한 번에 저장) → 쪽마다 별도 호출(동시 OCR_CONCURRENCY개, 순서 유지)
    before = len(context.attachments)
    surfaces: dict[str, Attachment] = {}
    for surface_name, page, _ in targets:
        if (surface_name, question) not in context.analysis_cache:
            surfaces[surface_name] = await _resolve_visual_surface(context, record, page, why="analyze_pages 요청", save=False)
    if len(context.attachments) > before:
        await _save_rendered_pages(context)
    limiter = asyncio.Semaphore(config.OCR_CONCURRENCY)
    finished = 0

    async def one(surface: Attachment) -> str:
        nonlocal finished
        # 따로 보기의 이미지는 답변 호출과 같은 규칙(전체 한 장)이다. 타일은 전사·bbox 호출에만.
        images = await assemble_model_images(surface.name, surface.mime, surface.data or b"", purpose="analysis",
                                             mode=context.image_mode)
        async with limiter:
            context.analyzed_pages += 1
            text = await _analyze_one(context, surface, images[0], question)
        finished += 1
        context.on_progress(f"쪽을 따로 보는 중… {record.name} ({finished}/{len(surfaces)}쪽 · 이 턴 {_limit_note(context)})")
        return text

    ordered = list(surfaces.values())
    outcomes = await asyncio.gather(*(one(surface) for surface in ordered), return_exceptions=True)
    for outcome in outcomes:
        # 취소, 그리고 서버가 추론 수준을 받지 않은 경우(설정 오류 — 모든 쪽이 같은 값이다)는 "한 쪽의 실패"가 아니다.
        if isinstance(outcome, (asyncio.CancelledError, ReasoningEffortError)):
            raise outcome
    errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    if ordered and len(errors) == len(ordered):
        raise errors[0]      # 한 쪽도 보지 못했다 → 도구 오류로 되돌린다(모델이 다시 부를 수 있게)
    for surface, outcome in zip(ordered, outcomes):
        context.analysis_cache[(surface.name, question)] = (f"ERROR: {outcome}" if isinstance(outcome, BaseException) else outcome)
        if surface.name not in context.analyzed:
            context.analyzed.append(surface.name)
        # 그린 쪽의 바이트는 저장돼 있다(필요하면 도구가 다시 읽는다). 수십 쪽을 메모리에 붙들지 않는다.
        if surface.id is not None and surface.page_number and surface.name not in context.attached_image_names():
            surface.data = None

    answers = [(marker, context.analysis_cache[(surface_name, question)]) for surface_name, _, marker in targets]
    remaining = config.MAX_ANALYZED_PAGES - context.analyzed_pages
    trace.note("tool", f"따로 보기 결과 · {len(answers)}쪽 (이 턴 {_limit_note(context)})",
               results=[{"page": marker, "chars": len(answer), "failed": answer.startswith(("ERROR:", "(not analyzed"))}
                        for marker, answer in answers], remaining=remaining)
    return analyze_pages_result(record.name, question, answers, remaining=remaining, limit=config.MAX_ANALYZED_PAGES,
                                deferred=page_ranges(deferred), not_analyzed=page_ranges(not_analyzed), beyond=beyond,
                                per_call=per_call)


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
