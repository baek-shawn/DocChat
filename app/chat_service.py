"""/api/chat 한 턴의 전체 흐름.

    업로드 정제 → 전처리(§5.1·5.2) → 첨부 저장 → 시각 OCR(§5.3) → 증거 선택
      → 단일 tool-calling 루프(§6) → 거짓 거절 교정 → 대화 저장

진행 상황은 `emit({"type": "progress", ...})`로 흘려보낸다. 전송 방식(동기 JSON / NDJSON)은 라우터가 정한다.

이미지 처리 방식(전체/타일)은 요청마다 고른다. 전사와 bbox 호출에만 적용되고, 어떤 방식으로 처리했는지는
답변의 메타데이터(`meta`)에 남긴다.
답변(추론) 호출에 어떤 이미지를 실을지(끔 / 업로드 이미지만 / 전체, Step 8)도 요청마다 고르고 메타데이터에 남긴다.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from . import config, trace
from .agent.loop import run_tool_loop
from .agent.prompts import FALSE_REFUSAL_CORRECTION, system_prompt
from .agent.tools import ToolContext, available_tools, describe_tool_call, execute_tool
from .attachments import Attachment, UploadError, sanitize_uploads
from .db import ChatStore, now_ms, sanitize_messages, valid_id
from .pipeline.evidence import (attachment_context_for_prompt, attachment_manifest, attachment_root_name, clip,
                                is_pending_page_image, is_visual_ocr, merge_attachment_sets)
from .pipeline.images import ImageError, ModelImage, TileSource, VisionUsage, assemble_model_images
from .pipeline.ocr import TileSink, build_ocr_reader, prepare_visual_ocr_evidence
from .pipeline.pdf import PdfError
from .pipeline.preprocess import preprocess_attachments, render_page_attachment
from .providers import Provider, ProviderError, create_provider
from .providers.traced import TracedProvider
from .storage import StorageError, extension_for, safe_filename

logger = logging.getLogger("docchat.chat")

Emit = Callable[[dict[str, Any]], None]

ERROR_PREFIX = "오류: "
_STORED_ERROR = re.compile(r"^(?:오류|error)\s*:", re.IGNORECASE)
# 첨부 내용을 이미 받았는데도 "파일을 볼 수 없다"고 답하는 경우를 잡는다.
_FALSE_REFUSAL = [re.compile(pattern, re.IGNORECASE) for pattern in (
    r"cannot (?:directly )?(?:access|view|open|read|see)",
    r"(?:don't|do not) have (?:the )?(?:ability|capability|access) to (?:access|view|open|read|see)",
    r"unable to (?:access|view|open|read|see) (?:the |any )?(?:file|attachment|pdf|image|document)",
    r"paste (?:the )?(?:text|content)", r"copy and paste", r"share the (?:text|content)",
    r"(?:파일|첨부|문서|pdf|이미지)[^.\n]{0,20}(?:볼|열|읽을|접근할|확인할) 수 없",
    r"(?:내용|텍스트)을 (?:직접 )?(?:붙여|복사해)",
)]


class ChatError(Exception):
    """사용자에게 그대로 보여 줄 오류. status는 동기 모드의 HTTP 상태 코드."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class ChatRequest:
    provider: str = "openaiCompatible"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    conversation_id: str = ""
    context_size: int | None = None
    disable_thinking: bool = True      # 모든 호출의 추론을 끈다
    # 호출 종류별 추론 끄기(Step 6-0). None이면 서버 기본값(config). disable_thinking이 켜져 있으면 의미가 없다.
    disable_thinking_grounding: bool | None = None
    disable_thinking_ocr: bool | None = None
    image_mode: str = ""      # "whole" | "tile". 비어 있으면 config.DEFAULT_IMAGE_MODE
    # 답변 호출의 이미지(Step 8): "off" | "uploads" | "whole". 비어 있으면 config.DEFAULT_ANSWER_IMAGE_MODE
    answer_image_mode: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    attachments: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ThinkingPlan:
    """이번 턴에서 호출 종류별로 추론을 끄고 보내는가. 추론을 끌 방법이 없는 provider면 controllable이 False다."""
    answer: bool = False
    grounding: bool = False
    ocr: bool = False
    controllable: bool = False

    def to_public(self) -> dict[str, bool]:
        return {"answer": self.answer, "grounding": self.grounding, "ocr": self.ocr}


def plan_thinking(request: ChatRequest, provider: Provider) -> ThinkingPlan:
    """"추론 끄기"(모든 호출)가 켜져 있으면 전부 끈다. 꺼져 있으면 bbox·전사만 각자의 선택을 따른다."""
    if not provider.can_disable_thinking():
        return ThinkingPlan()
    everything = bool(request.disable_thinking)
    return ThinkingPlan(
        answer=everything,
        grounding=everything or config.resolve_switch(request.disable_thinking_grounding, config.GROUNDING_DISABLE_THINKING),
        ocr=everything or config.resolve_switch(request.disable_thinking_ocr, config.OCR_DISABLE_THINKING),
        controllable=True,
    )


@dataclass
class Answer:
    text: str
    artifacts: list[dict[str, Any]]
    attachments: list[Attachment]
    usage: VisionUsage
    image_names: list[str] = field(default_factory=list)   # 답변 호출에 실은 이미지(순서대로)
    image_candidates: int = 0                              # 모드상 실을 수 있었던 이미지 수(상한 때문에 뺀 수를 알 수 있게)


def looks_like_false_attachment_refusal(text: str) -> bool:
    value = str(text or "")
    return len(value) < 1500 and any(pattern.search(value) for pattern in _FALSE_REFUSAL)


def has_usable_attachment_content(attachments: list[Attachment]) -> bool:
    return any((item.text or "").strip() or item.is_image for item in attachments)


def compact_conversation_messages(messages: list[dict[str, Any]], total_chars: int) -> list[dict[str, str]]:
    """최근 메시지부터 예산 안에 담는다. 저장돼 있던 오류 메시지는 모델에게 보내지 않는다."""
    useful = [m for m in messages if not (m["role"] == "assistant" and _STORED_ERROR.match(m["content"].strip()))]
    selected: list[dict[str, str]] = []
    remaining = max(1000, total_chars)
    for message in reversed(useful):
        if remaining <= 0:
            break
        allowance = min(remaining, max(600, int(total_chars * 0.45)))
        content = clip(message["content"], allowance)
        selected.append({"role": message["role"], "content": content})
        remaining -= len(content) + 40
    selected.reverse()
    # 첫 메시지가 assistant면 떼어 낸다(역할 교대를 요구하는 템플릿 대비).
    while selected and selected[0]["role"] == "assistant":
        selected.pop(0)
    return selected


def _budgets(provider: Provider, context_size: int | None) -> tuple[int, int, int]:
    """(전체 글자 예산, 첨부 텍스트 예산, 대화 이력 예산)"""
    if provider.is_local:
        tokens = context_size if context_size and context_size > 0 else config.DEFAULT_LOCAL_CONTEXT_TOKENS
        char_budget = config.estimate_context_char_budget(min(tokens, 1_048_576))
        return char_budget, max(6000, int(char_budget * 0.42)), max(3500, int(char_budget * 0.35))
    return config.CLOUD_CHAR_BUDGET, config.CLOUD_ATTACHMENT_TEXT_BUDGET, config.CLOUD_CHAR_BUDGET // 2


def reply_language_hint(text: str) -> str:
    """사용자 글의 문자로 답변 언어를 짚어 준다.

    영어 시스템 프롬프트 + 영어 문서가 컨텍스트를 채우면 소형 로컬 모델은 한국어 질문에도 영어로
    답하곤 한다(gemma3 4B에서 실측). 질문 바로 뒤에 한 줄을 덧붙이는 편이 시스템 프롬프트보다 잘 듣는다.
    """
    if re.search(r"[가-힣]", text):
        language = "Korean (한국어)"
    elif re.search(r"[぀-ヿ]", text):
        language = "Japanese (日本語)"
    elif re.search(r"[一-鿿]", text):
        language = "Chinese (中文)"
    else:
        return ""
    # 이 힌트를 메시지에 미리 박아 두면 안 된다: JSON 폴백에서 도구를 제공하는 호출에 언어 지시가 섞이면
    # gemma3가 도구 호출을 건너뛴다(실측 0/6 vs 6/6). 언제 붙일지는 `run_tool_loop(language_hint=…)`가 정한다.
    return (f"[Language: write your final answer in {language}. "
            "Keep identifiers and values from the documents exactly as written.]")


def _user_content(text: str, documents: list[Attachment], has_visual_ocr: bool,
                  images: list[Attachment] | None = None) -> str:
    parts = [text or "Please analyze the attached files."]
    for document in documents:
        if document.text:
            parts.append(f"[Attachment: {document.name}]\n{document.text}")
    if has_visual_ocr:
        parts.append(
            "[EVIDENCE NOTE: pages without enough native text were transcribed from whole-page images; see the "
            '"visual OCR" attachment. Account for every visual source, keep the page order, and do not invent '
            "values that are unreadable.]"
        )
    # PDF 쪽 이미지를 실을 때(Step 8 전체 모드)만: 이미지가 여러 장이면 몇 번째가 어느 쪽인지 알려 줘야 한다.
    # 업로드 이미지만 실리는 기본 모드에서는 붙지 않는다(Step 8 이전과 같은 프롬프트).
    if images and any(image.page_number for image in images):
        listing = "; ".join(f"{index}: {image.name}" for index, image in enumerate(images, start=1))
        parts.append(
            f"[PAGE IMAGES: attached to this message in this order - {listing}. They show the drawings, symbols "
            "and layout that the extracted text cannot convey.]"
        )
    return "\n\n".join(parts)


async def run_chat(store: ChatStore, request: ChatRequest, emit: Emit) -> dict[str, Any]:
    if request.provider not in config.ALL_PROVIDERS:
        raise ChatError(f"지원하지 않는 provider입니다: {request.provider}")
    if not request.model.strip():
        raise ChatError("모델을 선택하거나 입력하세요.")
    messages = sanitize_messages(request.messages)
    if not messages or messages[-1]["role"] != "user":
        raise ChatError("마지막 메시지는 사용자 메시지여야 합니다.")
    try:
        image_mode = config.resolve_image_mode(request.image_mode)
    except ValueError as error:
        raise ChatError(f"지원하지 않는 이미지 처리 방식입니다: {request.image_mode!r}. "
                        "'whole'(전체) 또는 'tile'(타일) 중에서 고르세요.") from error
    try:
        answer_image_mode = config.resolve_answer_image_mode(request.answer_image_mode)
    except ValueError as error:
        raise ChatError(f"지원하지 않는 답변 이미지 방식입니다: {request.answer_image_mode!r}. "
                        "'off'(끔), 'uploads'(업로드 이미지만), 'whole'(전체) 중에서 고르세요.") from error
    try:
        uploads = sanitize_uploads(request.attachments)
        provider = create_provider(request.provider, model=request.model, api_key=request.api_key,
                                   base_url=request.base_url, disable_thinking=request.disable_thinking)
    except (UploadError, ProviderError) as error:
        raise ChatError(str(error)) from error

    conversation_id = request.conversation_id if valid_id(request.conversation_id) else ""
    conversation_id = await store.save_conversation(
        conversation_id=conversation_id or None, provider=request.provider, model=request.model, messages=messages)
    # 개발용 턴 트레이스(Step 7): 켜져 있을 때만 만든다. id를 먼저 알려 줘야 화면이 진행 중에도 "과정 보기"를 열 수 있다.
    turn = trace.TurnTrace(conversation_id, store.save_trace) if config.debug_trace_enabled() else None
    emit({"type": "conversation", "conversationId": conversation_id, **({"traceId": turn.id} if turn else {})})
    trace_meta = {"traceId": turn.id} if turn else {}

    started = time.monotonic()
    if turn is not None:
        provider = TracedProvider(provider, turn)
        turn.activate()
    thinking = plan_thinking(request, provider)
    if turn is not None:
        _record_input(turn, request, messages, uploads, image_mode, answer_image_mode, thinking, provider)
    try:
        answer = await _answer(store, provider, request, conversation_id, messages, uploads, emit, image_mode,
                               answer_image_mode, thinking)
    except (ProviderError, UploadError, PdfError, ImageError, StorageError) as error:
        failure = ChatError(str(error), status=502 if isinstance(error, ProviderError) else 400)
        await _save_reply(store, request, conversation_id, messages, f"{ERROR_PREFIX}{error}", [], trace_meta)
        if turn is not None:
            await turn.close("failed", reason=str(error))
        raise failure from error
    except asyncio.CancelledError:
        if turn is not None:
            await turn.close("cancelled", reason=trace.CANCELLED_REASON)
        raise
    except Exception as error:      # 예상 밖 오류도 트레이스에는 남긴다(라우터가 사용자에게 알린다)
        if turn is not None:
            await turn.close("failed", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        await provider.aclose()
        if turn is not None:
            turn.deactivate()

    # 이 답을 어떤 방식으로 만들었는지 남긴다 — 모드를 바꿔 가며 비교할 때 어느 답이 어느 모드였는지 알 수 있어야 한다.
    meta: dict[str, Any] = {"imageMode": image_mode, "vision": answer.usage.to_public(),
                            "elapsedMs": int((time.monotonic() - started) * 1000), **trace_meta}
    if image_mode == "tile":
        meta["tiling"] = config.tile_settings()
    # 답변 호출에 실은 이미지(Step 8): 모드, 실은 수·이름, 모드상 실을 수 있었던 수(상한 때문에 뺀 것이 있는지)
    meta["answerImageMode"] = answer_image_mode
    meta["answerImages"] = {"sent": len(answer.image_names), "candidates": answer.image_candidates,
                            "names": answer.image_names}
    if thinking.controllable and provider.can_disable_thinking():
        # 호출 종류별로 추론을 끄고 보냈는지. 서버가 그 요청을 거절했으면(can_disable_thinking이 False로 바뀐다) 적지 않는다.
        meta["thinkingDisabled"] = thinking.to_public()
    if provider.is_local and config.vision_max_tokens():
        meta["visionMaxTokens"] = config.vision_max_tokens()
    if turn is not None:
        turn.note("answer", "최종 답변", text=trace.clip(answer.text), chars=len(answer.text), meta=meta,
                  artifacts=[{"name": item.get("name"), "boxes": len(item.get("boxes") or [])} for item in answer.artifacts])
    await _save_reply(store, request, conversation_id, messages, answer.text, answer.artifacts, meta)
    if turn is not None:
        await turn.close("done")
    return {
        "type": "final",
        "conversationId": conversation_id,
        "text": answer.text,
        "artifacts": answer.artifacts,
        "attachments": [item.to_public() for item in answer.attachments],
        "files": messages[-1].get("files") or [],
        "meta": meta,
    }


def _record_input(turn: trace.TurnTrace, request: ChatRequest, messages: list[dict[str, Any]], uploads: list[Attachment],
                  image_mode: str, answer_image_mode: str, thinking: ThinkingPlan, provider: Provider) -> None:
    """턴의 입력: 질문, 새 첨부, 이번 턴의 설정. API key는 넣지 않는다."""
    turn.note(
        "input", "입력", question=trace.clip(messages[-1]["content"]), historyMessages=len(messages) - 1,
        attachments=[{"name": item.name, "kind": item.kind, "mime": item.mime, "size": item.size} for item in uploads],
        provider=request.provider, model=request.model, baseUrl=request.base_url or None, imageMode=image_mode,
        answerImageMode=answer_image_mode, maxModelImages=config.MAX_MODEL_IMAGES,
        contextSize=request.context_size, thinkingDisabled=thinking.to_public() if thinking.controllable else None,
        thinkingControl=provider.can_disable_thinking(), visionMaxTokens=config.vision_max_tokens(),
        tiling=config.tile_settings() if image_mode == "tile" else None,
    )


async def _save_reply(store: ChatStore, request: ChatRequest, conversation_id: str, messages: list[dict[str, Any]],
                      text: str, artifacts: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    reply = {"role": "assistant", "content": text, "artifacts": artifacts, "meta": meta, "createdAt": now_ms()}
    await store.save_conversation(conversation_id=conversation_id, provider=request.provider, model=request.model,
                                  messages=[*messages, reply])


def _tile_source_loader(store: ChatStore, attachments: list[Attachment]):
    """전사할 쪽 이미지 → 타일을 렌더할 원본 PDF. 같은 PDF는 한 요청 안에서 한 번만 읽는다."""
    loaded: dict[str, bytes | None] = {}

    async def load(page: Attachment) -> TileSource | None:
        root = attachment_root_name(page.name)
        if root not in loaded:
            pdf = next((item for item in attachments if item.name == root and item.is_pdf), None)
            data = pdf.data if pdf is not None else None
            if data is None and pdf is not None and pdf.id is not None:
                data = await store.load_attachment_data(pdf.id)
            loaded[root] = data
        data = loaded[root]
        return TileSource(kind="pdf", data=data, page_number=page.page_number or 1) if data else None

    return load


def _tile_sink(store: ChatStore, conversation_id: str) -> TileSink | None:
    """트레이스를 켰을 때만: 모델에 보낸 타일을 대화 폴더에 남긴다("모델이 실제로 본 이미지" 확인용).

    같은 설정으로 만든 타일은 내용이 같으므로 이미 있으면 다시 쓰지 않는다.
    """
    if not config.debug_trace_enabled():
        return None
    settings = f"{config.TILE_SIZE}px-{config.TILE_RENDER_DPI}dpi-o{config.TILE_OVERLAP:g}"

    async def save(name: str, purpose: str, images: list[ModelImage]) -> None:
        folder = f"tiles/{safe_filename(name, max_length=60)}/{purpose}-{settings}"
        for image in images:
            row, col = image.tile or (1, 1)
            target = f"{folder}/r{row:02d}c{col:02d}.{extension_for(image.mime, 'png')}"
            try:
                await asyncio.to_thread(store.files.write, conversation_id, target, image.data, keep_existing=True)
            except StorageError as error:      # 확인용 파일이다. 못 써도 답변은 계속한다.
                logger.warning("타일을 저장하지 못했습니다: %s (%s)", target, error)
        trace.note("files", f"모델에 보낸 타일 {len(images)}장을 저장 · {name}", folder=f"{conversation_id}/{folder}",
                   purpose=purpose)

    return save


async def _render_pending_pages(store: ChatStore, attachments: list[Attachment], images: list[Attachment],
                                progress: Callable[[str], None]) -> int:
    """전체 모드(Step 8)에서 아직 렌더하지 않은 PDF 쪽(자리표시)을 원본 PDF에서 그려 첨부 목록에 넣는다.

    자리표시는 그 자리에서 렌더된 첨부로 바뀐다. 그리지 못한 쪽(원본이 없거나 PDF 오류)은 자리표시로 남아
    호출부가 걸러 낸다 — 쪽 하나 때문에 턴 전체를 실패시키지 않는다. 그린 수를 돌려준다.
    """
    pending = [index for index, image in enumerate(images) if is_pending_page_image(image)]
    if not pending:
        return 0
    loaded: dict[str, bytes | None] = {}      # 같은 PDF는 한 번만 읽는다. pdf.data에 붙들어 두지 않는다(메모리).
    rendered = 0
    for count, index in enumerate(pending, start=1):
        progress(f"답변 호출에 실을 쪽 이미지를 렌더하는 중… ({count}/{len(pending)})")
        placeholder = images[index]
        root = attachment_root_name(placeholder.name)
        pdf = next((item for item in attachments if item.name == root and item.is_pdf), None)
        if pdf is None:
            continue
        if root not in loaded:
            loaded[root] = pdf.data if pdf.data else (await store.load_attachment_data(pdf.id) if pdf.id is not None else None)
        if not loaded[root]:
            trace.note("evidence", f"{placeholder.name}을(를) 렌더하지 못함 · 원본 PDF 바이트 없음")
            continue
        try:
            page = await render_page_attachment(replace(pdf, data=loaded[root]), placeholder.page_number or 1,
                                                why="답변 호출에 실음", trace_kind="evidence")
        except PdfError as error:
            logger.warning("답변 호출용 쪽 렌더 실패: %s (%s)", placeholder.name, error)
            trace.note("evidence", f"{placeholder.name}을(를) 렌더하지 못함", error=str(error))
            continue
        images[index] = page
        attachments.append(page)
        rendered += 1
    return rendered


async def _answer(store: ChatStore, provider: Provider, request: ChatRequest, conversation_id: str,
                  messages: list[dict[str, Any]], uploads: list[Attachment], emit: Emit,
                  image_mode: str, answer_image_mode: str, thinking: ThinkingPlan) -> Answer:
    def progress(message: str) -> None:
        emit({"type": "progress", "message": message})
        trace.note("progress", message)      # 화면에 보인 진행 단계가 트레이스의 시간축에도 남는다

    progress("요청을 준비하는 중…")
    usage = VisionUsage()
    tile_sink = _tile_sink(store, conversation_id) if image_mode == "tile" else None
    attachments = await store.list_attachments(conversation_id)

    # 1) 새 업로드 전처리: 네이티브 텍스트 우선, 검사에서 탈락한 페이지만 이미지로 렌더
    if uploads:
        progress(f"첨부 {len(uploads)}개를 분석하는 중… (페이지 판별·렌더링)")
        parsed = await preprocess_attachments(uploads)
        attachments, replaced = merge_attachment_sets(attachments, parsed)
        await store.delete_attachments(conversation_id, [item.id for item in replaced if item.id is not None])
        await store.save_attachments(conversation_id, attachments)
        _link_uploaded_files(messages[-1], uploads)
    turn = trace.current()
    if turn is not None:
        turn.register_attachments(attachments)      # 모델에 보낸 이미지를 첨부 ID로 가리키기 위해

    # 2) 시각 OCR: needs_vlm 페이지만, 메인 답변 전에, 선택된 VLM에게 전사시킨다
    #    (다른 방식 — 이미지 처리 방식·추론 여부 — 으로 전사해 둔 쪽은 이번 요청의 방식으로 다시 전사한다)
    attachments, transcribed = await prepare_visual_ocr_evidence(
        attachments,
        read_image=build_ocr_reader(provider, image_mode=image_mode, usage=usage, on_progress=progress,
                                    load_tile_source=_tile_source_loader(store, attachments), on_tiles=tile_sink,
                                    disable_thinking=thinking.ocr),
        load_data=lambda item: store.load_attachment_data(item.id) if item.id is not None else _none(),
        on_progress=progress,
        cache_namespace=provider.cache_namespace,
        image_mode=image_mode,
        # 추론을 끌 수 있는 provider인데 끄지 않고 보내는 경우만 따로 표시한다(그 밖에는 Step 6-0 이전과 같은 기록).
        thinking=thinking.controllable and not thinking.ocr,
    )
    if transcribed:
        await store.save_attachments(conversation_id, attachments)
        if turn is not None:
            turn.register_attachments(attachments)
    # 큰 바이트는 메모리에서 내려놓는다. 필요하면 도구가 DB에서 다시 읽는다.
    for item in attachments:
        if item.id is not None and not item.send_to_model:
            item.data = None

    # 3) 프롬프트 예산 안에서 증거 선택. 답변 호출에 실을 이미지는 요청의 답변 이미지 모드(Step 8)가 정한다.
    char_budget, attachment_budget, history_budget = _budgets(provider, request.context_size)
    latest_user = messages[-1]["content"]
    context = attachment_context_for_prompt(latest_user, attachments, attachment_budget, config.MAX_MODEL_IMAGES,
                                            answer_images=answer_image_mode)
    # 전체 모드: 전처리에서 렌더하지 않은 쪽(네이티브 글자가 충분한 쪽)은 실을 것만 지금 그려 첨부로 저장한다.
    if await _render_pending_pages(store, attachments, context.images, progress):
        await store.save_attachments(conversation_id, attachments)
        if turn is not None:
            turn.register_attachments(attachments)
    context.images = [image for image in context.images if not is_pending_page_image(image)]
    model_images = []
    for image in context.images:
        if image.data is None and image.id is not None:
            image.data = await store.load_attachment_data(image.id)
        if image.data:
            # 답변 호출의 이미지는 모드와 무관하게 전체 한 장이다(타일은 전사·bbox 호출에만 적용, Step 8 나머지에서 다룬다).
            model_images += await assemble_model_images(image.name, image.mime, image.data, purpose="analysis",
                                                        mode=image_mode)
    context.images = [image for image in context.images if image.data]

    tools = available_tools(attachments)
    history = compact_conversation_messages(messages[-config.MAX_HISTORY_MESSAGES:], history_budget)
    if not history or history[-1]["role"] != "user":
        history.append({"role": "user", "content": latest_user})
    has_visual_ocr = any(is_visual_ocr(item) for item in attachments)
    history[-1] = {"role": "user", "content": _user_content(history[-1]["content"], context.documents, has_visual_ocr,
                                                            context.images)}
    model_messages = [{"role": "system", "content": system_prompt(attachment_manifest(attachments),
                                                                 tools_enabled=bool(tools),
                                                                 model_name=request.model)}, *history]
    if turn is not None:
        _record_evidence(turn, attachments, context, model_images, tools, history, model_messages[0]["content"],
                         budgets=(char_budget, attachment_budget, history_budget), has_visual_ocr=has_visual_ocr,
                         answer_image_mode=answer_image_mode)

    # 4) 단일 tool-calling 루프
    tool_context = ToolContext(
        provider=provider, attachments=attachments, store=store, conversation_id=conversation_id,
        default_read_chars=max(1000, min(16_000, attachment_budget // 3)), on_progress=progress,
        image_mode=image_mode, usage=usage, on_tiles=tile_sink, disable_thinking=thinking.grounding,
    )
    progress("답변을 생성하는 중…")
    language_hint = reply_language_hint(latest_user)
    result = await run_tool_loop(
        provider, model_messages, images=model_images or None, tools=tools,
        execute=lambda call: execute_tool(tool_context, call), on_progress=progress, describe=describe_tool_call,
        language_hint=language_hint,
    )
    text = result.text
    usage.answer_calls += result.model_calls

    # 5) 내용을 이미 줬는데 "첨부를 볼 수 없다"고 하면 한 번만 바로잡는다
    if has_usable_attachment_content(attachments) and looks_like_false_attachment_refusal(text):
        progress("첨부 내용을 근거로 다시 답변하는 중…")
        trace.note("cleanup", "첨부를 볼 수 없다는 답 → 첨부 내용을 근거로 다시 요청", rejected=trace.clip(text))
        usage.answer_calls += 1
        retry = await provider.analyze(
            [*model_messages, {"role": "assistant", "content": text},
             {"role": "user", "content": f"{FALSE_REFUSAL_CORRECTION}\n{language_hint}".strip()}],
            images=model_images or None,
        )
        text = retry.text.strip() or text
    if not text.strip():
        raise ProviderError("모델이 빈 응답을 돌려주었습니다. 다시 시도하거나 다른 모델을 선택하세요.")
    return Answer(text=text, artifacts=tool_context.artifacts, attachments=attachments, usage=usage,
                  image_names=[image.name for image in model_images], image_candidates=context.image_candidates)


def _record_evidence(turn: trace.TurnTrace, attachments: list[Attachment], context: Any, model_images: list[ModelImage],
                     tools: list[Any], history: list[dict[str, Any]], system: str, *, budgets: tuple[int, int, int],
                     has_visual_ocr: bool, answer_image_mode: str) -> None:
    """증거 조립: 어떤 텍스트를 얼마나 실었고 무엇이 잘렸는지, 어떤 이미지를 보냈는지, 예산은 얼마였는지."""
    full = {item.name: len(item.text or "") for item in attachments}
    documents = []
    for item in context.documents:
        sent = len(item.text or "")
        documents.append({"name": item.name, "kind": item.kind, "chars": full.get(item.name, sent), "sentChars": sent,
                          "clipped": sent < full.get(item.name, sent), "visualOcr": is_visual_ocr(item)})
    turn.note(
        "evidence", "증거 조립", budgets={"chars": budgets[0], "attachmentText": budgets[1], "history": budgets[2]},
        documents=documents, images=turn.describe_images(model_images), tools=trace.describe_tools(tools),
        answerImageMode=answer_image_mode, imageCandidates=context.image_candidates, maxModelImages=config.MAX_MODEL_IMAGES,
        historyMessages=len(history), visualOcrNote=has_visual_ocr, systemPrompt=trace.clip(system),
        manifest=attachment_manifest(attachments),
    )


async def _none() -> None:
    return None


def _link_uploaded_files(message: dict[str, Any], uploads: list[Attachment]) -> None:
    """사용자 메시지의 파일 칩이 저장된 첨부를 가리키게 한다(칩을 눌러 원본을 다시 열 수 있도록)."""
    by_root = {attachment_root_name(item.name): item for item in uploads}
    files = message.get("files") or []
    known = {file.get("name") for file in files}
    files += [{"name": item.name, "kind": item.kind, "size": item.size} for item in uploads if item.name not in known]
    for file in files:
        saved = by_root.get(file.get("name"))
        if saved is not None and saved.id is not None:
            file["attachmentId"], file["mime"], file["kind"] = saved.id, saved.mime, saved.kind
    message["files"] = files
