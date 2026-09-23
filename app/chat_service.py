"""/api/chat 한 턴의 전체 흐름.

    업로드 정제 → 전처리(§5.1·5.2) → 첨부 저장 → 시각 OCR(§5.3) → 증거 선택
      → 단일 tool-calling 루프(§6) → 거짓 거절 교정 → 대화 저장

진행 상황은 `emit({"type": "progress", ...})`로 흘려보낸다. 전송 방식(동기 JSON / NDJSON)은 라우터가 정한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from . import config
from .agent.loop import run_tool_loop
from .agent.prompts import FALSE_REFUSAL_CORRECTION, system_prompt
from .agent.tools import ToolContext, available_tools, describe_tool_call, execute_tool
from .attachments import Attachment, UploadError, sanitize_uploads
from .db import ChatStore, now_ms, sanitize_messages, valid_id
from .pipeline.evidence import (attachment_context_for_prompt, attachment_manifest, attachment_root_name, clip,
                                is_visual_ocr, merge_attachment_sets)
from .pipeline.images import ImageError, assemble_model_images
from .pipeline.ocr import build_ocr_reader, prepare_visual_ocr_evidence
from .pipeline.pdf import PdfError
from .pipeline.preprocess import preprocess_attachments
from .providers import Provider, ProviderError, create_provider

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
    disable_thinking: bool = True
    messages: list[dict[str, Any]] = field(default_factory=list)
    attachments: list[dict[str, Any]] = field(default_factory=list)


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


def _user_content(text: str, documents: list[Attachment], has_visual_ocr: bool) -> str:
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
        uploads = sanitize_uploads(request.attachments)
        provider = create_provider(request.provider, model=request.model, api_key=request.api_key,
                                   base_url=request.base_url, disable_thinking=request.disable_thinking)
    except (UploadError, ProviderError) as error:
        raise ChatError(str(error)) from error

    conversation_id = request.conversation_id if valid_id(request.conversation_id) else ""
    conversation_id = await store.save_conversation(
        conversation_id=conversation_id or None, provider=request.provider, model=request.model, messages=messages)
    emit({"type": "conversation", "conversationId": conversation_id})

    try:
        text, artifacts, attachments = await _answer(store, provider, request, conversation_id, messages, uploads, emit)
    except (ProviderError, UploadError, PdfError, ImageError) as error:
        failure = ChatError(str(error), status=502 if isinstance(error, ProviderError) else 400)
        await _save_reply(store, request, conversation_id, messages, f"{ERROR_PREFIX}{error}", [])
        raise failure from error
    finally:
        await provider.aclose()

    await _save_reply(store, request, conversation_id, messages, text, artifacts)
    return {
        "type": "final",
        "conversationId": conversation_id,
        "text": text,
        "artifacts": artifacts,
        "attachments": [item.to_public() for item in attachments],
        "files": messages[-1].get("files") or [],
    }


async def _save_reply(store: ChatStore, request: ChatRequest, conversation_id: str,
                      messages: list[dict[str, Any]], text: str, artifacts: list[dict[str, Any]]) -> None:
    reply = {"role": "assistant", "content": text, "artifacts": artifacts, "createdAt": now_ms()}
    await store.save_conversation(conversation_id=conversation_id, provider=request.provider, model=request.model,
                                  messages=[*messages, reply])


async def _answer(store: ChatStore, provider: Provider, request: ChatRequest, conversation_id: str,
                  messages: list[dict[str, Any]], uploads: list[Attachment],
                  emit: Emit) -> tuple[str, list[dict[str, Any]], list[Attachment]]:
    def progress(message: str) -> None:
        emit({"type": "progress", "message": message})

    progress("요청을 준비하는 중…")
    attachments = await store.list_attachments(conversation_id)

    # 1) 새 업로드 전처리: 네이티브 텍스트 우선, 검사에서 탈락한 페이지만 이미지로 렌더
    if uploads:
        progress(f"첨부 {len(uploads)}개를 분석하는 중… (페이지 판별·렌더링)")
        parsed = await preprocess_attachments(uploads)
        attachments, replaced = merge_attachment_sets(attachments, parsed)
        await store.delete_attachments(conversation_id, [item.id for item in replaced if item.id is not None])
        await store.save_attachments(conversation_id, attachments)
        _link_uploaded_files(messages[-1], uploads)

    # 2) 시각 OCR: needs_vlm 페이지만, 메인 답변 전에, 선택된 VLM에게 전사시킨다
    attachments, transcribed = await prepare_visual_ocr_evidence(
        attachments,
        read_image=build_ocr_reader(provider),
        load_data=lambda item: store.load_attachment_data(item.id) if item.id is not None else _none(),
        on_progress=progress,
        cache_namespace=provider.cache_namespace,
    )
    if transcribed:
        await store.save_attachments(conversation_id, attachments)
    # 큰 바이트는 메모리에서 내려놓는다. 필요하면 도구가 DB에서 다시 읽는다.
    for item in attachments:
        if item.id is not None and not item.send_to_model:
            item.data = None

    # 3) 프롬프트 예산 안에서 증거 선택
    _, attachment_budget, history_budget = _budgets(provider, request.context_size)
    latest_user = messages[-1]["content"]
    context = attachment_context_for_prompt(latest_user, attachments, attachment_budget, config.MAX_MODEL_IMAGES)
    model_images = []
    for image in context.images:
        if image.data is None and image.id is not None:
            image.data = await store.load_attachment_data(image.id)
        if image.data:
            model_images += assemble_model_images(image.name, image.mime, image.data, purpose="analysis")

    tools = available_tools(attachments)
    history = compact_conversation_messages(messages[-config.MAX_HISTORY_MESSAGES:], history_budget)
    if not history or history[-1]["role"] != "user":
        history.append({"role": "user", "content": latest_user})
    history[-1] = {"role": "user", "content": _user_content(
        history[-1]["content"], context.documents, any(is_visual_ocr(item) for item in attachments))}
    model_messages = [{"role": "system", "content": system_prompt(attachment_manifest(attachments),
                                                                 tools_enabled=bool(tools),
                                                                 model_name=request.model)}, *history]

    # 4) 단일 tool-calling 루프
    tool_context = ToolContext(
        provider=provider, attachments=attachments, store=store, conversation_id=conversation_id,
        default_read_chars=max(1000, min(16_000, attachment_budget // 3)), on_progress=progress,
    )
    progress("답변을 생성하는 중…")
    language_hint = reply_language_hint(latest_user)
    result = await run_tool_loop(
        provider, model_messages, images=model_images or None, tools=tools,
        execute=lambda call: execute_tool(tool_context, call), on_progress=progress, describe=describe_tool_call,
        language_hint=language_hint,
    )
    text = result.text

    # 5) 내용을 이미 줬는데 "첨부를 볼 수 없다"고 하면 한 번만 바로잡는다
    if has_usable_attachment_content(attachments) and looks_like_false_attachment_refusal(text):
        progress("첨부 내용을 근거로 다시 답변하는 중…")
        retry = await provider.analyze(
            [*model_messages, {"role": "assistant", "content": text},
             {"role": "user", "content": f"{FALSE_REFUSAL_CORRECTION}\n{language_hint}".strip()}],
            images=model_images or None,
        )
        text = retry.text.strip() or text
    if not text.strip():
        raise ProviderError("모델이 빈 응답을 돌려주었습니다. 다시 시도하거나 다른 모델을 선택하세요.")
    return text, tool_context.artifacts, attachments


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
