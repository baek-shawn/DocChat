"""POST /api/chat — 기본은 동기 요청-응답, `stream: true`면 NDJSON 청크.

정식 SSE(`text/event-stream`, `data:` 접두사)도 WebSocket도 쓰지 않는다. 스트리밍은 진행률 표시용으로
한 줄에 JSON 하나씩만 흘려보내는 단순 청크다.
    {"type":"conversation","conversationId":"..."}
    {"type":"progress","message":"페이지 전사 중… (1/3)"}
    {"type":"final","text":"...","artifacts":[...],"attachments":[...]}
    {"type":"error","error":"..."}
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..chat_service import ChatError, ChatRequest, run_chat

router = APIRouter(prefix="/api", tags=["chat"])


class ChatBody(BaseModel):
    provider: str = "openaiCompatible"
    apiKey: str = ""
    baseUrl: str = ""
    model: str = ""
    conversationId: str = ""
    contextSize: int | None = None
    disableThinking: bool = True
    # 호출 종류별 추론 끄기(bbox / 전사). 비우면 서버 기본값(DOCCHAT_GROUNDING_DISABLE_THINKING, DOCCHAT_OCR_DISABLE_THINKING).
    # disableThinking이 true면 모든 호출이 꺼지므로 이 둘은 쓰이지 않는다.
    disableThinkingGrounding: bool | None = None
    disableThinkingOcr: bool | None = None
    # 이미지 처리 방식: "whole"(전체) | "tile"(타일). 비우면 서버 기본값(DOCCHAT_IMAGE_MODE).
    imageMode: str = ""
    # 답변(추론) 호출의 이미지(Step 8): "off"(끔) | "uploads"(업로드 이미지만) | "whole"(전체, PDF 쪽 포함)
    # | "auto"(자동 — 업로드 이미지만 + 모델이 보기 도구로 쪽을 요청, Step 10). 비우면 서버 기본값(DOCCHAT_ANSWER_IMAGE_MODE).
    answerImageMode: str = ""
    # 따로 보기 도구 analyze_pages(Step 10 2차) — 자동 모드에서만 의미 있다. 비우면(null) 서버 기본값(DOCCHAT_ANALYZE_TOOL).
    analyzeTool: bool | None = None
    # 묶음 보기: 따로 보기 한 VLM 호출에 넣는 쪽 수 — 1(쪽마다) | k | "auto"(모델이 고름). 비우면 서버 기본값(DOCCHAT_ANALYZE_GROUP).
    analyzeGroup: str | int | None = None
    # 범위 고정: "free"(모델이 넘긴 범위 그대로) | "fixed"(첫 쪽부터 한 호출 상한만큼 꽉 채움). 비우면 서버 기본값(DOCCHAT_ANALYZE_RANGE).
    analyzeRange: str = ""
    # 호출 종류별 추론 수준(Step 6 2차): 모델의 채팅 템플릿이 받는 값(예: Qwen3.8의 "low" | "medium" | "xhigh").
    # 비우면(null) 서버 기본값(DOCCHAT_REASONING_EFFORT_*), ""이면 보내지 않는다. 추론을 켠 로컬 호출에만 실린다.
    reasoningEffortAnswer: str | None = None
    reasoningEffortGrounding: str | None = None
    reasoningEffortOcr: str | None = None
    stream: bool = False
    messages: list[dict[str, Any]] = Field(default_factory=list)
    attachments: list[dict[str, Any]] = Field(default_factory=list)


def _to_request(body: ChatBody) -> ChatRequest:
    return ChatRequest(
        provider=body.provider, api_key=body.apiKey, base_url=body.baseUrl, model=body.model,
        conversation_id=body.conversationId, context_size=body.contextSize, disable_thinking=body.disableThinking,
        disable_thinking_grounding=body.disableThinkingGrounding, disable_thinking_ocr=body.disableThinkingOcr,
        image_mode=body.imageMode, answer_image_mode=body.answerImageMode, analyze_tool=body.analyzeTool,
        analyze_group=body.analyzeGroup, analyze_range=body.analyzeRange,
        reasoning_effort_answer=body.reasoningEffortAnswer, reasoning_effort_grounding=body.reasoningEffortGrounding,
        reasoning_effort_ocr=body.reasoningEffortOcr, messages=body.messages, attachments=body.attachments,
    )


def _line(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


@router.post("/chat")
async def chat(request: Request, body: ChatBody) -> Any:
    store = request.app.state.store
    chat_request = _to_request(body)

    if not body.stream:
        try:
            result = await run_chat(store, chat_request, lambda _event: None)
        except ChatError as error:
            return JSONResponse({"error": str(error)}, status_code=error.status)
        return {key: value for key, value in result.items() if key != "type"}

    async def stream() -> AsyncIterator[bytes]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def work() -> None:
            try:
                queue.put_nowait(await run_chat(store, chat_request, queue.put_nowait))
            except ChatError as error:
                queue.put_nowait({"type": "error", "error": str(error)})
            except asyncio.CancelledError:
                raise
            except Exception as error:  # 예상 밖 오류도 스트림으로 알려야 프론트가 멈추지 않는다
                queue.put_nowait({"type": "error", "error": f"서버 오류: {error}"})
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(work())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield _line(event)
        finally:
            # 사용자가 "중지"를 누르거나 탭을 닫으면 진행 중인 모델 호출도 취소한다.
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
