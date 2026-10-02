"""모델 호출을 트레이스에 기록하는 provider 겉싸개(Step 7).

`chat_service`가 트레이스를 켠 턴에만 실제 provider를 이걸로 감싼다. 호출마다
  - 시작할 때: 보낸 메시지(내용은 잘라서), 이미지(첨부 ID·타일 위치), 제공한 도구, temperature·추론·출력 상한
  - 끝날 때  : 본문, 추론 글, 도구 호출, finish_reason, 토큰 수, 걸린 시간 — 실패·취소면 그 사유
를 남긴다. 어떤 종류의 호출인지(답변 / 전사 / 도구 안의 호출)는 트레이스의 부모 이벤트로 안다.

기록 코드는 실제 호출에 영향을 주지 않는다: 안쪽 provider의 예외는 그대로 다시 던지고, 기록 쪽 예외는 `TurnTrace`가 삼킨다.
"""
from __future__ import annotations

import asyncio
from typing import Any

from .. import trace
from ..pipeline.images import ModelImage
from .base import Message, ModelResponse, Provider, ToolSpec
from .reasoning import OnReasoning, ReasoningProgress

_LABELS = {"answer": "답변 호출", "ocr": "전사 호출", "grounding": "위치 확인 호출"}
_REASONS = {"budget": "추론 예산 초과", "repeat": "추론 반복"}


class TracedProvider(Provider):
    def __init__(self, inner: Provider, turn: trace.TurnTrace):
        # API key는 겉싸개에 두지 않는다(어디에도 기록되지 않도록). 호출은 안쪽 provider가 한다.
        super().__init__(model=inner.model, api_key="", base_url=inner.base_url, timeout=inner.timeout)
        self.inner = inner
        self.turn = turn
        self.name = inner.name
        self.is_local = inner.is_local

    @property
    def cache_namespace(self) -> str:
        return self.inner.cache_namespace

    def can_disable_thinking(self) -> bool:
        return self.inner.can_disable_thinking()

    def thinking_off_for_every_call(self) -> bool:
        return self.inner.thinking_off_for_every_call()

    async def list_models(self) -> list[str]:
        return await self.inner.list_models()

    async def aclose(self) -> None:
        await self.inner.aclose()

    def _kind(self) -> tuple[str, str]:
        """(호출 종류, 표시 이름). 부모 이벤트가 전사면 ocr, 도구면 그 도구의 호출, 없으면 답변 호출이다."""
        parent = self.turn.parent()
        if parent is None:
            return "answer", _LABELS["answer"]
        if parent.kind == "ocr":
            return "ocr", _LABELS["ocr"]
        if parent.kind == "tool":
            tool = str(parent.data.get("name") or "")
            if tool == "inspect_visual":
                return "grounding", _LABELS["grounding"]
            return f"tool:{tool}", f"{tool} 안의 모델 호출"
        return "answer", _LABELS["answer"]

    async def analyze(self, messages: list[Message], images: list[ModelImage] | None = None,
                      tools: list[ToolSpec] | None = None, *, temperature: float = 0.2,
                      disable_thinking: bool = False, max_tokens: int | None = None,
                      reasoning_budget: int | None = None, on_reasoning: OnReasoning | None = None) -> ModelResponse:
        kind, label = self._kind()
        number = self.turn.count(f"model:{kind}")
        image_notes = self.turn.describe_images(images)
        tile = next((item.get("tile") for item in image_notes if item.get("tile")), None)
        where = image_notes[0]["name"] if image_notes else ""
        title = f"{label} #{number}" if kind == "answer" else f"{label} · {where}" if where else f"{label} #{number}"
        # 이 호출이 실제로 추론을 끄고 나가는가: 호출별 값 또는 "모든 호출" 설정, 단 서버가 그 요청을 받을 때만.
        control = self.inner.can_disable_thinking()
        thinking_off = control and (bool(disable_thinking) or self.inner.thinking_off_for_every_call())
        event = self.turn.start(
            "model", title, kind=kind, number=number, provider=self.name, model=self.model,
            messages=trace.describe_messages(messages), images=image_notes, tile=tile,
            tools=trace.describe_tools(tools), temperature=temperature, disableThinking=thinking_off,
            thinkingControl=control, maxTokens=max_tokens if self.is_local else None,
            reasoningBudget=reasoning_budget if (self.is_local and not thinking_off) else None,
        )

        def watched(info: ReasoningProgress) -> None:
            """추론 진행은 호출 이벤트에 덧쓰고(진행 중에도 보이게), 소프트·하드 조치는 따로 한 줄 남긴다."""
            self.turn.update(event, reasoningTokens=info.tokens, reasoningSeconds=round(info.seconds, 1),
                             reasoningStage=info.stage)
            if info.stage == "forced":
                self.turn.update(event, forcedCycle=info.cycle or None)      # 반복이면 되풀이된 묶음의 첫 줄
                self.turn.note("cleanup", f"{_REASONS.get(info.reason, info.reason)} → 추론을 끊고 답으로 넘김" + (f" · {where}" if where else ""),
                               reason=info.reason, tokens=info.tokens, seconds=round(info.seconds, 1),
                               cycle=info.cycle or None)
            elif info.stage == "runaway":
                self.turn.note("cleanup", "이어 쓰기로도 답을 받지 못해 중단" + (f" · {where}" if where else ""),
                               reason=info.reason, tokens=info.tokens)
            if on_reasoning is not None:
                on_reasoning(info)

        try:
            response = await self.inner.analyze(messages, images, tools, temperature=temperature,
                                                disable_thinking=disable_thinking, max_tokens=max_tokens,
                                                reasoning_budget=reasoning_budget, on_reasoning=watched)
        except asyncio.CancelledError:
            self.turn.finish(event, "cancelled", reason=trace.CANCELLED_REASON)
            raise
        except Exception as error:
            self.turn.finish(event, "failed", error=f"{type(error).__name__}: {error}"[:2000])
            raise
        self.turn.finish(event, "done", **self._describe_response(response))
        return response

    @staticmethod
    def _describe_response(response: ModelResponse) -> dict[str, Any]:
        return {
            "text": trace.clip(response.text), "textChars": len(response.text or ""),
            "reasoning": trace.clip(response.reasoning) if response.reasoning else None,
            "reasoningChars": len(response.reasoning or "") or None,
            "toolCalls": [trace.describe_tool_call(call) for call in response.tool_calls] or None,
            "finishReason": response.finish_reason or None,
            "promptTokens": response.prompt_tokens, "completionTokens": response.completion_tokens,
            "reasoningTokens": response.reasoning_tokens, "reasoningSeconds": response.reasoning_seconds,
            "reasoningStage": "done" if response.reasoning_tokens is not None else None,
            "forced": response.forced or None, "runaway": response.runaway or None,
        }
