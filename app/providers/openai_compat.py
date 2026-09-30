"""OpenAI 호환 provider — `openai` SDK에서 `base_url`만 바꿔 쓴다.

하나의 구현으로 llama.cpp / Ollama / vLLM / LM Studio 같은 로컬 런타임과 OpenAI 클라우드를 모두 다룬다.
"""
from __future__ import annotations

import base64
import json
import re
from typing import Any

import openai
from openai import AsyncOpenAI

from ..pipeline.images import ModelImage
from .base import (ContextWindowError, Message, ModelResponse, Provider, ProviderError, ToolCall, ToolSpec,
                   ToolsUnsupportedError, compact_messages, is_context_window_error, image_anchor_index,
                   looks_like_tools_unsupported)

OPENAI_BASE_URL = "https://api.openai.com/v1"


def to_wire_messages(messages: list[Message], images: list[ModelImage] | None) -> list[dict[str, Any]]:
    """내부 메시지 → Chat Completions 형식. 이미지는 마지막 user 메시지에 image_url 파트로 붙인다."""
    image_index = image_anchor_index(messages) if images else -1
    wire: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "tool":
            wire.append({"role": "tool", "tool_call_id": message.get("tool_call_id") or "", "content": content})
        elif role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": content}
            calls = message.get("tool_calls") or []
            if calls:
                item["content"] = content or None
                item["tool_calls"] = [
                    {"id": call.id, "type": "function",
                     "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}}
                    for call in calls
                ]
            wire.append(item)
        elif role == "user" and index == image_index:
            parts: list[dict[str, Any]] = [{"type": "text", "text": content}]
            for image in images or []:
                encoded = base64.b64encode(image.data).decode("ascii")
                parts.append({"type": "image_url", "image_url": {"url": f"data:{image.mime};base64,{encoded}"}})
            wire.append({"role": "user", "content": parts})
        else:
            wire.append({"role": "system" if role == "system" else "user", "content": content})
    return wire


def to_wire_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"type": "function",
         "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters}}
        for tool in tools
    ]


def parse_tool_calls(raw_calls: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for raw in raw_calls or []:
        function = getattr(raw, "function", None)
        name = getattr(function, "name", None)
        if not name:
            continue
        arguments = getattr(function, "arguments", None)
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments or "{}")
            except ValueError:
                parsed = {}
        else:
            parsed = arguments or {}
        call = ToolCall(name=name, arguments=parsed if isinstance(parsed, dict) else {})
        if getattr(raw, "id", None):
            call.id = raw.id
        calls.append(call)
    return calls


class OpenAICompatProvider(Provider):
    def __init__(self, *, name: str, model: str, api_key: str = "", base_url: str = "", timeout: float = 180.0,
                 is_local: bool = False, disable_thinking: bool = False):
        super().__init__(model=model, api_key=api_key, base_url=base_url or OPENAI_BASE_URL, timeout=timeout)
        self.name = name
        self.is_local = is_local
        # 인증이 없는 로컬 서버도 SDK는 비어 있지 않은 키를 요구한다.
        self._client = AsyncOpenAI(
            base_url=self.base_url, api_key=api_key or "local", timeout=timeout,
            max_retries=0 if is_local else 2,
        )
        self._send_temperature = True
        # 추론형 모델(Qwen3 계열 등)은 "안녕" 한마디에도 수천 토큰을 생각한다(실측: vLLM Qwen3.5에서 0.3초 vs 90초 초과).
        # vLLM·llama.cpp는 chat_template_kwargs로 끌 수 있다. 이 필드를 거절하는 서버면 한 번 실패한 뒤 빼고 다시 보낸다.
        # _disable_thinking은 모든 호출에 적용되는 설정이고, 호출 하나만 끄는 것은 analyze(disable_thinking=True)다.
        self._disable_thinking = disable_thinking and is_local
        self._thinking_control = is_local      # 서버가 그 필드를 거절하면 False — 이후로는 보내지 않는다

    def can_disable_thinking(self) -> bool:
        return self._thinking_control

    async def aclose(self) -> None:
        await self._client.close()

    async def list_models(self) -> list[str]:
        try:
            page = await self._client.models.list()
        except openai.APIError as error:
            raise ProviderError(self._explain(error)) from error
        return sorted({item.id for item in page.data if getattr(item, "id", None)})

    def _explain(self, error: Exception) -> str:
        """연결·시간 초과처럼 사용자가 직접 조치할 수 있는 오류는 한국어 안내로 바꾼다."""
        if isinstance(error, openai.APITimeoutError):
            return f"모델 응답이 {int(self.timeout)}초 안에 오지 않았습니다."
        if isinstance(error, openai.APIConnectionError):
            return (f"모델 서버({self.base_url})에 연결할 수 없습니다. 서버가 실행 중인지, "
                    "주소에 /v1이 포함돼 있는지 확인하세요.")
        return self._describe(error)

    async def analyze(self, messages: list[Message], images: list[ModelImage] | None = None,
                      tools: list[ToolSpec] | None = None, *, temperature: float = 0.2,
                      disable_thinking: bool = False, max_tokens: int | None = None) -> ModelResponse:
        # 출력 상한은 폭주가 확인된 로컬 서버에만 보낸다. 클라우드 API는 이름도 의미도 달라(예: max_completion_tokens)
        # 실제로 확인하지 않고는 넣지 않는다.
        limit = int(max_tokens) if max_tokens and max_tokens > 0 and self.is_local else None
        try:
            return await self._complete(messages, images, tools, temperature, disable_thinking, limit)
        except ContextWindowError:
            pass
        # 로컬 서버는 예산을 넘긴 프롬프트를 잘라 주지 않는다 → 두 단계로 줄여 다시 보낸다.
        try:
            return await self._complete(compact_messages(messages, 1), images, tools, temperature, disable_thinking, limit)
        except ContextWindowError:
            return await self._complete(compact_messages(messages, 2), (images or [])[:1] or None, tools, temperature,
                                        disable_thinking, limit)

    async def _complete(self, messages: list[Message], images: list[ModelImage] | None,
                        tools: list[ToolSpec] | None, temperature: float, disable_thinking: bool = False,
                        max_tokens: int | None = None) -> ModelResponse:
        request: dict[str, Any] = {"model": self.model, "messages": to_wire_messages(messages, images)}
        if self._send_temperature:
            request["temperature"] = temperature
        if tools:
            request["tools"] = to_wire_tools(tools)
            request["tool_choice"] = "auto"
        thinking_off = self._thinking_control and (self._disable_thinking or disable_thinking)
        if thinking_off:
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        if max_tokens:
            request["max_tokens"] = max_tokens
        try:
            completion = await self._client.chat.completions.create(**request)
        except openai.APIStatusError as error:
            detail = self._describe(error)
            if is_context_window_error(detail):
                if max_tokens:
                    # vLLM은 "입력 + 출력 상한"이 컨텍스트를 넘으면 생성하지 않고 거절한다. 남은 자리가 상한보다 작다는
                    # 뜻이므로 상한을 빼고 보낸다 — 그래도 출력은 남은 자리(< 상한)를 넘지 못한다.
                    return await self._complete(messages, images, tools, temperature, disable_thinking, None)
                raise ContextWindowError(detail) from error
            if thinking_off and error.status_code in (400, 422) and re.search(
                    r"chat_template_kwargs|enable_thinking|extra|unknown|unrecognized|unexpected", detail, re.IGNORECASE):
                self._thinking_control = False
                return await self._complete(messages, images, tools, temperature, disable_thinking, max_tokens)
            if self._send_temperature and error.status_code == 400 and "temperature" in detail.lower():
                # 일부 추론형 모델은 기본값 외의 temperature를 거절한다.
                self._send_temperature = False
                return await self._complete(messages, images, tools, temperature, disable_thinking, max_tokens)
            if tools and looks_like_tools_unsupported(error.status_code, detail):
                raise ToolsUnsupportedError(detail) from error
            raise ProviderError(detail) from error
        except openai.APIError as error:  # 시간 초과·연결 실패 포함
            raise ProviderError(self._explain(error)) from error

        choice = completion.choices[0] if getattr(completion, "choices", None) else None
        if choice is None or choice.message is None:
            raise ProviderError("모델 엔드포인트가 assistant 메시지를 돌려주지 않았습니다.")
        reasoning = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None)
        usage = getattr(completion, "usage", None)
        return ModelResponse(
            text=(choice.message.content or "").strip(),
            tool_calls=parse_tool_calls(getattr(choice.message, "tool_calls", None)),
            finish_reason=str(choice.finish_reason or ""),
            reasoning=reasoning.strip() if isinstance(reasoning, str) else "",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    @staticmethod
    def _describe(error: Exception) -> str:
        status = getattr(error, "status_code", None)
        body = getattr(error, "body", None)
        message = ""
        if isinstance(body, dict):
            inner = body.get("error", body)
            message = str(inner.get("message") if isinstance(inner, dict) else inner)
        message = message or getattr(error, "message", "") or str(error)
        return f"HTTP {status}: {message[:2000]}" if status else message[:2000]
