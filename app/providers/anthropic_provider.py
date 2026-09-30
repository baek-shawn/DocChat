"""Anthropic(Claude) provider — 공식 SDK. system은 별도 인자, 도구 결과는 user 턴의 tool_result 블록."""
from __future__ import annotations

import base64
from typing import Any

from ..pipeline.images import ModelImage, fit_image_bytes
from .base import (Message, ModelResponse, Provider, ProviderError, ToolCall, ToolSpec, is_context_window_error,
                   image_anchor_index, ContextWindowError, compact_messages)

MAX_OUTPUT_TOKENS = 8192
MAX_IMAGE_BYTES = 5 * 1024 * 1024 - 64 * 1024  # API 한도는 이미지당 5MB


def split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    system = "\n\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "system")
    return system, [m for m in messages if m.get("role") != "system"]


def to_anthropic_messages(messages: list[Message], images: list[ModelImage] | None) -> list[dict[str, Any]]:
    """내부 메시지 → Messages API 형식. 같은 역할이 연달아 오면 한 턴으로 합친다."""
    image_index = image_anchor_index(messages) if images else -1
    turns: list[dict[str, Any]] = []

    def push(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": blocks})

    for index, message in enumerate(messages):
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "tool":
            push("user", [{"type": "tool_result", "tool_use_id": message.get("tool_call_id") or "", "content": content}])
        elif role == "assistant":
            blocks: list[dict[str, Any]] = [{"type": "text", "text": content}] if content.strip() else []
            for call in message.get("tool_calls") or []:
                blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})
            push("assistant", blocks)
        else:
            blocks = [{"type": "text", "text": content or "Please analyze the attached files."}]
            if index == image_index:
                for image in images or []:
                    data, mime = fit_image_bytes(image.data, image.mime, MAX_IMAGE_BYTES)
                    blocks.append({"type": "image", "source": {
                        "type": "base64", "media_type": mime, "data": base64.b64encode(data).decode("ascii")}})
            push("user", blocks)
    return turns


def to_anthropic_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]


def parse_anthropic_response(response: Any) -> ModelResponse:
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in getattr(response, "content", None) or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            texts.append(getattr(block, "text", "") or "")
        elif kind == "tool_use":
            arguments = getattr(block, "input", None)
            call = ToolCall(name=block.name, arguments=arguments if isinstance(arguments, dict) else {})
            if getattr(block, "id", None):
                call.id = block.id
            calls.append(call)
    return ModelResponse(text="\n".join(texts).strip(), tool_calls=calls,
                         finish_reason=str(getattr(response, "stop_reason", "") or ""))


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, *, model: str, api_key: str = "", base_url: str = "", timeout: float = 180.0):
        super().__init__(model=model, api_key=api_key, base_url=base_url, timeout=timeout)
        try:
            import anthropic
        except ImportError as error:  # pragma: no cover - 의존성 누락 안내
            raise ProviderError("anthropic 패키지가 설치돼 있지 않습니다. `uv sync`를 실행하세요.") from error
        self._sdk = anthropic
        options: dict[str, Any] = {"api_key": api_key, "timeout": timeout, "max_retries": 2}
        if self.base_url:
            options["base_url"] = self.base_url
        self._client = anthropic.AsyncAnthropic(**options)

    async def aclose(self) -> None:
        await self._client.close()

    async def list_models(self) -> list[str]:
        try:
            page = await self._client.models.list(limit=100)
        except self._sdk.APIError as error:
            raise ProviderError(self._describe(error)) from error
        return [item.id for item in page.data if getattr(item, "id", None)]

    async def analyze(self, messages: list[Message], images: list[ModelImage] | None = None,
                      tools: list[ToolSpec] | None = None, *, temperature: float = 0.2,
                      disable_thinking: bool = False, max_tokens: int | None = None) -> ModelResponse:
        # disable_thinking·max_tokens는 로컬 provider용이다. 여기서는 쓰지 않는다(확장 추론을 요청하지 않고,
        # 출력 상한은 MAX_OUTPUT_TOKENS로 이미 있다).
        try:
            return await self._complete(messages, images, tools, temperature)
        except ContextWindowError:
            return await self._complete(compact_messages(messages, 1), images, tools, temperature)

    async def _complete(self, messages, images, tools, temperature) -> ModelResponse:
        system, turns = split_system(messages)
        request: dict[str, Any] = {
            "model": self.model, "max_tokens": MAX_OUTPUT_TOKENS, "temperature": temperature,
            "messages": to_anthropic_messages(turns, images),
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = to_anthropic_tools(tools)
        try:
            response = await self._client.messages.create(**request)
        except self._sdk.APITimeoutError as error:
            raise ProviderError(f"Anthropic 응답이 {int(self.timeout)}초 안에 오지 않았습니다.") from error
        except self._sdk.APIConnectionError as error:
            raise ProviderError("Anthropic API에 연결할 수 없습니다. 네트워크를 확인하세요.") from error
        except self._sdk.APIError as error:
            detail = self._describe(error)
            if is_context_window_error(detail):
                raise ContextWindowError(detail) from error
            raise ProviderError(detail) from error
        return parse_anthropic_response(response)

    @staticmethod
    def _describe(error: Exception) -> str:
        status = getattr(error, "status_code", None)
        message = getattr(error, "message", "") or str(error)
        return f"HTTP {status}: {message[:2000]}" if status else message[:2000]
