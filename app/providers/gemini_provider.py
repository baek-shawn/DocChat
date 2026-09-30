"""Google Gemini provider — 공식 `google-genai` SDK.

함수 호출 턴은 응답의 원본 Content(`raw_assistant`)를 그대로 되돌려 보낸다.
(사고형 모델은 function_call 파트에 thought_signature가 붙어 있어, 직접 재구성하면 거절될 수 있다.)
"""
from __future__ import annotations

from typing import Any

from ..pipeline.images import ModelImage
from .base import (ContextWindowError, Message, ModelResponse, Provider, ProviderError, ToolCall, ToolSpec,
                   compact_messages, image_anchor_index, is_context_window_error)


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, *, model: str, api_key: str = "", base_url: str = "", timeout: float = 180.0):
        super().__init__(model=model, api_key=api_key, base_url=base_url, timeout=timeout)
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as error:  # pragma: no cover - 의존성 누락 안내
            raise ProviderError("google-genai 패키지가 설치돼 있지 않습니다. `uv sync`를 실행하세요.") from error
        self._types, self._errors = types, errors
        http_options: dict[str, Any] = {"timeout": int(timeout * 1000)}
        if self.base_url:
            http_options["base_url"] = self.base_url
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(**http_options))

    async def list_models(self) -> list[str]:
        names: list[str] = []
        try:
            pager = await self._client.aio.models.list()
            async for model in pager:
                actions = getattr(model, "supported_actions", None) or []
                if actions and "generateContent" not in actions:
                    continue
                names.append(str(model.name or "").removeprefix("models/"))
        except self._errors.APIError as error:
            raise ProviderError(self._describe(error)) from error
        return [name for name in names if name]

    def to_contents(self, messages: list[Message], images: list[ModelImage] | None) -> list[Any]:
        types = self._types
        image_index = image_anchor_index(messages) if images else -1
        contents: list[Any] = []
        for index, message in enumerate(messages):
            role = message.get("role")
            text = str(message.get("content") or "")
            if role == "system":
                continue
            if role == "tool":
                part = types.Part.from_function_response(name=message.get("name") or "tool", response={"result": text})
                contents.append(types.Content(role="user", parts=[part]))
            elif role == "assistant":
                raw = (message.get("raw") or {}).get(self.name)
                if raw is not None:
                    contents.append(raw)
                    continue
                parts = [types.Part.from_text(text=text)] if text.strip() else []
                parts += [types.Part.from_function_call(name=call.name, args=call.arguments)
                          for call in message.get("tool_calls") or []]
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
            else:
                parts = [types.Part.from_text(text=text or "Please analyze the attached files.")]
                if index == image_index:
                    parts += [types.Part.from_bytes(data=image.data, mime_type=image.mime) for image in images or []]
                contents.append(types.Content(role="user", parts=parts))
        return contents

    async def analyze(self, messages: list[Message], images: list[ModelImage] | None = None,
                      tools: list[ToolSpec] | None = None, *, temperature: float = 0.2,
                      disable_thinking: bool = False, max_tokens: int | None = None) -> ModelResponse:
        # disable_thinking·max_tokens는 로컬 provider용이다. 여기서는 쓰지 않는다(실제 API로 확인한 적이 없다).
        try:
            return await self._complete(messages, images, tools, temperature)
        except ContextWindowError:
            return await self._complete(compact_messages(messages, 1), images, tools, temperature)

    async def _complete(self, messages, images, tools, temperature) -> ModelResponse:
        types = self._types
        system = "\n\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "system")
        options: dict[str, Any] = {
            "temperature": temperature,
            # 도구 실행은 우리 루프가 맡는다 → SDK의 자동 함수 호출을 끈다.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if system:
            options["system_instruction"] = system
        if tools:
            options["tools"] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
                for t in tools
            ])]
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model, contents=self.to_contents(messages, images),
                config=types.GenerateContentConfig(**options),
            )
        except self._errors.APIError as error:
            detail = self._describe(error)
            if is_context_window_error(detail):
                raise ContextWindowError(detail) from error
            raise ProviderError(detail) from error
        except Exception as error:  # 네트워크 계층 예외(httpx 등)
            raise ProviderError(f"Gemini API 호출에 실패했습니다: {error}") from error
        return self.parse_response(response)

    def parse_response(self, response: Any) -> ModelResponse:
        candidate = (getattr(response, "candidates", None) or [None])[0]
        content = getattr(candidate, "content", None)
        texts: list[str] = []
        calls: list[ToolCall] = []
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "thought", False):
                continue
            if getattr(part, "function_call", None) is not None:
                function_call = part.function_call
                call = ToolCall(name=function_call.name or "", arguments=dict(function_call.args or {}))
                if getattr(function_call, "id", None):
                    call.id = function_call.id
                calls.append(call)
            elif getattr(part, "text", None):
                texts.append(part.text)
        finish = getattr(candidate, "finish_reason", None)
        finish_name = str(getattr(finish, "name", finish) or "").lower()
        return ModelResponse(
            text="\n".join(texts).strip(), tool_calls=calls,
            finish_reason="length" if finish_name == "max_tokens" else finish_name,
            raw_assistant={self.name: content} if calls and content is not None else None,
        )

    @staticmethod
    def _describe(error: Exception) -> str:
        status = getattr(error, "code", None) or getattr(error, "status_code", None)
        message = getattr(error, "message", "") or str(error)
        return f"HTTP {status}: {str(message)[:2000]}" if status else str(message)[:2000]
