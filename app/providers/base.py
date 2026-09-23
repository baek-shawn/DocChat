"""모델 연결의 단일 인터페이스: `analyze(messages, images=None, tools=None) -> ModelResponse`.

내부 메시지 형식(모든 provider가 공유):
    {"role": "system",    "content": str}
    {"role": "user",      "content": str}
    {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...], "raw": provider별 원본(선택)}
    {"role": "tool",      "tool_call_id": str, "name": str, "content": str}
`images`는 **마지막 user 메시지**에 붙는다.
"""
from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..pipeline.evidence import clip
from ..pipeline.images import ModelImage

Message = dict[str, Any]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (type=object)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    raw_assistant: Any = None  # provider가 다음 턴에 그대로 돌려받고 싶은 원본(예: Gemini Content)


class ProviderError(Exception):
    """사용자에게 보여 줄 수 있는 모델 호출 오류."""


class ToolsUnsupportedError(ProviderError):
    """엔드포인트/모델이 네이티브 tool-calling을 지원하지 않는다 → 루프가 JSON 폴백으로 전환한다."""


class ContextWindowError(ProviderError):
    """프롬프트가 모델 컨텍스트를 넘었다."""


_CONTEXT_PATTERN = re.compile(
    r"context.?window|maximum context length|context length|context size|too many tokens|input tokens|"
    r"prompt is too long|exceeds? the (?:available )?context|n_ctx|token limit",
    re.IGNORECASE,
)
_TOOLS_PATTERN = re.compile(r"tool|function.?call|chat template|jinja", re.IGNORECASE)


def is_context_window_error(message: str) -> bool:
    return bool(_CONTEXT_PATTERN.search(message or ""))


def looks_like_tools_unsupported(status: int | None, message: str) -> bool:
    return status in (400, 404, 422, 501) or bool(_TOOLS_PATTERN.search(message or ""))


def is_output_length_stop(reason: str) -> bool:
    return str(reason or "").lower() in {"length", "max_tokens", "max_output_tokens"}


def last_user_index(messages: list[Message]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return -1


def image_anchor_index(messages: list[Message]) -> int:
    """이미지를 붙일 user 메시지의 위치.

    도구 루프에서는 (JSON 폴백일 때) 도구 결과가 user 메시지로 뒤에 쌓인다. 그때도 이미지는
    원래 질문에 붙어 있어야 하므로, 루프가 `images_anchor=True`로 표시한 메시지를 우선한다.
    """
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("images_anchor") and messages[index].get("role") == "user":
            return index
    return last_user_index(messages)


def compact_messages(messages: list[Message], level: int = 1) -> list[Message]:
    """컨텍스트 초과 시 재시도용 축약본. system은 남기고, 최근 몇 개만 짧게 잘라 보낸다."""
    system = next((message for message in messages if message.get("role") == "system"), None)
    tail = [message for message in messages if message.get("role") != "system"][-(3 if level >= 2 else 5):]
    # tool 결과는 짝이 되는 assistant tool_calls 없이 보낼 수 없다.
    while tail and tail[0].get("role") == "tool":
        tail = tail[1:]
    system_limit = 1800 if level >= 2 else 3200
    content_limit = 1000 if level >= 2 else 1800
    result: list[Message] = []
    if system is not None:
        result.append({**system, "content": clip(str(system.get("content") or ""), system_limit)})
    for message in tail:
        limit = content_limit * 2 if message.get("role") == "tool" else content_limit
        result.append({**message, "content": clip(str(message.get("content") or ""), limit)})
    return result


class Provider(ABC):
    """provider 하나 = (엔드포인트, 키, 모델) 조합 하나. 요청마다 새로 만들고 끝나면 닫는다."""

    name: str = ""
    is_local: bool = False

    def __init__(self, *, model: str, api_key: str = "", base_url: str = "", timeout: float = 180.0):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def cache_namespace(self) -> str:
        return f"{self.name}:{self.base_url}:{self.model}"

    @abstractmethod
    async def analyze(self, messages: list[Message], images: list[ModelImage] | None = None,
                      tools: list[ToolSpec] | None = None, *, temperature: float = 0.2) -> ModelResponse:
        ...

    @abstractmethod
    async def list_models(self) -> list[str]:
        ...

    async def aclose(self) -> None:
        return None
