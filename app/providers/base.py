"""모델 연결의 단일 인터페이스: `analyze(messages, images=None, tools=None) -> ModelResponse`.

호출마다 달라지는 선택 사항은 키워드 인자로 받는다.
    temperature       : 표본 온도
    disable_thinking  : 이 호출만 추론을 끈다(Step 6-0). provider를 만들 때 이미 껐다면 그대로 꺼져 있다.
                        추론을 끄는 방법이 있는 로컬(OpenAI 호환) provider만 따르고 나머지는 무시한다
    max_tokens        : 이 호출의 출력 토큰 상한(추론 토큰 포함). None이면 보내지 않는다. 역시 로컬 provider만 따른다
    reasoning_budget  : 이 호출의 추론 토큰 예산(Step 6). 추론을 켠 로컬 호출만 스트리밍으로 받으며 센다 — 넘거나 반복이
                        보이면 추론을 끊고 답만 이어 쓰게 한 뒤(소프트), 그래도 안 되면 finish_reason="reasoning_runaway"(하드)
    on_reasoning      : 추론 진행(토큰 수·초)과 소프트/하드 조치를 알리는 콜백(`reasoning.ReasoningProgress`)
    reasoning_effort  : 이 호출의 추론 수준(Step 6 2차, 예: Qwen3.8의 "low" / "medium" / "xhigh"). 추론을 켠 로컬 호출에만
                        실린다. 비어 있으면 보내지 않는다(모델 기본 수준). 서버가 값을 받지 않으면 `ReasoningEffortError`

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
from .reasoning import OnReasoning

Message = dict[str, Any]
# 추론이 끝나지 않아(예산 초과·반복) 소프트 조치로도 답을 받지 못한 호출의 finish_reason(Step 6).
REASONING_RUNAWAY = "reasoning_runaway"


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
    # 서버가 본문과 따로 떼어 준 추론 글(vLLM·llama.cpp의 reasoning_content). 떼어 주지 않는 서버면 빈 문자열이다.
    # 사용자에게 보여 주지 않는다 — 출력 한도에서 끊긴 본문이 추론인지 답인지 가릴 때만 쓴다.
    reasoning: str = ""
    # 서버가 알려 준 토큰 수(없으면 None). 출력 토큰에는 추론 토큰이 포함된다.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # 추론 제어(Step 6): 추론을 켠 로컬 호출을 스트리밍으로 받았을 때만 채워진다.
    reasoning_tokens: int | None = None     # 추론 토큰 수(스트림 조각 수 기준, 근사)
    reasoning_seconds: float | None = None  # 호출 시작부터 추론이 끝나기(또는 끊기)까지
    forced: str = ""                        # 추론을 끊고 답으로 넘겼으면 그 사유: "budget" | "repeat"
    runaway: str = ""                       # 하드 중단 사유(finish_reason == REASONING_RUNAWAY): "budget" | "repeat" | "empty" | "rejected"


class ProviderError(Exception):
    """사용자에게 보여 줄 수 있는 모델 호출 오류."""


class ToolsUnsupportedError(ProviderError):
    """엔드포인트/모델이 네이티브 tool-calling을 지원하지 않는다 → 루프가 JSON 폴백으로 전환한다."""


class ContextWindowError(ProviderError):
    """프롬프트가 모델 컨텍스트를 넘었다."""


class ReasoningEffortError(ProviderError):
    """서버(모델의 채팅 템플릿)가 요청한 추론 수준을 받지 않았다(Step 6 2차).

    설정이 틀린 것이라 다시 보내도 같은 결과다 → 전사 재시도·도구 오류·타일의 "일부 실패"로 다루지 않고
    턴의 오류로 올린다. 값을 빼고 다시 보내지도 않는다(요청한 수준과 다른 수준으로 돈 답이 기록에 섞인다).
    """


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


def is_reasoning_runaway(reason: str) -> bool:
    """추론이 끝나지 않아 하드 중단한 호출인가(Step 6). 다시 보내지 않고 실패로 다룬다."""
    return str(reason or "") == REASONING_RUNAWAY


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

    def can_disable_thinking(self) -> bool:
        """이 provider가 추론을 끄라는 요청을 실제로 보내는가(답변 메타데이터에 적을 때 쓴다)."""
        return False

    def thinking_off_for_every_call(self) -> bool:
        """"추론 끄기 — 모든 호출"로 만들어졌는가. 그러면 `analyze(disable_thinking=False)`여도 추론을 끄고 보낸다."""
        return False

    def can_set_reasoning_effort(self) -> bool:
        """이 provider가 추론 수준(`reasoning_effort`)을 실어 보내는가(Step 6 2차). 로컬(OpenAI 호환)만 그렇다."""
        return False

    @abstractmethod
    async def analyze(self, messages: list[Message], images: list[ModelImage] | None = None,
                      tools: list[ToolSpec] | None = None, *, temperature: float = 0.2,
                      disable_thinking: bool = False, max_tokens: int | None = None,
                      reasoning_budget: int | None = None, on_reasoning: OnReasoning | None = None,
                      reasoning_effort: str | None = None) -> ModelResponse:
        ...

    @abstractmethod
    async def list_models(self) -> list[str]:
        ...

    async def aclose(self) -> None:
        return None
