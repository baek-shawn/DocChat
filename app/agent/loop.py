"""단일 tool-calling 루프.

    모델 호출 → tool_call 감지 → 실행 → 결과 재주입 → 반복

프레임워크도 서브에이전트도 없다. 여기에 더해 실사용에서 필요한 안전장치만 둔다.
  - 네이티브 tool-calling을 지원하지 않는 모델(예: Ollama의 gemma3) → 텍스트 JSON 규약으로 자동 전환
  - 같은 호출을 같은 인자로 되풀이하면 중단하고 최종 답을 요구
  - 최대 스텝 초과 시 도구 없이 최종 답을 요구
  - 출력 길이 한도로 끊기면 이어 쓰게 함
  - `<think>` 같은 내부 추론 블록은 사용자에게 보이지 않게 제거
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .. import config, trace
from ..pipeline.images import ModelImage
from ..providers.base import (Message, ModelResponse, Provider, ToolCall, ToolSpec, ToolsUnsupportedError,
                              is_output_length_stop, is_reasoning_runaway, last_user_index)
from ..providers.reasoning import describe_reasoning_progress
from .prompts import (AFTER_TOOL_RESULT, CONTINUE_ANSWER, FORCE_FINAL_ANSWER, RESEND_VALID_TOOL_JSON,
                      json_tool_protocol, json_tool_reminder)

_MAX_MALFORMED_ENVELOPES = 2

ExecuteTool = Callable[[ToolCall], Awaitable[str]]
Progress = Callable[[str], None]

# 추론 제어(Step 6)로 답을 받지 못했을 때 사용자에게 보이는 안내문. 추론 글은 답으로 내보내지 않는다.
REASONING_RUNAWAY_NOTICE = ("모델의 추론이 끝나지 않아(같은 내용을 반복하거나 추론 예산을 넘어) 답변을 받지 못했습니다. "
                            "다시 보내거나, 설정에서 \"추론 끄기\"를 켜 보세요.")
REASONING_LENGTH_NOTICE = ("모델이 추론만 하다 출력 한도에 닿아 답변을 내지 못했습니다. "
                           "다시 보내거나, 설정에서 \"추론 끄기\"를 켜 보세요.")

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_TAIL = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE_HEAD = re.compile(r"^.*?</think>", re.IGNORECASE | re.DOTALL)
_TOOL_CALL_BLOCK = re.compile(r"<tool_call\b[^>]*>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
_FENCE = re.compile(r"```(?:json|tool_call|tool_code)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass
class LoopResult:
    text: str
    steps: int = 0
    used_json_fallback: bool = False
    stopped_reason: str = ""  # "" | "max_steps" | "repeated_tool_call" | "malformed_tool_call" | "reasoning_runaway" | "reasoning_length"
    model_calls: int = 0      # 답변 모델을 부른 횟수(도구 안의 비전 호출은 포함하지 않는다)
    reasoning_forced: int = 0  # 추론을 끊고 답으로 넘긴 호출 수(Step 6, 소프트)
    reasoning_stops: int = 0   # 그래도 답을 받지 못해 중단한 호출 수(하드)
    # 답변 호출의 응답들 — 호출부가 추론 토큰·조치를 집계한다(`VisionUsage.count_reasoning`)
    responses: list[ModelResponse] = field(default_factory=list)


def strip_reasoning(text: str) -> str:
    """추론형 모델의 <think> 블록을 없앤다(닫는 태그만 남았거나 열린 채 끝난 경우 포함)."""
    raw = _THINK_BLOCK.sub("", str(text or ""))
    if re.search(r"</think>", raw, re.IGNORECASE):
        raw = _THINK_CLOSE_HEAD.sub("", raw, count=1)
    return _THINK_OPEN_TAIL.sub("", raw).strip()


# --------------------------------------------------------------------------- 본문 속 도구 호출 파싱
def _calls_from_value(value: Any, tool_names: set[str]) -> list[ToolCall]:
    if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
        items = value["tool_calls"]
    elif isinstance(value, list):
        items = value
    else:
        items = [value]
    calls: list[ToolCall] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else item
        name = function.get("name") or function.get("tool")
        if name not in tool_names:
            continue
        arguments = function.get("arguments", function.get("args", function.get("parameters", {})))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        calls.append(ToolCall(name=name, arguments=arguments if isinstance(arguments, dict) else {}))
    return calls


def parse_embedded_tool_calls(text: str, tool_names: set[str]) -> tuple[list[ToolCall], str]:
    """본문에 JSON으로 적힌 도구 호출을 찾는다. (호출 목록, 호출 부분을 뺀 나머지 글)을 돌려준다.

    알려진 도구 이름과 일치할 때만 호출로 인정한다 — 평범한 JSON 답변을 도구 호출로 오인하지 않기 위해서다.
    """
    if not text or not tool_names:
        return [], text
    candidates: list[tuple[str, str]] = []  # (JSON 후보, 본문에서 지울 원문)
    candidates += [(match.group(1).strip(), match.group(0)) for match in _TOOL_CALL_BLOCK.finditer(text)]
    candidates += [(match.group(1).strip(), match.group(0)) for match in _FENCE.finditer(text)]
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            _, end = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        candidates.append((text[match.start():end], text[match.start():end]))
        break  # 가장 바깥 객체 하나면 충분하다
    for raw, span in candidates:
        try:
            value = json.loads(raw)
        except ValueError:
            continue
        calls = _calls_from_value(value, tool_names)
        if calls:
            return calls, text.replace(span, "").strip()
    salvaged = _salvage_tool_calls(text, tool_names)
    if salvaged:
        return salvaged, ""
    return [], text


_NAMED_CALL = re.compile(r'"(?:name|tool)"\s*:\s*"([A-Za-z_][\w]*)"')
_ARGUMENTS_KEY = re.compile(r'"(?:arguments|args|parameters)"\s*:\s*')


def _salvage_tool_calls(text: str, tool_names: set[str]) -> list[ToolCall]:
    """바깥 괄호가 어긋난 봉투에서 호출을 건져 낸다.

    gemma3 실측: `{"tool_calls": [{"name": "inspect_visual", "arguments": {...}]} ]}` — 호출 객체를 닫는 `}`를
    빼먹어 전체는 JSON이 아니지만, 도구 이름과 **arguments 객체 자체는 멀쩡하다.** 그래서 알려진 도구 이름 뒤의
    arguments 객체만 따로 디코드한다. 이름이 알려진 도구일 때만 인정하므로 평범한 JSON 답변을 오인하지 않는다.
    """
    decoder = json.JSONDecoder()
    calls: list[ToolCall] = []
    for match in _NAMED_CALL.finditer(text):
        if match.group(1) not in tool_names:
            continue
        key = _ARGUMENTS_KEY.search(text, match.end())
        if key is None or key.end() >= len(text) or text[key.end()] != "{":
            continue
        try:
            arguments, _ = decoder.raw_decode(text, key.end())
        except ValueError:
            continue
        if isinstance(arguments, dict):
            calls.append(ToolCall(name=match.group(1), arguments=arguments))
    return calls


def looks_like_tool_envelope(text: str, tool_names: set[str]) -> bool:
    """호출로 읽어 내지는 못했지만 도구 봉투를 쓰려던 흔적이 있는가 — 이런 글은 절대 사용자에게 보여 주면 안 된다."""
    stripped = str(text or "").strip()
    mentions_tool = any(f'"{name}"' in stripped for name in tool_names)
    return mentions_tool and ('"tool_calls"' in stripped or "<tool_call" in stripped.lower()
                              or stripped.startswith(("{", "```")))


# --------------------------------------------------------------------------- 루프
def _with_protocol(messages: list[Message], tools: list[ToolSpec]) -> list[Message]:
    schema = json.dumps([{"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools],
                        ensure_ascii=False)
    protocol = json_tool_protocol(schema)
    result = [dict(message) for message in messages]
    # 소형 모델은 시스템 프롬프트보다 **질문의 마지막 줄**을 훨씬 강하게 따른다. 규약을 시스템 프롬프트에만 두면
    # gemma3는 도구를 건너뛰고 바로 산문으로 답했다(실측) → 질문 끝에 짧은 리마인더를 한 번 더 붙인다.
    anchor = next((m for m in reversed(result) if m.get("role") == "user" and m.get("images_anchor")), None)
    if anchor is None:
        anchor = next((m for m in reversed(result) if m.get("role") == "user"), None)
    if anchor is not None:
        anchor["content"] = f"{anchor.get('content') or ''}\n\n{json_tool_reminder([tool.name for tool in tools])}"
    for message in result:
        if message.get("role") == "system":
            message["content"] = f"{message.get('content') or ''}\n\n{protocol}"
            return result
    return [{"role": "system", "content": protocol}, *result]


def _signature(calls: list[ToolCall]) -> str:
    return json.dumps([[call.name, call.arguments] for call in calls], sort_keys=True, ensure_ascii=False)


def _append_user(transcript: list[Message], content: str) -> None:
    """user 메시지가 연달아 오면 하나로 합친다 — 역할 교대를 강제하는 챗 템플릿(gemma 등)을 위해서다."""
    if transcript and transcript[-1].get("role") == "user":
        transcript[-1]["content"] = f"{transcript[-1].get('content') or ''}\n\n{content}"
    else:
        transcript.append({"role": "user", "content": content})


def _plain(transcript: list[Message]) -> list[Message]:
    """네이티브 tool 메시지를 평문으로 바꾼다.

    도구 없이(tools=None) 최종 답을 요구할 때 쓴다. 일부 API는 tools 정의 없이
    tool_calls/tool 메시지가 들어오면 요청을 거절하기 때문이다.
    """
    plain: list[Message] = []
    for message in transcript:
        role = message.get("role")
        if role == "tool":
            _append_user(plain, f"TOOL RESULT ({message.get('name') or 'tool'}):\n{message.get('content') or ''}")
        elif role == "assistant" and message.get("tool_calls"):
            calls = "; ".join(f"{call.name}({json.dumps(call.arguments, ensure_ascii=False)})"
                              for call in message["tool_calls"])
            text = str(message.get("content") or "").strip()
            plain.append({"role": "assistant", "content": f"{text}\n[called tools: {calls}]".strip()})
        elif role == "user":
            _append_user(plain, str(message.get("content") or ""))
        else:
            plain.append({"role": role, "content": str(message.get("content") or "")})
    return plain


async def run_tool_loop(
    provider: Provider,
    messages: list[Message],
    *,
    images: list[ModelImage] | None = None,
    tools: list[ToolSpec] | None = None,
    execute: ExecuteTool | None = None,
    on_progress: Progress | None = None,
    describe: Callable[[ToolCall], str] | None = None,
    max_steps: int | None = None,
    temperature: float = 0.2,
    language_hint: str = "",
    on_live: Progress | None = None,
) -> LoopResult:
    """language_hint: "최종 답변을 한국어로" 같은 한 줄. 메시지에 미리 박지 않고 루프가 상황을 보고 붙인다.
    on_live: 생성 중의 추론 진행("추론 중… n토큰")처럼 같은 줄을 갱신해 보여 줄 문구(Step 6).

    gemma3 실측(각 6회): JSON 폴백에서 도구를 제공하는 호출의 user 턴에 언어 지시가 **어디에든** 있으면
    도구 호출이 0/6으로 죽고, 빼면 위치 질문 6/6 · 비위치 질문 오호출 0/6이었다. 그래서
      - JSON 폴백 + 도구 제공 호출  → 힌트 생략(대신 도구 리마인더)
      - 그 밖의 모든 호출            → 힌트 부착 (네이티브 tool-calling, 도구 실행 후 최종 답, 강제 종료, 이어쓰기)
    네이티브 tool-calling 모델에서 힌트가 무해한지는 아직 실측하지 못했다.
    """
    notify = on_progress or (lambda _message: None)
    tools = list(tools or []) if execute is not None else []
    tool_names = {tool.name for tool in tools}
    max_steps = max_steps or config.MAX_TOOL_STEPS

    base = [dict(message) for message in messages]
    anchor = last_user_index(base)
    if anchor >= 0:
        base[anchor]["images_anchor"] = True  # 도구 결과가 뒤에 쌓여도 이미지는 원래 질문에 붙는다

    transcript: list[Message] = []   # 루프 중에 쌓이는 assistant/tool 메시지
    use_fallback = False
    last_signature, repeats, steps = "", 0, 0
    stopped = ""
    malformed = 0
    model_calls = 0
    reasoning_forced = 0
    reasoning_stops = 0
    responses: list[ModelResponse] = []
    notice_given = False      # 추론 제어가 답 대신 안내문을 냈다 → 이어 쓰기·정리를 더 하지 않는다
    # 추론을 켠 답변 호출의 추론 예산(Step 6). 넘거나 반복하면 provider가 추론을 끊고 답만 이어 쓰게 한다.
    watch = (lambda info: on_live(describe_reasoning_progress(info))) if on_live is not None else None

    async def ask(request: list[Message], offered: list[ToolSpec] | None) -> ModelResponse:
        nonlocal model_calls, reasoning_forced
        model_calls += 1
        reply = await provider.analyze(request, images, offered, temperature=temperature,
                                       reasoning_budget=config.reasoning_budget("answer"), on_reasoning=watch)
        if reply.forced:
            reasoning_forced += 1
        responses.append(reply)
        return reply

    def hinted() -> list[Message]:
        """원래 질문(anchor) 끝에 언어 힌트를 붙인 사본."""
        if not language_hint or anchor < 0:
            return base
        copy = [dict(message) for message in base]
        copy[anchor]["content"] = f"{copy[anchor].get('content') or ''}\n\n{language_hint}"
        return copy

    async def call_model(with_tools: bool) -> ModelResponse:
        nonlocal use_fallback
        if with_tools and not use_fallback:
            try:
                return await ask(hinted() + transcript, tools)
            except ToolsUnsupportedError as error:
                use_fallback = True
                notify("이 모델은 네이티브 도구 호출을 지원하지 않아 JSON 방식으로 전환합니다…")
                trace.note("loop", "네이티브 도구 호출 거절 → JSON 방식으로 전환", error=str(error)[:500])
        if with_tools:
            # 도구가 이미 한 번 실행된 뒤라면(transcript 있음) 최종 답을 기대하는 상황이라 힌트는 결과 뒤에 붙어 있다.
            return await ask(_with_protocol(base, tools) + transcript, None)
        return await ask(hinted() + _plain(transcript), None)

    response = ModelResponse()
    text = ""
    while True:
        offer_tools = bool(tools) and not stopped
        response = await call_model(offer_tools)
        text = strip_reasoning(response.text)
        if "<think" in (response.text or "").lower() or "</think>" in (response.text or "").lower():
            trace.note("cleanup", "본문에서 추론 블록 제거", removedChars=len(response.text or "") - len(text))
        if is_reasoning_runaway(response.finish_reason):
            # 추론이 끝나지 않아 이어 쓰기로도 답을 받지 못했다(Step 6 하드). 추론 글을 답으로 내보내지 않는다.
            reasoning_stops += 1
            stopped, notice_given = "reasoning_runaway", True
            trace.note("loop", "추론이 끝나지 않아 답을 받지 못함 → 안내문으로 대체", reason=response.runaway)
            text = REASONING_RUNAWAY_NOTICE
            break
        if is_output_length_stop(response.finish_reason) and not text and response.reasoning:
            # 추론만 하다 출력 한도에 닿았다(예산 없이 서버 상한만 있을 때). 이어 쓸 답이 없으니 안내만 한다.
            stopped, notice_given = "reasoning_length", True
            trace.note("loop", "추론 도중 출력 한도에 닿아 답을 받지 못함 → 안내문으로 대체",
                       reasoningChars=len(response.reasoning))
            text = REASONING_LENGTH_NOTICE
            break
        calls = list(response.tool_calls) if offer_tools else []
        if offer_tools and not calls:
            # 네이티브 모드여도 일부 서버는 호출을 본문에 적어 보낸다 → 두 형태를 모두 받아 준다.
            calls, remainder = parse_embedded_tool_calls(text, tool_names)
            if calls:
                text = remainder
                trace.note("loop", f"본문에 적힌 도구 호출 JSON을 읽음 ({len(calls)}건)",
                           calls=[trace.describe_tool_call(call) for call in calls])
        if offer_tools and not calls and looks_like_tool_envelope(text, tool_names):
            # 도구를 부르려다 JSON을 망가뜨린 경우다. 이 글을 '최종 답'으로 사용자에게 내보내면 안 된다.
            malformed += 1
            transcript.append({"role": "assistant", "content": response.text})
            if malformed <= _MAX_MALFORMED_ENVELOPES:
                notify("도구 호출 형식이 올바르지 않아 다시 요청하는 중…")
                trace.note("loop", f"도구 호출 JSON이 깨져 다시 요청 ({malformed}/{_MAX_MALFORMED_ENVELOPES})",
                           text=trace.clip(text))
                _append_user(transcript, RESEND_VALID_TOOL_JSON)
            else:
                stopped = "malformed_tool_call"
                trace.note("loop", "도구 호출 JSON이 계속 깨져 도구 없이 최종 답을 요구", text=trace.clip(text))
                _append_user(transcript, FORCE_FINAL_ANSWER)
            continue
        if not calls:
            break

        signature = _signature(calls)
        repeats = repeats + 1 if signature == last_signature else 1
        last_signature = signature
        if repeats >= config.REPEATED_TOOL_CALL_LIMIT:
            stopped = "repeated_tool_call"
        elif steps >= max_steps:
            stopped = "max_steps"
        if stopped:
            notify("도구 호출을 멈추고 답변을 정리하는 중…")
            trace.note("loop", ("같은 도구 호출이 되풀이돼 중단" if stopped == "repeated_tool_call"
                                else f"도구 호출 상한({max_steps}회)에 닿아 중단") + " → 도구 없이 최종 답을 요구",
                       reason=stopped, steps=steps, repeats=repeats)
            transcript.append({"role": "assistant", "content": text or f"[called tools: {signature}]"})
            _append_user(transcript, FORCE_FINAL_ANSWER)
            continue

        steps += 1
        if use_fallback:
            transcript.append({"role": "assistant", "content": response.text or _signature(calls)})
        else:
            transcript.append({"role": "assistant", "content": text, "tool_calls": calls,
                               "raw": response.raw_assistant})
        for call in calls:
            notify(describe(call) if describe else f"도구 실행 중… ({call.name})")
            async with trace.scope("tool", describe(call) if describe else f"도구 실행 · {call.name}", name=call.name,
                                   arguments=call.arguments, step=steps) as span:
                try:
                    result = await execute(call)  # type: ignore[misc]
                except Exception as error:  # 도구의 예기치 못한 실패도 모델에게 알려 스스로 복구하게 한다.
                    result = f"ERROR: {error}"
                    span.status = "failed"
                span.set(result=trace.clip(result), resultChars=len(result))
            if use_fallback:
                _append_user(transcript, f"TOOL RESULT ({call.name}):\n{result}")
            else:
                transcript.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result})
        if use_fallback:
            # 질문에 붙인 리마인더 때문에 같은 도구를 다시 부르지 않도록, 결과 뒤에서 방향을 돌려 준다.
            # 도구는 이미 실행됐으므로 여기서는 언어 힌트를 붙여도 도구 호출을 방해하지 않는다.
            _append_user(transcript, f"{AFTER_TOOL_RESULT}\n{language_hint}".strip())
        notify("결과를 바탕으로 답변을 작성하는 중…")

    # 출력 길이 한도에서 끊겼으면 이어 쓰게 한다.
    continuations = 0
    while not notice_given and is_output_length_stop(response.finish_reason) and continuations < config.MAX_CONTINUATIONS:
        continuations += 1
        notify(f"답변이 길어 이어서 작성하는 중… ({continuations})")
        trace.note("loop", f"답변이 출력 한도에서 끊겨 이어 쓰기 요청 ({continuations}/{config.MAX_CONTINUATIONS})")
        transcript.append({"role": "assistant", "content": text[-24_000:]})
        _append_user(transcript, CONTINUE_ANSWER)
        response = await ask(hinted() + _plain(transcript), None)
        addition = strip_reasoning(response.text)
        if not addition or text.endswith(addition):
            break
        text = f"{text}\n\n{addition}"

    if tool_names and not notice_given and looks_like_tool_envelope(text, tool_names):
        # 마지막 안전망: 끝까지 도구 봉투만 내놓는 모델. 날 JSON을 답변이라고 보여 주지 않는다.
        trace.note("cleanup", "최종 답이 도구 호출 JSON이라 안내문으로 대체", text=trace.clip(text))
        text = ("모델이 도구 호출 형식(JSON)을 올바르게 만들지 못해 답변을 완성하지 못했습니다. "
                "다시 시도하거나, 도구 호출을 더 안정적으로 지원하는 모델을 선택해 주세요.")
        stopped = stopped or "malformed_tool_call"
    trace.note("loop", "도구 루프 종료", steps=steps, modelCalls=model_calls, jsonFallback=use_fallback,
               stoppedReason=stopped or None, continuations=continuations,
               reasoningForced=reasoning_forced or None, reasoningStops=reasoning_stops or None)
    return LoopResult(text=text, steps=steps, used_json_fallback=use_fallback, stopped_reason=stopped,
                      model_calls=model_calls, reasoning_forced=reasoning_forced, reasoning_stops=reasoning_stops,
                      responses=responses)
