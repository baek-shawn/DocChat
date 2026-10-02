"""Step 6 1차 — 추론 제어: 추론을 켠 로컬 호출을 스트리밍으로 받으며 예산 초과·반복을 잡고,
추론을 끊고 답만 이어 쓰게 한 뒤(소프트), 그래도 안 되면 중단한다(하드). 답변·bbox·전사 호출 모두.

실측(사용자 vLLM Qwen3.5, 2026-10-01): 스트림 조각 하나가 토큰 하나(평균 3.8자), 이어 쓰기(continue_final_message)는
스트리밍으로 받아야 답이 content로 온다. 반복 표본은 test_runaway_guards.RUNAWAY_REASONING(타일 r2c3의 8줄 순환).
"""
from __future__ import annotations

import json

from app import config
from app.agent.loop import REASONING_LENGTH_NOTICE, REASONING_RUNAWAY_NOTICE
from app.agent.tools import ToolContext, execute_tool
from app.attachments import Attachment
from app.pipeline.images import VisionUsage
from app.pipeline.ocr import build_ocr_reader
from app.providers.base import ToolCall, ToolSpec, is_reasoning_runaway
from app.providers.openai_compat import OpenAICompatProvider
from app.providers.reasoning import (FORCED_END_NOTE, InlineThinkSplitter, ReasoningMonitor, ReasoningProgress,
                                     RepetitionDetector, describe_reasoning_progress)
from conftest import chat_body, upload
from mock_openai import is_grounding_call, is_ocr_call, thinking_disabled
from pdf_factory import png_bytes
from test_ocr_evidence import page
from test_runaway_guards import LOOP, QUESTION, RUNAWAY_REASONING, local

# 실측 4줄 순환(Qwen3.5가 "안녕"에 답하다 맴돈 추론)
CYCLE4 = ["Wait, I'll check if I should add a greeting in English too.", "No.", "Okay.", "Let me write the reply."]
# 400줄이 모두 다른 긴 추론 — 반복은 아니고 예산만 넘긴다
LONG_REASONING = "\n".join(f"Consider item {number} of the drawing carefully before deciding." for number in range(400))
INSPECT = ToolSpec("inspect_visual", "inspect", {"type": "object", "properties": {}})


def answer_calls(mock):
    return [body for body in mock.requests if not is_ocr_call(body) and not is_grounding_call(body)]


# --------------------------------------------------------------------------- 감지(단위)
def test_repetition_detector_catches_the_real_loops_and_ignores_normal_reasoning():
    detector, fed = RepetitionDetector(), 0
    for line in RUNAWAY_REASONING.split("\n"):
        fed += 1
        if detector.feed(line + "\n"):
            break
    # 첫 줄 + 8줄 묶음 × 3번 = 25줄째에 잡힌다. 40번 되풀이를 다 기다리지 않는다.
    assert detector.cycle == LOOP and fed == 1 + len(LOOP) * config.REASONING_REPEAT_COUNT

    detector, fed = RepetitionDetector(), 0
    for line in CYCLE4 * 5:
        fed += 1
        if detector.feed(line + "\n"):
            break
    assert detector.cycle == CYCLE4 and fed == len(CYCLE4) * config.REASONING_REPEAT_COUNT

    # 정상 추론(서로 다른 줄)과 짧은 줄의 반복("Okay." 셋)은 반복이 아니다.
    detector = RepetitionDetector()
    assert not any(detector.feed(f"Step {number}: compute part {number} of the drawing.\n") for number in range(300))
    detector = RepetitionDetector()
    assert not any(detector.feed("Okay.\n") for _ in range(10))
    # 줄이 조각 경계에서 잘려 와도 같다. 개행이 와야 한 줄로 센다.
    detector = RepetitionDetector()
    text = "\n".join(LOOP * 3) + "\n"
    assert any(detector.feed(text[index:index + 3]) for index in range(0, len(text), 3))
    detector = RepetitionDetector()
    assert not detector.feed("\n".join(LOOP * 3))         # 마지막 줄에 개행이 없으면 아직 완성된 줄이 아니다


def test_monitor_counts_tokens_by_chunks_or_chars_and_keeps_its_verdict():
    seen: list[ReasoningProgress] = []
    monitor = ReasoningMonitor(budget=10, on_progress=seen.append)
    verdicts = [monitor.feed("ab") for _ in range(11)]           # 조각 11개 > 예산 10
    assert verdicts[-1] == "budget" and verdicts[-2] == "" and monitor.tokens == 11
    assert monitor.feed("more") == "budget"                        # 한 번 걸리면 그대로다
    assert seen and seen[0].stage == "reasoning" and seen[0].tokens >= 1
    monitor = ReasoningMonitor(budget=10)
    assert monitor.feed("x" * 100) == "budget"                     # 조각은 하나지만 글자 수/4 = 25 > 10
    monitor = ReasoningMonitor(budget=0)                           # 예산 없음: 크기로는 걸리지 않는다
    assert not any(monitor.feed(f"word {number} ") for number in range(500))
    assert monitor.feed("") == "" and monitor.chunks == 500        # 빈 조각은 세지 않는다


def test_inline_think_splitter_handles_tags_split_across_chunks():
    splitter = InlineThinkSplitter()
    text = "<think>\nplan A\nplan B\n</think>\nThe answer is 42."
    reasoning, content = "", ""
    for index in range(0, len(text), 3):
        piece_reasoning, piece_content = splitter.feed(text[index:index + 3])
        reasoning += piece_reasoning
        content += piece_content
    tail_reasoning, tail_content = splitter.flush()
    assert reasoning + tail_reasoning == "\nplan A\nplan B\n" and content + tail_content == "\nThe answer is 42."
    assert splitter.saw_think and not splitter.in_think

    splitter = InlineThinkSplitter()
    reasoning, content = splitter.feed("a < b and c > d <")
    tail_reasoning, tail_content = splitter.flush()
    assert reasoning + tail_reasoning == "" and content + tail_content == "a < b and c > d <"
    splitter = InlineThinkSplitter()                               # 닫히지 않은 채 끝나면 전부 추론이다
    reasoning, content = splitter.feed("<think>still thinking <")
    tail_reasoning, tail_content = splitter.flush()
    assert reasoning + tail_reasoning == "still thinking <" and content + tail_content == ""


def test_progress_messages():
    info = ReasoningProgress(stage="reasoning", tokens=1234, chars=5000, seconds=12.4, image="scan.pdf · page 1 · tile r1c2")
    assert describe_reasoning_progress(info) == "추론 중… 1,234토큰 · 12초 · scan.pdf · page 1 · tile r1c2"
    assert describe_reasoning_progress(ReasoningProgress("reasoning", 20, 80, 1.0)) == "추론 중… 20토큰 · 1초"
    assert "반복" in describe_reasoning_progress(ReasoningProgress("forced", 500, 0, 3, reason="repeat"))
    assert "예산(500토큰)" in describe_reasoning_progress(ReasoningProgress("forced", 500, 0, 3, reason="budget"))
    assert "중단" in describe_reasoning_progress(ReasoningProgress("runaway", 500, 0, 3, reason="repeat"))


# --------------------------------------------------------------------------- provider: 스트리밍 수신
async def test_only_thinking_on_local_calls_are_streamed(mock_llm):
    mock_llm.reset(lambda body: {"text": "answer", "reasoning": "short thought"})
    provider = local(mock_llm)
    cloud = OpenAICompatProvider(name="openai", model="gpt", api_key="k", base_url=mock_llm.base_url)
    try:
        off = await provider.analyze(QUESTION, disable_thinking=True, reasoning_budget=1000)
        on = await provider.analyze(QUESTION, reasoning_budget=1000)
        remote = await cloud.analyze(QUESTION, reasoning_budget=1000)
    finally:
        await provider.aclose()
        await cloud.aclose()
    assert [body.get("stream") for body in mock_llm.requests] == [None, True, None]
    assert mock_llm.requests[1]["stream_options"] == {"include_usage": True}
    assert thinking_disabled(mock_llm.requests[0]) and not thinking_disabled(mock_llm.requests[1])
    assert off.text == on.text == remote.text == "answer" and off.reasoning == on.reasoning == "short thought"
    assert on.finish_reason == "stop" and on.completion_tokens == 6                 # usage는 마지막 조각으로 온다
    assert on.reasoning_tokens == 4 and on.reasoning_seconds is not None and not on.forced
    assert off.reasoning_tokens is None and remote.reasoning_tokens is None


async def test_streamed_tool_calls_and_inline_think_are_assembled(mock_llm):
    mock_llm.reset(lambda body: {"text": "", "reasoning": "think first", "inline_think": True,
                                 "tool_calls": [{"name": "inspect_visual", "arguments": {"name": "a.png", "task": "find"}}]})
    provider = local(mock_llm)
    try:
        response = await provider.analyze(QUESTION, tools=[INSPECT])
    finally:
        await provider.aclose()
    (call,) = response.tool_calls
    assert call.name == "inspect_visual" and call.arguments == {"name": "a.png", "task": "find"} and call.id == "call_0"
    assert response.finish_reason == "tool_calls"
    assert response.reasoning == "think first" and response.text == ""     # 본문 속 <think>…</think>는 추론으로 갈라진다


# --------------------------------------------------------------------------- 소프트: 추론을 끊고 답만 이어 쓰기
async def test_budget_overrun_cuts_the_reasoning_and_continues_with_the_answer_only(mock_llm):
    def model(body):
        if body.get("continue_final_message"):
            return {"text": "The drawing number is PS-1."}
        return {"text": "never reached", "reasoning": LONG_REASONING}

    mock_llm.reset(model)
    seen: list[ReasoningProgress] = []
    provider = local(mock_llm)
    try:
        response = await provider.analyze(QUESTION, reasoning_budget=300, on_reasoning=seen.append)
    finally:
        await provider.aclose()
    assert response.text == "The drawing number is PS-1." and response.finish_reason == "stop"
    assert response.forced == "budget" and not response.runaway
    assert 300 < response.reasoning_tokens <= 330 and 0 < len(response.reasoning) < len(LONG_REASONING)
    first, follow = mock_llm.requests
    assert first.get("stream") and not first.get("continue_final_message")
    assert follow["continue_final_message"] is True and follow["add_generation_prompt"] is False and follow["stream"]
    assert follow["messages"][:-1] == first["messages"]                # 같은 요청 뒤에 assistant 메시지 하나만 더
    prefill = follow["messages"][-1]
    assert prefill["role"] == "assistant" and prefill["content"].startswith("<think>\n")
    assert prefill["content"].endswith(f"\n\n{FORCED_END_NOTE}\n</think>\n\n") and response.reasoning in prefill["content"]
    stages = [info.stage for info in seen]
    assert stages[0] == "reasoning" and stages[-1] == "forced" and seen[-1].reason == "budget"


async def test_repetition_cuts_the_reasoning_early(mock_llm):
    mock_llm.reset(lambda body: {"text": "There are two door symbols."} if body.get("continue_final_message")
                   else {"text": "", "reasoning": RUNAWAY_REASONING})
    provider = local(mock_llm)
    try:
        response = await provider.analyze(QUESTION, reasoning_budget=100_000)
    finally:
        await provider.aclose()
    assert response.text == "There are two door symbols." and response.forced == "repeat"
    # 40번 되풀이를 다 받지 않았다: 묶음이 3번 보인 데서 끊었다.
    assert response.reasoning.count("These are door symbols.") == config.REASONING_REPEAT_COUNT
    assert response.reasoning.startswith("The user wants me to identify")


async def test_hard_stop_when_the_continuation_also_fails(mock_llm):
    """이어 쓰기에서도 반복 / 빈 답 / 서버가 이어 쓰기를 거절 → 하드 중단. 더 보내지 않는다."""
    cases = {
        "repeat": lambda body: {"text": "", "reasoning": RUNAWAY_REASONING},
        "empty": lambda body: {"text": ""} if body.get("continue_final_message") else {"text": "", "reasoning": RUNAWAY_REASONING},
        "rejected": lambda body: ({"status": 400, "body": {"error": {"message": "unknown field continue_final_message"}}}
                                  if body.get("continue_final_message") else {"text": "", "reasoning": RUNAWAY_REASONING}),
    }
    provider = local(mock_llm)
    try:
        for expected, handler in cases.items():
            mock_llm.reset(handler)
            seen: list[ReasoningProgress] = []
            response = await provider.analyze(QUESTION, on_reasoning=seen.append)
            assert is_reasoning_runaway(response.finish_reason) and response.text == "" and response.tool_calls == []
            assert response.forced == "repeat" and response.runaway.startswith(expected), expected
            assert len(mock_llm.requests) == 2 and seen[-1].stage == "runaway"
    finally:
        await provider.aclose()


async def test_continuation_keeps_the_tools_and_only_the_output_share_of_the_limit(mock_llm):
    mock_llm.reset(lambda body: ({"text": "", "tool_calls": [{"name": "inspect_visual", "arguments": {"name": "a"}}]}
                                 if body.get("continue_final_message") else {"text": "", "reasoning": RUNAWAY_REASONING}))
    provider = local(mock_llm)
    try:
        response = await provider.analyze(QUESTION, tools=[INSPECT], max_tokens=4096, reasoning_budget=4000)
    finally:
        await provider.aclose()
    first, follow = mock_llm.requests
    assert first["max_tokens"] == 8096 and follow["max_tokens"] == 4096      # 상한 = 예산 + 출력 몫, 이어 쓰기는 출력 몫만
    assert "tools" in follow and [call.name for call in response.tool_calls] == ["inspect_visual"]
    assert response.forced == "repeat" and not response.runaway


# --------------------------------------------------------------------------- 전사·bbox 호출
async def test_ocr_runaway_fails_the_page_without_retry_and_a_forced_page_is_kept(mock_llm):
    mock_llm.reset(lambda body: {"text": "", "reasoning": RUNAWAY_REASONING})
    usage = VisionUsage()
    provider = local(mock_llm)
    try:
        read = build_ocr_reader(provider, usage=usage, disable_thinking=False)
        text = await read(page(1, png_bytes()), "Transcribe.")
        assert text.startswith("[OCR FAILED: the model's reasoning did not finish (repeat)") and "Not retried" in text
        assert len(mock_llm.requests) == 2                                  # 첫 호출 + 이어 쓰기, 재시도 없음
        assert (usage.ocr_calls, usage.ocr_reasoning_forced, usage.ocr_reasoning_stops) == (1, 1, 1)

        mock_llm.reset(lambda body: {"text": "DWG NO FA-7731-B"} if body.get("continue_final_message")
                       else {"text": "", "reasoning": RUNAWAY_REASONING})
        text = await read(page(2, png_bytes(text="OTHER")), "Transcribe.")
        assert text == "DWG NO FA-7731-B"
        assert (usage.ocr_calls, usage.ocr_reasoning_forced, usage.ocr_reasoning_stops) == (2, 2, 1)
        assert all(body["max_tokens"] in (4096 + config.REASONING_BUDGET_OCR, 4096) for body in mock_llm.requests)
    finally:
        await provider.aclose()


async def test_bbox_runaway_yields_no_boxes_and_a_warning_without_retry(mock_llm):
    mock_llm.reset(lambda body: {"text": "", "reasoning": RUNAWAY_REASONING})
    sheet = Attachment(name="sheet.png", mime="image/png", kind="image", data=png_bytes(), has_data=True,
                       width=320, height=200, send_to_model=True)
    provider = local(mock_llm)
    try:
        context = ToolContext(provider=provider, attachments=[sheet], disable_thinking=False)
        result = json.loads(await execute_tool(context, ToolCall("inspect_visual", {"name": "sheet.png", "task": "find stamp"})))
    finally:
        await provider.aclose()
    assert result["regions"] == [] and "reasoning did not finish" in result["warning"] and "not retried" in result["warning"]
    assert len(mock_llm.requests) == 2                                      # 첫 호출 + 이어 쓰기, 다시 묻지 않는다
    assert (context.usage.grounding_calls, context.usage.grounding_reasoning_forced, context.usage.grounding_reasoning_stops) == (1, 1, 1)
    assert "door symbols" not in json.dumps(result)                         # 추론 글은 도구 결과에 넘어가지 않는다


# --------------------------------------------------------------------------- /api/chat
def test_chat_turn_records_reasoning_actions_and_streams_live_progress(client, mock_llm, monkeypatch):
    monkeypatch.setattr(config, "REASONING_BUDGET_ANSWER", 500)             # DOCCHAT_REASONING_BUDGET_ANSWER=500
    mock_llm.reset(lambda body: {"text": "도면 번호는 PS-1입니다."} if body.get("continue_final_message")
                   else {"text": "never", "reasoning": LONG_REASONING})
    body = chat_body(mock_llm, "도면 번호 알려줘", disableThinking=False, stream=True)
    lines = [json.loads(line) for line in client.post("/api/chat", json=body).text.splitlines() if line.strip()]
    final = lines[-1]
    assert final["type"] == "final" and final["text"] == "도면 번호는 PS-1입니다."
    vision = final["meta"]["vision"]
    assert (vision["answerCalls"], vision["answerReasoningForced"], vision["answerReasoningStops"]) == (1, 1, 0)
    assert vision["reasoningTokens"] > 500                                 # 턴의 추론 토큰 합계(화면 표시용)
    assert final["meta"]["reasoningActions"] == [{"kind": "answer", "image": "", "reason": "budget", "stopped": False}]
    assert final["meta"]["reasoning"] == config.reasoning_settings()
    live = [event for event in lines if event["type"] == "progress" and event.get("live")]
    assert live and live[0]["message"].startswith("추론 중… ")
    assert any(event["message"].startswith("추론 예산") and "답으로 넘기는 중" in event["message"] for event in live)
    saved = client.get(f"/api/sessions/{final['conversationId']}").json()["messages"]
    assert saved[1]["meta"] == final["meta"]                                # 설정과 횟수가 답변과 함께 저장된다
    assert client.get("/api/health").json()["reasoning"] == config.reasoning_settings()


def test_chat_turn_that_cannot_be_rescued_shows_a_notice_instead_of_the_reasoning(client, mock_llm):
    mock_llm.reset(lambda body: {"text": "", "reasoning": RUNAWAY_REASONING})
    data = client.post("/api/chat", json=chat_body(mock_llm, "안녕", disableThinking=False)).json()
    assert data["text"] == REASONING_RUNAWAY_NOTICE
    assert len(answer_calls(mock_llm)) == 2                                 # 첫 호출 + 이어 쓰기. 이어 쓰기 루프를 더 돌지 않는다
    vision = data["meta"]["vision"]
    assert (vision["answerReasoningForced"], vision["answerReasoningStops"]) == (1, 1)
    assert "door symbols" not in json.dumps(data, ensure_ascii=False)       # 추론 글은 어디에도 나가지 않는다


def test_reasoning_that_runs_into_the_server_limit_gives_a_notice_not_a_continuation(client, mock_llm):
    """예산 없이 서버 상한만 있을 때(또는 예산보다 상한이 작을 때): 추론만 하다 length → 안내문, 이어 쓰기 요청 없음."""
    mock_llm.reset(lambda body: {"text": "", "reasoning": "x" * 200, "finish_reason": "length"})
    data = client.post("/api/chat", json=chat_body(mock_llm, "안녕", disableThinking=False)).json()
    assert data["text"] == REASONING_LENGTH_NOTICE and len(answer_calls(mock_llm)) == 1
    assert data["meta"]["vision"]["answerReasoningForced"] == 0


def test_trace_records_reasoning_tokens_and_the_soft_action(client, mock_llm, monkeypatch):
    monkeypatch.setenv("DOCCHAT_DEBUG_TRACE", "1")
    monkeypatch.setattr(config, "REASONING_BUDGET_ANSWER", 500)
    mock_llm.reset(lambda body: {"text": "답"} if body.get("continue_final_message")
                   else {"text": "never", "reasoning": LONG_REASONING})
    data = client.post("/api/chat", json=chat_body(mock_llm, "안녕", disableThinking=False)).json()
    document = client.get(f"/api/traces/{data['meta']['traceId']}").json()
    (call,) = [event for event in document["events"] if event["kind"] == "model"]
    assert call["data"]["reasoningBudget"] == config.REASONING_BUDGET_ANSWER and call["data"]["forced"] == "budget"
    assert call["data"]["reasoningTokens"] > config.REASONING_BUDGET_ANSWER and call["data"]["reasoningStage"] == "done"
    assert "forcedCycle" not in call["data"]                                # 예산으로 끊긴 것이라 되풀이된 줄이 없다
    assert any(event["label"] == "추론 예산 초과 → 추론을 끊고 답으로 넘김" for event in document["events"] if event["kind"] == "cleanup")
    (entry,) = [event for event in document["events"] if event["kind"] == "input"]
    assert entry["data"]["reasoning"] == config.reasoning_settings()
    assert not [event for event in document["events"] if event["kind"] == "progress" and event["label"].startswith("추론 중")]
