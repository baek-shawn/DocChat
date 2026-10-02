"""Step 6 2차-a — 추론 수준(reasoning_effort): 모델의 채팅 템플릿이 받는 값을 호출 종류별로 실어 보낸다.

  - 추론을 켠 로컬 호출에만 `chat_template_kwargs.reasoning_effort`로 실린다(추론을 끈 호출의 요청은 이전과 같다).
  - 값은 모델마다 달라 모양만 검사한다. 서버가 받지 않으면 빼고 다시 보내지 않고 **턴의 오류**로 끝낸다.
  - 원인이 추론 수준인지는 서버의 오류 문구가 아니라 1토큰짜리 확인 요청 둘(실은 것 / 뺀 것)로 가린다.
  - 전사의 OCR 캐시 키·쪽 기록, 답변 메타데이터, 트레이스에 수준이 남는다.

mock 서버의 거절 문구는 실제 Qwen3.8 서버(vLLM)가 `high`에 낸 문구 그대로다(2026-10-02 실모델 확인). 앱은 문구에 기대지 않는다.
"""
from __future__ import annotations

import json

import pytest

from app import config
from app.agent.tools import ToolContext, execute_tool
from app.chat_service import ChatRequest, plan_effort, plan_thinking
from app.db import sanitize_meta
from app.pipeline.ocr import OcrCache, build_ocr_reader, prepare_visual_ocr_evidence
from app.providers.base import ProviderError, ReasoningEffortError, ToolCall, ToolSpec, ToolsUnsupportedError
from app.providers.openai_compat import OpenAICompatProvider
from conftest import chat_body, upload
from mock_openai import (is_effort_probe, is_grounding_call, is_ocr_call, reasoning_effort, request_images,
                         thinking_disabled)
from pdf_factory import colors_in
from test_env_file import example_values, run_config
from test_ocr_evidence import ScriptedProvider, page
from test_runaway_guards import QUESTION, RUNAWAY_REASONING, local, locate, plan_upload
from test_tiling import PDF, SeeingProvider, box_reply, plan_image, scanned_drawing, seeing_model, seen

INSPECT = ToolSpec("inspect_visual", "inspect", {"type": "object", "properties": {}})
# Qwen3.8의 템플릿이 받는 값(2026-09-30 확인). 그 밖의 값은 템플릿이 예외를 던진다 → 서버가 400으로 돌려준다.
ACCEPTED = (None, "low", "medium", "xhigh")
REFUSAL = {"status": 400, "body": {"error": {"message": (
    "Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.")}}}


def qwen38(reply=seeing_model):
    """Qwen3.8 흉내: 받는 값이 아니면 템플릿이 실패한다. 그 밖에는 reply가 답한다."""
    def handler(body):
        if reasoning_effort(body) not in ACCEPTED:
            return REFUSAL
        return "p" if is_effort_probe(body) else reply(body)
    return handler


def real_calls(mock):
    """앱이 보낸 요청 중 확인 요청(1토큰)을 뺀 것."""
    return [body for body in mock.requests if not is_effort_probe(body)]


def answer_calls(mock):
    return [body for body in real_calls(mock) if not is_ocr_call(body) and not is_grounding_call(body)]


# --------------------------------------------------------------------------- 값·설정
def test_effort_values_are_checked_by_shape_only():
    assert config.normalize_reasoning_effort("  XHigh ") == "xhigh"
    assert config.normalize_reasoning_effort("") == "" and config.normalize_reasoning_effort(None) == ""
    assert [config.normalize_reasoning_effort(value) for value in ("low", "x-high", "level_2", "8")] == [
        "low", "x-high", "level_2", "8"]                 # 모델이 받는 값인지는 서버만 안다 — 여기서는 모양만 본다
    for bad in ("very high", "<b>", "low;", "-low", "a" * 25, "높게"):
        with pytest.raises(ValueError):
            config.normalize_reasoning_effort(bad)


def test_defaults_come_from_config_and_an_explicit_empty_value_means_do_not_send(monkeypatch):
    assert config.reasoning_effort_settings() == {"answer": "", "grounding": "", "ocr": ""}      # 기본: 보내지 않는다
    monkeypatch.setattr(config, "REASONING_EFFORT_GROUNDING", "medium")       # DOCCHAT_REASONING_EFFORT_GROUNDING=medium
    assert config.resolve_reasoning_effort(None, "grounding") == "medium"     # 요청에 없으면 서버 기본값
    assert config.resolve_reasoning_effort("", "grounding") == ""             # 빈 값을 보냈으면 "보내지 않음"
    assert config.resolve_reasoning_effort("LOW", "grounding") == "low"
    assert config.resolve_reasoning_effort(None, "answer") == ""


def test_effort_settings_can_be_set_from_the_env_file(tmp_path):
    shown = "config.REASONING_EFFORT_ANSWER, config.REASONING_EFFORT_GROUNDING, config.REASONING_EFFORT_OCR"
    assert run_config("off", shown) == "('', '', '')"
    path = tmp_path / "effort.env"
    path.write_text("DOCCHAT_REASONING_EFFORT_ANSWER=Medium   # 답변\nDOCCHAT_REASONING_EFFORT_GROUNDING=low\n"
                    "DOCCHAT_REASONING_EFFORT_OCR=        # 비워 둠\n", encoding="utf-8")
    assert run_config(str(path), shown) == "('medium', 'low', '')"
    # 알아볼 수 없는 값은 보내지 않는 쪽으로 둔다(오타가 서버 오류로 이어지지 않게).
    assert run_config("off", shown, DOCCHAT_REASONING_EFFORT_ANSWER="very high") == "('', '', '')"
    example = example_values()
    assert [example[f"DOCCHAT_REASONING_EFFORT_{kind}"] for kind in ("ANSWER", "GROUNDING", "OCR")] == ["", "", ""]


def test_the_ocr_record_includes_the_effort_only_for_thinking_transcriptions():
    assert config.image_mode_variant("whole", thinking=True, budget=4000, effort="low") == "whole+thinking:b4000:elow"
    assert config.image_mode_variant("whole", thinking=True, effort="xhigh") == "whole+thinking:exhigh"
    # 수준이 없으면 Step 6 1차의 기록과 같은 값이다 → 이미 전사해 둔 쪽을 다시 전사하지 않는다.
    assert config.image_mode_variant("whole", thinking=True, budget=4000) == "whole+thinking:b4000"
    assert config.image_mode_variant("whole", thinking=False, budget=4000, effort="low") == "whole"
    tile = config.image_mode_variant("tile")
    assert config.image_mode_variant("tile", thinking=True, budget=4000, effort="medium") == f"{tile}+thinking:b4000:emedium"


def test_health_reports_the_default_efforts(client, monkeypatch):
    assert client.get("/api/health").json()["reasoningEffort"] == {"answer": "", "grounding": "", "ocr": ""}
    monkeypatch.setattr(config, "REASONING_EFFORT_ANSWER", "medium")
    assert client.get("/api/health").json()["reasoningEffort"] == {"answer": "medium", "grounding": "", "ocr": ""}


def test_stored_meta_keeps_only_well_formed_efforts():
    meta = sanitize_meta({"reasoningEffort": {"answer": " LOW ", "grounding": "<b>", "ocr": 3, "other": "x"}})
    assert meta == {"reasoningEffort": {"answer": "low"}}
    assert sanitize_meta({"reasoningEffort": {"answer": ""}}) == {} and sanitize_meta({"reasoningEffort": "low"}) == {}


# --------------------------------------------------------------------------- provider: 싣는 곳
async def test_the_effort_rides_only_on_thinking_on_local_calls(mock_llm):
    mock_llm.reset(lambda body: {"text": "answer", "reasoning": "thought"})
    provider, everything_off = local(mock_llm), local(mock_llm, disable_thinking=True)
    cloud = OpenAICompatProvider(name="openai", model="gpt", api_key="k", base_url=mock_llm.base_url)
    try:
        await provider.analyze(QUESTION, reasoning_effort="low")
        await provider.analyze(QUESTION, disable_thinking=True, reasoning_effort="low")
        await provider.analyze(QUESTION)
        await provider.analyze(QUESTION, reasoning_effort="")
        await everything_off.analyze(QUESTION, reasoning_effort="low")
        await cloud.analyze(QUESTION, reasoning_effort="low")
    finally:
        for item in (provider, everything_off, cloud):
            await item.aclose()
    on, off, plain, empty, all_off, remote = mock_llm.requests
    assert on["chat_template_kwargs"] == {"reasoning_effort": "low"} and on["stream"] is True
    assert off["chat_template_kwargs"] == {"enable_thinking": False}          # 추론을 끈 호출의 요청은 이전과 같다
    assert all_off["chat_template_kwargs"] == {"enable_thinking": False}
    assert all("chat_template_kwargs" not in body for body in (plain, empty, remote))
    assert "reasoning_effort" not in on and "reasoning_effort" not in remote  # 요청 최상위 필드로는 보내지 않는다
    assert provider.can_set_reasoning_effort() and not cloud.can_set_reasoning_effort()


async def test_the_continuation_request_keeps_the_effort(mock_llm):
    """추론을 끊고 답만 이어 쓰는 요청(소프트)도 같은 수준을 싣는다 — 템플릿이 넣는 지시문이 같아야 한다."""
    mock_llm.reset(lambda body: {"text": "PS-1"} if body.get("continue_final_message")
                   else {"text": "", "reasoning": RUNAWAY_REASONING})
    provider = local(mock_llm)
    try:
        response = await provider.analyze(QUESTION, reasoning_effort="medium")
    finally:
        await provider.aclose()
    first, follow = mock_llm.requests
    assert response.text == "PS-1" and response.forced == "repeat"
    assert reasoning_effort(first) == reasoning_effort(follow) == "medium" and follow["continue_final_message"] is True


# --------------------------------------------------------------------------- provider: 서버가 받지 않는 값
async def test_a_refused_effort_is_an_error_and_is_never_dropped_or_sent_again(mock_llm):
    mock_llm.reset(qwen38(lambda body: "answer"))
    provider = local(mock_llm)
    try:
        with pytest.raises(ReasoningEffortError) as refused:
            await provider.analyze(QUESTION, tools=[INSPECT], reasoning_effort="high")
        # 원래 요청 + 확인 요청 둘(수준을 실은 것은 거절, 뺀 것은 통과). 수준을 빼고 다시 보낸 "진짜" 요청은 없다.
        original, with_effort, without = mock_llm.requests
        assert reasoning_effort(original) == "high" and not is_effort_probe(original) and "tools" in original
        assert is_effort_probe(with_effort) and reasoning_effort(with_effort) == "high" and "tools" not in with_effort
        assert is_effort_probe(without) and "chat_template_kwargs" not in without
        message = str(refused.value)
        assert '추론 수준 "high"' in message and "low · medium · xhigh" in message
        assert "Supported types are xhigh (default), medium, and low." in message   # 서버가 한 말을 그대로 보여 준다

        # 같은 값은 다시 보내지 않는다(타일 20장이 저마다 서버에 물어보지 않게).
        with pytest.raises(ReasoningEffortError):
            await provider.analyze(QUESTION, reasoning_effort="high")
        assert len(mock_llm.requests) == 3
        # 다른 값과, 추론을 끈 호출(수준이 실리지 않는다)은 그대로 된다.
        assert (await provider.analyze(QUESTION, reasoning_effort="xhigh")).text == "answer"
        assert (await provider.analyze(QUESTION, disable_thinking=True, reasoning_effort="high")).text == "answer"
    finally:
        await provider.aclose()


async def test_a_server_that_does_not_know_the_field_also_refuses_the_effort(mock_llm):
    """`chat_template_kwargs` 자체를 모르는 서버: 수준을 실을 방법이 없다 → 같은 오류."""
    mock_llm.reset(lambda body: {"status": 400, "body": {"error": {"message": "Unrecognized request argument: chat_template_kwargs"}}}
                   if "chat_template_kwargs" in body else "ok")
    provider = local(mock_llm)
    try:
        with pytest.raises(ReasoningEffortError):
            await provider.analyze(QUESTION, reasoning_effort="low")
        assert (await provider.analyze(QUESTION)).text == "ok"
    finally:
        await provider.aclose()


async def test_rejections_from_other_causes_are_not_blamed_on_the_effort(mock_llm):
    # 도구를 받지 않는 서버: 수준을 실은 확인 요청은 통과한다 → 수준 탓이 아니다 → 기존 처리(JSON 폴백으로 전환)로 간다.
    mock_llm.reset(lambda body: {"status": 400, "body": {"error": {"message": "this model does not support tools"}}}
                   if body.get("tools") else "ok")
    provider = local(mock_llm)
    try:
        with pytest.raises(ToolsUnsupportedError):
            await provider.analyze(QUESTION, tools=[INSPECT], reasoning_effort="low")
        assert [is_effort_probe(body) for body in mock_llm.requests] == [False, True]      # 확인 요청은 하나면 가려진다

        # 무엇을 보내도 거절하는 서버(모델 이름이 틀린 경우 등): 수준을 뺀 확인 요청도 거절된다 → 원래 오류 그대로.
        mock_llm.reset(lambda body: {"status": 400, "body": {"error": {"message": "model not loaded"}}})
        with pytest.raises(ProviderError) as other:
            await provider.analyze(QUESTION, reasoning_effort="low")
        assert not isinstance(other.value, ReasoningEffortError) and "model not loaded" in str(other.value)
        assert [is_effort_probe(body) for body in mock_llm.requests] == [False, True, True]
    finally:
        await provider.aclose()


# --------------------------------------------------------------------------- 호출 지점
async def test_ocr_and_bbox_calls_pass_the_effort_only_when_they_think():
    provider = ScriptedProvider(["TEXT", "TEXT", "TEXT"])
    await build_ocr_reader(provider, disable_thinking=False, reasoning_effort="low")(page(1, b"img"), "INSTRUCTION")
    await build_ocr_reader(provider, disable_thinking=True, reasoning_effort="low")(page(1, b"img"), "INSTRUCTION")
    await build_ocr_reader(provider, disable_thinking=False)(page(1, b"img"), "INSTRUCTION")
    assert [call["reasoning_effort"] for call in provider.calls] == ["low", None, None]

    surface, _ = plan_image([(3000, 1500, 3200, 1700, "red")])
    seeing = SeeingProvider(lambda call: box_reply(seen(call)))
    task = ToolCall("inspect_visual", {"name": "plan.png", "task": "find squares"})
    await execute_tool(ToolContext(provider=seeing, attachments=[surface], disable_thinking=False,
                                   reasoning_effort="medium"), task)
    await execute_tool(ToolContext(provider=seeing, attachments=[surface], disable_thinking=True,
                                   reasoning_effort="medium"), task)
    assert [call["reasoning_effort"] for call in seeing.calls] == ["medium", None]


async def test_a_refused_effort_is_not_swallowed_by_ocr_retries_or_tile_failures():
    """전사는 실패를 3번 다시 시도하고, 타일 bbox는 일부 타일의 실패를 넘어간다 — 설정 오류는 그렇게 다루지 않는다."""
    refusal = ReasoningEffortError("refused")
    ocr = ScriptedProvider([refusal, "never", "never"])
    with pytest.raises(ReasoningEffortError):
        await build_ocr_reader(ocr, disable_thinking=False, reasoning_effort="high")(page(1, b"img"), "INSTRUCTION")
    assert len(ocr.calls) == 1                                               # 다시 보내지 않았다

    surface, _ = plan_image([(3000, 1500, 3200, 1700, "red"), (200, 300, 600, 500, "blue")])
    tiles = SeeingProvider(lambda call: refusal if "red" in _colors(call) else box_reply(seen(call)))
    context = ToolContext(provider=tiles, attachments=[surface], image_mode="tile", disable_thinking=False,
                          reasoning_effort="high")
    with pytest.raises(ReasoningEffortError):                                # "ERROR: …" 문자열로 바뀌지 않는다
        await execute_tool(context, ToolCall("inspect_visual", {"name": "plan.png", "task": "find squares"}))


def _colors(call):
    return set(colors_in(seen(call)))


async def test_changing_the_effort_transcribes_thinking_pages_again():
    calls, progress = [], []

    async def read(attachment, _instruction):
        calls.append(attachment.name)
        return f"read #{len(calls)}"

    options = dict(read_image=read, cache=OcrCache(), thinking=True)
    output, _ = await prepare_visual_ocr_evidence([page(1, b"p1")], **options, reasoning_effort="low")
    assert output[0].ocr_variant == f"whole+thinking:b{config.REASONING_BUDGET_OCR}:elow" and len(calls) == 1
    _, processed = await prepare_visual_ocr_evidence(output, **options, reasoning_effort="low")
    assert not processed and len(calls) == 1                                 # 같은 수준: 다시 전사하지 않는다
    output, processed = await prepare_visual_ocr_evidence(output, **options, reasoning_effort="xhigh",
                                                          on_progress=progress.append)
    assert processed and len(calls) == 2 and output[0].ocr_variant.endswith(":exhigh")
    assert progress[0] == "전사 호출의 추론 설정이 바뀌어 1쪽을 다시 전사하는 중…"
    output, processed = await prepare_visual_ocr_evidence(output, **options)  # 수준을 비우면 그것도 다른 방식이다
    assert processed and output[0].ocr_variant == f"whole+thinking:b{config.REASONING_BUDGET_OCR}"


# --------------------------------------------------------------------------- 요청 → 호출별 계획
def test_effort_plan_follows_the_thinking_plan(mock_llm, monkeypatch):
    server = local(mock_llm)
    cloud = OpenAICompatProvider(name="openai", model="gpt", api_key="k", base_url=mock_llm.base_url)

    def plan(provider, **options):
        request = ChatRequest(**options)
        effort = plan_effort(request, provider, plan_thinking(request, provider))
        return (effort.answer, effort.grounding, effort.ocr), effort.to_public()

    levels = dict(reasoning_effort_answer="low", reasoning_effort_grounding="medium", reasoning_effort_ocr="xhigh")
    # 기본값: bbox·전사는 추론을 끄고 나간다 → 수준은 답변 호출에만 실린다.
    assert plan(server, disable_thinking=False, **levels) == (("low", "", ""), {"answer": "low"})
    assert plan(server, disable_thinking=False, disable_thinking_grounding=False, disable_thinking_ocr=False, **levels) == (
        ("low", "medium", "xhigh"), {"answer": "low", "grounding": "medium", "ocr": "xhigh"})
    assert plan(server, disable_thinking=True, **levels) == (("", "", ""), {})            # 모든 호출 추론 끔
    assert plan(cloud, disable_thinking=False, **levels) == (("", "", ""), {})            # 클라우드에는 연결하지 않았다
    monkeypatch.setattr(config, "REASONING_EFFORT_ANSWER", "medium")
    assert plan(server, disable_thinking=False)[0] == ("medium", "", "")                  # 요청에 없으면 서버 기본값
    assert plan(server, disable_thinking=False, reasoning_effort_answer="")[0] == ("", "", "")


# --------------------------------------------------------------------------- /api/chat
def test_each_call_kind_gets_its_own_effort_and_the_answer_records_it(client, mock_llm):
    mock_llm.reset(seeing_model)
    scan = [upload("scan.pdf", scanned_drawing(), PDF)]
    options = dict(disableThinking=False, disableThinkingOcr=False, reasoningEffortAnswer="Low",
                   reasoningEffortGrounding="xhigh", reasoningEffortOcr="medium")
    data = client.post("/api/chat", json=chat_body(mock_llm, "도면 번호 알려줘", scan, **options)).json()

    (ocr,) = [body for body in mock_llm.requests if is_ocr_call(body)]
    (answer,) = answer_calls(mock_llm)
    assert reasoning_effort(ocr) == "medium" and reasoning_effort(answer) == "low"
    # bbox 호출은 이 턴에서 추론을 끄고 나가므로(기본값) 수준이 적용되지 않는다 → 기록에도 없다.
    assert data["meta"]["reasoningEffort"] == {"answer": "low", "ocr": "medium"}
    saved = client.get(f"/api/sessions/{data['conversationId']}").json()["messages"]
    assert saved[1]["meta"] == data["meta"]

    def follow(text, **changed):
        mock_llm.reset(seeing_model)
        body = chat_body(mock_llm, text, conversationId=data["conversationId"], **{**options, **changed})
        body["messages"] = [*saved, {"role": "user", "content": text}]
        reply = client.post("/api/chat", json=body).json()
        return reply, [reasoning_effort(body) for body in mock_llm.requests if is_ocr_call(body)]

    assert follow("다시")[1] == []                                            # 같은 수준: 저장된 전사를 쓴다
    reply, ocr_levels = follow("전사 수준을 바꿔서", reasoningEffortOcr="low")
    assert ocr_levels == ["low"] and reply["meta"]["reasoningEffort"] == {"answer": "low", "ocr": "low"}
    # 화면이 meta를 되돌려 보내도 첫 답의 기록은 그대로다.
    assert client.get(f"/api/sessions/{data['conversationId']}").json()["messages"][1]["meta"] == data["meta"]


def test_bbox_calls_get_the_grounding_effort(client, mock_llm):
    mock_llm.reset(locate)
    body = chat_body(mock_llm, "빨간 사각형 위치를 표시해줘", [plan_upload([(3000, 1500, 3200, 1700, "red")])],
                     disableThinking=False, disableThinkingGrounding=False, reasoningEffortGrounding="low")
    data = client.post("/api/chat", json=body).json()
    (grounding,) = [request for request in mock_llm.requests if is_grounding_call(request)]
    assert reasoning_effort(grounding) == "low" and grounding["stream"] is True
    assert all(reasoning_effort(request) is None for request in answer_calls(mock_llm))    # 답변 호출에는 고르지 않았다
    assert data["meta"]["reasoningEffort"] == {"grounding": "low"} and data["artifacts"][0]["boxes"]


def test_nothing_changes_when_no_effort_is_chosen_or_thinking_is_off(client, mock_llm, monkeypatch):
    mock_llm.reset(seeing_model)
    scan = [upload("scan.pdf", scanned_drawing(), PDF)]
    # 수준을 고르지 않은 턴: 어떤 요청에도 수준이 없고 기록도 없다(Step 6 1차와 같은 요청).
    data = client.post("/api/chat", json=chat_body(mock_llm, "도면 번호", scan, disableThinking=False,
                                                    disableThinkingOcr=False)).json()
    assert all(reasoning_effort(body) is None for body in mock_llm.requests) and "reasoningEffort" not in data["meta"]
    assert all("chat_template_kwargs" not in body for body in mock_llm.requests)

    # "추론 끄기"(모든 호출)가 켜져 있으면 고른 수준은 실리지 않는다.
    mock_llm.reset(seeing_model)
    data = client.post("/api/chat", json=chat_body(mock_llm, "도면 번호", reasoningEffortAnswer="low",
                                                    reasoningEffortOcr="low")).json()
    assert all(thinking_disabled(body) and reasoning_effort(body) is None for body in mock_llm.requests)
    assert "reasoningEffort" not in data["meta"]

    # 서버 기본값(.env)은 요청에 값이 없을 때만 쓰인다.
    monkeypatch.setattr(config, "REASONING_EFFORT_ANSWER", "medium")
    mock_llm.reset(seeing_model)
    data = client.post("/api/chat", json=chat_body(mock_llm, "안녕", disableThinking=False)).json()
    assert reasoning_effort(mock_llm.requests[-1]) == "medium" and data["meta"]["reasoningEffort"] == {"answer": "medium"}
    mock_llm.reset(seeing_model)
    data = client.post("/api/chat", json=chat_body(mock_llm, "안녕", disableThinking=False, reasoningEffortAnswer="")).json()
    assert reasoning_effort(mock_llm.requests[-1]) is None and "reasoningEffort" not in data["meta"]


def test_a_malformed_effort_is_rejected_before_any_model_call(client, mock_llm):
    response = client.post("/api/chat", json=chat_body(mock_llm, "안녕", disableThinking=False,
                                                       reasoningEffortAnswer="very high"))
    assert response.status_code == 400 and "추론 수준 값이 올바르지 않습니다" in response.json()["error"]
    assert mock_llm.requests == []


def test_a_turn_whose_effort_the_server_refuses_ends_with_an_error_not_an_answer(client, mock_llm):
    mock_llm.reset(qwen38())
    body = chat_body(mock_llm, "도면 번호 알려줘", [upload("scan.pdf", scanned_drawing(), PDF)], disableThinking=False,
                     reasoningEffortAnswer="high", stream=True)
    lines = [json.loads(line) for line in client.post("/api/chat", json=body).text.splitlines() if line.strip()]
    assert lines[-1]["type"] == "error" and '추론 수준 "high"' in lines[-1]["error"]
    # 답변 호출 한 번(+ 확인 요청 둘)으로 끝났다: JSON 폴백으로 바꿔 다시 보내지도, 수준을 빼고 다시 보내지도 않았다.
    assert [reasoning_effort(request) for request in answer_calls(mock_llm)] == ["high"]
    assert sum(1 for request in mock_llm.requests if is_effort_probe(request)) == 2
    # 전사(추론 끔 — 수준이 실리지 않는다)는 이미 끝나 저장돼 있다 → 수준을 고쳐 다시 보내면 전사는 다시 하지 않는다.
    conversation = next(line["conversationId"] for line in lines if line["type"] == "conversation")
    saved = client.get(f"/api/sessions/{conversation}").json()["messages"]
    assert saved[-1]["content"].startswith('오류: 모델 서버가 추론 수준 "high"')
    mock_llm.reset(qwen38())
    retry = chat_body(mock_llm, "도면 번호 알려줘", conversationId=conversation, disableThinking=False,
                      reasoningEffortAnswer="xhigh")
    data = client.post("/api/chat", json=retry).json()
    assert data["text"] == "도면 번호는 FA-7731-B입니다." and not [request for request in mock_llm.requests if is_ocr_call(request)]
    assert data["meta"]["reasoningEffort"] == {"answer": "xhigh"}


def test_a_refused_effort_inside_the_bbox_tool_or_the_ocr_step_fails_the_turn(client, mock_llm):
    # bbox 호출의 수준이 거절되면 도구 오류로 답변 모델에게 돌아가지 않는다(모델이 고칠 수 없는 설정 오류다).
    mock_llm.reset(qwen38(locate))
    image = plan_upload([(200, 300, 600, 500, "blue"), (2000, 920, 2200, 1080, "green"), (3000, 1500, 3200, 1700, "red")])
    body = chat_body(mock_llm, "사각형 위치를 표시해줘", [image], imageMode="tile", disableThinking=False,
                     disableThinkingGrounding=False, reasoningEffortGrounding="high")
    response = client.post("/api/chat", json=body)
    assert response.status_code == 400 and '추론 수준 "high"' in response.json()["error"]
    grounding = [request for request in mock_llm.requests if is_grounding_call(request)]
    assert 1 <= len(grounding) <= config.OCR_CONCURRENCY                     # 타일 4장이 저마다 물어보지 않는다
    assert len(answer_calls(mock_llm)) == 1                                  # 도구를 부른 호출뿐. 결과를 받아 다시 답하지 않았다

    # 전사 호출의 수준이 거절되면 3번 다시 보내지 않고 턴이 오류로 끝난다("[OCR FAILED …]" 증거로 남지 않는다).
    mock_llm.reset(qwen38())
    body = chat_body(mock_llm, "도면 번호", [upload("scan.pdf", scanned_drawing(), PDF)], disableThinking=False,
                     disableThinkingOcr=False, reasoningEffortOcr="high")
    response = client.post("/api/chat", json=body)
    assert response.status_code == 400 and '추론 수준 "high"' in response.json()["error"]
    assert len([request for request in mock_llm.requests if is_ocr_call(request)]) == 1 and not answer_calls(mock_llm)


def test_trace_records_the_effort_of_each_call(client, mock_llm, monkeypatch):
    monkeypatch.setenv("DOCCHAT_DEBUG_TRACE", "1")
    mock_llm.reset(seeing_model)
    body = chat_body(mock_llm, "도면 번호 알려줘", [upload("scan.pdf", scanned_drawing(), PDF)], disableThinking=False,
                     reasoningEffortAnswer="low", reasoningEffortOcr="medium")
    data = client.post("/api/chat", json=body).json()
    document = client.get(f"/api/traces/{data['meta']['traceId']}").json()
    calls = {event["data"]["kind"]: event["data"] for event in document["events"] if event["kind"] == "model"}
    assert calls["answer"]["reasoningEffort"] == "low"
    assert "reasoningEffort" not in calls["ocr"] and calls["ocr"]["disableThinking"] is True   # 추론을 끈 호출에는 실리지 않았다
    (entry,) = [event for event in document["events"] if event["kind"] == "input"]
    assert entry["data"]["reasoningEffort"] == {"answer": "low"}
    assert request_images(mock_llm.requests[0])                              # (전사 호출은 그대로 이미지를 싣고 나갔다)
