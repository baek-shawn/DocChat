"""Step 6-0 — bbox·전사 호출의 폭주 막기: 호출별 추론 끄기, 출력 상한, 상한에 닿은 호출은 다시 보내지 않기.

배경(실측, vLLM Qwen3.5-35B, 추론 켬): 타일 6장 중 2장에서 추론이 같은 내용을 맴돌다 출력 한도까지 갔다.
본문(content)은 비어 있고 추론(reasoning_content)만 4만 자였다. 앱은 이를 "JSON이 아님"으로 보고 같은 타일을
3번씩 다시 보냈고, 출력 상한이 없어 한 번에 컨텍스트가 찰 때까지 생성했다 → 요청이 끝나지 않았다.
"""
from __future__ import annotations

import json

import pytest

from app import config
from app.agent.tools import ToolContext, execute_tool
from app.chat_service import ChatRequest, plan_thinking
from app.pipeline.images import VisionUsage
from app.pipeline.ocr import OcrCache, build_ocr_reader, cut_off_transcription, prepare_visual_ocr_evidence
from app.providers.base import ContextWindowError, ModelResponse, ToolCall
from app.providers.openai_compat import OpenAICompatProvider
from conftest import chat_body, upload
from mock_openai import all_text, is_grounding_call, is_ocr_call, request_images, thinking_disabled
from pdf_factory import colors_in, encode, marked_image
from test_ocr_evidence import ScriptedProvider, page
from test_tiling import (PDF, RED_TEXT, SeeingProvider, box_reply, page_of, plan_image, scanned_drawing, seeing_model,
                         seen, tile_loader, transcribe)

# 실제 폭주 출력(Qwen3.5, 타일 r2c3)의 되풀이 구간. 1,112줄 중 서로 다른 줄은 115개였다.
LOOP = ["Wait, looking at the symbol at x=350, y=350.", "It's a circle with a line.",
        "Looking at the symbol at x=30, y=750.", "It's a circle with a line.", "These are door symbols.",
        "I will output these two.", "Let's check the window detail at x=0, y=400.", "It's a window."]
RUNAWAY_REASONING = "\n".join(["The user wants me to identify FSD window and door symbols in the provided image.",
                               *(LOOP * 40), "Let's check the window detail at x=0, y"])
INSPECT = ToolCall("inspect_visual", {"name": "plan.png", "task": "find squares"})
QUESTION = [{"role": "user", "content": "q"}]
LIMIT_FAILURE = ("[OCR FAILED: the model reached the output limit of 4096 tokens without producing a transcription. "
                 "Not retried.]")
INCOMPLETE = ("[TRANSCRIPTION INCOMPLETE: the model reached the output limit of 4096 tokens here. "
              "Text after this point was not transcribed.]")


def cut(text: str = "", reasoning: str = "") -> ModelResponse:
    """출력 상한에 닿아 끊긴 응답."""
    return ModelResponse(text=text, finish_reason="length", reasoning=reasoning)


class LocalScripted(ScriptedProvider):
    """추론을 끌 수 있는 로컬 서버 흉내(기본 가짜 provider는 클라우드처럼 is_local=False다)."""
    is_local = True
    accepts_thinking_field = True

    def can_disable_thinking(self):
        return self.accepts_thinking_field


def local(mock, **options) -> OpenAICompatProvider:
    return OpenAICompatProvider(name="openaiCompatible", model="mock-vlm", base_url=mock.base_url, is_local=True, **options)


def test_defaults():
    assert config.GROUNDING_DISABLE_THINKING is True and config.OCR_DISABLE_THINKING is True
    assert config.VISION_MAX_TOKENS == 4096 == config.vision_max_tokens()
    assert config.resolve_switch(None, True) is True and config.resolve_switch(False, True) is False
    assert config.resolve_switch(True, False) is True and config.resolve_switch(None, False) is False


# --------------------------------------------------------------------------- provider: 실제 HTTP 요청에 실리는 것
async def test_thinking_can_be_turned_off_for_a_single_call(mock_llm):
    provider = local(mock_llm)                                       # "추론 끄기"(모든 호출)는 꺼져 있다
    everything = local(mock_llm, disable_thinking=True)
    try:
        await provider.analyze(QUESTION)
        await provider.analyze(QUESTION, disable_thinking=True)
        await provider.analyze(QUESTION)                             # 한 호출만 끈 것이라 다음 호출은 그대로다
        await everything.analyze(QUESTION)
        await everything.analyze(QUESTION, disable_thinking=False)   # 모든 호출을 껐으면 호출별 값과 무관하게 꺼진다
    finally:
        await provider.aclose()
        await everything.aclose()
    assert [thinking_disabled(body) for body in mock_llm.requests] == [False, True, False, True, True]
    assert all("max_tokens" not in body for body in mock_llm.requests)


async def test_output_limit_is_sent_only_when_given_and_only_to_local_servers(mock_llm):
    provider = local(mock_llm)
    cloud = OpenAICompatProvider(name="openai", model="gpt", api_key="k", base_url=mock_llm.base_url, is_local=False)
    try:
        for limit in (None, 4096, 0):
            await provider.analyze(QUESTION, max_tokens=limit)
        await cloud.analyze(QUESTION, disable_thinking=True, max_tokens=4096)
    finally:
        await provider.aclose()
        await cloud.aclose()
    assert [body.get("max_tokens") for body in mock_llm.requests] == [None, 4096, None, None]
    assert not {"max_completion_tokens", "chat_template_kwargs"} & set(mock_llm.requests[-1])
    assert provider.can_disable_thinking() and not cloud.can_disable_thinking()


async def test_a_limit_that_does_not_fit_the_context_is_dropped_instead_of_failing(mock_llm):
    """vLLM은 "입력 + max_tokens"가 컨텍스트를 넘으면 생성하지 않고 400으로 거절한다(도면 한 장이 6,700토큰쯤 든다)."""
    rejection = {"status": 400, "body": {"error": {"message": (
        "This model's maximum context length is 8192 tokens. However, you requested 10789 tokens "
        "(6693 in the messages, 4096 in the completion). Please reduce the length of the messages or completion.")}}}
    mock_llm.reset(lambda body: rejection if body.get("max_tokens") else "measured")
    provider = local(mock_llm)
    try:
        response = await provider.analyze(QUESTION, disable_thinking=True, max_tokens=4096)
        assert response.text == "measured"
        assert [body.get("max_tokens") for body in mock_llm.requests] == [4096, None]
        assert all(thinking_disabled(body) for body in mock_llm.requests)       # 추론 끄기는 그대로 간다

        # 입력만으로 넘치는 경우는 기존 경로(메시지를 줄여 다시 보내기)로 넘어가고, 그래도 안 되면 오류다.
        mock_llm.reset(lambda body: rejection)
        with pytest.raises(ContextWindowError):
            await provider.analyze(QUESTION, max_tokens=4096)
        assert [body.get("max_tokens") for body in mock_llm.requests] == [4096, None] * 3
    finally:
        await provider.aclose()


async def test_servers_that_reject_the_thinking_field_are_remembered(mock_llm):
    def strict(body):
        if "chat_template_kwargs" in body:
            return {"status": 400, "body": {"error": {"message": "Unrecognized request argument: chat_template_kwargs"}}}
        return "ok"

    mock_llm.reset(strict)
    provider = local(mock_llm)
    try:
        for _ in range(2):
            assert (await provider.analyze(QUESTION, disable_thinking=True, max_tokens=4096)).text == "ok"
    finally:
        await provider.aclose()
    assert ["chat_template_kwargs" in body for body in mock_llm.requests] == [True, False, False]   # 다시 시도하지 않는다
    assert [body.get("max_tokens") for body in mock_llm.requests] == [4096] * 3                      # 출력 상한은 그대로
    assert not provider.can_disable_thinking()


async def test_reasoning_split_off_by_the_server_never_becomes_the_answer_text(mock_llm):
    mock_llm.reset(lambda body: {"text": "", "reasoning": RUNAWAY_REASONING, "finish_reason": "length"})
    provider = local(mock_llm)
    try:
        # 추론을 끈 호출(Step 6 이전 경로 그대로): 서버가 떼어 준 추론은 reasoning에, 본문은 비어 있다.
        response = await provider.analyze(QUESTION, disable_thinking=True, max_tokens=4096)
        assert response.text == "" and response.finish_reason == "length"
        assert response.reasoning == RUNAWAY_REASONING
        # 추론을 켠 호출(Step 6): 스트리밍으로 받다가 같은 묶음의 되풀이를 잡아 추론을 끊고 이어 쓰기를 시키는데,
        # 이 서버는 이어 쓰기에서도 같은 반복을 내놓는다 → 하드 중단. 어느 쪽이든 추론 글이 답이 되지는 않는다.
        response = await provider.analyze(QUESTION, max_tokens=4096)
        assert response.text == "" and response.finish_reason == "reasoning_runaway"
        assert response.forced == "repeat" and response.runaway == "repeat"
        assert response.reasoning.startswith("The user wants me to identify") and len(response.reasoning) < len(RUNAWAY_REASONING)
    finally:
        await provider.aclose()


# --------------------------------------------------------------------------- bbox 호출
async def test_bbox_calls_carry_the_thinking_choice_and_the_output_limit(monkeypatch):
    surface, _ = plan_image([(3000, 1500, 3200, 1700, "red"), (200, 300, 600, 500, "blue")])
    provider = SeeingProvider(lambda call: box_reply(seen(call)))
    await execute_tool(ToolContext(provider=provider, attachments=[surface], disable_thinking=True), INSPECT)
    assert [(call["disable_thinking"], call["max_tokens"]) for call in provider.calls] == [(True, 4096)]

    monkeypatch.setattr(config, "VISION_MAX_TOKENS", 16000)
    await execute_tool(ToolContext(provider=provider, attachments=[surface], image_mode="tile"), INSPECT)
    assert [(call["disable_thinking"], call["max_tokens"]) for call in provider.calls[1:]] == [(False, 16000)] * 2

    monkeypatch.setattr(config, "VISION_MAX_TOKENS", 0)              # 0 = 상한을 보내지 않는다
    await execute_tool(ToolContext(provider=provider, attachments=[surface]), INSPECT)
    assert provider.calls[-1]["max_tokens"] is None


async def test_a_bbox_call_cut_off_at_the_limit_is_not_sent_again():
    """실측 재현: 본문은 비어 있고 추론만 상한까지 찼다. 예전에는 같은 호출을 3번 보냈다."""
    surface, _ = plan_image([(3000, 1500, 3200, 1700, "red")])
    provider = SeeingProvider(lambda call: cut(reasoning=RUNAWAY_REASONING))
    context = ToolContext(provider=provider, attachments=[surface])
    raw = await execute_tool(context, INSPECT)
    output = json.loads(raw)

    assert len(provider.calls) == 1 and output["regions"] == [] and output["text"] == ""
    assert output["warning"] == ("The vision model stopped at the output limit of 4096 tokens before answering, "
                                 "so no boxes could be measured. The call was not retried.")
    assert "door symbols" not in raw                                  # 추론 글은 결과 어디에도 들어가지 않는다
    assert (context.usage.grounding_calls, context.usage.grounding_length_stops) == (1, 1)
    assert "boxes" not in context.artifacts[0] and context.artifacts[0]["text"] == ""


async def test_unfinished_json_from_a_cut_off_call_is_not_used():
    """끊긴 JSON 안에 온전한 영역이 있어도 쓰지 않는다 — 어디까지가 전부인지 알 수 없다."""
    truncated = ('{"text": "two doors", "regions": [{"type": "object", "label": "door", "bbox": [100, 100, 200, 200]}, '
                 '{"type": "object", "label": "do')
    surface, _ = plan_image([(3000, 1500, 3200, 1700, "red")])
    provider = SeeingProvider(lambda call: cut(text=truncated))
    context = ToolContext(provider=provider, attachments=[surface])
    output = json.loads(await execute_tool(context, INSPECT))
    assert len(provider.calls) == 1 and output["regions"] == [] and "output limit" in output["warning"]
    assert "two doors" not in json.dumps(output) and context.artifacts[0]["text"] == ""


async def test_broken_json_is_still_retried_when_the_model_finished_normally():
    surface, _ = plan_image([(3000, 1500, 3200, 1700, "red")])
    provider = SeeingProvider(lambda call: "I could not find anything to mark.")
    context = ToolContext(provider=provider, attachments=[surface])
    output = json.loads(await execute_tool(context, INSPECT))
    assert len(provider.calls) == 1 + config.GROUNDING_RETRY_COUNT == 3
    assert output["warning"] == "The vision model did not return structured regions; no boxes could be measured."
    assert context.usage.grounding_length_stops == 0


async def test_tiles_that_hit_the_limit_are_skipped_and_the_rest_are_still_measured():
    surface, _ = plan_image([(3000, 1500, 3200, 1700, "red"), (200, 300, 600, 500, "blue")])
    attempts: dict[tuple, int] = {}

    def reply(call):
        tile = call["images"][0].tile
        attempts[tile] = attempts.get(tile, 0) + 1
        return cut(reasoning=RUNAWAY_REASONING) if tile == (1, 1) else box_reply(seen(call))

    context = ToolContext(provider=SeeingProvider(reply), attachments=[surface], image_mode="tile")
    output = json.loads(await execute_tool(context, INSPECT))
    assert attempts == {(1, 1): 1, (2, 3): 1}                          # 끊긴 타일도 한 번만
    assert [region["label"] for region in output["regions"]] == ["red"]
    assert output["warning"] == ("1 of 2 tiles did not return structured regions, so targets inside those tiles may be "
                                 "missing. 1 of 2 tiles stopped at the output limit of 4096 tokens before answering "
                                 "and were not retried.")
    assert (context.usage.grounding_calls, context.usage.grounding_length_stops) == (2, 1)

    every = ToolContext(provider=SeeingProvider(lambda call: cut(reasoning=RUNAWAY_REASONING)), attachments=[
        plan_image([(3000, 1500, 3200, 1700, "red"), (200, 300, 600, 500, "blue")])[0]], image_mode="tile")
    failed = json.loads(await execute_tool(every, INSPECT))
    assert len(every.provider.calls) == 2 and failed["regions"] == [] and "boxes" not in every.artifacts[0]
    assert failed["warning"] == ("The vision model did not return structured regions for any tile; no boxes could be "
                                 "measured. 2 of 2 tiles stopped at the output limit of 4096 tokens before answering "
                                 "and were not retried.")


async def test_the_warning_does_not_name_a_limit_that_was_not_sent(monkeypatch):
    """상한을 껐는데(0) 서버가 자기 한도에서 끊은 경우."""
    monkeypatch.setattr(config, "VISION_MAX_TOKENS", 0)
    surface, _ = plan_image([(3000, 1500, 3200, 1700, "red")])
    context = ToolContext(provider=SeeingProvider(lambda call: cut()), attachments=[surface])
    output = json.loads(await execute_tool(context, INSPECT))
    assert "stopped at its output limit before answering" in output["warning"] and len(context.provider.calls) == 1


# --------------------------------------------------------------------------- 전사 호출
async def test_ocr_calls_carry_the_thinking_choice_and_the_output_limit(monkeypatch):
    provider = ScriptedProvider(["TEXT A", "TEXT B", "TEXT C"])
    await build_ocr_reader(provider, disable_thinking=True)(page(1, b"img"), "INSTRUCTION")
    monkeypatch.setattr(config, "VISION_MAX_TOKENS", 12000)
    await build_ocr_reader(provider)(page(1, b"img"), "INSTRUCTION")
    monkeypatch.setattr(config, "VISION_MAX_TOKENS", 0)
    await build_ocr_reader(provider)(page(1, b"img"), "INSTRUCTION")
    assert [(call["disable_thinking"], call["max_tokens"]) for call in provider.calls] == [
        (True, 4096), (False, 12000), (False, None)]


async def test_a_transcription_cut_off_at_the_limit_keeps_what_was_read_and_is_not_retried():
    """추론 없이 상한에 닿았다 = 글자가 많은 쪽이다. 읽은 데까지는 전사이므로 버리지 않고, 끊긴 자리를 표시한다."""
    provider = LocalScripted([cut("DWG NO: FA-7731-B\nITEM 1  FLANGE\nITEM 2  GASK")])
    usage = VisionUsage()
    text = await build_ocr_reader(provider, usage=usage, disable_thinking=True)(page(1, b"img"), "INSTRUCTION")
    assert text == f"DWG NO: FA-7731-B\nITEM 1  FLANGE\nITEM 2  GASK\n{INCOMPLETE}"
    assert len(provider.calls) == 1 and (usage.ocr_calls, usage.ocr_length_stops) == (1, 1)


async def test_a_repetition_loop_that_runs_into_the_limit_is_cleaned_and_kept():
    """소형 VLM의 퇴화 루프(실측: gemma3가 `[UNCLEAR]`를 수백 줄). 앞의 본문은 옳게 받아쓴 것이다."""
    looping = "PARTS LIST\nFL-0420  FLANGE  2\n" + "[UNCLEAR]\n" * 400
    provider = ScriptedProvider([cut(looping)])
    text = await build_ocr_reader(provider)(page(1, b"img"), "INSTRUCTION")
    assert text == f"PARTS LIST\nFL-0420  FLANGE  2\n[UNCLEAR]\n{INCOMPLETE}" and len(provider.calls) == 1


async def test_runaway_reasoning_is_never_kept_as_a_transcription():
    # 서버가 추론을 따로 떼어 줬고 본문은 비어 있다(실측과 같은 형태).
    separated = LocalScripted([cut(reasoning=RUNAWAY_REASONING)])
    usage = VisionUsage()
    assert await build_ocr_reader(separated, usage=usage)(page(1, b"img"), "INSTRUCTION") == LIMIT_FAILURE
    assert len(separated.calls) == 1 and (usage.ocr_calls, usage.ocr_length_stops) == (1, 1)

    # 추론을 본문에 섞어 주는 서버: <think>가 열린 채 끝났다 → 답은 시작도 못 했다.
    inline = LocalScripted([cut(f"<think>\n{RUNAWAY_REASONING}")])
    assert await build_ocr_reader(inline, disable_thinking=True)(page(1, b"img"), "INSTRUCTION") == LIMIT_FAILURE

    # 챗 템플릿이 <think>를 프롬프트 쪽에 넣는 모델은 본문에 태그 없이 추론만 온다. 추론을 끄지 않았다면
    # 이 글이 추론인지 전사인지 가릴 수 없다 → 쓰지 않는다.
    untagged = LocalScripted([cut(RUNAWAY_REASONING)])
    assert await build_ocr_reader(untagged)(page(1, b"img"), "INSTRUCTION") == LIMIT_FAILURE
    assert len(inline.calls) == len(untagged.calls) == 1

    # 추론을 끄라고 보냈지만 서버가 그 요청을 거절한 경우도 마찬가지다.
    refused = LocalScripted([cut(RUNAWAY_REASONING)])
    refused.accepts_thinking_field = False
    assert await build_ocr_reader(refused, disable_thinking=True)(page(1, b"img"), "INSTRUCTION") == LIMIT_FAILURE


async def test_text_after_the_reasoning_is_the_transcription():
    closed = LocalScripted([cut(f"{RUNAWAY_REASONING}\n</think>\n\nDWG NO: FA-7731-B\nSCALE 1:5")])
    assert await build_ocr_reader(closed)(page(1, b"img"), "INSTRUCTION") == f"DWG NO: FA-7731-B\nSCALE 1:5\n{INCOMPLETE}"

    # 서버가 추론을 떼어 줬으면 본문은 답이다(추론을 켠 호출이어도).
    separated = LocalScripted([cut("DWG NO: FA-7731-B", reasoning="The title block says FA-7731-B.")])
    assert await build_ocr_reader(separated)(page(1, b"img"), "INSTRUCTION") == f"DWG NO: FA-7731-B\n{INCOMPLETE}"

    # 끊긴 글이 전사가 아니라 설명이면 쓰지 않는다(기존 규칙 그대로). 그래도 다시 보내지는 않는다.
    chatter = LocalScripted([cut("The image shows a floor plan with several rooms and")])
    assert await build_ocr_reader(chatter, disable_thinking=True)(page(1, b"img"), "INSTRUCTION") == LIMIT_FAILURE
    assert len(chatter.calls) == 1


def test_which_cut_off_text_can_be_trusted():
    cloud, server = ScriptedProvider([]), LocalScripted([])
    assert cut_off_transcription(cloud, cut("LINE ONE"), thinking_disabled=False) == "LINE ONE"
    assert cut_off_transcription(server, cut("LINE ONE"), thinking_disabled=True) == "LINE ONE"
    assert cut_off_transcription(server, cut("LINE ONE"), thinking_disabled=False) == ""
    assert cut_off_transcription(server, cut("LINE ONE", reasoning="r"), thinking_disabled=False) == "LINE ONE"
    assert cut_off_transcription(server, cut("a</think>b</THINK >LINE TWO"), thinking_disabled=False) == "LINE TWO"
    assert cut_off_transcription(cloud, cut("<think>half a thought"), thinking_disabled=True) == ""
    assert cut_off_transcription(cloud, cut(""), thinking_disabled=True) == ""


async def test_failures_from_other_causes_are_retried_as_before():
    provider = ScriptedProvider(["Sorry, I cannot read this.", RuntimeError("timeout"), "DWG NO: FA-7731-B"])
    usage = VisionUsage()
    text = await build_ocr_reader(provider, usage=usage)(page(1, b"img"), "INSTRUCTION")
    assert text == "DWG NO: FA-7731-B" and (usage.ocr_calls, usage.ocr_length_stops) == (3, 0)


async def test_a_cut_off_tile_is_marked_inside_its_own_block():
    attachments = page_of(scanned_drawing())

    def reply(call):
        if call["images"][0].tile == (2, 3):
            return cut("APPROVED BY S. L")
        if call["images"][0].tile == (1, 2):
            return cut(reasoning=RUNAWAY_REASONING)
        return transcribe(seen(call))

    provider, usage = SeeingProvider(reply), VisionUsage()
    reader = build_ocr_reader(provider, image_mode="tile", load_tile_source=tile_loader(attachments), usage=usage)
    text = await reader(attachments[1], "INSTRUCTION")
    assert len(provider.calls) == 3 and (usage.ocr_calls, usage.ocr_length_stops) == (3, 2)
    assert "2 tiles contain text, 3 contain none, 1 could not be read" in text.splitlines()[0]
    assert f"[TILE r1c1]\n{RED_TEXT}" in text and f"[TILE r1c2]\n{LIMIT_FAILURE}" in text
    assert f"[TILE r2c3]\nAPPROVED BY S. L\n{INCOMPLETE}" in text


async def test_cut_off_pages_with_text_are_cached_but_failures_are_not():
    cache = OcrCache()
    partial = ScriptedProvider([cut("LINE ONE\nLINE TW")])
    for _ in range(2):
        output, _ = await prepare_visual_ocr_evidence([page(1, b"same")], read_image=build_ocr_reader(partial), cache=cache)
    assert len(partial.calls) == 1 and len(cache) == 1 and output[-1].text.endswith(INCOMPLETE)

    failing = LocalScripted([cut(reasoning=RUNAWAY_REASONING), cut(reasoning=RUNAWAY_REASONING)])
    for _ in range(2):
        output, _ = await prepare_visual_ocr_evidence([page(1, b"other")], read_image=build_ocr_reader(failing), cache=cache)
    assert len(failing.calls) == 2 and len(cache) == 1 and output[-1].text.endswith(LIMIT_FAILURE)


# --------------------------------------------------------------------------- 전사 기록에 추론 여부 포함
def test_transcriptions_made_with_thinking_are_kept_apart():
    tile = config.image_mode_variant("tile")
    assert config.image_mode_variant("whole", thinking=False) == "whole"      # Step 6-0 이전의 기록과 같은 값
    assert config.image_mode_variant("whole", thinking=True) == "whole+thinking"
    assert config.image_mode_variant("tile", thinking=True) == f"{tile}+thinking" and tile.startswith("tile:")
    keys = {OcrCache.key("m", "image/png", b"page", config.image_mode_variant(mode, thinking=thinking))
            for mode in ("whole", "tile") for thinking in (False, True)}
    assert len(keys) == 4


async def test_pages_transcribed_with_another_thinking_choice_are_transcribed_again():
    calls, progress = [], []

    async def read(attachment, _instruction):
        calls.append(attachment.name)
        return f"read #{len(calls)}"

    pages = [page(1, b"p1"), page(2, b"p2")]
    output, _ = await prepare_visual_ocr_evidence(pages, read_image=read, cache=OcrCache())
    assert [item.ocr_variant for item in output[:2]] == ["whole", "whole"] and len(calls) == 2
    output, processed = await prepare_visual_ocr_evidence(output, read_image=read, cache=OcrCache(), thinking=True,
                                                          on_progress=progress.append)
    # 추론을 켠 전사의 기록에는 추론 예산도 들어간다(Step 6): 예산을 바꾸면 끊기는 자리가 달라 다시 전사해야 한다.
    with_budget = config.image_mode_variant("whole", thinking=True, budget=config.REASONING_BUDGET_OCR)
    assert with_budget == f"whole+thinking:b{config.REASONING_BUDGET_OCR}"
    assert processed and len(calls) == 4 and [item.ocr_variant for item in output[:2]] == [with_budget] * 2
    assert progress[0] == "전사 호출의 추론 설정이 바뀌어 2쪽을 다시 전사하는 중…"
    _, processed = await prepare_visual_ocr_evidence(output, read_image=read, cache=OcrCache(), thinking=True)
    assert not processed and len(calls) == 4

    progress.clear()
    await prepare_visual_ocr_evidence(output, read_image=read, cache=OcrCache(), image_mode="tile", thinking=True,
                                      on_progress=progress.append)
    assert progress[0] == "이미지 처리 방식이 바뀌어 2쪽을 다시 전사하는 중…"

    # 예산만 바꿔도(Step 6) 추론을 켠 전사는 다시 한다 — 끊기는 자리가 달라 결과가 달라진다.
    progress.clear()
    _, processed = await prepare_visual_ocr_evidence(output, read_image=read, cache=OcrCache(), image_mode="tile",
                                                     thinking=True, reasoning_budget=1234, on_progress=progress.append)
    assert processed and len(calls) == 8 and progress[0] == "전사 호출의 추론 설정이 바뀌어 2쪽을 다시 전사하는 중…"
    assert all(item.ocr_variant.endswith("+thinking:b1234") for item in output[:2])


# --------------------------------------------------------------------------- 요청 → 호출별 추론 계획
def test_thinking_plan():
    server, cloud = LocalScripted([]), ScriptedProvider([])
    plan = plan_thinking(ChatRequest(disable_thinking=False), server)               # 기본값: bbox·전사만 끈다
    assert (plan.answer, plan.grounding, plan.ocr, plan.controllable) == (False, True, True, True)
    plan = plan_thinking(ChatRequest(disable_thinking=False, disable_thinking_grounding=False,
                                     disable_thinking_ocr=False), server)
    assert (plan.answer, plan.grounding, plan.ocr) == (False, False, False)
    plan = plan_thinking(ChatRequest(disable_thinking=True, disable_thinking_grounding=False,
                                     disable_thinking_ocr=False), server)            # 모든 호출 끄기가 우선
    assert (plan.answer, plan.grounding, plan.ocr) == (True, True, True)
    # 추론을 끌 방법이 없는 provider에는 아무것도 적지 않는다(끈 것처럼 보이면 안 된다).
    plan = plan_thinking(ChatRequest(disable_thinking=True), cloud)
    assert (plan.answer, plan.grounding, plan.ocr, plan.controllable) == (False, False, False, False)


# --------------------------------------------------------------------------- /api/chat
def locate(body):
    if is_grounding_call(body):
        return box_reply(request_images(body)[0])
    if any(message.get("role") == "tool" for message in body["messages"]):
        return "표시했습니다."
    return {"tool_calls": [{"name": "inspect_visual", "arguments": {"name": "plan.png", "task": "find the squares"}}]}


def plan_upload(marks):
    return upload("plan.png", encode(marked_image(4000, 2000, marks)), "image/png")


def test_health_reports_the_runaway_guards(client, monkeypatch):
    assert client.get("/api/health").json()["vision"] == {
        "disableThinkingGrounding": True, "disableThinkingOcr": True, "maxTokens": 4096}
    monkeypatch.setattr(config, "GROUNDING_DISABLE_THINKING", False)
    monkeypatch.setattr(config, "VISION_MAX_TOKENS", 0)
    assert client.get("/api/health").json()["vision"] == {
        "disableThinkingGrounding": False, "disableThinkingOcr": True, "maxTokens": 0}


def test_by_default_ocr_runs_without_thinking_even_when_the_answer_may_think(client, mock_llm):
    mock_llm.reset(seeing_model)
    body = chat_body(mock_llm, "도면 번호 알려줘", [upload("scan.pdf", scanned_drawing(), PDF)], disableThinking=False)
    data = client.post("/api/chat", json=body).json()

    (ocr,) = [request for request in mock_llm.requests if is_ocr_call(request)]
    (answer,) = [request for request in mock_llm.requests if not is_ocr_call(request)]
    assert thinking_disabled(ocr) and ocr["max_tokens"] == 4096
    assert not thinking_disabled(answer) and "max_tokens" not in answer     # 답변 호출에는 상한을 두지 않는다
    assert data["meta"]["thinkingDisabled"] == {"answer": False, "grounding": True, "ocr": True}
    assert data["meta"]["visionMaxTokens"] == 4096
    assert data["meta"]["vision"]["ocrLengthStops"] == 0 and data["meta"]["vision"]["groundingLengthStops"] == 0
    saved = client.get(f"/api/sessions/{data['conversationId']}").json()["messages"]
    assert saved[1]["meta"] == data["meta"]

    # 화면이 meta를 되돌려 보내도(다음 턴) 그대로 남는다.
    follow = chat_body(mock_llm, "다시", conversationId=data["conversationId"])
    follow["messages"] = [*saved, {"role": "user", "content": "다시"}]
    client.post("/api/chat", json=follow)
    assert client.get(f"/api/sessions/{data['conversationId']}").json()["messages"][1]["meta"] == data["meta"]


def test_thinking_for_bbox_calls_is_chosen_per_request(client, mock_llm):
    image = plan_upload([(3000, 1500, 3200, 1700, "red")])

    def run(**options):
        mock_llm.reset(locate)
        data = client.post("/api/chat", json=chat_body(mock_llm, "빨간 사각형 위치를 표시해줘", [image], **options)).json()
        grounding = [request for request in mock_llm.requests if is_grounding_call(request)]
        answers = [request for request in mock_llm.requests if not is_grounding_call(request)]
        # 추론을 켠 bbox 호출의 상한은 출력 몫 + 추론 예산이다(Step 6). 추론을 끈 호출은 출력 몫만.
        assert all(request["max_tokens"] == 4096 + (0 if thinking_disabled(request) else config.REASONING_BUDGET_GROUNDING)
                   for request in grounding)
        assert all("max_tokens" not in request for request in answers)
        return (data["meta"]["thinkingDisabled"], [thinking_disabled(request) for request in grounding],
                [thinking_disabled(request) for request in answers])

    assert run(disableThinking=False) == ({"answer": False, "grounding": True, "ocr": True}, [True], [False, False])
    assert run(disableThinking=False, disableThinkingGrounding=False) == (
        {"answer": False, "grounding": False, "ocr": True}, [False], [False, False])
    assert run(disableThinking=False, disableThinkingGrounding=True, disableThinkingOcr=False) == (
        {"answer": False, "grounding": True, "ocr": False}, [True], [False, False])
    # "추론 끄기"(모든 호출)가 켜져 있으면 호출별 선택과 무관하게 전부 꺼진다 — Step 6-0 이전과 같은 동작.
    assert run(disableThinkingGrounding=False, disableThinkingOcr=False) == (
        {"answer": True, "grounding": True, "ocr": True}, [True], [True, True])


def test_defaults_for_the_per_call_choice_and_the_limit_come_from_config(client, mock_llm, monkeypatch):
    monkeypatch.setattr(config, "GROUNDING_DISABLE_THINKING", False)      # DOCCHAT_GROUNDING_DISABLE_THINKING=0
    monkeypatch.setattr(config, "VISION_MAX_TOKENS", 9000)                # DOCCHAT_VISION_MAX_TOKENS=9000
    mock_llm.reset(locate)
    body = chat_body(mock_llm, "위치 표시", [plan_upload([(3000, 1500, 3200, 1700, "red")])], disableThinking=False)
    data = client.post("/api/chat", json=body).json()
    (grounding,) = [request for request in mock_llm.requests if is_grounding_call(request)]
    assert not thinking_disabled(grounding) and grounding["max_tokens"] == 9000 + config.REASONING_BUDGET_GROUNDING
    assert data["meta"]["thinkingDisabled"] == {"answer": False, "grounding": False, "ocr": True}
    assert data["meta"]["visionMaxTokens"] == 9000

    monkeypatch.setattr(config, "VISION_MAX_TOKENS", 0)                   # 상한 없음
    mock_llm.reset(locate)
    data = client.post("/api/chat", json=body).json()
    assert all("max_tokens" not in request for request in mock_llm.requests) and "visionMaxTokens" not in data["meta"]


def test_changing_the_ocr_thinking_choice_transcribes_again(client, mock_llm):
    """추론을 켠 전사와 끈 전사를 비교하려는데 앞의 결과가 그대로 쓰이면 비교가 무의미하다."""
    mock_llm.reset(seeing_model)
    scan = [upload("scan.pdf", scanned_drawing(), PDF)]
    first = client.post("/api/chat", json=chat_body(mock_llm, "도면 번호", scan, disableThinking=False)).json()
    conversation = first["conversationId"]

    def follow(text, **options):
        mock_llm.reset(seeing_model)
        body = chat_body(mock_llm, text, conversationId=conversation, disableThinking=False, **options)
        body["messages"] = [{"role": "user", "content": "도면 번호"}, {"role": "assistant", "content": first["text"]},
                            {"role": "user", "content": text}]
        data = client.post("/api/chat", json=body).json()
        return data, [thinking_disabled(request) for request in mock_llm.requests if is_ocr_call(request)]

    assert follow("다시")[1] == []                                     # 같은 선택: 저장된 전사를 쓴다
    thinking, calls = follow("추론 켜고 다시", disableThinkingOcr=False)
    assert calls == [False] and thinking["meta"]["thinkingDisabled"]["ocr"] is False
    assert follow("한 번 더", disableThinkingOcr=False)[1] == []
    back, calls = follow("끄고 다시", disableThinkingOcr=True)           # 끈 전사는 캐시에 남아 있다
    assert calls == [] and back["meta"]["vision"]["ocrCalls"] == 0


def test_the_tile_bbox_request_that_never_finished_now_ends_after_one_call_per_tile(client, mock_llm):
    """사용자 테스트의 재현: 타일 모드 bbox, 추론 켬, 일부 타일에서 추론이 맴돈다."""
    def model(body):
        if is_grounding_call(body):
            image = request_images(body)[0]
            if {"red", "blue"} & set(colors_in(image)):
                return {"text": "", "reasoning": RUNAWAY_REASONING, "finish_reason": "length"}
            return box_reply(image)
        return locate(body)

    mock_llm.reset(model)
    # 4000 x 2000 → 2행 x 3열. 파랑은 r1c1, 초록은 r1c2·r2c2(세로 겹침에 걸침), 빨강은 r2c3에 있다.
    image = plan_upload([(200, 300, 600, 500, "blue"), (2000, 920, 2200, 1080, "green"), (3000, 1500, 3200, 1700, "red")])
    body = chat_body(mock_llm, "사각형 위치를 표시해줘", [image], imageMode="tile", disableThinking=False,
                     disableThinkingGrounding=False)
    data = client.post("/api/chat", json=body).json()

    grounding = [request for request in mock_llm.requests if is_grounding_call(request)]
    first = [request for request in grounding if not request.get("continue_final_message")]
    follow = [request for request in grounding if request.get("continue_final_message")]
    assert len(first) == 4                                            # 타일마다 한 번. 예전 규칙이면 2 + 2 x 3 = 8번
    # Step 6: 맴도는 두 타일은 추론 반복이 잡혀 "추론을 끊고 답만 이어 쓰기" 요청을 한 번씩 더 받는다(재시도가 아니라
    # 소프트 조치). 이 서버는 이어 쓰기에서도 같은 반복을 내놓아 하드 중단된다 — 역시 다시 묻지 않는다.
    assert len(follow) == 2 and all(request["messages"][-1]["content"].endswith("</think>\n\n") for request in follow)
    assert all(not thinking_disabled(request) and request["max_tokens"] == 4096 + config.REASONING_BUDGET_GROUNDING
               for request in first)
    assert all(request["max_tokens"] == 4096 for request in follow)   # 이어 쓰기는 출력 몫만
    assert data["text"] == "표시했습니다."
    assert [box["label"] for box in data["artifacts"][0]["boxes"]] == ["green"]
    vision = data["meta"]["vision"]
    assert (vision["tiles"], vision["groundingCalls"], vision["groundingLengthStops"]) == (4, 4, 0)
    assert (vision["groundingReasoningForced"], vision["groundingReasoningStops"]) == (2, 2)
    # 어느 타일이 왜 끊겼는지는 화면용 메타에만 남는다(도구 결과에는 타일 이름을 넣지 않는다 — 아래 told).
    actions = data["meta"]["reasoningActions"]
    assert sorted(item["image"].split(" · tile ")[1] for item in actions) == ["r1c1", "r2c3"]
    assert all(item == {"kind": "grounding", "image": item["image"], "reason": "repeat", "stopped": True} for item in actions)
    saved = client.get(f"/api/sessions/{data['conversationId']}").json()["messages"]
    assert saved[1]["meta"]["reasoningActions"] == actions
    assert data["meta"]["thinkingDisabled"]["grounding"] is False

    # 답변 모델은 어느 타일을 재지 못했는지 안다. 추론 글은 넘어가지 않는다.
    told = all_text(mock_llm.requests[-1])
    assert "2 of 4 tiles were stopped because the model's reasoning did not finish" in told
    assert "door symbols" not in told and "door symbols" not in json.dumps(data, ensure_ascii=False)
