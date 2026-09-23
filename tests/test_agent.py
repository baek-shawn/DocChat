"""Step 4 — grounding 파싱, 본문 속 도구 호출 파싱, 단일 tool-calling 루프."""
from __future__ import annotations

import json

import pytest

from app import config
from app.agent.grounding import map_box_to_source, parse_visual_inspection, valid_box
from app.agent.loop import looks_like_tool_envelope, parse_embedded_tool_calls, run_tool_loop, strip_reasoning
from app.agent.tools import INSPECT_VISUAL, READ_ATTACHMENT, ToolContext, available_tools, execute_tool
from app.attachments import Attachment
from app.providers.base import ModelResponse, ToolCall, ToolsUnsupportedError
from pdf_factory import build_pdf, png_bytes
from test_ocr_evidence import ScriptedProvider


def approx(box, **expected):
    return all(box[key] == pytest.approx(value) for key, value in expected.items())


# --------------------------------------------------------------------------- grounding 파싱
def test_regions_become_fractional_boxes():
    result = parse_visual_inspection(json.dumps({"text": "two stamps", "regions": [
        {"type": "stamp", "label": "APPROVED", "bbox": [100, 200, 300, 260], "confidence": 0.93},
        {"type": "signature", "label": "J. Kim", "bbox": [700, 900, 1000, 1000]},
    ]}))
    assert result.structured and result.text == "two stamps" and len(result.boxes) == 2
    assert approx(result.boxes[0], x=0.1, y=0.2, w=0.2, h=0.06, confidence=0.93)
    assert result.boxes[0]["label"] == "APPROVED" and result.boxes[0]["type"] == "stamp"
    assert approx(result.boxes[1], x=0.7, y=0.9, w=0.3, h=0.1) and "confidence" not in result.boxes[1]
    assert all(valid_box(box) for box in result.boxes)


def test_json_is_recovered_from_fences_prose_and_think_blocks():
    payload = '{"text":"ok","regions":[{"label":"A","bbox":[0,0,500,500]}]}'
    for wrapped in (f"```json\n{payload}\n```", f"Sure! Here it is:\n{payload}\nHope that helps.",
                    f"<think>where is it?</think>{payload}"):
        result = parse_visual_inspection(wrapped)
        assert result.structured and len(result.boxes) == 1 and result.boxes[0]["type"] == "other"


def test_unstructured_answer_is_flagged_for_retry():
    result = parse_visual_inspection("The stamp is in the lower right corner.")
    assert not result.structured and result.boxes == [] and "lower right" in result.text


def test_invalid_boxes_are_dropped_and_coordinates_clamped():
    result = parse_visual_inspection(json.dumps({"regions": [
        {"label": "reversed", "bbox": [500, 500, 100, 100]},
        {"label": "zero area", "bbox": [10, 10, 10, 90]},
        {"label": "three numbers", "bbox": [1, 2, 3]},
        {"label": "not numbers", "bbox": ["a", "b", "c", "d"]},
        {"label": "nan", "bbox": [0, 0, float("nan"), 10]},
        "not a dict",
        {"label": "overflow", "bbox": [-50, 900, 1200, 1100], "confidence": 7},
    ]}).replace("NaN", "null"))
    assert [box["label"] for box in result.boxes] == ["overflow"]
    assert approx(result.boxes[0], x=0.0, y=0.9, w=1.0, h=0.1, confidence=1.0)


def test_region_count_and_label_lengths_are_bounded():
    regions = [{"label": "L" * 300, "type": "T" * 99, "bbox": [0, 0, 10, 10]}] * 500
    result = parse_visual_inspection(json.dumps({"regions": regions}))
    assert len(result.boxes) == config.MAX_GROUNDING_REGIONS == 200
    assert len(result.boxes[0]["label"]) == 80 and result.boxes[0]["type"] == "other"


def test_region_types_are_normalized_to_the_allowed_set():
    """실제 gemma3 응답에서 나온 사례: 스키마의 "text|object|table"을 그대로 베껴 온다."""
    regions = [{"type": raw, "label": "x", "bbox": [0, 0, 10, 10]}
               for raw in ("text|object|table", "STAMP", " Signature ", "logo", "", None, "dimension/text")]
    result = parse_visual_inspection(json.dumps({"regions": regions}))
    assert [box["type"] for box in result.boxes] == ["text", "stamp", "signature", "other", "other", "other", "dimension"]


def test_common_vlm_coordinate_dialects_are_normalized():
    # Qwen-VL식 키 + 픽셀 좌표(보낸 이미지 2000x1000 기준)
    pixels = parse_visual_inspection('[{"label":"bolt","bbox_2d":[200,100,1800,500]}]', image_width=2000, image_height=1000)
    assert approx(pixels.boxes[0], x=0.1, y=0.1, w=0.8, h=0.4)
    # Gemini 고유 형식은 y가 먼저다
    gemini = parse_visual_inspection('{"regions":[{"label":"title","box_2d":[100,200,300,800]}]}')
    assert approx(gemini.boxes[0], x=0.2, y=0.1, w=0.6, h=0.2)
    # 0~1 분수로 답한 경우
    fraction = parse_visual_inspection('{"regions":[{"label":"x","bbox":[0.25,0.5,0.75,1.0]}]}')
    assert approx(fraction.boxes[0], x=0.25, y=0.5, w=0.5, h=0.5)


def test_tile_boxes_map_back_to_the_full_image():
    """타일링을 위한 자리: 오른쪽 아래 1/4 타일의 박스가 전체 좌표로 옮겨진다."""
    mapped = map_box_to_source({"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2, "label": "a"}, (0.5, 0.5, 1.0, 1.0))
    assert approx(mapped, x=0.75, y=0.75, w=0.1, h=0.1)
    whole = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
    assert map_box_to_source(whole, (0.0, 0.0, 1.0, 1.0)) == whole


# --------------------------------------------------------------------------- 본문 속 도구 호출
TOOLS = {"inspect_visual", "read_attachment"}


def test_embedded_tool_calls_are_recognized_in_several_shapes():
    envelope = '{"tool_calls":[{"name":"inspect_visual","arguments":{"name":"a.png","task":"find stamp"}}]}'
    for text in (envelope, f"```json\n{envelope}\n```", f"I will check.\n{envelope}",
                 '<tool_call>{"name":"inspect_visual","arguments":{"name":"a.png","task":"find stamp"}}</tool_call>',
                 '{"name":"inspect_visual","arguments":"{\\"name\\":\\"a.png\\",\\"task\\":\\"find stamp\\"}"}',
                 '{"function":{"name":"inspect_visual","args":{"name":"a.png","task":"find stamp"}}}'):
        calls, _rest = parse_embedded_tool_calls(text, TOOLS)
        assert [(c.name, c.arguments) for c in calls] == [("inspect_visual", {"name": "a.png", "task": "find stamp"})], text


def test_ordinary_json_answers_are_not_mistaken_for_tool_calls():
    for text in ('{"name":"BRACKET","qty":4}', '{"tool_calls":[{"name":"delete_everything","arguments":{}}]}',
                 "No JSON here.", ""):
        assert parse_embedded_tool_calls(text, TOOLS) == ([], text)


# gemma3가 실제로 낸 문자열 그대로. 호출 객체를 닫는 `}`가 빠져 전체는 JSON이 아니다.
GEMMA_MALFORMED = ('{"tool_calls": [{"name": "inspect_visual", "arguments": {"name": "sheet_with_stamp", "page": 1, '
                   '"task": "mark APPROVED and signature"}]} ]}')


def test_tool_call_is_salvaged_from_an_envelope_with_mismatched_brackets():
    with pytest.raises(ValueError):
        json.loads(GEMMA_MALFORMED)                       # 전제: 정말로 깨진 JSON이다
    calls, rest = parse_embedded_tool_calls(GEMMA_MALFORMED, TOOLS)
    assert [(c.name, c.arguments) for c in calls] == [
        ("inspect_visual", {"name": "sheet_with_stamp", "page": 1, "task": "mark APPROVED and signature"})]
    assert rest == ""                                     # 깨진 봉투의 잔해를 답변으로 남기지 않는다
    # 알려진 도구 이름일 때만 건진다 → 평범한 JSON/깨진 JSON 답변은 그대로 둔다.
    assert parse_embedded_tool_calls('{"name": "BRACKET", "arguments": {"qty": 4}]}', TOOLS)[0] == []
    assert looks_like_tool_envelope(GEMMA_MALFORMED, TOOLS)
    assert looks_like_tool_envelope('{"tool_calls": [{"name": "inspect_visual", "arguments": {"name": "a', TOOLS)
    assert not looks_like_tool_envelope("도장은 왼쪽 아래에 있습니다.", TOOLS)
    assert not looks_like_tool_envelope('{"name": "BRACKET", "qty": 4}', TOOLS)


async def test_unparseable_tool_envelope_is_never_shown_to_the_user():
    """건질 수도 없을 만큼 깨진 봉투: 재요청 → 그래도 안 되면 도구 없이 최종 답을 요구한다."""
    broken = '{"tool_calls": [{"name": "read_attachment", "arguments": {"name": "a.pd'      # 중간에 잘림
    provider = ScriptedProvider([broken, '{"tool_calls":[{"name":"read_attachment","arguments":{"name":"a.pdf"}}]}', "정상 답변"])
    progress = []

    async def execute(_tool_call):
        return "TOOL OUTPUT"

    result = await run_tool_loop(provider, [{"role": "user", "content": "Q"}], tools=[READ_ATTACHMENT],
                                 execute=execute, on_progress=progress.append)
    assert result.text == "정상 답변" and result.steps == 1
    assert "was not valid JSON" in provider.calls[1]["messages"][-1]["content"]
    assert any("형식이 올바르지 않아" in message for message in progress)

    stubborn = ScriptedProvider([broken, broken, broken, "포기하고 쓴 답"])
    result = await run_tool_loop(stubborn, [{"role": "user", "content": "Q"}], tools=[READ_ATTACHMENT], execute=execute)
    assert result.text == "포기하고 쓴 답" and result.stopped_reason == "malformed_tool_call"
    assert stubborn.calls[-1]["tools"] is None

    hopeless = ScriptedProvider([broken] * 4)
    result = await run_tool_loop(hopeless, [{"role": "user", "content": "Q"}], tools=[READ_ATTACHMENT], execute=execute)
    assert "tool_calls" not in result.text and "도구 호출 형식" in result.text      # 날 JSON 대신 안내문


def test_reasoning_blocks_are_hidden():
    assert strip_reasoning("<think>secret</think>Answer") == "Answer"
    assert strip_reasoning("leaked reasoning</think>Answer") == "Answer"
    assert strip_reasoning("Answer<think>unfinished") == "Answer"


# --------------------------------------------------------------------------- 루프
def call(tool="read_attachment", **arguments):
    return ToolCall(name=tool, arguments=arguments)


async def test_loop_without_tools_is_a_single_call():
    provider = ScriptedProvider(["<think>hmm</think>안녕하세요"])
    result = await run_tool_loop(provider, [{"role": "user", "content": "hi"}])
    assert result.text == "안녕하세요" and result.steps == 0 and len(provider.calls) == 1
    assert provider.calls[0]["tools"] is None


async def test_native_tool_call_result_is_fed_back_until_final_answer():
    provider = ScriptedProvider([ModelResponse(text="", tool_calls=[call(name="a.pdf", start=0)]), "최종 답변"])
    executed = []

    async def execute(tool_call):
        executed.append(tool_call)
        return '{"content":"PAGE TEXT"}'

    result = await run_tool_loop(provider, [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}],
                                 tools=[READ_ATTACHMENT], execute=execute)
    assert result.text == "최종 답변" and result.steps == 1 and not result.used_json_fallback
    assert [c.arguments for c in executed] == [{"name": "a.pdf", "start": 0}]
    second = provider.calls[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    assert second[2]["tool_calls"][0].id == second[3]["tool_call_id"] and second[3]["content"] == '{"content":"PAGE TEXT"}'
    assert second[1]["images_anchor"] is True          # 이미지는 원래 질문에 붙는다


async def test_models_without_native_tools_fall_back_to_json_protocol():
    provider = ScriptedProvider([
        ToolsUnsupportedError("HTTP 400: does not support tools"),
        '{"tool_calls":[{"name":"read_attachment","arguments":{"name":"a.pdf"}}]}',
        "폴백으로 완성한 답",
    ])
    progress = []

    async def execute(_tool_call):
        return "TOOL OUTPUT"

    result = await run_tool_loop(provider, [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}],
                                 tools=[READ_ATTACHMENT], execute=execute, on_progress=progress.append)
    assert result.text == "폴백으로 완성한 답" and result.used_json_fallback and result.steps == 1
    assert any("JSON 방식" in message for message in progress)
    fallback_call, final_call = provider.calls[1], provider.calls[2]
    assert fallback_call["tools"] is None and "TOOL PROTOCOL" in fallback_call["messages"][0]["content"]
    assert '"read_attachment"' in fallback_call["messages"][0]["content"]
    # 폴백에서는 tool 역할을 쓰지 않고 평문으로 결과를 돌려준다.
    assert [m["role"] for m in final_call["messages"]] == ["system", "user", "assistant", "user"]
    assert final_call["messages"][-1]["content"].startswith("TOOL RESULT (read_attachment):\nTOOL OUTPUT")


async def test_language_hint_never_rides_along_with_a_json_fallback_tool_offer():
    """gemma3 실측(각 6회): 폴백의 도구 제공 호출에 언어 지시가 있으면 도구 호출 0/6, 없으면 6/6(오호출 0/6).

    그래서 힌트는 (1) 폴백의 도구 제공 호출에서는 빠지고 도구 리마인더가 대신 붙으며,
    (2) 도구가 실행된 뒤 최종 답을 요청할 때 다시 붙는다. 네이티브 경로에서는 처음부터 붙는다.
    """
    hint = "[Language: write your final answer in Korean (한국어).]"
    provider = ScriptedProvider([
        ToolsUnsupportedError("HTTP 400: does not support tools"),
        '{"tool_calls":[{"name":"read_attachment","arguments":{"name":"a.pdf"}}]}',
        "한국어 최종 답",
    ])

    async def execute(_tool_call):
        return "TOOL OUTPUT"

    await run_tool_loop(provider, [{"role": "system", "content": "S"}, {"role": "user", "content": "질문"}],
                        tools=[READ_ATTACHMENT], execute=execute, language_hint=hint)
    native_attempt, fallback_offer, after_tool = provider.calls
    assert hint in native_attempt["messages"][1]["content"]                      # 네이티브 시도: 힌트 있음
    offer_question = fallback_offer["messages"][1]["content"]
    assert hint not in offer_question and "[TOOLS: read_attachment" in offer_question   # 폴백 제공: 힌트 없음 + 리마인더
    assert all(hint not in str(m["content"]) for m in fallback_offer["messages"])
    final_turn = after_tool["messages"][-1]["content"]
    assert "TOOL RESULT (read_attachment)" in final_turn and "Now write the final answer" in final_turn
    assert final_turn.endswith(hint)                                             # 도구 실행 후: 힌트 복귀

    # 도구가 없는 순수 대화에서는 그냥 붙는다.
    plain = ScriptedProvider(["답"])
    await run_tool_loop(plain, [{"role": "user", "content": "질문"}], language_hint=hint)
    assert plain.calls[0]["messages"][0]["content"].endswith(hint)


async def test_repeated_identical_calls_are_stopped_and_a_final_answer_is_forced():
    same = ModelResponse(text="", tool_calls=[call(name="a.pdf")])
    provider = ScriptedProvider([same, same, same, "정리된 답"])

    async def execute(_tool_call):
        return "same result"

    result = await run_tool_loop(provider, [{"role": "user", "content": "Q"}], tools=[READ_ATTACHMENT], execute=execute)
    assert result.text == "정리된 답" and result.stopped_reason == "repeated_tool_call" and result.steps == 2
    final = provider.calls[-1]
    assert final["tools"] is None
    # 도구 없이 부를 때는 네이티브 tool 메시지를 평문으로 바꾼다(일부 API가 거절하기 때문).
    assert all(m["role"] in ("user", "assistant") and not m.get("tool_calls") for m in final["messages"])
    assert "stop calling tools" in final["messages"][-1]["content"]


async def test_step_budget_is_enforced():
    replies = [ModelResponse(text="", tool_calls=[call(name="a.pdf", start=index)]) for index in range(10)]
    provider = ScriptedProvider([*replies[:3], "끝"])

    async def execute(_tool_call):
        return "chunk"

    result = await run_tool_loop(provider, [{"role": "user", "content": "Q"}], tools=[READ_ATTACHMENT],
                                 execute=execute, max_steps=2)
    assert result.text == "끝" and result.steps == 2 and result.stopped_reason == "max_steps"


async def test_tool_failures_are_reported_to_the_model_not_raised():
    provider = ScriptedProvider([ModelResponse(text="", tool_calls=[call(name="x")]), "복구한 답"])

    async def execute(_tool_call):
        raise RuntimeError("disk on fire")

    result = await run_tool_loop(provider, [{"role": "user", "content": "Q"}], tools=[READ_ATTACHMENT], execute=execute)
    assert result.text == "복구한 답" and provider.calls[1]["messages"][-1]["content"] == "ERROR: disk on fire"


async def test_length_stopped_answers_are_continued():
    provider = ScriptedProvider([ModelResponse(text="첫 부분", finish_reason="length"),
                                 ModelResponse(text="나머지 부분", finish_reason="stop")])
    result = await run_tool_loop(provider, [{"role": "user", "content": "Q"}])
    assert result.text == "첫 부분\n\n나머지 부분"
    assert "Continue the previous answer" in provider.calls[1]["messages"][-1]["content"]


# --------------------------------------------------------------------------- 도구
def test_tools_are_offered_only_when_they_can_work():
    assert available_tools([]) == []
    image_only = [Attachment(name="a.png", kind="image", mime="image/png", data=b"x")]
    assert [tool.name for tool in available_tools(image_only)] == ["inspect_visual"]
    pdf = [Attachment(name="a.pdf", kind="pdf", mime="application/pdf", text="body")]
    assert [tool.name for tool in available_tools(pdf)] == ["inspect_visual", "read_attachment", "search_attachments"]
    assert INSPECT_VISUAL.parameters["required"] == ["name", "task"]


async def test_inspect_visual_makes_a_separate_grounding_call_on_one_image():
    grounding = json.dumps({"text": "1 title block", "regions": [
        {"type": "table", "label": "TITLE BLOCK", "bbox": [600, 800, 990, 980], "confidence": 0.8}]})
    provider = ScriptedProvider(["not json at all", f"```json\n{grounding}\n```"])
    image = Attachment(name="plan.png", kind="image", mime="image/png", data=png_bytes(), id=5, width=320, height=200)
    context = ToolContext(provider=provider, attachments=[image])

    output = json.loads(await execute_tool(context, call("inspect_visual", name="PLAN.PNG", task="find the title block")))

    assert output["source"] == "plan.png" and output["regions"][0]["label"] == "TITLE BLOCK"
    # 분리된 호출: 전용 시스템 프롬프트, temperature 0, 이미지 한 장, 도구 없음
    first, retry = provider.calls
    assert "visual grounding engine" in first["messages"][0]["content"] and first["temperature"] == 0.0
    assert len(first["images"]) == 1 and first["tools"] is None
    assert "0 to 1000" in first["messages"][0]["content"] and '"bbox":[x1,y1,x2,y2]' in first["messages"][1]["content"]
    assert "Return only the JSON object" in retry["messages"][1]["content"]     # 구조화 실패 → 재시도
    artifact = context.artifacts[0]
    assert artifact["view"] == "image" and artifact["attachmentId"] == 5 and artifact["task"] == "find the title block"
    assert approx(artifact["boxes"][0], x=0.6, y=0.8, w=0.39, h=0.18)
    assert "base64" not in artifact                                             # 이미지는 URL로 참조한다


async def test_inspect_visual_gives_up_after_configured_retries():
    provider = ScriptedProvider(["prose"] * (1 + config.GROUNDING_RETRY_COUNT))
    image = Attachment(name="plan.png", kind="image", mime="image/png", data=png_bytes())
    context = ToolContext(provider=provider, attachments=[image])
    output = json.loads(await execute_tool(context, call("inspect_visual", name="plan.png", task="find x")))
    assert len(provider.calls) == 3 and output["regions"] == [] and "warning" in output
    assert "boxes" not in context.artifacts[0]


async def test_inspect_visual_renders_any_pdf_page_on_demand():
    pdf = Attachment(name="spec.pdf", kind="pdf", mime="application/pdf", data=build_pdf("native", "native"), text="t")
    provider = ScriptedProvider(['{"text":"","regions":[{"label":"REV C","bbox":[10,10,200,60]}]}'])
    context = ToolContext(provider=provider, attachments=[pdf])
    output = json.loads(await execute_tool(context, call("inspect_visual", name="spec.pdf", page=2, task="find revision")))
    assert output["source"] == "spec.pdf · page 2"
    rendered = context.attachments[-1]
    assert rendered.name == "spec.pdf · page 2" and rendered.data.startswith(b"\x89PNG")
    assert not rendered.ocr_required and not rendered.send_to_model
    assert provider.calls[0]["images"][0].data == rendered.data


async def test_attachment_names_without_extension_are_accepted_only_when_unambiguous():
    """gemma3는 "sheet_with_stamp.png"를 "sheet_with_stamp"로 넘긴다. 유일하면 받아 주고, 모호하면 추측하지 않는다."""
    provider = ScriptedProvider(['{"text":"","regions":[]}'])
    sheet = Attachment(name="sheet_with_stamp.png", kind="image", mime="image/png", data=png_bytes())
    context = ToolContext(provider=provider, attachments=[sheet])
    output = json.loads(await execute_tool(context, call("inspect_visual", name="Sheet_With_Stamp", task="find stamp")))
    assert output["source"] == "sheet_with_stamp.png"

    twins = ToolContext(provider=ScriptedProvider([]), attachments=[
        Attachment(name="plan.png", kind="image", mime="image/png", data=png_bytes()),
        Attachment(name="plan.pdf", kind="pdf", mime="application/pdf", data=build_pdf("native"), text="t")])
    ambiguous = await execute_tool(twins, call("inspect_visual", name="plan", task="find stamp"))
    assert ambiguous.startswith("ERROR:") and '"plan.png"' in ambiguous and '"plan.pdf"' in ambiguous


async def test_tool_errors_help_the_model_recover():
    context = ToolContext(provider=ScriptedProvider([]), attachments=[
        Attachment(name="spec.pdf", kind="pdf", mime="application/pdf", data=build_pdf("native"), text="ABC " * 1000)])
    missing = await execute_tool(context, call("inspect_visual", name="nope.pdf", task="x"))
    assert missing.startswith("ERROR:") and '"spec.pdf"' in missing
    assert (await execute_tool(context, call("inspect_visual", name="spec.pdf", page=7, task="x"))).startswith("ERROR:")
    assert (await execute_tool(context, call("inspect_visual", name="spec.pdf"))).startswith("ERROR:")
    assert "unknown tool" in await execute_tool(context, call("web_search", query="x"))


async def test_read_and_search_attachment_tools():
    text = "HEADER\n" + "filler line\n" * 300 + "ITEM 042 GASKET QTY 8\n" + "tail\n" * 300
    context = ToolContext(provider=ScriptedProvider([]), default_read_chars=1500, attachments=[
        Attachment(name="bom.pdf", kind="pdf", mime="application/pdf", text=text),
        Attachment(name="bom.pdf · page 1", kind="image", mime="image/png", text="ITEM 042 in image metadata")])
    first = json.loads(await execute_tool(context, call("read_attachment", name="bom.pdf")))
    assert first["start"] == 0 and first["end"] == 1500 and first["hasMore"] and first["totalCharacters"] == len(text)
    rest = json.loads(await execute_tool(context, call("read_attachment", name="bom.pdf", start=first["end"], maxChars=50000)))
    assert not rest["hasMore"] and first["content"] + rest["content"] == text
    found = json.loads(await execute_tool(context, call("search_attachments", query="item 042")))
    assert [hit["name"] for hit in found["results"]] == ["bom.pdf"] and "GASKET QTY 8" in found["results"][0]["excerpt"]
