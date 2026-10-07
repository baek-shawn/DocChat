"""Step 10 2차 — 따로 보기 도구(`analyze_pages`): 쪽마다 **별도 VLM 호출**로 모델이 넘긴 질문에 답하게 하고 그 글을 답변 호출에
돌려준다(답변 모델은 이미지를 보지 않는다). 보기 도구(`view_page`, 같이 보기)와 무엇을 쓸지는 모델이 고른다.

자동 모드에서만, 요청 옵션 `analyzeTool`(없으면 서버 기본값)로 끄고 켠다. 끄면 1차의 자동 모드(같이 보기만)와 도구 목록·프롬프트가
같아야 한다(실험 ③의 비교 기준). 상한 둘은 이유가 다르다: 한 호출의 쪽 수(끊어 보기 단위)와 한 턴의 총 쪽 수(시간 상한).
"""
from __future__ import annotations

import json
import re

import pytest

from conftest import chat_body, upload
from mock_openai import all_text, image_count, is_grounding_call, is_ocr_call, reasoning_effort, system_text, thinking_disabled
from pdf_factory import build_pdf, png_bytes
from test_view_tool import anchor_text, tool_names, tool_results, view

from app import config
from app.agent.tools import ToolContext, ToolError, available_tools, execute_tool, parse_page_spec
from app.attachments import Attachment
from app.db import sanitize_meta
from app.providers.base import ModelResponse, Provider, ReasoningEffortError, ToolCall

PDF = "application/pdf"
P1, P2, P3, P4 = (f"spec.pdf · page {number}" for number in (1, 2, 3, 4))


def is_analysis_call(body) -> bool:
    return "visual analysis engine" in system_text(body)


def analysis_calls(mock):
    return [body for body in mock.requests if is_analysis_call(body)]


def main_calls(mock):
    return [body for body in mock.requests if not (is_ocr_call(body) or is_grounding_call(body) or is_analysis_call(body))]


def source_of(body) -> str:
    match = re.search(r"Source: (.+)", all_text(body))
    return match.group(1).strip() if match else ""


def page_answer(body) -> str:
    """따로 보기 호출의 user 글에서 쪽을 읽어 그 쪽만의 답을 만든다(mock) — 어느 쪽의 답이 어디에 놓이는지 확인할 수 있게."""
    return f"answer for {source_of(body)}"


def analyze(name: str, pages: str | None, question: str) -> dict:
    arguments = {"name": name, "question": question, **({"pages": pages} if pages else {})}
    return {"tool_calls": [{"name": "analyze_pages", "arguments": arguments}]}


class PageProvider(Provider):
    """쪽 이름(`Source: …`)별로 정해 둔 답을 돌려주는 가짜 provider — 동시 호출이라 순서로는 짝을 맞출 수 없다."""
    name = "scripted"
    is_local = True

    def __init__(self, replies: dict[str, object]):
        super().__init__(model="m")
        self.replies, self.calls = replies, []

    def can_disable_thinking(self) -> bool:
        return True

    async def analyze(self, messages, images=None, tools=None, *, temperature=0.2, disable_thinking=False, max_tokens=None,
                      reasoning_budget=None, on_reasoning=None, reasoning_effort=None):
        source = re.search(r"Source: (.+)", messages[-1]["content"]).group(1).strip()
        self.calls.append({"source": source, "images": images, "temperature": temperature, "disable_thinking": disable_thinking,
                           "max_tokens": max_tokens, "reasoning_budget": reasoning_budget, "reasoning_effort": reasoning_effort,
                           "system": messages[0]["content"], "user": messages[-1]["content"]})
        reply = self.replies[source]
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, ModelResponse) else ModelResponse(text=str(reply))

    async def list_models(self):
        return ["m"]


# --------------------------------------------------------------------------- 제공 조건: 자동 모드 + 옵션
def test_analyze_tool_is_offered_in_auto_mode_and_can_be_switched_off(client, mock_llm):
    mock_llm.reset(lambda body: "OCR TEXT" if is_ocr_call(body) else "answer")
    pdf = upload("scan.pdf", build_pdf("native", "scanned"), PDF)
    data = client.post("/api/chat", json=chat_body(mock_llm, "형상은?", [pdf], answerImageMode="auto")).json()
    (main,) = main_calls(mock_llm)
    assert tool_names(main) == ["view_page", "analyze_pages", "inspect_visual", "read_attachment", "search_attachments"]
    system = system_text(main)
    assert "analyze_pages looks at pages for you in separate calls" in system
    assert f"up to {config.ANALYZE_PAGES_PER_CALL} pages per call" in system and f"up to {config.MAX_ANALYZED_PAGES} pages per turn" in system
    assert "go through all the pages in ranges" in system and "call view_page on the decisive pages" in system
    assert "view_page attaches the image of one page" in system                  # 같이 보기 안내는 그대로
    assert data["meta"]["analyzedPages"] == {"enabled": True, "names": [], "calls": 0, "limit": config.MAX_ANALYZED_PAGES, "refused": 0, "group": 1}
    assert data["meta"]["viewedPages"]["names"] == []
    # 끄면 1차의 자동 모드와 같다: 도구 목록, 그리고 시스템 프롬프트는 따로 보기 문단 하나만 빠진 것
    mock_llm.reset(lambda body: "OCR TEXT" if is_ocr_call(body) else "answer")
    off = client.post("/api/chat", json=chat_body(mock_llm, "형상은?", [pdf], answerImageMode="auto", analyzeTool=False)).json()
    (main_off,) = main_calls(mock_llm)
    assert tool_names(main_off) == ["view_page", "inspect_visual", "read_attachment", "search_attachments"]
    start, end = system.index("analyze_pages looks at"), system.index("If a supplied excerpt is truncated")
    assert system_text(main_off) == system[:start] + system[end:]
    assert off["meta"]["analyzedPages"] == {"enabled": False, "names": [], "calls": 0, "limit": config.MAX_ANALYZED_PAGES, "refused": 0, "group": 1}
    # 다른 모드에서는 옵션을 켜도 내놓지 않고 메타에도 없다(Step 8까지의 프롬프트 그대로)
    for mode in ("off", "uploads", "whole"):
        mock_llm.reset(lambda body: "OCR TEXT" if is_ocr_call(body) else "answer")
        other = client.post("/api/chat", json=chat_body(mock_llm, "형상은?", [pdf], answerImageMode=mode, analyzeTool=True)).json()
        (main_other,) = main_calls(mock_llm)
        assert "analyze_pages" not in tool_names(main_other) and "analyze_pages" not in system_text(main_other), mode
        assert "analyzedPages" not in other["meta"], mode
    health = client.get("/api/health").json()
    assert health["analyze"] == {"enabled": config.ANALYZE_TOOL, "pagesPerCall": config.ANALYZE_PAGES_PER_CALL,
                                 "maxPages": config.MAX_ANALYZED_PAGES, "group": config.ANALYZE_GROUP, "groupMax": config.ANALYZE_GROUP_MAX}


def test_analyze_tool_is_offered_only_with_a_visual_surface():
    pdf = [Attachment(name="a.pdf", kind="pdf", mime=PDF, text="body")]
    assert [tool.name for tool in available_tools(pdf, view_tool=True, analyze_tool=True)] == \
        ["view_page", "analyze_pages", "inspect_visual", "read_attachment", "search_attachments"]
    assert [tool.name for tool in available_tools(pdf, view_tool=True)] == ["view_page", "inspect_visual", "read_attachment", "search_attachments"]
    text_only = [Attachment(name="notes.txt", kind="document", mime="text/plain", text="body")]
    assert [tool.name for tool in available_tools(text_only, view_tool=True, analyze_tool=True)] == ["read_attachment", "search_attachments"]


# --------------------------------------------------------------------------- 쪽마다 별도 호출 → 글만 돌아온다
def test_analyze_pages_runs_one_call_per_page_and_returns_text_only(client, mock_llm):
    def handler(body):
        if is_analysis_call(body):
            return page_answer(body)
        return "글로 답" if tool_results(body) else analyze("spec.pdf", "1-3", "How many circles are drawn? Report the count.")

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "각 쪽의 원 개수", [upload("spec.pdf", build_pdf("native", "native", "native"), PDF)],
                                                   answerImageMode="auto")).json()
    first, final = main_calls(mock_llm)
    looks = analysis_calls(mock_llm)
    assert len(looks) == 3 and all(image_count(body) == 1 for body in looks)
    assert sorted(source_of(body) for body in looks) == [P1, P2, P3]
    assert all("Question: How many circles are drawn? Report the count." in all_text(body) for body in looks)
    assert all("Answer about this page only" in all_text(body) and body.get("temperature") == 0 for body in looks)
    assert all("tools" not in body for body in looks)
    assert image_count(first) == 0 and image_count(final) == 0 and "[IMAGES ATTACHED" not in anchor_text(final)   # 글만 간다
    (result,) = tool_results(final)
    assert result.startswith('Analyzed 3 pages of spec.pdf with the question: "How many circles are drawn? Report the count.". '
                             "Each answer below comes from a separate look at that page only; the pages are not attached to your call.")
    assert f"\n[page 1] answer for {P1}\n[page 2] answer for {P2}\n[page 3] answer for {P3}\n" in result
    assert f"You may analyze up to {config.MAX_ANALYZED_PAGES - 3} more pages this turn." in result
    assert "call view_page on the decisive pages to look at them yourself" in result
    assert data["text"] == "글로 답" and data["artifacts"] == []
    assert data["meta"]["analyzedPages"] == {"enabled": True, "names": [P1, P2, P3], "calls": 1, "limit": config.MAX_ANALYZED_PAGES, "refused": 0, "group": 1}
    assert data["meta"]["viewedPages"]["names"] == []
    vision = data["meta"]["vision"]
    assert vision["analysisCalls"] == 3 and vision["groundingCalls"] == 0 and vision["answerCalls"] == 2
    assert sorted(item["name"] for item in data["attachments"]) == ["spec.pdf", P1, P2, P3]   # 본 쪽은 첨부로 저장된다
    saved = client.get(f"/api/sessions/{data['conversationId']}").json()["messages"]
    assert saved[1]["meta"]["analyzedPages"] == data["meta"]["analyzedPages"]


def test_per_call_and_per_turn_limits(client, mock_llm, monkeypatch):
    monkeypatch.setattr(config, "ANALYZE_PAGES_PER_CALL", 2)
    monkeypatch.setattr(config, "MAX_ANALYZED_PAGES", 3)

    def handler(body):
        if is_analysis_call(body):
            return page_answer(body)
        results = tool_results(body)
        if not results:
            return analyze("spec.pdf", "1-4", "Q")
        if len(results) == 1:
            return analyze("spec.pdf", "3-4", "Q")
        if len(results) == 2:
            return analyze("spec.pdf", "4", "Q")
        return "답"

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "훑어줘", [upload("spec.pdf", build_pdf("native", "native", "native", "native"), PDF)],
                                                   answerImageMode="auto")).json()
    first, second, third = tool_results(main_calls(mock_llm)[-1])
    # 한 호출의 쪽 수: 앞 2쪽만 보고 다음 범위를 알려 준다
    assert f"[page 1] answer for {P1}\n[page 2] answer for {P2}\n" in first and "[page 3]" not in first
    assert 'Only the first 2 requested pages were analyzed in this call; call analyze_pages again with pages "3-4" to continue.' in first
    assert "You may analyze up to 1 more page this turn." in first
    # 턴의 총 쪽 수: 남은 1쪽만 보고 상한을 알려 준다
    assert f"[page 3] answer for {P3}" in second and "[page 4]" not in second
    assert "The per-turn limit of 3 analyzed pages is now reached; the remaining requested pages (4) were not analyzed." in second
    assert "call view_page" not in second
    # 상한에 닿은 뒤의 요청은 실행하지 않는다
    assert third == ("Page limit reached: 3 pages have already been analyzed this turn, so spec.pdf (pages 4) was not analyzed. "
                     "Answer from the results you already have, and tell the user that only 3 pages could be analyzed.")
    assert len(analysis_calls(mock_llm)) == 3
    assert data["meta"]["analyzedPages"] == {"enabled": True, "names": [P1, P2, P3], "calls": 3, "limit": 3, "refused": 1, "group": 1}
    assert [item["name"] for item in data["attachments"]] == ["spec.pdf", P1, P2, P3]      # 4쪽은 그리지 않았다


def test_same_page_and_question_is_not_asked_twice_but_a_new_question_is(client, mock_llm):
    def handler(body):
        if is_analysis_call(body):
            return page_answer(body)
        results = tool_results(body)
        if not results:
            return analyze("spec.pdf", "1", "Q1")
        if len(results) == 1:
            return analyze("spec.pdf", "1", "Q1")               # 같은 쪽·같은 질문 → 호출 없이 같은 답
        if len(results) == 2:
            return analyze("spec.pdf", "1", "Q2")               # 같은 쪽·다른 질문 → 다시 본다
        return "답"

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "질문", [upload("spec.pdf", build_pdf("native"), PDF)], answerImageMode="auto")).json()
    first, second, third = tool_results(main_calls(mock_llm)[-1])
    assert len(analysis_calls(mock_llm)) == 2
    assert first.split("\n")[1] == second.split("\n")[1] == f"[page 1] answer for {P1}"
    assert third.startswith('Analyzed 1 page of spec.pdf with the question: "Q2".')
    assert data["meta"]["analyzedPages"] == {"enabled": True, "names": [P1], "calls": 3, "limit": config.MAX_ANALYZED_PAGES, "refused": 0, "group": 1}
    assert data["meta"]["vision"]["analysisCalls"] == 2


def test_analyze_pages_on_an_uploaded_image(client, mock_llm):
    def handler(body):
        if is_analysis_call(body):
            return "two bolts, one on each side"
        return "답" if tool_results(body) else analyze("pic.png", None, "Count the bolts")

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "볼트 몇 개?", [upload("pic.png", png_bytes(), "image/png")], answerImageMode="auto")).json()
    first, final = main_calls(mock_llm)
    assert image_count(first) == 1 and image_count(final) == 1                  # 업로드 이미지는 처음부터 실려 있고 그대로다
    (look,) = analysis_calls(mock_llm)
    assert image_count(look) == 1 and source_of(look) == "pic.png"
    (result,) = tool_results(final)
    assert result.startswith('Analyzed 1 page of pic.png with the question: "Count the bolts".')
    assert "\n[pic.png] two bolts, one on each side\n" in result
    assert data["meta"]["analyzedPages"]["names"] == ["pic.png"] and data["meta"]["viewedPages"]["names"] == []


def test_analyze_then_view_page_attaches_the_decisive_page(client, mock_llm):
    """따로 본 글로 애매하면 모델이 view_page로 그 쪽을 같이 본다 — 따로 보기 결과 글은 대화에 남고 이미지는 그 뒤 호출에 붙는다."""
    def handler(body):
        if is_analysis_call(body):
            return page_answer(body)
        results = tool_results(body)
        if not results:
            return analyze("spec.pdf", "1-2", "Which page shows a section view?")
        if len(results) == 1:
            return view("spec.pdf", 2)
        return "2쪽을 직접 보고 답"

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "단면도 쪽은?", [upload("spec.pdf", build_pdf("native", "native"), PDF)],
                                                   answerImageMode="auto")).json()
    first, second, final = main_calls(mock_llm)
    assert image_count(first) == 0 and image_count(second) == 0 and image_count(final) == 1
    assert f"1: {P2} (requested with view_page)" in anchor_text(final)
    analyzed, viewed = tool_results(final)
    assert analyzed.startswith("Analyzed 2 pages of spec.pdf") and viewed.startswith(f"Attached {P2}")
    assert data["meta"]["viewedPages"]["names"] == [P2] and data["meta"]["analyzedPages"]["names"] == [P1, P2]
    assert [item["name"] for item in data["attachments"]] == ["spec.pdf", P1, P2]       # 2쪽은 한 번만 그린다


def test_analysis_calls_follow_the_bbox_thinking_setting(client, mock_llm):
    """추론 끄기·출력 상한·추론 수준은 bbox 호출과 같은 축을 따른다(2차 설계: 일단 bbox와 같이)."""
    def handler(body):
        if is_analysis_call(body):
            return page_answer(body)
        return "답" if tool_results(body) else analyze("spec.pdf", "1", "Q")

    mock_llm.reset(handler)
    client.post("/api/chat", json=chat_body(mock_llm, "q", [upload("spec.pdf", build_pdf("native"), PDF)], answerImageMode="auto",
                                            disableThinking=False, disableThinkingGrounding=True, reasoningEffortGrounding="medium"))
    (look,) = analysis_calls(mock_llm)
    assert thinking_disabled(look) and reasoning_effort(look) is None and look["max_tokens"] == config.VISION_MAX_TOKENS
    mock_llm.reset(handler)
    client.post("/api/chat", json=chat_body(mock_llm, "q", [upload("spec.pdf", build_pdf("native"), PDF)], answerImageMode="auto",
                                            disableThinking=False, disableThinkingGrounding=False, reasoningEffortGrounding="medium"))
    (look,) = analysis_calls(mock_llm)
    assert not thinking_disabled(look) and reasoning_effort(look) == "medium"


# --------------------------------------------------------------------------- JSON 폴백(gemma3 경로)
def test_json_fallback_reminder_and_tool_result(client, mock_llm):
    def handler(body):
        if "tools" in body:
            return {"status": 400, "body": {"error": {"message": "gemma3 does not support tools"}}}
        if is_analysis_call(body):
            return page_answer(body)
        if "TOOL RESULT (analyze_pages)" in all_text(body):
            return "훑고 답"
        return '{"tool_calls":[{"name":"analyze_pages","arguments":{"name":"spec.pdf","pages":"1-2","question":"Q"}}]}'

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "훑어줘", [upload("spec.pdf", build_pdf("native", "native"), PDF)], answerImageMode="auto")).json()
    assert data["text"] == "훑고 답"
    protocol = mock_llm.requests[1]
    assert "TOOL PROTOCOL" in system_text(protocol) and '"analyze_pages"' in system_text(protocol)
    assert "reply with ONLY the analyze_pages tool-call JSON with a page range and a concrete question" in anchor_text(protocol)
    final = mock_llm.requests[-1]
    users = [message for message in final["messages"] if message["role"] == "user"]
    assert image_count(final) == 0
    assert isinstance(users[-1]["content"], str) and users[-1]["content"].startswith("TOOL RESULT (analyze_pages):\nAnalyzed 2 pages of spec.pdf")
    assert data["meta"]["analyzedPages"]["names"] == [P1, P2] and len(analysis_calls(mock_llm)) == 2


# --------------------------------------------------------------------------- 트레이스
def test_trace_nests_the_per_page_calls_under_the_tool(client, mock_llm, monkeypatch):
    monkeypatch.setenv("DOCCHAT_DEBUG_TRACE", "1")

    def handler(body):
        if is_analysis_call(body):
            return page_answer(body)
        return "답" if tool_results(body) else analyze("spec.pdf", "1-2", "Q")

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "q", [upload("spec.pdf", build_pdf("native", "native"), PDF)], answerImageMode="auto")).json()
    document = client.get(f"/api/traces/{data['meta']['traceId']}").json()
    events = document["events"]
    (entry,) = [event for event in events if event["kind"] == "input"]
    assert entry["data"]["analyze"] == {"enabled": True, "pagesPerCall": config.ANALYZE_PAGES_PER_CALL, "maxPages": config.MAX_ANALYZED_PAGES, "group": config.ANALYZE_GROUP, "groupMax": config.ANALYZE_GROUP_MAX}
    (evidence,) = [event for event in events if event["label"] == "증거 조립"]
    assert evidence["data"]["analyzeTool"] is True and evidence["data"]["tools"][1] == "analyze_pages"
    assert evidence["data"]["maxAnalyzedPages"] == config.MAX_ANALYZED_PAGES
    (tool,) = [event for event in events if event["kind"] == "tool" and event["data"].get("name") == "analyze_pages"]
    assert tool["data"]["arguments"] == {"name": "spec.pdf", "pages": "1-2", "question": "Q"}
    assert tool["data"]["result"].startswith("Analyzed 2 pages of spec.pdf")
    looks = [event for event in events if event["kind"] == "model" and event["data"]["kind"] == "analysis"]
    assert len(looks) == 2 and all(event["parent"] == tool["id"] for event in looks)
    assert sorted(event["label"] for event in looks) == [f"따로 보기 호출 · {P1}", f"따로 보기 호출 · {P2}"]
    pages = {item["name"]: item for item in data["attachments"] if item.get("pageNumber")}
    for event in looks:
        (image,) = event["data"]["images"]
        assert image["attachmentId"] == pages[image["name"]]["id"]
    plan = next(event for event in events if event["label"].startswith("spec.pdf의 2쪽을 따로 보기 · 쪽마다"))
    assert plan["parent"] == tool["id"] and plan["data"]["question"] == "Q" and plan["data"]["pages"] == ["page 1", "page 2"]
    outcome = next(event for event in events if event["label"].startswith("따로 보기 결과 · 2쪽"))
    assert [item["page"] for item in outcome["data"]["results"]] == ["page 1", "page 2"] and not any(item["failed"] for item in outcome["data"]["results"])
    answers = [event for event in events if event["kind"] == "model" and event["data"]["kind"] == "answer"]
    assert [len(call["data"]["images"]) for call in answers] == [0, 0]
    assert "base64," not in json.dumps(document)


# --------------------------------------------------------------------------- 단위: 인자, 쪽별 실패, 메타
def test_page_spec_parsing():
    assert parse_page_spec("7") == [7] and parse_page_spec(7) == [7] and parse_page_spec("1-3") == [1, 2, 3]
    assert parse_page_spec("2, 5, 7-9") == [2, 5, 7, 8, 9] and parse_page_spec([3, "1-2"]) == [1, 2, 3]
    assert parse_page_spec("3~4") == [3, 4] and parse_page_spec("5-5") == [5]
    for bad in ("", "a", "3-1", "0", None, True, "1-5000", "1,,x"):
        with pytest.raises(ToolError):
            parse_page_spec(bad)


async def test_tool_arguments_are_checked_and_missing_pages_are_reported():
    pdf = Attachment(name="spec.pdf", kind="pdf", mime=PDF, data=build_pdf("native", "native", "native"), text="t", total_pages=3)
    provider = PageProvider({P1: "one", P2: "two", P3: "three"})
    context = ToolContext(provider=provider, attachments=[pdf])
    call = lambda **arguments: execute_tool(context, ToolCall(name="analyze_pages", arguments=arguments))  # noqa: E731
    assert (await call(name="spec.pdf", pages="1", question="")).startswith('ERROR: analyze_pages needs a non-empty "question"')
    assert (await call(name="spec.pdf", question="Q")).startswith('ERROR: "spec.pdf" has 3 pages: pass "pages"')
    assert 'Exact names: "spec.pdf"' in await call(name="x.pdf", pages="1", question="Q")
    assert (await call(name="spec.pdf", pages="4-6", question="Q")).startswith('ERROR: "spec.pdf" has only 3 pages; none of the requested pages exist.')
    assert (await call(name="spec.pdf", pages="x", question="Q")).startswith('ERROR: "pages" must be a page number')
    assert provider.calls == [] and context.analysis_calls == 5 and context.analyzed_pages == 0
    # 있는 쪽만 보고, 없는 쪽은 무시했다고 알려 준다
    result = await call(name="spec.pdf", pages="2-5", question="Q")
    assert "\n[page 2] two\n[page 3] three\n" in result and "2 requested pages beyond the end of the document were ignored." in result
    assert sorted(item["source"] for item in provider.calls) == [P2, P3]
    assert all("visual analysis engine" in item["system"] and item["user"].startswith("Question: Q\nSource: spec.pdf · page") for item in provider.calls)
    assert all(item["temperature"] == 0.0 and len(item["images"]) == 1 and item["images"][0].tile is None for item in provider.calls)
    assert context.analyzed == [P2, P3] and context.analyzed_pages == 2
    assert [item.name for item in context.attachments] == ["spec.pdf", P2, P3]       # 1쪽은 그리지 않았다
    # 한 쪽짜리 PDF는 쪽 번호 없이도 된다
    single = ToolContext(provider=PageProvider({"one.pdf · page 1": "only"}),
                         attachments=[Attachment(name="one.pdf", kind="pdf", mime=PDF, data=build_pdf("native"), text="t", total_pages=1)])
    assert "[page 1] only" in await execute_tool(single, ToolCall(name="analyze_pages", arguments={"name": "one.pdf", "question": "Q"}))


async def test_cut_off_runaway_and_errors_are_reported_per_page():
    pdf = Attachment(name="spec.pdf", kind="pdf", mime=PDF, data=build_pdf("native", "native", "native"), text="t", total_pages=3)
    provider = PageProvider({
        P1: ModelResponse(text="partial answer", finish_reason="length"),
        P2: ModelResponse(text="", finish_reason="reasoning_runaway", runaway="repeat"),
        P3: RuntimeError("boom"),
    })
    context = ToolContext(provider=provider, attachments=[pdf], disable_thinking=True)
    result = await execute_tool(context, ToolCall(name="analyze_pages", arguments={"name": "spec.pdf", "pages": "1-3", "question": "Q"}))
    assert "\n[page 1] partial answer\n[cut off at the output limit of 4096 tokens; the rest of this answer is missing]\n" in result
    assert "\n[page 2] (not analyzed: the model's reasoning did not finish and was stopped; the call was not retried)\n" in result
    assert "\n[page 3] ERROR: boom\n" in result
    assert context.usage.analysis_calls == 3 and context.usage.analysis_length_stops == 1 and context.usage.analysis_reasoning_stops == 1
    assert context.usage.reasoning_actions == [{"kind": "analysis", "image": P2, "reason": "repeat", "stopped": True}]
    assert context.analyzed == [P1, P2, P3] and context.analyzed_pages == 3           # 시간은 썼다 → 상한에 센다
    assert all(item["disable_thinking"] is True and item["max_tokens"] == 4096 and item["reasoning_budget"] is None for item in provider.calls)
    # 추론을 끄지 않은 로컬 호출에서 끊긴 글은 추론일 수 있어 버린다(Step 6-0 규칙)
    context = ToolContext(provider=PageProvider({P1: ModelResponse(text="maybe thinking", finish_reason="length")}), attachments=[pdf],
                          disable_thinking=False, reasoning_effort="low")
    result = await execute_tool(context, ToolCall(name="analyze_pages", arguments={"name": "spec.pdf", "pages": "1", "question": "Q"}))
    assert "\n[page 1] (not analyzed: the model reached the output limit of 4096 tokens before answering; the call was not retried)\n" in result
    (call,) = context.provider.calls
    assert call["disable_thinking"] is False and call["reasoning_budget"] == config.REASONING_BUDGET_GROUNDING and call["reasoning_effort"] == "low"
    # 한 쪽도 보지 못하면 도구 오류(루프가 모델에게 ERROR로 돌려준다), 추론 수준 거절은 묻히지 않는다
    context = ToolContext(provider=PageProvider({P1: RuntimeError("down")}), attachments=[pdf])
    with pytest.raises(RuntimeError, match="down"):
        await execute_tool(context, ToolCall(name="analyze_pages", arguments={"name": "spec.pdf", "pages": "1", "question": "Q"}))
    context = ToolContext(provider=PageProvider({P1: ReasoningEffortError("no such effort")}), attachments=[pdf])
    with pytest.raises(ReasoningEffortError):
        await execute_tool(context, ToolCall(name="analyze_pages", arguments={"name": "spec.pdf", "pages": "1", "question": "Q"}))


def test_meta_keeps_only_well_formed_analyzed_pages():
    good = {"enabled": True, "names": ["a.pdf · page 1", 7], "calls": 2, "limit": 60, "refused": 0, "group": 1}
    assert sanitize_meta({"analyzedPages": good})["analyzedPages"] == {"enabled": True, "names": ["a.pdf · page 1"], "calls": 2, "limit": 60, "refused": 0, "group": 1}
    assert sanitize_meta({"analyzedPages": {**good, "enabled": False, "names": []}})["analyzedPages"]["enabled"] is False
    assert "analyzedPages" not in sanitize_meta({"analyzedPages": {**good, "enabled": "yes"}})
    assert "analyzedPages" not in sanitize_meta({"analyzedPages": {**good, "calls": "two"}})
    assert "analyzedPages" not in sanitize_meta({"analyzedPages": ["a.pdf · page 1"]})
    vision = sanitize_meta({"vision": {"analysisCalls": 3, "analysisLengthStops": 1, "analysisReasoningForced": 0, "analysisReasoningStops": 2}})["vision"]
    assert vision == {"analysisCalls": 3, "analysisLengthStops": 1, "analysisReasoningForced": 0, "analysisReasoningStops": 2}
    # 묶음 크기(실험 조건): 정수 또는 "auto". 틀린 값은 버린다(옛 메타처럼 없는 것과 같다)
    assert sanitize_meta({"analyzedPages": {**good, "group": "auto"}})["analyzedPages"]["group"] == "auto"
    assert sanitize_meta({"analyzedPages": {**good, "group": 5}})["analyzedPages"]["group"] == 5
    assert "group" not in sanitize_meta({"analyzedPages": {**good, "group": "lots"}})["analyzedPages"]
    assert "group" not in sanitize_meta({"analyzedPages": {**good, "group": 0}})["analyzedPages"]


# --------------------------------------------------------------------------- 묶음 보기: 한 VLM 호출에 쪽 여러 장
def grouped_answer(body) -> str:
    """묶음 호출의 user 글에서 "n: spec.pdf · page k" 목록을 읽어 쪽마다 [page k] 표식으로 답한다(mock)."""
    pages = re.findall(r"\d+: (spec\.pdf · page \d+)", all_text(body))
    if not pages:
        return page_answer(body)
    return "\n".join(f"[page {name.rsplit(' ', 1)[1]}] answer for {name}" for name in pages)


def test_fixed_group_sends_several_pages_in_one_call_and_splits_the_answers(client, mock_llm):
    """요청이 묶음을 숫자로 고정하면 그 크기로 묶고, 모델이 넘긴 group 인자는 무시한다(도구 정의에도 없다)."""
    def handler(body):
        if is_analysis_call(body):
            return grouped_answer(body)
        if tool_results(body):
            return "답"
        return {"tool_calls": [{"name": "analyze_pages", "arguments": {"name": "spec.pdf", "pages": "1-3", "question": "Q", "group": 50}}]}

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "훑어줘", [upload("spec.pdf", build_pdf("native", "native", "native"), PDF)],
                                                   answerImageMode="auto", analyzeGroup=2)).json()
    first, final = main_calls(mock_llm)
    assert "group" not in first["tools"][1]["function"]["parameters"]["properties"]
    assert 'Pass "group"' not in system_text(first)
    looks = analysis_calls(mock_llm)
    assert sorted(image_count(body) for body in looks) == [1, 2]
    grouped = next(body for body in looks if image_count(body) == 2)
    assert "Answer one question about each of the attached images" in system_text(grouped)
    assert f"Images attached to this message, in this order - 1: {P1}; 2: {P2}." in all_text(grouped)
    assert "exactly [page 1], [page 2], in that order" in all_text(grouped)
    single = next(body for body in looks if image_count(body) == 1)
    assert f"Source: {P3}" in all_text(single) and "single attached image" in system_text(single)
    (result,) = tool_results(final)
    assert result.startswith('Analyzed 3 pages of spec.pdf with the question: "Q". The pages were looked at 2 at a time in separate calls')
    assert f"\n[page 1] answer for {P1}\n[page 2] answer for {P2}\n[page 3] answer for {P3}\n" in result
    assert data["meta"]["analyzedPages"] == {"enabled": True, "names": [P1, P2, P3], "calls": 1, "limit": config.MAX_ANALYZED_PAGES,
                                             "refused": 0, "group": 2}
    assert data["meta"]["vision"]["analysisCalls"] == 2
    assert client.post("/api/chat", json=chat_body(mock_llm, "q", answerImageMode="auto", analyzeGroup="lots")).status_code == 400


def test_auto_group_lets_the_model_choose_within_the_cap(client, mock_llm, monkeypatch):
    monkeypatch.setattr(config, "ANALYZE_GROUP_MAX", 2)

    def handler(body):
        if is_analysis_call(body):
            return grouped_answer(body)
        results = tool_results(body)
        if not results:
            return {"tool_calls": [{"name": "analyze_pages", "arguments": {"name": "spec.pdf", "pages": "1-3", "question": "Q", "group": 50}}]}
        if len(results) == 1:
            return analyze("spec.pdf", "1-2", "Q2")                  # group 없음 → 쪽마다
        return "답"

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "훑어줘", [upload("spec.pdf", build_pdf("native", "native", "native"), PDF)],
                                                   answerImageMode="auto", analyzeGroup="auto")).json()
    first = main_calls(mock_llm)[0]
    assert first["tools"][1]["function"]["parameters"]["properties"]["group"]["maximum"] == 2
    assert 'Pass "group" (1-2) to have that many pages looked at in one call' in system_text(first)
    assert sorted(image_count(body) for body in analysis_calls(mock_llm)) == [1, 1, 1, 2]      # 50 → 상한 2로 자름, 둘째는 쪽마다
    assert data["meta"]["analyzedPages"]["group"] == "auto" and data["meta"]["analyzedPages"]["calls"] == 2
    assert data["meta"]["vision"]["analysisCalls"] == 4


class FlatProvider(Provider):
    """무엇을 묻든 같은 글을 돌려주는 가짜 provider(묶음 답에 [page n] 표식이 없는 경우)."""
    name = "flat"

    def __init__(self, text: str):
        super().__init__(model="m")
        self.text, self.calls = text, 0

    async def analyze(self, messages, images=None, tools=None, **_options):
        self.calls += 1
        return ModelResponse(text=self.text)

    async def list_models(self):
        return ["m"]


async def test_grouped_answer_without_markers_is_kept_as_one_block_and_a_bad_group_is_rejected():
    from app.agent.tools import split_page_answers
    assert split_page_answers("[page 1] a\n[Page 2]: b\nmore", [1, 2]) == {1: "a", 2: "b\nmore"}
    assert split_page_answers("[page 1] a", [1, 2]) is None and split_page_answers("", [1]) is None
    pdf = Attachment(name="spec.pdf", kind="pdf", mime=PDF, data=build_pdf("native", "native", "native"), text="t", total_pages=3)
    provider = FlatProvider("no markers here")
    context = ToolContext(provider=provider, attachments=[pdf], analysis_group=2)
    result = await execute_tool(context, ToolCall(name="analyze_pages", arguments={"name": "spec.pdf", "pages": "1-3", "question": "Q"}))
    assert "\n[pages 1-2] no markers here\n[page 3] no markers here\n" in result          # 묶음 답은 한 번만, 쪽들이 나눠 갖는다
    assert context.analysis_cache[(P1, "Q")] == ("pages 1-2", "no markers here") == context.analysis_cache[(P2, "Q")]
    assert provider.calls == 2 and context.usage.analysis_calls == 2 and context.analyzed_pages == 3
    assert context.analyzed == [P1, P2, P3]
    # auto에서 모델이 틀린 group을 넘기면 도구 오류(모델이 고칠 수 있게), 비우면 쪽마다
    context = ToolContext(provider=FlatProvider("x"), attachments=[pdf], analysis_group="auto")
    bad = await execute_tool(context, ToolCall(name="analyze_pages", arguments={"name": "spec.pdf", "pages": "1", "question": "Q", "group": "lots"}))
    assert bad.startswith('ERROR: "group" must be a whole number of pages from 1 to')
    await execute_tool(context, ToolCall(name="analyze_pages", arguments={"name": "spec.pdf", "pages": "1-2", "question": "Q"}))
    assert context.usage.analysis_calls == 2
