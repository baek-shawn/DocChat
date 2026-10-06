"""Step 10 — 필요할 때만 그림을 보기: 보기 도구(`view_page`)와 답변 이미지 모드 자동(`auto`), bbox 도구와의 역할 분담.

자동 모드는 업로드 이미지만 싣고 시작하고(기본 모드와 같다), PDF 쪽은 답변 모델이 보기 도구로 한 쪽씩 요청한다.
요청한 쪽은 그 턴 안에서 쌓여 다음 호출부터 모두 실리고, 상한을 넘으면 붙이지 않고 알려만 준다.
"""
from __future__ import annotations

import json

from conftest import chat_body, upload
from mock_openai import all_text, image_count, is_grounding_call, is_ocr_call, request_images, system_text
from pdf_factory import build_pdf, png_bytes
from test_ocr_evidence import ScriptedProvider

from app import config
from app.agent.tools import ToolContext, available_tools, execute_tool
from app.attachments import Attachment
from app.db import sanitize_meta
from app.pipeline.evidence import attachment_manifest, drawing_cue, drawing_pages, page_ranges
from app.pipeline.pdf import render_pdf_for_vision
from app.providers.base import ToolCall

PDF = "application/pdf"
BOX = {"text": "", "regions": [{"type": "text", "label": "REV C", "bbox": [100, 80, 400, 110]}]}


def main_calls(mock):
    return [body for body in mock.requests if not is_ocr_call(body) and not is_grounding_call(body)]


def tool_names(body) -> list[str]:
    return [tool["function"]["name"] for tool in body.get("tools", [])]


def tool_results(body) -> list[str]:
    return [str(message.get("content") or "") for message in body["messages"] if message.get("role") == "tool"]


def anchor_text(body) -> str:
    """원래 질문(이미지가 붙는 user 메시지 — 이력 뒤의 마지막 질문)의 글."""
    users = [message for message in body["messages"] if message["role"] == "user"]
    with_images = [message for message in users if isinstance(message["content"], list)]
    message = with_images[0] if with_images else [m for m in users if not str(m["content"]).startswith("TOOL RESULT")][-1]
    content = message["content"]
    return "\n".join(part["text"] for part in content if part.get("type") == "text") if isinstance(content, list) else content


def view(name: str, page: int | None = None) -> dict:
    arguments = {"name": name, **({"page": page} if page else {})}
    return {"tool_calls": [{"name": "view_page", "arguments": arguments}]}


def follow_up(mock, text: str, conversation_id: str, previous: list[tuple[str, str]], **extra) -> dict:
    body = chat_body(mock, text, conversationId=conversation_id, **extra)
    body["messages"] = [*({"role": role, "content": content} for role, content in previous), {"role": "user", "content": text}]
    return body


# --------------------------------------------------------------------------- 모드: 자동에서만 보기 도구와 판단 재료
def test_auto_mode_offers_the_view_tool_and_marks_drawing_pages(client, mock_llm):
    mock_llm.reset(lambda body: "OCR TEXT" if is_ocr_call(body) else "answer")
    data = client.post("/api/chat", json=chat_body(mock_llm, "이 도면의 형상은?", [upload("scan.pdf", build_pdf("native", "scanned"), PDF)],
                                                   answerImageMode="auto")).json()
    (main,) = main_calls(mock_llm)
    assert tool_names(main) == ["view_page", "inspect_visual", "read_attachment", "search_attachments"]
    system = system_text(main)
    assert "view_page attaches the image of one page" in system and f"up to {config.MAX_VIEWED_PAGES} pages" in system
    # 판단 재료: 그림이 있는 쪽과 그 쪽의 글이 무엇을 담는지(전사한 2쪽) — 네이티브 1쪽은 그림이 없다.
    assert ('"scan.pdf": parts=3, pages=2, images=1, parsedText=' in system
            and "drawings on pages 2 (text = transcription of the visible labels only) - the shapes" in system)
    assert image_count(main) == 0 and "[PAGE IMAGES" not in all_text(main)         # 처음에는 업로드 이미지만(여기선 없음)
    assert data["meta"]["answerImageMode"] == "auto"
    assert data["meta"]["answerImages"] == {"sent": 0, "candidates": 0, "names": []}
    assert data["meta"]["viewedPages"] == {"names": [], "limit": config.MAX_VIEWED_PAGES, "refused": 0}
    health = client.get("/api/health").json()
    assert health["view"] == {"maxViewedPages": config.MAX_VIEWED_PAGES, "drawingMinRasterArea": config.DRAWING_MIN_RASTER_AREA,
                              "drawingMinVectorOperations": config.DRAWING_MIN_VECTOR_OPERATIONS}


def test_other_modes_keep_the_step8_prompt_without_the_view_tool(client, mock_llm):
    """끔 / 업로드만 / 전체는 실험 ①의 비교 기준이다 — 도구 목록·시스템 프롬프트·매니페스트가 Step 8까지와 같아야 한다."""
    for mode in ("off", "uploads", "whole"):
        mock_llm.reset(lambda body: "OCR TEXT" if is_ocr_call(body) else "answer")
        data = client.post("/api/chat", json=chat_body(mock_llm, "형상은?", [upload("scan.pdf", build_pdf("native", "scanned"), PDF)],
                                                       answerImageMode=mode)).json()
        (main,) = main_calls(mock_llm)
        assert tool_names(main) == ["inspect_visual", "read_attachment", "search_attachments"], mode
        system = system_text(main)
        assert "view_page" not in system and "drawings on pages" not in system, mode
        assert "viewedPages" not in data["meta"], mode
    bad = client.post("/api/chat", json=chat_body(mock_llm, "hi", answerImageMode="view"))
    assert bad.status_code == 400 and "'auto'" in bad.json()["error"]


# --------------------------------------------------------------------------- 보기 도구: 다음 호출부터 실리고 쌓인다
def test_view_page_attaches_the_page_to_the_next_call_and_pages_accumulate(client, mock_llm):
    def handler(body):
        results = tool_results(body)
        if not results:
            return view("spec.pdf", 2)
        if len(results) == 1:
            return view("spec.pdf", 3)
        return "2쪽과 3쪽을 보고 답합니다."

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "2쪽과 3쪽의 그림을 비교해줘",
                                                   [upload("spec.pdf", build_pdf("native", "native", "native"), PDF)],
                                                   answerImageMode="auto")).json()
    first, second, third = main_calls(mock_llm)
    assert image_count(first) == 0
    # 첫 보기 → 다음 호출에 그 쪽이 원래 질문에 붙고, 질문 끝에 몇 번째 이미지가 어느 쪽인지 적힌다.
    assert image_count(second) == 1
    assert "[IMAGES ATTACHED TO THIS MESSAGE, in this order - 1: spec.pdf · page 2 (requested with view_page)." in anchor_text(second)
    assert isinstance(second["messages"][1]["content"], list)
    result = tool_results(second)[0]
    assert result.startswith("Attached spec.pdf · page 2 to the user's message as image #1 (images attached, in order: 1: spec.pdf · page 2).")
    assert f"up to {config.MAX_VIEWED_PAGES - 1} more pages this turn" in result
    # 두 번째 보기 → 쌓여서 두 장이 모두, 요청한 순서로 실린다.
    assert image_count(third) == 2
    assert "1: spec.pdf · page 2 (requested with view_page); 2: spec.pdf · page 3 (requested with view_page)." in anchor_text(third)
    assert "image #2 (images attached, in order: 1: spec.pdf · page 2; 2: spec.pdf · page 3)" in tool_results(third)[1]
    pages = {item["name"]: item for item in data["attachments"] if item.get("pageNumber")}
    assert sorted(pages) == ["spec.pdf · page 2", "spec.pdf · page 3"]          # 1쪽은 그리지 않았다
    assert [image.size for image in request_images(third)] == [(pages[name]["width"], pages[name]["height"])
                                                               for name in ("spec.pdf · page 2", "spec.pdf · page 3")]
    assert data["text"] == "2쪽과 3쪽을 보고 답합니다." and data["artifacts"] == []     # 보기는 bbox 아티팩트를 만들지 않는다
    assert data["meta"]["viewedPages"] == {"names": ["spec.pdf · page 2", "spec.pdf · page 3"], "limit": config.MAX_VIEWED_PAGES, "refused": 0}
    saved = client.get(f"/api/sessions/{data['conversationId']}").json()["messages"]
    assert saved[1]["meta"]["viewedPages"] == data["meta"]["viewedPages"]


def test_view_limit_and_duplicate_requests(client, mock_llm, monkeypatch):
    monkeypatch.setattr(config, "MAX_VIEWED_PAGES", 1)

    def handler(body):
        results = tool_results(body)
        if not results:
            return view("spec.pdf", 1)
        if len(results) == 1:
            return view("spec.pdf", 1)                   # 같은 쪽을 다시
        if len(results) == 2:
            return view("spec.pdf", 2)                   # 상한 초과
        return "1쪽만 보고 답합니다."

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "그림 설명", [upload("spec.pdf", build_pdf("native", "native"), PDF)],
                                                   answerImageMode="auto")).json()
    final = main_calls(mock_llm)[-1]
    results = tool_results(final)
    assert results[0].startswith("Attached spec.pdf · page 1") and "No more pages can be attached this turn." in results[0]
    assert results[1] == ("spec.pdf · page 1 is already attached to the user's message as image #1; nothing was added. "
                          "Look at that image to answer.")
    assert results[2] == ("Page limit reached: 1 pages are already attached for this turn, so spec.pdf · page 2 was not attached. "
                          "Answer from the pages you can see, and tell the user that only 1 pages were viewed.")
    assert image_count(final) == 1
    assert data["meta"]["viewedPages"] == {"names": ["spec.pdf · page 1"], "limit": 1, "refused": 1}
    assert [item["name"] for item in data["attachments"]] == ["spec.pdf", "spec.pdf · page 1"]     # 2쪽은 그리지 않았다


def test_view_page_on_an_already_attached_upload_adds_nothing(client, mock_llm):
    mock_llm.reset(lambda body: "보고 답" if tool_results(body) else view("pic.png"))
    data = client.post("/api/chat", json=chat_body(mock_llm, "뭐가 보여?", [upload("pic.png", png_bytes(), "image/png")],
                                                   answerImageMode="auto")).json()
    first, final = main_calls(mock_llm)
    assert image_count(first) == 1 and image_count(final) == 1                  # 두 장이 되지 않는다
    assert tool_results(final)[0].startswith("pic.png is already attached to the user's message as image #1; nothing was added.")
    assert "[IMAGES ATTACHED" not in anchor_text(final)                         # 보기 도구로 더한 것이 없으면 줄도 없다
    assert data["meta"]["viewedPages"] == {"names": [], "limit": config.MAX_VIEWED_PAGES, "refused": 0}


def test_follow_up_turn_does_not_carry_pages_but_can_request_them_again(client, mock_llm):
    mock_llm.reset(lambda body: "답" if tool_results(body) else view("spec.pdf", 2))
    data = client.post("/api/chat", json=chat_body(mock_llm, "2쪽 그림은?", [upload("spec.pdf", build_pdf("native", "native"), PDF)],
                                                   answerImageMode="auto")).json()
    names = [item["name"] for item in data["attachments"]]
    assert names == ["spec.pdf", "spec.pdf · page 2"]
    conversation_id = data["conversationId"]

    mock_llm.reset(lambda body: "다시 보고 답" if tool_results(body) else view("spec.pdf", 2))
    again = client.post("/api/chat", json=follow_up(mock_llm, "3번 치수는 어느 선의 것이야?", conversation_id,
                                                    [("user", "2쪽 그림은?"), ("assistant", "답")], answerImageMode="auto")).json()
    first, final = main_calls(mock_llm)
    assert image_count(first) == 0                                               # 이전 턴에서 본 쪽은 자동으로 실리지 않는다
    assert image_count(final) == 1 and "1: spec.pdf · page 2 (requested with view_page)" in anchor_text(final)
    assert [item["name"] for item in again["attachments"]] == names              # 렌더해 둔 쪽을 다시 쓴다(중복 없음)
    assert again["meta"]["viewedPages"]["names"] == ["spec.pdf · page 2"]


# --------------------------------------------------------------------------- bbox 도구와 섞이지 않는다
def test_viewed_pages_stay_out_of_the_bbox_call_and_bbox_results_stay_out_of_viewing(client, mock_llm):
    def handler(body):
        if is_grounding_call(body):
            return json.dumps(BOX)
        results = tool_results(body)
        if not results:
            return view("spec.pdf", 1)
        if len(results) == 1:
            return {"tool_calls": [{"name": "inspect_visual", "arguments": {"name": "spec.pdf", "page": 2, "task": "find REV C"}}]}
        return "1쪽을 보고, 2쪽의 REV C를 표시했습니다."

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "1쪽 형상을 보고 2쪽의 REV C를 표시해줘",
                                                   [upload("spec.pdf", build_pdf("native", "native"), PDF)], answerImageMode="auto")).json()
    first, second, grounding, final = mock_llm.requests
    assert image_count(grounding) == 1 and "IMAGES ATTACHED" not in all_text(grounding)   # bbox 호출은 그 쪽 한 장만
    page2 = next(item for item in data["attachments"] if item["name"] == "spec.pdf · page 2")
    assert request_images(grounding)[0].size == (page2["width"], page2["height"])
    assert image_count(final) == 1 and "1: spec.pdf · page 1 (requested with view_page)" in anchor_text(final)
    (artifact,) = data["artifacts"]
    assert artifact["name"] == "spec.pdf · page 2" and artifact["boxes"][0]["label"] == "REV C"
    assert data["meta"]["viewedPages"]["names"] == ["spec.pdf · page 1"]


# --------------------------------------------------------------------------- JSON 폴백(gemma3 경로)
def test_json_fallback_attaches_viewed_pages_to_the_original_question(client, mock_llm):
    def handler(body):
        if "tools" in body:
            return {"status": 400, "body": {"error": {"message": "gemma3 does not support tools"}}}
        if "TOOL RESULT (view_page)" in all_text(body):
            return "그림을 보고 답합니다."
        return '{"tool_calls":[{"name":"view_page","arguments":{"name":"spec.pdf","page":1}}]}'

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "형상 설명", [upload("spec.pdf", build_pdf("native"), PDF)], answerImageMode="auto")).json()
    assert data["text"] == "그림을 보고 답합니다."
    protocol = mock_llm.requests[1]
    assert "TOOL PROTOCOL" in system_text(protocol) and '"view_page"' in system_text(protocol)
    assert "reply with ONLY the view_page tool-call JSON" in anchor_text(protocol)
    final = mock_llm.requests[-1]
    users = [message for message in final["messages"] if message["role"] == "user"]
    assert isinstance(users[0]["content"], list) and image_count(final) == 1           # 이미지는 원래 질문에
    assert "[IMAGES ATTACHED TO THIS MESSAGE, in this order - 1: spec.pdf · page 1 (requested with view_page)." in anchor_text(final)
    assert isinstance(users[-1]["content"], str) and users[-1]["content"].startswith("TOOL RESULT (view_page):\nAttached spec.pdf · page 1")
    assert data["meta"]["viewedPages"]["names"] == ["spec.pdf · page 1"]


# --------------------------------------------------------------------------- 트레이스
def test_trace_records_which_page_was_viewed_and_when(client, mock_llm, monkeypatch):
    monkeypatch.setenv("DOCCHAT_DEBUG_TRACE", "1")
    mock_llm.reset(lambda body: "답" if tool_results(body) else view("spec.pdf", 1))
    data = client.post("/api/chat", json=chat_body(mock_llm, "형상은?", [upload("spec.pdf", build_pdf("native"), PDF)], answerImageMode="auto")).json()
    document = client.get(f"/api/traces/{data['meta']['traceId']}").json()
    events = document["events"]
    (entry,) = [event for event in events if event["kind"] == "input"]
    assert entry["data"]["answerImageMode"] == "auto" and entry["data"]["view"]["maxViewedPages"] == config.MAX_VIEWED_PAGES
    (evidence,) = [event for event in events if event["label"] == "증거 조립"]
    assert evidence["data"]["viewTool"] is True and evidence["data"]["tools"][0] == "view_page"
    (tool,) = [event for event in events if event["kind"] == "tool" and event["data"].get("name") == "view_page"]
    assert tool["data"]["arguments"] == {"name": "spec.pdf", "page": 1} and tool["data"]["result"].startswith("Attached spec.pdf · page 1")
    note = next(event for event in events if event["label"].startswith("spec.pdf · page 1을(를) 다음 답변 호출부터 실음"))
    page = next(item for item in data["attachments"] if item["name"] == "spec.pdf · page 1")
    assert note["parent"] == tool["id"] and note["data"]["attachmentId"] == page["id"]
    answers = [event for event in events if event["kind"] == "model" and event["data"]["kind"] == "answer"]
    assert [len(call["data"]["images"]) for call in answers] == [0, 1]
    assert answers[1]["data"]["images"][0] == {"name": "spec.pdf · page 1", "mime": "image/png", "bytes": answers[1]["data"]["images"][0]["bytes"],
                                               "attachmentId": page["id"], "width": page["width"], "height": page["height"]}
    assert "base64," not in json.dumps(document)


# --------------------------------------------------------------------------- 단위: 도구, 판단 재료, 메타
async def test_view_page_tool_renders_the_page_and_reports_its_position():
    pdf = Attachment(name="spec.pdf", kind="pdf", mime=PDF, data=build_pdf("native", "native"), text="t", total_pages=2)
    context = ToolContext(provider=ScriptedProvider([]), attachments=[pdf], base_image_names=["pic.png"])
    result = await execute_tool(context, ToolCall(name="view_page", arguments={"name": "spec.pdf", "page": 2}))
    assert result.startswith("Attached spec.pdf · page 2 to the user's message as image #2 (images attached, in order: 1: pic.png; 2: spec.pdf · page 2).")
    assert [image.name for image in context.viewed] == ["spec.pdf · page 2"] and context.viewed[0].tile is None
    rendered = context.attachments[-1]
    assert rendered.name == "spec.pdf · page 2" and rendered.data.startswith(b"\x89PNG") and not rendered.send_to_model
    assert context.viewed[0].data == rendered.data and context.attached_image_names() == ["pic.png", "spec.pdf · page 2"]
    # 쪽을 고르지 않은 여러 쪽 PDF, 모르는 이름 → 모델이 고칠 수 있는 오류
    assert (await execute_tool(context, ToolCall(name="view_page", arguments={"name": "spec.pdf"}))).startswith(
        'ERROR: "spec.pdf" has 2 pages: pass "page"')
    assert 'Exact names: "spec.pdf", "spec.pdf · page 2"' in await execute_tool(context, ToolCall(name="view_page", arguments={"name": "x.pdf"}))
    # 한 쪽짜리 PDF는 쪽 번호 없이도 된다.
    single = ToolContext(provider=ScriptedProvider([]), attachments=[Attachment(name="one.pdf", kind="pdf", mime=PDF, data=build_pdf("native"), text="t", total_pages=1)])
    assert (await execute_tool(single, ToolCall(name="view_page", arguments={"name": "one.pdf"}))).startswith("Attached one.pdf · page 1")


def test_view_tool_is_offered_only_with_a_visual_surface():
    pdf = [Attachment(name="a.pdf", kind="pdf", mime=PDF, text="body")]
    assert [tool.name for tool in available_tools(pdf, view_tool=True)] == ["view_page", "inspect_visual", "read_attachment", "search_attachments"]
    assert [tool.name for tool in available_tools(pdf)] == ["inspect_visual", "read_attachment", "search_attachments"]
    text_only = [Attachment(name="notes.txt", kind="document", mime="text/plain", text="body")]
    assert [tool.name for tool in available_tools(text_only, view_tool=True)] == ["read_attachment", "search_attachments"]


def test_drawing_judgement_uses_raster_area_and_vector_volume_not_counts():
    """머리글 로고(쪽 넓이의 0.1~0.2%)는 그림이 아니고, 본문 그림(3% 이상)·CAD 도면(벡터 수천)·글자 없는 벡터 쪽은 그림이다."""
    assert not config.is_drawing_page(raster_area=0.002, vector_operations=3, classification="mixed-native")
    assert config.is_drawing_page(raster_area=0.03, vector_operations=0, classification="mixed-native")
    assert config.is_drawing_page(raster_area=0.0, vector_operations=config.DRAWING_MIN_VECTOR_OPERATIONS, classification="native-vector")
    assert not config.is_drawing_page(raster_area=0.0, vector_operations=66, classification="native-vector")     # 벡터 데이터시트(표)
    assert config.is_drawing_page(raster_area=0.0, vector_operations=26, classification="vector-outlines")
    assert config.is_drawing_page(raster_area=0.0, vector_operations=0, classification="scanned-raster")

    inspection = render_pdf_for_vision(build_pdf("native", "scanned", "vector"), render_images=False)
    native, scanned, vector = inspection.page_analysis
    assert native.raster_area == 0.0 and not native.is_drawing()
    assert 0.3 < scanned.raster_area < 0.4 and scanned.is_drawing()           # 320x200 그림을 524pt 너비에 맞춤 → 524x327pt ≈ 34%
    assert vector.raster_area == 0.0 and vector.is_drawing()                   # 글자 없는 벡터 쪽
    assert scanned.to_public() == {"page": 2, "classification": "scanned-raster", "chars": 0, "raster": 1,
                                   "rasterArea": round(scanned.raster_area, 4), "vector": 0, "vlm": True}


def test_manifest_drawing_cue_lists_pages_by_what_their_text_holds():
    pdf = Attachment(name="a.pdf", kind="pdf", mime=PDF, text="x" * 50, total_pages=6, page_analysis=[
        {"page": 1, "classification": "mixed-native", "chars": 500, "raster": 1, "rasterArea": 0.3, "vector": 2, "vlm": False},
        {"page": 2, "classification": "scanned-raster", "chars": 0, "raster": 1, "rasterArea": 0.99, "vector": 0, "vlm": True},
        {"page": 3, "classification": "native-vector", "chars": 800, "raster": 1, "rasterArea": 0.001, "vector": 5, "vlm": False},
        {"page": 4, "classification": "vector-outlines", "chars": 2, "raster": 0, "rasterArea": 0.0, "vector": 30, "vlm": True},
        {"page": 5, "classification": "mixed-needs-vision", "chars": 30, "raster": 2, "rasterArea": 0.4, "vector": 5, "vlm": True},
        {"page": 6, "classification": "native-vector", "chars": 900, "raster": 0, "rasterArea": 0.0, "vector": 300, "vlm": False},
    ])
    assert [page["page"] for page in drawing_pages(pdf)] == [1, 2, 4, 5, 6]
    assert drawing_cue(pdf) == ("drawings on pages 2, 4-5 (text = transcription of the visible labels only); 1, 6 (native text beside "
                                "the drawing) - the shapes, their positions and counts, and which label or dimension belongs to which "
                                "feature are NOT in the text; call view_page to see such a page")
    plain = '"a.pdf": parts=1, pages=6, images=0, parsedText=50 chars, visualOcr=0 chars, pendingVision=0'
    assert attachment_manifest([pdf]) == plain                                           # 기본: Step 8까지와 같다
    assert attachment_manifest([pdf], drawing_cues=True) == f"{plain}; {drawing_cue(pdf)}"
    no_drawing = Attachment(name="t.pdf", kind="pdf", mime=PDF, text="x", total_pages=1,
                            page_analysis=[{"page": 1, "classification": "native-vector", "chars": 900, "raster": 0, "rasterArea": 0.0, "vector": 3, "vlm": False}])
    assert attachment_manifest([no_drawing], drawing_cues=True) == '"t.pdf": parts=1, pages=1, images=0, parsedText=1 chars, visualOcr=0 chars, pendingVision=0'
    # Step 10 이전에 올린 PDF(메타데이터 없음): 본문의 [PAGE ANALYSIS] 줄에서 읽는다 — 래스터 면적을 몰라 개수로 대신한다.
    legacy = Attachment(name="old.pdf", kind="pdf", mime=PDF, total_pages=3, text=(
        "[PAGE ANALYSIS]\nPage 1: native-vector; native characters=435; raster images=0; vector operations=0; vision OCR=skipped\n"
        "Page 2: scanned-raster; native characters=0; raster images=1; vector operations=0; vision OCR=required\n"
        "Page 3: mixed-native; native characters=700; raster images=1; vector operations=4; vision OCR=skipped\n\nbody"))
    assert [page["page"] for page in drawing_pages(legacy)] == [2, 3]
    assert page_ranges([8, 1, 2, 3, 5, 7, 3]) == "1-3, 5, 7-8" and page_ranges([]) == ""


def test_meta_keeps_only_well_formed_viewed_pages():
    assert sanitize_meta({"viewedPages": {"names": ["a.pdf · page 1", 7], "limit": 12, "refused": 0}})["viewedPages"] == \
        {"names": ["a.pdf · page 1"], "limit": 12, "refused": 0}
    assert "viewedPages" not in sanitize_meta({"viewedPages": {"names": "a", "limit": 12, "refused": 0}})
    assert "viewedPages" not in sanitize_meta({"viewedPages": {"names": [], "limit": "many"}})
    assert "viewedPages" not in sanitize_meta({"viewedPages": ["a.pdf · page 1"]})
