"""Step 8 1차 — 답변(추론) 호출에 이미지 포함: 끔(off) / 업로드 이미지만(uploads, 기본) / 전체(whole).

전사·bbox 호출의 이미지 처리 방식(전체/타일)과는 별개의 축이다. 기본값은 Step 8 이전과 같은 동작이어야 한다.
"""
from __future__ import annotations

import json

from conftest import chat_body, upload
from mock_openai import all_text, image_count, is_grounding_call, is_ocr_call, request_images
from pdf_factory import build_pdf, png_bytes

from app import config
from app.attachments import Attachment
from app.pipeline.evidence import attachment_context_for_prompt, is_pending_page_image

PDF = "application/pdf"
STAMP = {"text": "", "regions": [{"type": "stamp", "label": "APPROVED", "bbox": [620, 780, 940, 930]}]}


def main_calls(mock):
    return [body for body in mock.requests if not is_ocr_call(body) and not is_grounding_call(body)]


def follow_up(mock, text: str, conversation_id: str, previous: list[tuple[str, str]], **extra) -> dict:
    body = chat_body(mock, text, conversationId=conversation_id, **extra)
    body["messages"] = [*({"role": role, "content": content} for role, content in previous), {"role": "user", "content": text}]
    return body


# --------------------------------------------------------------------------- 기본값 = 이전 동작
def test_default_mode_is_the_previous_behaviour(client, mock_llm):
    """uploads(기본): PDF 쪽은 싣지 않고 업로드 이미지는 전체 한 장. 프롬프트도 Step 8 이전과 같다."""
    mock_llm.reset(lambda body: "OCR TEXT" if is_ocr_call(body) else "answer")
    data = client.post("/api/chat", json=chat_body(mock_llm, "분석해줘", [
        upload("scan.pdf", build_pdf("native", "scanned"), PDF), upload("pic.png", png_bytes(), "image/png")])).json()
    (main,) = main_calls(mock_llm)
    assert image_count(main) == 1 and "[PAGE IMAGES" not in all_text(main)
    assert data["meta"]["answerImageMode"] == "uploads"
    assert data["meta"]["answerImages"] == {"sent": 1, "candidates": 1, "names": ["pic.png"]}
    names = [item["name"] for item in data["attachments"]]
    assert "scan.pdf · page 1" not in names                     # 네이티브 쪽을 렌더하지 않는다

    health = client.get("/api/health").json()
    assert health["answerImageMode"] == "uploads" and health["maxModelImages"] == config.MAX_MODEL_IMAGES


def test_invalid_answer_image_mode_is_rejected_readably(client, mock_llm):
    bad = client.post("/api/chat", json=chat_body(mock_llm, "hi", answerImageMode="tiles"))
    assert bad.status_code == 400 and "답변 이미지" in bad.json()["error"] and "'whole'" in bad.json()["error"]
    assert mock_llm.requests == []


# --------------------------------------------------------------------------- 전체(whole)
def test_whole_mode_sends_every_pdf_page_after_transcription(client, mock_llm):
    """전사는 그대로 먼저 하고, 답변 호출에 쪽 이미지(네이티브 쪽 + 전사한 쪽)를 쪽 순서로 싣는다."""
    mock_llm.reset(lambda body: "DWG NO B-77" if is_ocr_call(body) else "answer")
    pdf = build_pdf("native", "scanned")       # 1쪽 네이티브(전처리에서 렌더하지 않음), 2쪽 스캔(전사)
    data = client.post("/api/chat", json=chat_body(mock_llm, "이 도면의 형상을 설명해줘", [upload("scan.pdf", pdf, PDF)],
                                                   answerImageMode="whole")).json()
    ocr = [body for body in mock_llm.requests if is_ocr_call(body)]
    assert len(ocr) == 1 and image_count(ocr[0]) == 1
    (main,) = main_calls(mock_llm)
    assert image_count(main) == 2
    text = all_text(main)
    assert "[PAGE IMAGES: attached to this message in this order - 1: scan.pdf · page 1; 2: scan.pdf · page 2." in text
    assert "DWG NO B-77" in text and "[Attachment: scan.pdf · visual OCR]" in text       # 전사 글도 그대로
    assert data["meta"]["answerImageMode"] == "whole"
    assert data["meta"]["answerImages"] == {"sent": 2, "candidates": 2, "names": ["scan.pdf · page 1", "scan.pdf · page 2"]}
    saved = client.get(f"/api/sessions/{data['conversationId']}").json()["messages"]
    assert saved[1]["meta"] == data["meta"]                       # 모드 기록이 답변과 함께 저장된다(비교 실험용)

    # 네이티브 1쪽은 지금 렌더돼 첨부로 저장됐다(뷰어에서 열 수 있고, 다음 턴에는 다시 그리지 않는다).
    page1 = next(item for item in data["attachments"] if item["name"] == "scan.pdf · page 1")
    served = client.get(page1["url"])
    assert served.status_code == 200 and served.content.startswith(b"\x89PNG")
    assert [image.size for image in request_images(main)] == [(page1["width"], page1["height"])] + [
        (item["width"], item["height"]) for item in data["attachments"] if item["name"] == "scan.pdf · page 2"]

    # 후속 턴(전체): 첨부 수가 그대로(다시 렌더하지 않음), 같은 두 장이 실린다.
    conversation_id = data["conversationId"]
    mock_llm.reset(lambda body: "answer 2")
    again = client.post("/api/chat", json=follow_up(mock_llm, "치수는?", conversation_id, [("user", "이 도면의 형상을 설명해줘"), ("assistant", "answer")],
                                                    answerImageMode="whole")).json()
    (request,) = mock_llm.requests
    assert image_count(request) == 2 and not is_ocr_call(request)
    assert [item["name"] for item in again["attachments"]] == [item["name"] for item in data["attachments"]]

    # 후속 턴(기본 모드): 렌더해 둔 쪽이 기본 모드로 새지 않는다 → 이미지 0장, [PAGE IMAGES] 없음.
    mock_llm.reset(lambda body: "answer 3")
    client.post("/api/chat", json=follow_up(mock_llm, "재질은?", conversation_id, [("user", "이 도면의 형상을 설명해줘"), ("assistant", "answer")]))
    (request,) = mock_llm.requests
    assert image_count(request) == 0 and "[PAGE IMAGES" not in all_text(request)


def test_whole_mode_images_ride_along_every_loop_call_and_the_page_is_rendered_once(client, mock_llm):
    """도구 루프의 매 호출(도구 결과 뒤의 호출 포함)에 같은 이미지가 원래 질문에 붙어 가고, bbox 도구는 렌더해 둔 쪽을 다시 그리지 않는다."""
    def handler(body):
        if is_grounding_call(body):
            return json.dumps({"text": "", "regions": [{"type": "text", "label": "REVISION C", "bbox": [100, 80, 400, 110]}]})
        if any(message.get("role") == "tool" for message in body["messages"]):
            return "REVISION C 위치를 표시했습니다."
        return {"tool_calls": [{"name": "inspect_visual", "arguments": {"name": "spec.pdf", "page": 1, "task": "find REVISION C"}}]}

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "REVISION C 위치를 표시해줘", [upload("spec.pdf", build_pdf("native"), PDF)],
                                                   answerImageMode="whole")).json()
    first, grounding, final = mock_llm.requests
    assert image_count(first) == 1 and image_count(final) == 1 and image_count(grounding) == 1
    assert isinstance(final["messages"][1]["content"], list)                   # 이미지는 원래 질문(닻)에 붙어 있다
    assert data["artifacts"][0]["name"] == "spec.pdf · page 1" and data["artifacts"][0]["boxes"][0]["label"] == "REVISION C"
    assert [item["name"] for item in data["attachments"]].count("spec.pdf · page 1") == 1
    assert data["meta"]["answerImages"]["names"] == ["spec.pdf · page 1"]


def test_image_cap_keeps_page_order_and_renders_only_what_is_sent(client, mock_llm, monkeypatch):
    monkeypatch.setattr(config, "MAX_MODEL_IMAGES", 2)
    mock_llm.reset(lambda body: "answer")
    data = client.post("/api/chat", json=chat_body(mock_llm, "요약해줘", [upload("spec.pdf", build_pdf("native", "native", "native"), PDF)],
                                                   answerImageMode="whole")).json()
    (main,) = main_calls(mock_llm)
    assert image_count(main) == 2
    assert "1: spec.pdf · page 1; 2: spec.pdf · page 2." in all_text(main)
    assert data["meta"]["answerImages"] == {"sent": 2, "candidates": 3, "names": ["spec.pdf · page 1", "spec.pdf · page 2"]}
    assert [item["name"] for item in data["attachments"]] == ["spec.pdf", "spec.pdf · page 1", "spec.pdf · page 2"]   # 3쪽은 그리지 않았다
    assert client.get("/api/health").json()["maxModelImages"] == 2


def test_whole_mode_without_pdf_pages_keeps_the_prompt_unchanged(client, mock_llm):
    """업로드 이미지만 있으면 전체 모드라도 [PAGE IMAGES] 줄은 붙지 않는다(쪽이 없으니 알려 줄 것이 없다)."""
    mock_llm.reset(lambda body: "answer")
    data = client.post("/api/chat", json=chat_body(mock_llm, "뭐가 보여?", [upload("pic.png", png_bytes(), "image/png")], answerImageMode="whole")).json()
    (main,) = main_calls(mock_llm)
    assert image_count(main) == 1 and "[PAGE IMAGES" not in all_text(main)
    assert data["meta"]["answerImages"] == {"sent": 1, "candidates": 1, "names": ["pic.png"]}


# --------------------------------------------------------------------------- 끔(off)
def test_off_mode_sends_no_images_but_keeps_the_bbox_tool(client, mock_llm):
    def handler(body):
        if is_grounding_call(body):
            return json.dumps(STAMP)
        if any(message.get("role") == "tool" for message in body["messages"]):
            return "표시했습니다."
        return {"tool_calls": [{"name": "inspect_visual", "arguments": {"name": "sheet.png", "task": "locate the stamp"}}]}

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "도장 위치 표시해줘", [upload("sheet.png", png_bytes(), "image/png")],
                                                   answerImageMode="off")).json()
    first, grounding, final = mock_llm.requests
    assert image_count(first) == 0 and image_count(final) == 0                  # 답변 호출에는 이미지가 없다
    assert image_count(grounding) == 1                                           # 도구는 그 이미지를 따로 본다
    assert [tool["function"]["name"] for tool in first["tools"]] == ["inspect_visual"]
    assert data["artifacts"][0]["boxes"][0]["label"] == "APPROVED"
    assert data["meta"]["answerImageMode"] == "off"
    assert data["meta"]["answerImages"] == {"sent": 0, "candidates": 0, "names": []}


# --------------------------------------------------------------------------- 증거 선택(단위)
def test_answer_image_planning_by_mode():
    pdf = Attachment(name="a.pdf", kind="pdf", mime=PDF, text="T", total_pages=3, id=1, has_data=True)
    page2 = Attachment(name="a.pdf · page 2", kind="image", mime="image/png", page_number=2, id=2, has_data=True)
    photo = Attachment(name="p.png", kind="image", mime="image/png", send_to_model=True, id=3, has_data=True)
    items = [pdf, page2, photo]

    off = attachment_context_for_prompt("", items, 10_000, 12, answer_images="off")
    assert off.images == [] and off.image_candidates == 0
    uploads = attachment_context_for_prompt("", items, 10_000, 12, answer_images="uploads")
    assert [item.name for item in uploads.images] == ["p.png"] and uploads.image_candidates == 1

    # 전체: 업로드 이미지와 PDF 쪽을 묶음별로 돌아가며, 쪽은 쪽 순서. 렌더돼 있는 2쪽은 그 첨부, 1·3쪽은 자리표시.
    whole = attachment_context_for_prompt("", items, 10_000, 12, answer_images="whole")
    assert [item.name for item in whole.images] == ["p.png", "a.pdf · page 1", "a.pdf · page 2", "a.pdf · page 3"]
    assert whole.image_candidates == 4
    assert whole.images[2] is page2 and not is_pending_page_image(page2)
    assert all(is_pending_page_image(whole.images[index]) and whole.images[index].page_number == number
               for index, number in ((1, 1), (3, 3)))
    capped = attachment_context_for_prompt("", items, 10_000, 2, answer_images="whole")
    assert [item.name for item in capped.images] == ["p.png", "a.pdf · page 1"] and capped.image_candidates == 4
    # 질문에서 파일 이름을 짚으면 그 묶음만 본다(기존 규칙 그대로).
    narrowed = attachment_context_for_prompt("a.pdf만 봐줘", items, 10_000, 12, answer_images="whole")
    assert [item.name for item in narrowed.images] == ["a.pdf · page 1", "a.pdf · page 2", "a.pdf · page 3"]
    # 쪽 수 기록이 없는 옛 PDF는 렌더돼 있는 쪽까지만.
    old = Attachment(name="b.pdf", kind="pdf", mime=PDF, text="T", id=4, has_data=True)
    legacy = attachment_context_for_prompt("", [old, Attachment(name="b.pdf · page 1", kind="image", mime="image/png", page_number=1, id=5, has_data=True)],
                                           10_000, 12, answer_images="whole")
    assert [item.name for item in legacy.images] == ["b.pdf · page 1"] and legacy.image_candidates == 1


# --------------------------------------------------------------------------- 트레이스
def test_trace_records_the_answer_image_mode_and_the_rendered_page(client, mock_llm, monkeypatch):
    monkeypatch.setenv("DOCCHAT_DEBUG_TRACE", "1")
    mock_llm.reset(lambda body: "answer")
    data = client.post("/api/chat", json=chat_body(mock_llm, "형상 설명해줘", [upload("spec.pdf", build_pdf("native"), PDF)],
                                                   answerImageMode="whole")).json()
    document = client.get(f"/api/traces/{data['meta']['traceId']}").json()
    (entry,) = [event for event in document["events"] if event["kind"] == "input"]
    assert entry["data"]["answerImageMode"] == "whole" and entry["data"]["maxModelImages"] == config.MAX_MODEL_IMAGES
    assert any(event["label"] == "spec.pdf · page 1을(를) 지금 렌더 · 답변 호출에 실음" for event in document["events"] if event["kind"] == "evidence")
    (evidence,) = [event for event in document["events"] if event["label"] == "증거 조립"]
    page = next(item for item in data["attachments"] if item["name"] == "spec.pdf · page 1")
    assert evidence["data"]["answerImageMode"] == "whole" and evidence["data"]["imageCandidates"] == 1
    assert evidence["data"]["images"][0]["attachmentId"] == page["id"]        # 지금 렌더한 쪽도 첨부 ID로 가리킨다
    (call,) = [event for event in document["events"] if event["kind"] == "model"]
    assert call["data"]["images"][0]["name"] == "spec.pdf · page 1"
    assert "base64," not in json.dumps(document)
