"""Step 2~4 통합 — mock OpenAI 호환 서버를 실제 HTTP로 호출하며 /api/chat 전체 흐름을 검증한다."""
from __future__ import annotations

import json

from conftest import chat_body, upload
from mock_openai import all_text, image_count, is_grounding_call, is_ocr_call, system_text
from pdf_factory import build_pdf, png_bytes

PDF = "application/pdf"


def main_calls(mock):
    return [body for body in mock.requests if not is_ocr_call(body) and not is_grounding_call(body)]


# --------------------------------------------------------------------------- Step 2: 모델 연결
def test_models_and_connection_test(client, mock_llm):
    body = {"provider": "openaiCompatible", "baseUrl": mock_llm.base_url}
    assert client.post("/api/models", json=body).json() == {"models": ["mock-vlm"]}
    assert client.get("/api/models", params={"provider": "openaiCompatible", "baseUrl": mock_llm.base_url}).json() == {"models": ["mock-vlm"]}

    ok = client.post("/api/test-connection", json={**body, "model": "mock-vlm"}).json()
    assert ok["ok"] and "1개" in ok["message"]
    unknown = client.post("/api/test-connection", json={**body, "model": "other"}).json()
    assert unknown["ok"] and "other" in unknown["message"]

    # 실패도 200 + ok:false 로 돌려준다(프론트가 한 가지 방식으로 표시).
    down = client.post("/api/test-connection", json={"provider": "openaiCompatible", "baseUrl": "http://127.0.0.1:9/v1"})
    assert down.status_code == 200 and down.json()["ok"] is False and "연결할 수 없습니다" in down.json()["message"]
    no_key = client.post("/api/test-connection", json={"provider": "anthropic"}).json()
    assert no_key["ok"] is False and "API key" in no_key["message"]
    assert client.post("/api/models", json={"provider": "openaiCompatible"}).status_code == 400


def test_plain_chat_round_trip_and_persistence(client, mock_llm):
    mock_llm.reset(lambda body: "<think>hidden</think>안녕하세요! 무엇을 도와드릴까요?")
    response = client.post("/api/chat", json=chat_body(mock_llm, "안녕"))
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "안녕하세요! 무엇을 도와드릴까요?" and data["artifacts"] == [] and data["attachments"] == []

    request = mock_llm.requests[0]
    assert request["model"] == "mock-vlm" and "tools" not in request          # 첨부가 없으면 도구도 없다
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    assert "ATTACHMENT MANIFEST: none" in system_text(request)

    saved = client.get(f"/api/sessions/{data['conversationId']}").json()
    assert saved["title"] == "안녕" and saved["model"] == "mock-vlm"
    assert [(m["role"], m["content"]) for m in saved["messages"]] == [("user", "안녕"), ("assistant", data["text"])]


def test_reply_language_follows_the_users_script(client, mock_llm):
    """영어 문서가 컨텍스트를 채워도 한국어 질문에는 한국어로 답하도록 질문 바로 뒤에 힌트를 붙인다."""
    from app.chat_service import reply_language_hint
    assert "Korean" in reply_language_hint("도면 번호 알려줘") and "Japanese" in reply_language_hint("図面番号を教えて")
    assert "Chinese" in reply_language_hint("请告诉我图号") and reply_language_hint("What is the drawing number?") == ""

    # 네이티브 tool-calling 경로: 힌트는 원래 질문 메시지의 끝에 붙는다.
    client.post("/api/chat", json=chat_body(mock_llm, "도면 번호가 뭐야?", [upload("spec.pdf", build_pdf("native"), PDF)]))
    text = all_text(mock_llm.requests[-1])
    assert text.index("도면 번호가 뭐야?") < text.index("[Attachment: spec.pdf]") < text.index("[Language: write your final answer in Korean")
    client.post("/api/chat", json=chat_body(mock_llm, "What is the drawing number?"))
    assert "[Language:" not in all_text(mock_llm.requests[-1])


def test_thinking_is_disabled_for_local_models_and_survives_servers_that_reject_the_field(client, mock_llm):
    """실측(vLLM Qwen3.5-35B): thinking을 켜 두면 "안녕"에 90초 넘게 걸리고, 끄면 0.3초."""
    client.post("/api/chat", json=chat_body(mock_llm, "안녕"))
    assert mock_llm.requests[-1]["chat_template_kwargs"] == {"enable_thinking": False}
    client.post("/api/chat", json=chat_body(mock_llm, "안녕", disableThinking=False))
    assert "chat_template_kwargs" not in mock_llm.requests[-1]

    # 모르는 필드를 거절하는 서버: 한 번 실패한 뒤 필드를 빼고 다시 보낸다.
    def strict(body):
        if "chat_template_kwargs" in body:
            return {"status": 400, "body": {"error": {"message": "Unrecognized request argument: chat_template_kwargs"}}}
        return "정상 응답"

    mock_llm.reset(strict)
    assert client.post("/api/chat", json=chat_body(mock_llm, "안녕")).json()["text"] == "정상 응답"
    assert len(mock_llm.requests) == 2


def test_assistant_identifies_as_docchat_not_the_vendor_product(client, mock_llm):
    """실측: 클라우드 GPT 모델이 "저는 ChatGPT입니다"라고 답했다."""
    client.post("/api/chat", json=chat_body(mock_llm, "너 이름이 뭐니?"))
    system = system_text(mock_llm.requests[-1])
    assert "Your name is DocChat" in system and "never introduce yourself as ChatGPT" in system
    assert 'served by the model "mock-vlm"' in system


def test_request_validation_errors_are_readable(client, mock_llm):
    assert "모델" in client.post("/api/chat", json=chat_body(mock_llm, "hi", model="")).json()["error"]
    assert "provider" in client.post("/api/chat", json=chat_body(mock_llm, "hi", provider="skynet")).json()["error"]
    bad = client.post("/api/chat", json={**chat_body(mock_llm, "hi"), "messages": [{"role": "assistant", "content": "x"}]})
    assert bad.status_code == 400 and "사용자 메시지" in bad.json()["error"]
    docx = client.post("/api/chat", json=chat_body(mock_llm, "hi", [upload("a.docx", b"PK", "application/msword")]))
    assert docx.status_code == 400 and "PDF와 이미지" in docx.json()["error"]
    assert mock_llm.requests == []                                             # 검증 실패는 모델을 부르지 않는다


def test_provider_failure_is_reported_and_stored(client, mock_llm):
    mock_llm.reset(lambda body: {"status": 500, "body": {"error": {"message": "model crashed"}}})
    response = client.post("/api/chat", json=chat_body(mock_llm, "hi"))
    assert response.status_code == 502 and "model crashed" in response.json()["error"]
    session = client.get("/api/sessions").json()["sessions"][0]
    messages = client.get(f"/api/sessions/{session['id']}").json()["messages"]
    assert messages[-1]["role"] == "assistant" and messages[-1]["content"].startswith("오류: ")

    # 저장된 오류 메시지는 다음 턴의 모델 입력에서 빠진다.
    mock_llm.reset(lambda body: "recovered")
    follow = chat_body(mock_llm, "다시", conversationId=session["id"])
    follow["messages"] = [*messages, {"role": "user", "content": "다시"}]
    assert client.post("/api/chat", json=follow).json()["text"] == "recovered"
    assert "오류" not in all_text(mock_llm.requests[-1])


# --------------------------------------------------------------------------- Step 3: 문서 전처리
def test_native_pdf_uses_text_only_and_never_sends_images(client, mock_llm):
    mock_llm.reset(lambda body: "도면 번호는 A-1024입니다.")
    data = client.post("/api/chat", json=chat_body(mock_llm, "도면 번호가 뭐야?", [upload("spec.pdf", build_pdf("native"), PDF)])).json()

    assert len(mock_llm.requests) == 1                                          # 전사 호출 없음
    request = mock_llm.requests[0]
    assert image_count(request) == 0                                            # 이미지는 만들지도 보내지도 않는다
    text = all_text(request)
    assert "[Attachment: spec.pdf]" in text and "DRAWING NO A-1024" in text and "vision OCR=skipped" in text
    assert '"spec.pdf": parts=1, pages=1, images=0' in system_text(request)
    assert [item["name"] for item in data["attachments"]] == ["spec.pdf"]
    assert data["attachments"][0]["parsedCharacters"] > 0 and "base64" not in json.dumps(data)


def test_scanned_pdf_is_transcribed_first_then_answered_from_text_only(client, mock_llm):
    def handler(body):
        if is_ocr_call(body):
            return "SCANNED TITLE BLOCK\nDWG NO  B-77   REV  D"
        return "이 도면은 B-77 REV D입니다."

    mock_llm.reset(handler)
    pdf = build_pdf("native", "scanned", "vector")
    data = client.post("/api/chat", json=chat_body(mock_llm, "도면 번호 알려줘", [upload("scan.pdf", pdf, PDF)])).json()

    ocr = [body for body in mock_llm.requests if is_ocr_call(body)]
    assert len(ocr) == 2                                                        # needs_vlm 페이지(2, 3쪽)만 전사
    assert all(image_count(body) == 1 and body["temperature"] == 0 and "tools" not in body for body in ocr)
    assert {"page 2" in all_text(body) for body in ocr} == {True, False}

    (main,) = main_calls(mock_llm)
    assert image_count(main) == 0                                               # 2단계 구조: 메인 요청엔 이미지가 없다
    text = all_text(main)
    assert "[Attachment: scan.pdf · visual OCR]" in text and "DWG NO  B-77   REV  D" in text
    assert "[VISUAL SOURCE: scan.pdf · page 2]" in text and "[CLASSIFICATION: scanned-raster]" in text
    assert "DRAWING NO A-1024" in text                                          # 1쪽 네이티브 텍스트도 함께
    assert "EVIDENCE NOTE" in text
    assert [tool["function"]["name"] for tool in main["tools"]] == ["inspect_visual", "read_attachment", "search_attachments"]

    names = [item["name"] for item in data["attachments"]]
    assert names == ["scan.pdf", "scan.pdf · page 2", "scan.pdf · page 3", "scan.pdf · visual OCR"]
    page2 = data["attachments"][1]
    served = client.get(page2["url"])                                           # 표시용 이미지는 URL로 제공
    assert served.status_code == 200 and served.headers["content-type"] == "image/png"
    assert served.content.startswith(b"\x89PNG")


def test_ocr_retries_commentary_and_survives_total_failure(client, mock_llm):
    attempts = {"count": 0}

    def handler(body):
        if is_ocr_call(body):
            attempts["count"] += 1
            return "Here is the transcription: I guess it says hello" if attempts["count"] == 1 else "LITERAL TEXT 123"
        return "done"

    mock_llm.reset(handler)
    client.post("/api/chat", json=chat_body(mock_llm, "읽어줘", [upload("one.pdf", build_pdf("scanned"), PDF)]))
    assert attempts["count"] == 2 and "LITERAL TEXT 123" in all_text(main_calls(mock_llm)[0])
    assert "previous attempt was not a valid transcription" in all_text(mock_llm.requests[1])

    mock_llm.reset(lambda body: "Sorry, I cannot do that." if is_ocr_call(body) else "일부 페이지를 읽지 못했습니다.")
    data = client.post("/api/chat", json=chat_body(mock_llm, "읽어줘", [upload("two.pdf", build_pdf("vector"), PDF)])).json()
    assert data["text"] == "일부 페이지를 읽지 못했습니다."
    assert "[OCR FAILED AFTER 3 ATTEMPTS" in all_text(main_calls(mock_llm)[0])    # 실패를 숨기지 않고 증거에 남긴다


def test_plain_image_goes_straight_to_the_vlm_without_transcription(client, mock_llm):
    mock_llm.reset(lambda body: "사진에는 스캔된 페이지가 보입니다.")
    data = client.post("/api/chat", json=chat_body(mock_llm, "뭐가 보여?", [upload("photo.png", png_bytes(), "image/png")])).json()
    assert len(mock_llm.requests) == 1 and not is_ocr_call(mock_llm.requests[0])  # §5.4: 전사 사전 처리 없음
    request = mock_llm.requests[0]
    assert image_count(request) == 1
    assert [tool["function"]["name"] for tool in request["tools"]] == ["inspect_visual"]
    assert data["attachments"][0]["width"] == 320 and data["files"][0]["attachmentId"] == data["attachments"][0]["id"]


def test_attachments_persist_for_follow_up_turns_and_reupload_replaces(client, mock_llm):
    def handler(body):
        return "OCR TEXT V1" if is_ocr_call(body) else "answer"

    mock_llm.reset(handler)
    first = client.post("/api/chat", json=chat_body(mock_llm, "분석해줘", [upload("doc.pdf", build_pdf("scanned"), PDF),
                                                                       upload("pic.png", png_bytes(), "image/png")])).json()
    conversation_id = first["conversationId"]

    # 후속 턴: 첨부를 다시 보내지 않아도 증거와 이미지가 그대로 쓰이고, 전사는 다시 하지 않는다.
    mock_llm.reset(handler)
    follow = chat_body(mock_llm, "수량은?", conversationId=conversation_id)
    follow["messages"] = [{"role": "user", "content": "분석해줘"}, {"role": "assistant", "content": "answer"},
                          {"role": "user", "content": "수량은?"}]
    client.post("/api/chat", json=follow)
    (request,) = mock_llm.requests
    assert "OCR TEXT V1" in all_text(request) and image_count(request) == 1
    assert [m["role"] for m in request["messages"]] == ["system", "user", "assistant", "user"]

    # 같은 이름으로 다시 올리면 파생 첨부까지 통째로 교체된다.
    mock_llm.reset(lambda body: "native now")
    again = chat_body(mock_llm, "교체본", [upload("doc.pdf", build_pdf("native"), PDF)], conversationId=conversation_id)
    names = [item["name"] for item in client.post("/api/chat", json=again).json()["attachments"]]
    assert names == ["pic.png", "doc.pdf"]
    assert "OCR TEXT V1" not in all_text(mock_llm.requests[-1])

    saved = client.get(f"/api/sessions/{conversation_id}").json()
    assert [item["name"] for item in saved["attachments"]] == names
    assert client.get("/api/sessions").json()["sessions"][0]["fileCount"] == 2


def test_attachment_inspect_reports_the_parse_route(client):
    report = client.post("/api/attachments/inspect", json={"attachment": upload("d.pdf", build_pdf("native", "scanned"), PDF)}).json()
    assert report["totalPages"] == 2 and report["visualPages"] == 1 and report["parsedCharacters"] > 0
    assert [(p["classification"], p["needsVlm"]) for p in report["pages"]] == [("native-vector", False), ("scanned-raster", True)]
    image = client.post("/api/attachments/inspect", json={"attachment": upload("big.png", png_bytes(5000, 2500), "image/png")}).json()
    assert image["resized"] and image["width"] == 3072 and image["sourceWidth"] == 5000
    broken = client.post("/api/attachments/inspect", json={"attachment": upload("x.pdf", b"%PDF-broken", PDF)})
    assert broken.status_code == 400 and "PDF" in broken.json()["error"]


def test_context_overflow_is_retried_with_a_compact_prompt(client, mock_llm):
    seen = []

    def handler(body):
        seen.append(len(json.dumps(body)))
        if len(seen) <= 2:
            return {"status": 400, "body": {"error": {"message": "the request exceeds the available context size (8192 tokens)"}}}
        return "요약 완료"

    mock_llm.reset(handler)
    long_text = "아주 긴 질문 " * 4000
    data = client.post("/api/chat", json=chat_body(mock_llm, long_text)).json()
    assert data["text"] == "요약 완료" and len(seen) == 3 and seen[2] < seen[1] < seen[0]
    assert "max_tokens" not in mock_llm.requests[-1]


def test_false_attachment_refusal_gets_one_grounded_retry(client, mock_llm):
    replies = iter(["죄송하지만 첨부된 PDF 파일을 직접 열 수 없습니다. 내용을 붙여 넣어 주세요.", "문서에는 A-1024가 적혀 있습니다."])
    mock_llm.reset(lambda body: next(replies))
    data = client.post("/api/chat", json=chat_body(mock_llm, "이 PDF 뭐라고 써 있어?", [upload("e.pdf", build_pdf("native"), PDF)])).json()
    assert data["text"] == "문서에는 A-1024가 적혀 있습니다." and len(mock_llm.requests) == 2
    assert "RUNTIME CORRECTION" in all_text(mock_llm.requests[1])


# --------------------------------------------------------------------------- Step 4: inspect_visual
GROUNDING = {"text": "승인 도장 1개", "regions": [
    {"type": "stamp", "label": "APPROVED", "bbox": [620, 780, 940, 930], "confidence": 0.91}]}


def test_bbox_comes_from_a_separate_tool_call_not_the_main_answer(client, mock_llm):
    def handler(body):
        if is_grounding_call(body):
            return json.dumps(GROUNDING)
        if any(message.get("role") == "tool" for message in body["messages"]):
            return "승인 도장은 오른쪽 아래에 있습니다. 뷰어에 표시했습니다."
        return {"tool_calls": [{"name": "inspect_visual", "arguments": {"name": "sheet.png", "task": "locate the approval stamp"}}]}

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "승인 도장 위치를 표시해줘", [upload("sheet.png", png_bytes(), "image/png")])).json()

    first, grounding, final = mock_llm.requests
    assert "tools" in first and "tools" not in grounding                       # grounding은 완전히 분리된 호출
    assert grounding["temperature"] == 0 and image_count(grounding) == 1
    assert "locate the approval stamp" in all_text(grounding) and "승인 도장 위치" not in all_text(grounding)
    tool_message = final["messages"][-1]
    assert tool_message["role"] == "tool" and json.loads(tool_message["content"])["regions"][0]["label"] == "APPROVED"

    assert data["text"].startswith("승인 도장은")
    (artifact,) = data["artifacts"]
    box = artifact["boxes"][0]
    assert artifact["view"] == "image" and artifact["name"] == "sheet.png"
    assert (round(box["x"], 3), round(box["y"], 3), round(box["w"], 3), round(box["h"], 3)) == (0.62, 0.78, 0.32, 0.15)
    assert box["label"] == "APPROVED" and box["type"] == "stamp"
    assert client.get(f"/api/attachments/{artifact['attachmentId']}/content").status_code == 200

    # 아티팩트는 대화와 함께 저장돼 세션을 다시 열어도 오버레이를 볼 수 있다.
    saved = client.get(f"/api/sessions/{data['conversationId']}").json()
    assert saved["messages"][-1]["artifacts"][0]["boxes"][0]["label"] == "APPROVED"


def test_bbox_works_on_a_native_pdf_page_rendered_on_demand(client, mock_llm):
    def handler(body):
        if is_grounding_call(body):
            return json.dumps({"text": "", "regions": [{"type": "text", "label": "REVISION C", "bbox": [100, 80, 400, 110]}]})
        if any(message.get("role") == "tool" for message in body["messages"]):
            return "REVISION C 위치를 표시했습니다."
        return {"tool_calls": [{"name": "inspect_visual", "arguments": {"name": "spec.pdf", "page": 1, "task": "find REVISION C"}}]}

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "REVISION C가 어디 있는지 표시해줘", [upload("spec.pdf", build_pdf("native"), PDF)])).json()
    (artifact,) = data["artifacts"]
    assert artifact["name"] == "spec.pdf · page 1" and artifact["boxes"][0]["label"] == "REVISION C"
    image = client.get(f"/api/attachments/{artifact['attachmentId']}/content")
    assert image.headers["content-type"] == "image/png"
    assert "spec.pdf · page 1" in [item["name"] for item in data["attachments"]]


def test_models_without_tool_support_use_the_json_fallback(client, mock_llm):
    def handler(body):
        if "tools" in body:                                                   # Ollama의 gemma3 등과 같은 응답
            return {"status": 400, "body": {"error": {"message": "registry.ollama.ai/library/gemma3:latest does not support tools"}}}
        if is_grounding_call(body):
            return json.dumps(GROUNDING)
        if "TOOL RESULT (inspect_visual)" in all_text(body):
            return "도장 위치를 표시했습니다."
        return '```json\n{"tool_calls":[{"name":"inspect_visual","arguments":{"name":"sheet.png","task":"find stamp"}}]}\n```'

    mock_llm.reset(handler)
    data = client.post("/api/chat", json=chat_body(mock_llm, "도장 위치 표시", [upload("sheet.png", png_bytes(), "image/png")])).json()
    assert data["text"] == "도장 위치를 표시했습니다." and data["artifacts"][0]["boxes"][0]["label"] == "APPROVED"
    protocol_call = mock_llm.requests[1]
    assert "tools" not in protocol_call and "TOOL PROTOCOL" in system_text(protocol_call)
    # 폴백에서도 이미지는 도구 결과가 아니라 원래 질문에 붙어 있어야 한다.
    final = mock_llm.requests[-1]
    user_messages = [m for m in final["messages"] if m["role"] == "user"]
    assert isinstance(user_messages[0]["content"], list) and isinstance(user_messages[-1]["content"], str)


# --------------------------------------------------------------------------- 스트리밍(NDJSON)
def test_streaming_emits_ndjson_progress_then_final(client, mock_llm):
    mock_llm.reset(lambda body: "PAGE TEXT" if is_ocr_call(body) else "스트리밍 답변")
    body = chat_body(mock_llm, "읽어줘", [upload("s.pdf", build_pdf("scanned", "scanned"), PDF)], stream=True)
    with client.stream("POST", "/api/chat", json=body) as response:
        assert response.status_code == 200 and response.headers["content-type"].startswith("application/x-ndjson")
        raw = "".join(response.iter_text())
    assert "data:" not in raw                                                   # 정식 SSE 형식을 쓰지 않는다
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert events[0]["type"] == "conversation" and events[-1]["type"] == "final"
    assert events[-1]["text"] == "스트리밍 답변" and events[-1]["conversationId"] == events[0]["conversationId"]
    progress = [event["message"] for event in events if event["type"] == "progress"]
    assert any("(2/2)" in message for message in progress) and any("답변을 생성" in message for message in progress)


def test_streaming_reports_errors_as_events(client, mock_llm):
    with client.stream("POST", "/api/chat", json=chat_body(mock_llm, "hi", model="", stream=True)) as response:
        events = [json.loads(line) for line in response.iter_lines() if line.strip()]
    assert events == [{"type": "error", "error": "모델을 선택하거나 입력하세요."}]
