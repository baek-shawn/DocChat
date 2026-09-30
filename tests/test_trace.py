"""Step 7 — 개발용 턴 트레이스: 한 턴의 과정을 시간순으로 기록하고, 진행 중에도 볼 수 있고, 꺼져 있으면 비용이 없다."""
from __future__ import annotations

import asyncio
import json
import threading

import pytest

from app import config, trace
from app.chat_service import ChatRequest, run_chat
from app.db import ChatStore, sanitize_meta
from app.providers.base import ModelResponse, ProviderError
from app.providers.openai_compat import OpenAICompatProvider
from app.providers.traced import TracedProvider
from app.trace import TurnTrace
from conftest import chat_body, upload
from mock_openai import is_grounding_call, is_ocr_call, request_images
from pdf_factory import build_pdf, encode, marked_image
from test_tiling import PDF, box_reply, scanned_drawing, seeing_model, transcribe

SECRET = "sk-this-key-must-never-be-recorded"


@pytest.fixture()
def traced(client, monkeypatch):
    """트레이스를 켠 앱(스위치는 요청마다 읽으므로 앱을 만든 뒤에 켜도 된다)."""
    monkeypatch.setenv("DOCCHAT_DEBUG_TRACE", "1")
    return client


def events_of(document: dict, kind: str | None = None) -> list[dict]:
    return [event for event in document["events"] if kind is None or event["kind"] == kind]


def locate(body):
    if is_grounding_call(body):
        return box_reply(request_images(body)[0])
    if any(message.get("role") == "tool" for message in body["messages"]):
        return "표시했습니다."
    return {"tool_calls": [{"name": "inspect_visual", "arguments": {"name": "plan.png", "task": "find the squares"}}]}


# --------------------------------------------------------------------------- 꺼져 있으면 아무것도 하지 않는다
def test_nothing_is_recorded_when_the_trace_is_off(client, mock_llm):
    assert client.get("/api/health").json()["debugTrace"] is False
    data = client.post("/api/chat", json=chat_body(mock_llm, "안녕")).json()
    assert "traceId" not in data["meta"]
    assert client.get(f"/api/sessions/{data['conversationId']}/traces").json() == {"traces": []}
    assert trace.current() is None and trace.note("progress", "x") is None and trace.start("progress", "x") is None


async def test_recording_outside_a_turn_is_a_no_op():
    async with trace.scope("tool", "없는 트레이스", name="t") as span:
        span.set(result="ignored")
        assert span.event_id is None
    trace.finish(None)


# --------------------------------------------------------------------------- 한 턴의 기록
def test_a_scanned_pdf_turn_is_traced_from_input_to_answer(traced, mock_llm):
    mock_llm.reset(seeing_model)
    assert traced.get("/api/health").json()["debugTrace"] is True
    body = chat_body(mock_llm, "도면 번호 알려줘", [upload("scan.pdf", scanned_drawing(), PDF)], apiKey=SECRET,
                     contextSize=16384, disableThinking=False)
    data = traced.post("/api/chat", json=body).json()
    trace_id = data["meta"]["traceId"]
    assert trace_id and data["meta"]["vision"]["ocrCalls"] == 1

    response = traced.get(f"/api/traces/{trace_id}")
    document = response.json()
    assert document["status"] == "done" and document["conversationId"] == data["conversationId"]
    assert document["version"] == 1 and document["elapsedMs"] >= 0 and document["reason"] == ""
    raw = json.dumps(document, ensure_ascii=False)
    assert SECRET not in raw and "base64," not in raw                       # API key도 이미지 바이트도 없다

    (entry,) = events_of(document, "input")
    assert entry["data"]["question"] == "도면 번호 알려줘" and entry["data"]["model"] == "mock-vlm"
    assert entry["data"]["attachments"] == [{"name": "scan.pdf", "kind": "pdf", "mime": PDF, "size": len(scanned_drawing())}]
    assert entry["data"]["thinkingDisabled"] == {"answer": False, "grounding": True, "ocr": True}
    assert entry["data"]["contextSize"] == 16384 and entry["data"]["visionMaxTokens"] == 4096

    (prep,) = events_of(document, "preprocess")
    assert prep["status"] == "done" and prep["data"]["totalPages"] == 1 and prep["data"]["visualPages"] == 1
    (page,) = prep["data"]["pages"]
    assert page["classification"] == "scanned-raster" and page["needsVlm"] is True and page["rendered"]["width"] == 3072
    assert prep["data"]["thresholds"] == {"nativeMinChars": config.NATIVE_MIN_CHARS, "sparseOverlayChars": config.SPARSE_OVERLAY_CHARS}

    # 전사: 쪽마다 하나의 ocr 이벤트, 그 안에 전사 모델 호출이 자식으로 달린다.
    transcriptions = [event for event in events_of(document, "ocr") if event["label"].startswith("전사 · ")]
    (ocr,) = transcriptions
    assert ocr["data"]["result"] == "ok" and "FA-7731-B" in ocr["data"]["text"] and ocr["elapsedMs"] >= 0
    ocr_calls = [event for event in events_of(document, "model") if event["data"]["kind"] == "ocr"]
    assert [event["parent"] for event in ocr_calls] == [ocr["id"]]
    assert ocr_calls[0]["data"]["disableThinking"] is True and ocr_calls[0]["data"]["maxTokens"] == 4096
    assert ocr_calls[0]["data"]["images"][0]["name"] == "scan.pdf · page 1"
    assert isinstance(ocr_calls[0]["data"]["images"][0]["attachmentId"], int)
    assert ocr_calls[0]["data"]["finishReason"] == "stop" and ocr_calls[0]["data"]["textChars"] > 0
    assert [message["role"] for message in ocr_calls[0]["data"]["messages"]] == ["system", "user"]

    (evidence,) = events_of(document, "evidence")
    names = {item["name"]: item for item in evidence["data"]["documents"]}
    assert names["scan.pdf · visual OCR"]["visualOcr"] is True and names["scan.pdf · visual OCR"]["clipped"] is False
    assert evidence["data"]["budgets"]["attachmentText"] > 0 and evidence["data"]["images"] == []
    assert evidence["data"]["tools"] == ["inspect_visual", "read_attachment", "search_attachments"]

    answers = [event for event in events_of(document, "model") if event["data"]["kind"] == "answer"]
    assert len(answers) == 1 and "parent" not in answers[0]                  # 답변 호출은 최상위
    assert answers[0]["label"] == "답변 호출 #1" and answers[0]["data"]["tools"] == evidence["data"]["tools"]
    assert answers[0]["data"]["messages"][0]["role"] == "system" and answers[0]["data"]["disableThinking"] is False
    assert answers[0]["data"]["text"] == "도면 번호는 FA-7731-B입니다."

    (final,) = events_of(document, "answer")
    assert final["data"]["text"] == data["text"] and final["data"]["meta"]["traceId"] == trace_id
    progress = [event["label"] for event in events_of(document, "progress")]
    assert progress[0] == "요청을 준비하는 중…" and any("답변을 생성" in label for label in progress)
    starts = [event["startedMs"] for event in document["events"]]
    assert starts == sorted(starts)                                          # 시간순

    # 저장된 대화에서도 답변이 트레이스를 가리킨다 → 세션을 다시 열어도 "과정 보기"가 된다.
    saved = traced.get(f"/api/sessions/{data['conversationId']}").json()["messages"]
    assert saved[1]["meta"]["traceId"] == trace_id
    listed = traced.get(f"/api/sessions/{data['conversationId']}/traces").json()["traces"]
    assert [(item["id"], item["status"]) for item in listed] == [(trace_id, "done")]


def test_a_cached_transcription_shows_up_as_no_model_call(traced, mock_llm):
    mock_llm.reset(seeing_model)
    scan = [upload("scan.pdf", scanned_drawing(), PDF)]
    first = traced.post("/api/chat", json=chat_body(mock_llm, "도면 번호", scan)).json()
    conversation = first["conversationId"]
    follow = chat_body(mock_llm, "다시", conversationId=conversation)
    follow["messages"] = [{"role": "user", "content": "도면 번호"}, {"role": "assistant", "content": first["text"],
                                                                       "meta": first["meta"]},
                          {"role": "user", "content": "다시"}]
    data = traced.post("/api/chat", json=follow).json()
    assert data["meta"]["traceId"] != first["meta"]["traceId"]
    document = traced.get(f"/api/traces/{data['meta']['traceId']}").json()
    assert not [event for event in events_of(document, "model") if event["data"]["kind"] == "ocr"]
    assert not events_of(document, "preprocess")                             # 새 첨부가 없다
    (entry,) = events_of(document, "input")
    assert entry["data"]["historyMessages"] == 2 and entry["data"]["attachments"] == []


def test_tool_calls_nest_the_grounding_calls_under_the_tool_event(traced, mock_llm):
    mock_llm.reset(locate)
    marks = [(3000, 1500, 3200, 1700, "red"), (200, 300, 600, 500, "blue")]
    image = upload("plan.png", encode(marked_image(4000, 2000, marks)), "image/png")
    data = traced.post("/api/chat", json=chat_body(mock_llm, "빨간 사각형 위치를 표시해줘", [image])).json()
    document = traced.get(f"/api/traces/{data['meta']['traceId']}").json()

    (tool,) = [event for event in events_of(document, "tool") if event["data"].get("name") == "inspect_visual"]
    assert tool["status"] == "done" and tool["data"]["arguments"] == {"name": "plan.png", "task": "find the squares"}
    assert '"regions"' in tool["data"]["result"] and tool["data"]["step"] == 1
    (grounding,) = [event for event in events_of(document, "model") if event["data"]["kind"] == "grounding"]
    assert grounding["parent"] == tool["id"] and grounding["label"] == "위치 확인 호출 · plan.png"
    assert grounding["data"]["temperature"] == 0.0 and grounding["data"]["images"][0]["width"] == 3072
    assert grounding["data"]["images"][0]["attachmentId"] == data["attachments"][0]["id"]
    outcomes = [event for event in events_of(document, "tool") if event["label"].startswith("inspect_visual 결과")]
    assert outcomes[0]["parent"] == tool["id"] and outcomes[0]["data"]["boxes"] == 2
    answers = [event for event in events_of(document, "model") if event["data"]["kind"] == "answer"]
    assert [event["label"] for event in answers] == ["답변 호출 #1", "답변 호출 #2"]
    assert answers[0]["data"]["toolCalls"][0]["name"] == "inspect_visual"
    assert answers[1]["data"]["messages"][-1]["role"] == "tool"
    assert answers[0]["data"]["images"][0]["attachmentId"] == data["attachments"][0]["id"]
    (loop_end,) = [event for event in events_of(document, "loop") if event["label"] == "도구 루프 종료"]
    assert loop_end["data"] == {"steps": 1, "modelCalls": 2, "jsonFallback": False, "continuations": 0}

    # 타일 모드: 타일마다 자식 호출 하나, 타일 위치가 적힌다.
    mock_llm.reset(locate)
    tiled = traced.post("/api/chat", json=chat_body(mock_llm, "빨간 사각형 위치", [image], imageMode="tile")).json()
    document = traced.get(f"/api/traces/{tiled['meta']['traceId']}").json()
    calls = [event for event in events_of(document, "model") if event["data"]["kind"] == "grounding"]
    assert len(calls) == tiled["meta"]["vision"]["tiles"] > 1
    assert sorted(tuple(event["data"]["tile"]) for event in calls) == sorted(tuple(event["data"]["images"][0]["tile"]) for event in calls)
    assert len({event["parent"] for event in calls}) == 1
    assert [event for event in events_of(document, "files") if "타일" in event["label"]]     # 타일 저장 기록


def test_a_page_rendered_on_demand_is_referenced_by_id_and_size(traced, mock_llm):
    """사용자 관찰(2026-09-30): 네이티브 PDF 쪽을 도구가 그 자리에서 렌더하면 트레이스의 '보낸 이미지'에 크기가 비어 있었다."""
    def model(body):
        if is_grounding_call(body):
            return '{"text": "nothing", "regions": []}'
        if any(message.get("role") == "tool" for message in body["messages"]):
            return "확인했습니다."
        return {"tool_calls": [{"name": "inspect_visual", "arguments": {"name": "spec.pdf", "page": 1, "task": "find the table"}}]}

    mock_llm.reset(model)
    data = traced.post("/api/chat", json=chat_body(mock_llm, "표 위치", [upload("spec.pdf", build_pdf("native"), PDF)])).json()
    document = traced.get(f"/api/traces/{data['meta']['traceId']}").json()
    (grounding,) = [event for event in events_of(document, "model") if event["data"]["kind"] == "grounding"]
    (image,) = grounding["data"]["images"]
    page = next(item for item in data["attachments"] if item["name"] == "spec.pdf · page 1")
    assert image["attachmentId"] == page["id"] and (image["width"], image["height"]) == (page["width"], page["height"])
    assert any(event["label"].endswith("요청 시점에 렌더") for event in events_of(document, "tool"))


def test_the_json_fallback_and_the_stop_reasons_are_recorded(traced, mock_llm):
    def stubborn(body):
        """네이티브 도구를 거절하고, JSON 방식에서는 같은 도구 호출만 되풀이하는 모델."""
        if "tools" in body:
            return {"status": 400, "body": {"error": {"message": "tools are not supported"}}}
        return '{"tool_calls": [{"name": "read_attachment", "arguments": {"name": "spec.pdf"}}]}'

    mock_llm.reset(stubborn)
    data = traced.post("/api/chat", json=chat_body(mock_llm, "읽어줘", [upload("spec.pdf", build_pdf("native"), PDF)])).json()
    document = traced.get(f"/api/traces/{data['meta']['traceId']}").json()
    labels = [event["label"] for event in events_of(document, "loop")]
    assert labels[0] == "네이티브 도구 호출 거절 → JSON 방식으로 전환"
    assert any(label.startswith("본문에 적힌 도구 호출 JSON을 읽음") for label in labels)
    assert any("되풀이돼 중단" in label for label in labels)
    failed_call = [event for event in events_of(document, "model") if event["status"] == "failed"]
    assert len(failed_call) == 1 and "ToolsUnsupportedError" in failed_call[0]["data"]["error"]
    tools = events_of(document, "tool")
    assert [event["data"]["name"] for event in tools if "name" in event["data"]] == ["read_attachment"] * 2
    (loop_end,) = [event for event in events_of(document, "loop") if event["label"] == "도구 루프 종료"]
    assert loop_end["data"]["jsonFallback"] is True and loop_end["data"]["stoppedReason"] == "repeated_tool_call"


def test_reasoning_blocks_and_false_refusals_are_recorded_as_cleanup(traced, mock_llm):
    replies = iter(["<think>hidden</think>I cannot access the file.", "도면 번호는 PS-2210-A입니다."])
    mock_llm.reset(lambda body: next(replies))
    data = traced.post("/api/chat", json=chat_body(mock_llm, "도면 번호", [upload("spec.pdf", build_pdf("native"), PDF)])).json()
    document = traced.get(f"/api/traces/{data['meta']['traceId']}").json()
    labels = [event["label"] for event in events_of(document, "cleanup")]
    assert labels == ["본문에서 추론 블록 제거", "첨부를 볼 수 없다는 답 → 첨부 내용을 근거로 다시 요청"]
    assert len([event for event in events_of(document, "model") if event["data"]["kind"] == "answer"]) == 2
    assert data["text"] == "도면 번호는 PS-2210-A입니다."


def test_a_failed_turn_keeps_its_trace_with_the_error(traced, mock_llm):
    mock_llm.reset(lambda body: {"status": 500, "body": {"error": {"message": "backend exploded"}}})
    response = traced.post("/api/chat", json=chat_body(mock_llm, "안녕"))
    assert response.status_code == 502
    conversation = traced.get("/api/sessions").json()["sessions"][0]["id"]
    saved = traced.get(f"/api/sessions/{conversation}").json()["messages"]
    trace_id = saved[1]["meta"]["traceId"]
    document = traced.get(f"/api/traces/{trace_id}").json()
    assert document["status"] == "failed" and "backend exploded" in document["reason"]
    (call,) = events_of(document, "model")
    assert call["status"] == "failed" and "backend exploded" in call["data"]["error"]


def test_traces_are_deleted_with_the_conversation_and_can_be_downloaded(traced, mock_llm):
    data = traced.post("/api/chat", json=chat_body(mock_llm, "안녕")).json()
    trace_id = data["meta"]["traceId"]
    download = traced.get(f"/api/traces/{trace_id}", params={"download": "1"})
    assert download.headers["content-disposition"] == f'attachment; filename="trace-{trace_id}.json"'
    assert json.loads(download.content)["id"] == trace_id
    assert traced.get("/api/traces/no-such-trace-id").status_code == 404
    traced.delete(f"/api/sessions/{data['conversationId']}")
    assert traced.get(f"/api/traces/{trace_id}").status_code == 404


def test_meta_keeps_only_a_valid_trace_id():
    assert sanitize_meta({"traceId": "3f2b9c1e-0000-4000-8000-000000000000"}) == {"traceId": "3f2b9c1e-0000-4000-8000-000000000000"}
    assert sanitize_meta({"traceId": "../x"}) == {} and sanitize_meta({"traceId": 12}) == {}


# --------------------------------------------------------------------------- 진행 중인 턴: 시작 시점 기록
class Gate:
    """모델 응답을 붙들어 두는 mock 핸들러. 테스트가 `release()`를 부를 때까지 요청이 끝나지 않는다."""

    def __init__(self):
        self.reached, self.open = threading.Event(), threading.Event()

    def __call__(self, body):
        self.reached.set()
        self.open.wait(timeout=10)
        return "늦은 답"

    def release(self):
        self.open.set()


async def run_turn(store, mock, gate, *, cancel: bool = False):
    events: list[dict] = []
    request = ChatRequest(base_url=mock.base_url, model="mock-vlm", messages=[{"role": "user", "content": "안녕"}])
    task = asyncio.create_task(run_chat(store, request, events.append))
    await asyncio.to_thread(gate.reached.wait, 10)        # 모델 호출이 나갔고 아직 답이 없다
    trace_id = next(event["traceId"] for event in events if event["type"] == "conversation")
    for _ in range(40):                                   # 시작 기록이 DB에 쓰일 때까지(지연 저장 0.25초)
        document = await store.get_trace(trace_id)
        if document and any(event["kind"] == "model" for event in document["events"]):
            break
        await asyncio.sleep(0.05)
    assert document is not None and document["status"] == "running"
    (call,) = [event for event in document["events"] if event["kind"] == "model"]
    assert call["status"] == "running" and call["endedMs"] is None and call["data"]["model"] == "mock-vlm"
    assert call["data"]["messages"][-1]["content"].startswith("안녕")     # 뒤에는 언어 힌트가 붙는다
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        gate.release()
        await task
    return trace_id


async def test_an_unfinished_model_call_is_already_in_the_trace(mock_llm, monkeypatch, tmp_path):
    monkeypatch.setenv("DOCCHAT_DEBUG_TRACE", "1")
    gate = Gate()
    mock_llm.reset(gate)
    store = await ChatStore(":memory:", files_dir=tmp_path / "files").open()
    try:
        trace_id = await run_turn(store, mock_llm, gate)
        document = await store.get_trace(trace_id)
        assert document["status"] == "done"
        (call,) = [event for event in document["events"] if event["kind"] == "model"]
        assert call["status"] == "done" and call["data"]["text"] == "늦은 답" and call["elapsedMs"] >= 0
    finally:
        gate.release()
        await store.close()


async def test_a_cancelled_turn_marks_what_was_still_running(mock_llm, monkeypatch, tmp_path):
    monkeypatch.setenv("DOCCHAT_DEBUG_TRACE", "1")
    gate = Gate()
    mock_llm.reset(gate)
    store = await ChatStore(":memory:", files_dir=tmp_path / "files").open()
    try:
        trace_id = await run_turn(store, mock_llm, gate, cancel=True)
        document = await store.get_trace(trace_id)
        assert document["status"] == "cancelled" and document["reason"] == trace.CANCELLED_REASON
        (call,) = [event for event in document["events"] if event["kind"] == "model"]
        assert call["status"] == "cancelled" and call["data"]["reason"] == trace.CANCELLED_REASON
        assert call["endedMs"] is not None
        assert trace.current() is None                                           # 컨텍스트가 정리됐다
    finally:
        gate.release()
        await store.close()


# --------------------------------------------------------------------------- 살아 있는 턴 vs 기록만 남은 턴
async def test_a_live_turn_is_reported_and_a_stale_record_is_not(mock_llm, monkeypatch, tmp_path):
    """사용자 관찰(2026-09-30): 끝난 지 오래된 기록이 DB에 running으로 남아 화면의 경과 시간이 계속 올라갔다."""
    monkeypatch.setenv("DOCCHAT_DEBUG_TRACE", "1")
    gate = Gate()
    mock_llm.reset(gate)
    store = await ChatStore(":memory:", files_dir=tmp_path / "files").open()
    try:
        events: list[dict] = []
        request = ChatRequest(base_url=mock_llm.base_url, model="mock-vlm", messages=[{"role": "user", "content": "안녕"}])
        task = asyncio.create_task(run_chat(store, request, events.append))
        await asyncio.to_thread(gate.reached.wait, 10)
        trace_id = next(event["traceId"] for event in events if event["type"] == "conversation")
        assert trace.is_live(trace_id)
        gate.release()
        await task
        assert not trace.is_live(trace_id)

        # 지난 프로세스가 running으로 남긴 기록: 서버 시작 때 "중단됨"으로 정리된다.
        stale = TurnTrace("3f2b9c1e-0000-4000-8000-000000000000")
        stale.conversation_id = (await store.list_conversations())[0]["id"]
        stale.start("model", "끝나지 않은 호출")
        await store.save_trace(stale.id, stale.conversation_id, "running", stale.to_json(), stale.created_at)
        assert await store.interrupt_running_traces(trace.interrupt_document) == 1
        assert await store.interrupt_running_traces(trace.interrupt_document) == 0
        document = await store.get_trace(stale.id)
        assert document["status"] == "interrupted" and document["reason"] == trace.INTERRUPTED_REASON
        (call,) = document["events"]
        assert call["status"] == "interrupted" and call["data"]["reason"] == trace.INTERRUPTED_REASON and call["endedMs"] is not None
        listed = await store.list_traces(stale.conversation_id)
        assert {item["id"]: item["status"] for item in listed}[stale.id] == "interrupted"
    finally:
        gate.release()
        await store.close()


def test_the_api_marks_running_records_that_this_server_is_not_running(traced, mock_llm):
    data = traced.post("/api/chat", json=chat_body(mock_llm, "안녕")).json()
    trace_id = data["meta"]["traceId"]
    assert "live" not in traced.get(f"/api/traces/{trace_id}").json()          # 끝난 턴에는 표시하지 않는다
    store = traced.app.state.store
    stale = TurnTrace("3f2b9c1e-0000-4000-8000-000000000000")
    stale.start("model", "끝나지 않은 호출")
    traced.portal.call(store.save_trace, stale.id, data["conversationId"], "running", stale.to_json(), stale.created_at)
    document = traced.get(f"/api/traces/{stale.id}").json()
    assert document["status"] == "running" and document["live"] is False


# --------------------------------------------------------------------------- TurnTrace 자체
async def test_events_are_flushed_while_running_and_once_more_on_close():
    writes: list[tuple[str, str]] = []

    async def writer(trace_id, conversation_id, status, document_json, created_at):
        writes.append((status, document_json))

    turn = TurnTrace("3f2b9c1e-0000-4000-8000-000000000000", writer)
    turn.activate()
    try:
        event = trace.start("model", "느린 호출", model="m")
        await asyncio.sleep(trace.FLUSH_DELAY * 3)
        assert [status for status, _ in writes] == ["running"]
        assert json.loads(writes[0][1])["events"][0]["status"] == "running"
        trace.finish(event, "done", text="ok")
        trace.note("progress", "거의 끝")
        await turn.close("done")
    finally:
        turn.deactivate()
    assert writes[-1][0] == "done"
    document = json.loads(writes[-1][1])
    assert [event["status"] for event in document["events"]] == ["done", "done"]
    assert document["events"][0]["data"]["text"] == "ok" and document["events"][0]["elapsedMs"] >= 0
    assert trace.current() is None


async def test_saves_are_serialized_so_the_last_write_is_the_newest_state():
    """실제로 겪은 것(2026-09-30): 먼저 시작된 저장이 DB 잠금에 밀려 늦게 끝나며 취소 기록을 running으로 되돌렸다."""
    writes, active, peak = [], 0, 0

    async def slow_writer(trace_id, conversation_id, status, document_json, created_at):
        nonlocal active, peak
        active += 1; peak = max(peak, active)
        await asyncio.sleep(0.3 if not writes else 0.01)     # 첫 저장이 오래 걸린다
        writes.append(status); active -= 1

    turn = TurnTrace("3f2b9c1e-0000-4000-8000-000000000000", slow_writer)
    turn.start("model", "느린 호출")
    await asyncio.sleep(trace.FLUSH_DELAY + 0.05)             # 첫 저장이 시작돼 느린 쓰기 중
    await turn.close("cancelled", reason=trace.CANCELLED_REASON)
    await asyncio.sleep(0.1)
    assert peak == 1 and writes[-1] == "cancelled" and writes[0] == "running"


async def test_the_final_save_survives_a_cancelled_request():
    writes = []

    async def slow_writer(trace_id, conversation_id, status, document_json, created_at):
        await asyncio.sleep(0.2)
        writes.append((status, json.loads(document_json)["events"][0]["status"]))

    turn = TurnTrace("3f2b9c1e-0000-4000-8000-000000000000", slow_writer)
    turn.start("model", "취소될 호출")

    async def closing():
        await turn.close("cancelled", reason=trace.CANCELLED_REASON)

    task = asyncio.create_task(closing())
    await asyncio.sleep(0.05)                                   # close()가 저장을 기다리는 중
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0.4)                                    # 저장 태스크는 shield 덕에 계속된다
    assert writes and writes[-1] == ("cancelled", "cancelled")


async def test_close_finishes_running_events_with_the_turn_status():
    turn = TurnTrace("3f2b9c1e-0000-4000-8000-000000000000")
    running = turn.start("tool", "끝나지 않은 도구", name="inspect_visual")
    turn.note("progress", "진행")
    await turn.close("failed", reason="서버 오류")
    document = turn.to_document()
    assert document["status"] == "failed" and document["reason"] == "서버 오류"
    first, second = document["events"]
    assert first["status"] == "failed" and first["data"]["reason"] == "서버 오류" and first["endedMs"] is not None
    assert second["status"] == "done" and "reason" not in second.get("data", {})
    await turn.close("done")                              # 두 번 닫아도 처음 상태가 남는다
    assert turn.to_document()["status"] == "failed" and running.status == "failed"


async def test_a_failing_writer_never_breaks_the_turn():
    async def writer(*_args):
        raise RuntimeError("disk full")

    turn = TurnTrace("3f2b9c1e-0000-4000-8000-000000000000", writer)
    turn.note("progress", "x")
    await turn.close("done")
    assert turn.to_document()["status"] == "done"


async def test_scopes_record_failures_and_nest_children():
    turn = TurnTrace("3f2b9c1e-0000-4000-8000-000000000000")
    turn.activate()
    try:
        async with trace.scope("tool", "바깥", name="read_attachment") as outer:
            trace.note("progress", "안쪽")
            async with trace.scope("ocr", "전사") as inner:
                inner.set(result="ok")
            outer.set(result="done")
        with pytest.raises(ValueError):
            async with trace.scope("tool", "실패하는 도구"):
                raise ValueError("boom")
        async with trace.scope("tool", "본문이 정한 상태") as span:
            span.status = "failed"
    finally:
        turn.deactivate()
    events = turn.to_document()["events"]
    assert [(event["label"], event.get("parent")) for event in events] == [
        ("바깥", None), ("안쪽", 1), ("전사", 1), ("실패하는 도구", None), ("본문이 정한 상태", None)]
    assert events[2]["data"] == {"result": "ok"} and events[0]["data"] == {"name": "read_attachment", "result": "done"}
    assert events[3]["status"] == "failed" and events[3]["data"]["error"] == "ValueError: boom"
    assert events[4]["status"] == "failed"


def test_text_is_clipped_and_images_are_referenced_by_attachment_id():
    long = "x" * 100_000
    clipped = trace.clip(long)
    assert len(clipped) < config.TRACE_TEXT_LIMIT + 100 and "자 생략" in clipped and clipped.startswith("xxx")
    assert trace.clip("short") == "short" and trace.clip(None) == ""

    from app.attachments import Attachment
    from app.pipeline.images import ModelImage
    from app.providers.base import ToolCall
    turn = TurnTrace("3f2b9c1e-0000-4000-8000-000000000000")
    turn.register_attachments([Attachment(name="plan.png", mime="image/png", kind="image", id=7, width=3072, height=1536)])
    whole = turn.describe_image(ModelImage(name="plan.png", mime="image/png", data=b"123"))
    assert whole == {"name": "plan.png", "mime": "image/png", "bytes": 3, "attachmentId": 7, "width": 3072, "height": 1536}
    tile = turn.describe_image(ModelImage(name="plan.png · tile r2c3", mime="image/png", data=b"1", width=1500,
                                          height=1000, tile=(2, 3), source_box=(0.5, 0.5, 1.0, 1.0)))
    assert tile["attachmentId"] == 7 and tile["tile"] == [2, 3] and tile["sourceBox"] == [0.5, 0.5, 1.0, 1.0]
    assert (tile["width"], tile["height"]) == (1500, 1000)
    assert "data" not in json.dumps(turn.describe_image(ModelImage(name="unknown.png", mime="image/png", data=b"?")))

    messages = trace.describe_messages([
        {"role": "system", "content": "S"}, {"role": "user", "content": long, "images_anchor": True},
        {"role": "assistant", "content": "", "tool_calls": [ToolCall("inspect_visual", {"name": "a"}, id="c1")], "raw": object()},
        {"role": "tool", "tool_call_id": "c1", "name": "inspect_visual", "content": "{}"},
    ])
    assert messages[1]["chars"] == 100_000 and messages[1]["imagesAnchor"] is True and "자 생략" in messages[1]["content"]
    assert messages[2]["toolCalls"] == [{"id": "c1", "name": "inspect_visual", "arguments": {"name": "a"}}]
    assert messages[3] == {"role": "tool", "content": "{}", "chars": 2, "name": "inspect_visual", "toolCallId": "c1"}
    assert "raw" not in json.dumps(messages)


async def test_the_traced_provider_forwards_everything_and_records_failures(mock_llm):
    turn = TurnTrace("3f2b9c1e-0000-4000-8000-000000000000")
    inner = OpenAICompatProvider(name="openaiCompatible", model="mock-vlm", api_key=SECRET, base_url=mock_llm.base_url,
                                 is_local=True)
    provider = TracedProvider(inner, turn)
    try:
        assert provider.is_local and provider.can_disable_thinking() and provider.cache_namespace == inner.cache_namespace
        assert provider.api_key == "" and await provider.list_models() == ["mock-vlm"]
        response = await provider.analyze([{"role": "user", "content": "q"}], disable_thinking=True, max_tokens=99)
        assert response.text == "mock reply"
        mock_llm.reset(lambda body: {"status": 500, "body": {"error": {"message": "down"}}})
        with pytest.raises(ProviderError):
            await provider.analyze([{"role": "user", "content": "q"}])
    finally:
        await provider.aclose()
    first, second = turn.to_document()["events"]
    assert first["data"]["disableThinking"] is True and first["data"]["maxTokens"] == 99 and first["status"] == "done"
    assert first["data"]["text"] == "mock reply" and first["data"]["promptTokens"] == 0
    assert second["status"] == "failed" and "down" in second["data"]["error"]
    assert SECRET not in turn.to_json()

    # "추론 끄기 — 모든 호출"로 만든 provider는 호출별 값이 False여도 끄고 나간다 → 기록도 실제 요청을 따른다.
    mock_llm.reset(lambda body: "ok")
    everything = TracedProvider(OpenAICompatProvider(name="openaiCompatible", model="mock-vlm", base_url=mock_llm.base_url,
                                                     is_local=True, disable_thinking=True), turn)
    cloud = TracedProvider(OpenAICompatProvider(name="openai", model="gpt", api_key="k", base_url=mock_llm.base_url), turn)
    try:
        await everything.analyze([{"role": "user", "content": "q"}], disable_thinking=False)
        await cloud.analyze([{"role": "user", "content": "q"}], disable_thinking=True, max_tokens=4096)
    finally:
        await everything.aclose()
        await cloud.aclose()
    third, fourth = turn.to_document()["events"][2:]
    assert third["data"]["disableThinking"] is True and third["data"]["thinkingControl"] is True
    assert fourth["data"]["disableThinking"] is False and fourth["data"]["thinkingControl"] is False
    assert "maxTokens" not in fourth["data"]                                  # 클라우드에는 상한을 보내지 않는다


def test_non_serializable_data_does_not_break_saving():
    turn = TurnTrace("3f2b9c1e-0000-4000-8000-000000000000")
    turn.note("progress", "bytes", raw=b"\x00\x01", response=ModelResponse(text="t"))
    document = json.loads(turn.to_json())
    assert document["events"][0]["data"]["raw"].startswith("b'")
