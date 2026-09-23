"""Step 1 — 세션 CRUD, 정적 파일, 교차 출처 차단."""
from __future__ import annotations


def _create(client, text="도면 A-1024를 검토해 주세요", **extra):
    response = client.post("/api/sessions", json={"provider": "openaiCompatible", "model": "demo",
                                                  "messages": [{"role": "user", "content": text}], **extra})
    assert response.status_code == 201
    return response.json()


def test_health_reports_reference_thresholds(client):
    limits = client.get("/api/health").json()["limits"]
    assert limits == {
        "nativeMinChars": 24, "sparseOverlayChars": 120, "pdfRenderDpi": 200, "maxVisionImageEdge": 3072,
        "maxVisionImagePixels": 8_000_000, "ocrRetryCount": 3, "ocrConcurrency": 2, "groundingRetryCount": 2,
        "pdfVisualPageLimit": 60,
    }


def test_session_create_list_get_delete(client):
    created = _create(client)
    assert created["title"] == "도면 A-1024를 검토해 주세요"          # 첫 사용자 메시지에서 제목을 만든다
    assert created["messages"][0]["content"] == "도면 A-1024를 검토해 주세요"

    listed = client.get("/api/sessions").json()["sessions"]
    assert [item["id"] for item in listed] == [created["id"]]
    assert listed[0]["messageCount"] == 1

    loaded = client.get(f"/api/sessions/{created['id']}").json()
    assert loaded["provider"] == "openaiCompatible" and loaded["model"] == "demo"
    assert loaded["attachments"] == []

    assert client.delete(f"/api/sessions/{created['id']}").json() == {"deleted": True}
    assert client.get(f"/api/sessions/{created['id']}").status_code == 404
    assert client.get("/api/sessions").json()["sessions"] == []


def test_update_replaces_messages_and_keeps_created_at(client):
    created = _create(client)
    updated = client.put(f"/api/sessions/{created['id']}", json={
        "title": "  이름을   바꾼 세션 ", "provider": "openai", "model": "gpt",
        "messages": [{"role": "user", "content": "첫 질문"},
                     {"role": "assistant", "content": "답변", "artifacts": [
                         {"name": "a.png", "mime": "image/png", "view": "image", "attachmentId": 3,
                          "boxes": [{"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4, "label": "치수", "type": "dimension"},
                                    {"x": "bad"}],
                          "base64": "SHOULD-NOT-BE-STORED"}]}],
    }).json()
    assert updated["title"] == "이름을 바꾼 세션"
    assert updated["createdAt"] == created["createdAt"]
    assert [m["role"] for m in updated["messages"]] == ["user", "assistant"]
    artifact = updated["messages"][1]["artifacts"][0]
    assert artifact["attachmentId"] == 3 and "base64" not in artifact
    assert artifact["boxes"] == [{"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4, "label": "치수", "type": "dimension"}]


def test_sessions_are_ordered_by_last_update(client):
    first, second = _create(client, "첫 번째"), _create(client, "두 번째")
    client.put(f"/api/sessions/{first['id']}", json={"messages": [{"role": "user", "content": "첫 번째"},
                                                                  {"role": "user", "content": "다시"}]})
    assert [item["id"] for item in client.get("/api/sessions").json()["sessions"]] == [first["id"], second["id"]]


def test_database_file_survives_restart(tmp_path, monkeypatch):
    """실제 파일 DB(WAL): 앱을 내렸다 올려도 대화·첨부 바이트가 남고, 세션을 지우면 첨부도 함께 지워진다."""
    from fastapi.testclient import TestClient

    from app.attachments import Attachment
    from app.main import create_app

    monkeypatch.setenv("DOCCHAT_DB_PATH", str(tmp_path / "nested" / "docchat.sqlite"))
    with TestClient(create_app()) as first:
        created = _create(first, "재시작 테스트")
        store = first.app.state.store
        first.portal.call(store.save_attachments, created["id"],
                          [Attachment(name="a.png", mime="image/png", kind="image", size=3, data=b"PNG", text="메모")])
    with TestClient(create_app()) as second:
        loaded = second.get(f"/api/sessions/{created['id']}").json()
        assert loaded["title"] == "재시작 테스트" and loaded["attachments"][0]["name"] == "a.png"
        content = second.get(loaded["attachments"][0]["url"])
        assert content.content == b"PNG" and content.headers["content-type"] == "image/png"
        assert second.delete(f"/api/sessions/{created['id']}").json() == {"deleted": True}
        assert second.get(loaded["attachments"][0]["url"]).status_code == 404      # FK cascade


def test_bulk_and_full_delete(client):
    ids = [_create(client, f"세션 {index}")["id"] for index in range(3)]
    assert client.request("DELETE", "/api/sessions", json={"ids": [ids[0], "nope", ids[0]]}).json() == {"deleted": 1}
    assert client.request("DELETE", "/api/sessions", json={"all": True}).json() == {"deleted": 2}
    assert client.get("/api/sessions").json()["sessions"] == []


def test_invalid_ids_are_rejected_not_crashed(client):
    assert client.get("/api/sessions/short").status_code == 404
    assert client.delete("/api/sessions/short").json() == {"deleted": False}
    assert client.get("/api/attachments/999/content").status_code == 404


def test_cross_origin_api_requests_are_blocked(client):
    assert client.get("/api/sessions", headers={"Origin": "http://evil.example"}).status_code == 403
    assert client.get("/api/sessions", headers={"Origin": "http://testserver"}).status_code == 200


def test_static_frontend_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "text/html" in response.headers["content-type"]
    assert client.get("/js/app.js").status_code == 200
    assert client.get("/styles.css").status_code == 200
