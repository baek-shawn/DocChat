"""Step 5-0 — 첨부 바이트를 `data/files/{대화ID}/` 아래 파일로 저장하고 DB에는 상대 경로만 둔다.

모든 테스트는 임시 폴더(`--basetemp=.pytest_tmp`)를 쓴다. 실제 `data/`는 건드리지 않는다.
"""
from __future__ import annotations

import io
import json
import logging
import sqlite3

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.attachments import Attachment
from app.db import ChatStore, storage_names
from app.storage import FileStore, StorageError, safe_filename
from conftest import chat_body, upload
from mock_openai import image_count, is_ocr_call, request_images
from pdf_factory import build_pdf, png_bytes

PDF = "application/pdf"


def query(client, sql: str, *parameters):
    store = client.app.state.store

    async def run():
        cursor = await store.db.execute(sql, parameters)
        found = [dict(row) for row in await cursor.fetchall()]
        await store.db.commit()
        return found

    return client.portal.call(run)


def attachment_rows(client) -> dict[str, dict]:
    found = query(client, "SELECT id, name, file_path, source_path, data IS NULL AS no_blob FROM attachments")
    return {row["name"]: row for row in found}


def tiff_bytes(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (250, 250, 250)).save(buffer, format="TIFF")
    return buffer.getvalue()


# --------------------------------------------------------------------------- 파일로 저장
def test_new_attachments_are_files_and_the_database_keeps_only_relative_paths(client, mock_llm, files_dir):
    mock_llm.reset(lambda body: "PAGE TEXT 0042" if is_ocr_call(body) else "answer")
    pdf, photo = build_pdf("native", "scanned"), png_bytes()
    data = client.post("/api/chat", json=chat_body(mock_llm, "분석해줘", [
        upload("scan.pdf", pdf, PDF), upload("photo.png", photo, "image/png")])).json()
    conversation = data["conversationId"]

    rows = attachment_rows(client)
    assert rows["scan.pdf"]["file_path"] == f"{conversation}/scan.pdf"
    assert rows["scan.pdf · page 2"]["file_path"] == f"{conversation}/scan.pdf.page-0002.png"
    assert rows["photo.png"]["file_path"] == f"{conversation}/photo.png"
    assert rows["scan.pdf · visual OCR"]["file_path"] is None          # 글뿐인 증거 문서에는 파일이 없다
    assert all(row["no_blob"] for row in rows.values())               # BLOB 컬럼에는 더 이상 쓰지 않는다
    assert all("\\" not in (row["file_path"] or "") and ":" not in (row["file_path"] or "") for row in rows.values())

    folder = files_dir / conversation
    assert (folder / "scan.pdf").read_bytes() == pdf                   # PDF 원본
    assert (folder / "photo.png").read_bytes() == photo                # 이미지 원본
    assert (folder / "scan.pdf.page-0002.png").read_bytes().startswith(b"\x89PNG")   # 페이지 렌더

    # 읽기 경로: 프론트는 여전히 URL로만 받는다(base64를 되돌려 보내지 않는다).
    by_name = {item["name"]: item for item in data["attachments"]}
    assert client.get(by_name["scan.pdf"]["url"]).content == pdf
    assert client.get(by_name["photo.png"]["url"]).content == photo
    assert "base64" not in json.dumps(data) and "file_path" not in json.dumps(data)


def test_uploaded_image_keeps_the_original_next_to_the_model_copy(client, mock_llm, files_dir):
    """예전에는 3072px 사본이 원본을 덮어썼다. 이제 원본(타일링 재료)과 모델 전송용 사본을 따로 둔다."""
    mock_llm.reset(lambda body: "큰 도면입니다.")
    original = png_bytes(5000, 2500)
    data = client.post("/api/chat", json=chat_body(mock_llm, "뭐가 보여?", [upload("plan.png", original, "image/png")])).json()
    conversation, item = data["conversationId"], data["attachments"][0]

    row = attachment_rows(client)["plan.png"]
    assert row["file_path"] == f"{conversation}/plan.model.png" and row["source_path"] == f"{conversation}/plan.png"
    assert (files_dir / row["source_path"]).read_bytes() == original               # 원본은 한 바이트도 바뀌지 않는다
    with Image.open(files_dir / row["file_path"]) as copy:
        assert copy.size == (3072, 1536)
    assert (item["width"], item["height"]) == (3072, 1536)

    # 답변 호출과 뷰어에는 사본이 가고, 원본은 저장소에서 따로 꺼낸다.
    (sent,) = request_images(mock_llm.requests[0])
    assert sent.size == (3072, 1536)
    assert client.get(item["url"]).content == (files_dir / row["file_path"]).read_bytes()
    store = client.app.state.store
    assert client.portal.call(store.load_attachment_source, item["id"]) == (original, "image/png")


def test_original_keeps_its_own_format_when_the_model_copy_is_converted(client, mock_llm, files_dir):
    mock_llm.reset(lambda body: "ok")
    original = tiff_bytes(640, 480)                       # 한도 이내지만 TIFF는 모델에 보낼 수 없어 PNG로 바꾼다
    data = client.post("/api/chat", json=chat_body(mock_llm, "봐줘", [upload("scan.tif", original, "image/tiff")])).json()
    row = attachment_rows(client)["scan.tif"]
    conversation = data["conversationId"]
    assert row["source_path"] == f"{conversation}/scan.tif" and row["file_path"] == f"{conversation}/scan.model.png"
    assert (files_dir / row["source_path"]).read_bytes() == original
    store = client.app.state.store
    assert client.portal.call(store.load_attachment_source, row["id"]) == (original, "image/tiff")


def test_small_image_that_is_sent_as_is_is_stored_once(client, mock_llm, files_dir):
    mock_llm.reset(lambda body: "ok")
    original = png_bytes()
    data = client.post("/api/chat", json=chat_body(mock_llm, "봐줘", [upload("photo.png", original, "image/png")])).json()
    row = attachment_rows(client)["photo.png"]
    assert row["source_path"] is None
    assert sorted(path.name for path in (files_dir / data["conversationId"]).iterdir()) == ["photo.png"]
    store = client.app.state.store
    assert client.portal.call(store.load_attachment_source, row["id"]) == (original, "image/png")


def test_reupload_writes_a_new_file_and_never_overwrites_the_old_one(client, mock_llm, files_dir):
    mock_llm.reset(lambda body: "OCR" if is_ocr_call(body) else "answer")
    first_pdf, second_pdf = build_pdf("scanned"), build_pdf("native")
    first = client.post("/api/chat", json=chat_body(mock_llm, "분석", [upload("doc.pdf", first_pdf, PDF)])).json()
    conversation = first["conversationId"]
    again = chat_body(mock_llm, "교체본", [upload("doc.pdf", second_pdf, PDF)], conversationId=conversation)
    client.post("/api/chat", json=again)

    folder = files_dir / conversation
    assert (folder / "doc.pdf").read_bytes() == first_pdf               # 먼저 올린 파일은 그대로 남는다
    assert (folder / "doc (2).pdf").read_bytes() == second_pdf
    rows = attachment_rows(client)
    assert list(rows) == ["doc.pdf"] and rows["doc.pdf"]["file_path"] == f"{conversation}/doc (2).pdf"
    assert client.get(f"/api/attachments/{rows['doc.pdf']['id']}/content").content == second_pdf


def test_files_written_by_a_failed_save_are_rolled_back(client, files_dir):
    session = client.post("/api/sessions", json={"messages": [{"role": "user", "content": "저장 실패 테스트"}]}).json()
    store = client.app.state.store
    good = Attachment(name="a.png", mime="image/png", kind="image", data=png_bytes(), size=1)
    broken = Attachment(name="b.png", mime="image/png", kind=None, data=png_bytes(), size=1)   # kind NOT NULL 위반
    with pytest.raises(sqlite3.IntegrityError):
        client.portal.call(store.save_attachments, session["id"], [good, broken])
    assert attachment_rows(client) == {}
    folder = files_dir / session["id"]
    assert not folder.exists() or list(folder.iterdir()) == []       # 아무도 가리키지 않는 파일을 남기지 않는다


# --------------------------------------------------------------------------- 재시작 후 읽기
def test_files_are_read_back_after_a_restart(tmp_path, monkeypatch, mock_llm):
    from app.main import create_app

    monkeypatch.setenv("DOCCHAT_DB_PATH", str(tmp_path / "data" / "docchat.sqlite"))
    monkeypatch.delenv("DOCCHAT_FILES_DIR", raising=False)        # 기본 위치: DB 파일 옆의 files/
    pdf, original = build_pdf("scanned"), png_bytes(4000, 2000)
    mock_llm.reset(lambda body: "DWG NO B-77" if is_ocr_call(body) else "answer")
    with TestClient(create_app()) as first:
        data = first.post("/api/chat", json=chat_body(mock_llm, "분석", [
            upload("scan.pdf", pdf, PDF), upload("plan.png", original, "image/png")])).json()
    conversation = data["conversationId"]
    assert (tmp_path / "data" / "files" / conversation / "scan.pdf").read_bytes() == pdf

    with TestClient(create_app()) as second:
        loaded = second.get(f"/api/sessions/{conversation}").json()
        by_name = {item["name"]: item for item in loaded["attachments"]}
        assert second.get(by_name["scan.pdf"]["url"]).content == pdf
        assert second.get(by_name["scan.pdf · page 1"]["url"]).content.startswith(b"\x89PNG")
        store = second.app.state.store
        assert second.portal.call(store.load_attachment_source, by_name["plan.png"]["id"]) == (original, "image/png")

        # 후속 턴: 재시작 뒤에도 저장된 이미지가 파일에서 읽혀 모델로 간다(전사는 다시 하지 않는다).
        mock_llm.reset(lambda body: "follow-up")
        follow = chat_body(mock_llm, "수량은?", conversationId=conversation)
        follow["messages"] = [{"role": "user", "content": "분석"}, {"role": "assistant", "content": "answer"},
                              {"role": "user", "content": "수량은?"}]
        assert second.post("/api/chat", json=follow).json()["text"] == "follow-up"
        (request,) = mock_llm.requests
        assert image_count(request) == 1 and request_images(request)[0].size == (3072, 1536)


def test_memory_database_without_a_folder_uses_a_temporary_one(monkeypatch):
    """폴더를 지정하지 않은 메모리 DB: 임시 폴더에 쓰고 닫을 때 함께 지운다(프로젝트 폴더를 더럽히지 않는다)."""
    import asyncio

    async def scenario():
        store = await ChatStore(":memory:").open()
        folder = store.files.root
        identifier = await store.save_conversation(messages=[{"role": "user", "content": "임시"}])
        item = Attachment(name="a.png", mime="image/png", kind="image", data=b"PNG-BYTES", size=9)
        await store.save_attachments(identifier, [item])
        assert (folder / identifier / "a.png").read_bytes() == b"PNG-BYTES"
        assert await store.load_attachment_data(item.id) == b"PNG-BYTES"
        await store.close()
        return folder

    folder = asyncio.run(scenario())
    assert not folder.exists()


# --------------------------------------------------------------------------- 기존 BLOB 폴백
OLD_SCHEMA = """
CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, provider TEXT NOT NULL DEFAULT '',
  model TEXT NOT NULL DEFAULT '', created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL);
CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE, position INTEGER NOT NULL,
  role TEXT NOT NULL, content TEXT NOT NULL, artifacts_json TEXT NOT NULL DEFAULT '[]',
  files_json TEXT NOT NULL DEFAULT '[]', created_at INTEGER NOT NULL, UNIQUE(conversation_id, position));
CREATE TABLE attachments (id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE, position INTEGER NOT NULL,
  name TEXT NOT NULL, mime TEXT NOT NULL, kind TEXT NOT NULL, size INTEGER NOT NULL DEFAULT 0,
  text_content TEXT NOT NULL DEFAULT '', data BLOB, metadata_json TEXT NOT NULL DEFAULT '{}',
  UNIQUE(conversation_id, name));
"""
LEGACY_ID = "legacy-conversation-0001"


def make_legacy_database(path, image: bytes) -> None:
    """Step 5-0 이전 버전이 만든 DB: 경로 컬럼이 없고 바이트가 BLOB에 들어 있다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(OLD_SCHEMA)
    connection.execute("INSERT INTO conversations VALUES (?, '예전 대화', 'openaiCompatible', 'mock-vlm', 1, 1)", (LEGACY_ID,))
    connection.execute("INSERT INTO messages (conversation_id, position, role, content, created_at) "
                       "VALUES (?, 0, 'user', '예전 질문', 1)", (LEGACY_ID,))
    connection.execute(
        "INSERT INTO attachments (conversation_id, position, name, mime, kind, size, text_content, data, metadata_json) "
        "VALUES (?, 0, 'old.png', 'image/png', 'image', ?, 'Original image', ?, ?)",
        (LEGACY_ID, len(image), image, json.dumps({"width": 320, "height": 200, "sendToModel": True})))
    connection.commit()
    connection.close()


def test_legacy_blob_rows_are_read_as_before_and_not_migrated(tmp_path, monkeypatch, mock_llm):
    from app.main import create_app

    database, image = tmp_path / "data" / "docchat.sqlite", png_bytes()
    make_legacy_database(database, image)
    monkeypatch.setenv("DOCCHAT_DB_PATH", str(database))
    monkeypatch.delenv("DOCCHAT_FILES_DIR", raising=False)
    mock_llm.reset(lambda body: "예전 이미지를 봤습니다.")

    with TestClient(create_app()) as client:                      # 열 때 없는 컬럼만 덧붙인다
        loaded = client.get(f"/api/sessions/{LEGACY_ID}").json()
        (item,) = loaded["attachments"]
        assert item["hasData"] and client.get(item["url"]).content == image
        assert loaded["messages"][0]["meta"] == {}
        store = client.app.state.store
        assert client.portal.call(store.load_attachment_source, item["id"]) == (image, "image/png")

        # 예전 첨부로 이어서 대화할 수 있다.
        follow = chat_body(mock_llm, "다시 봐줘", conversationId=LEGACY_ID)
        follow["messages"] = [{"role": "user", "content": "예전 질문"}, {"role": "user", "content": "다시 봐줘"}]
        assert client.post("/api/chat", json=follow).json()["text"] == "예전 이미지를 봤습니다."
        assert request_images(mock_llm.requests[-1])[0].size == (320, 200)

        # 기존 BLOB은 옮기지 않는다: 그대로 BLOB, 경로 없음, 파일도 만들지 않는다.
        row = attachment_rows(client)["old.png"]
        assert not row["no_blob"] and row["file_path"] is None
        assert not (tmp_path / "data" / "files" / LEGACY_ID).exists()

        # 같은 대화에 새로 올린 첨부는 파일로 간다(두 방식이 한 대화 안에 섞여도 된다).
        more = chat_body(mock_llm, "이것도", [upload("new.png", png_bytes(64, 64), "image/png")], conversationId=LEGACY_ID)
        more["messages"] = [{"role": "user", "content": "이것도"}]
        client.post("/api/chat", json=more)
        rows = attachment_rows(client)
        assert rows["new.png"]["file_path"] == f"{LEGACY_ID}/new.png" and rows["new.png"]["no_blob"]
        assert not rows["old.png"]["no_blob"]


# --------------------------------------------------------------------------- 경로 탈출 거부
@pytest.mark.parametrize("path", [
    "../secret.txt", "..\\secret.txt", "conversation-1/../../secret.txt", "conversation-1\\..\\..\\secret.txt",
    "/etc/passwd", "\\Windows\\win.ini", "C:\\Windows\\win.ini", "C:/Windows/win.ini", "c:secret.txt",
    "\\\\server\\share\\file.png", "//server/share/file.png", "conversation-1/./a.png", "conversation-1//a.png",
    "conversation-1/a.png:stream", "conversation-1/a\x00.png", ".", "..", "",
])
def test_paths_pointing_outside_the_files_folder_are_rejected(tmp_path, path):
    with pytest.raises(StorageError):
        FileStore(tmp_path / "files").resolve(path)


def test_paths_inside_the_files_folder_are_accepted(tmp_path):
    store = FileStore(tmp_path / "files")
    assert store.resolve("conversation-1/도면 (2).pdf") == (tmp_path / "files" / "conversation-1" / "도면 (2).pdf").resolve()
    assert store.resolve("conversation-1\\tiles\\r01c01.png").parent.name == "tiles"


def test_writes_cannot_leave_the_files_folder(tmp_path):
    store = FileStore(tmp_path / "files")
    for conversation, name in (("..", "a.png"), ("../outside", "a.png"), ("conversation-1", "../../a.png"),
                               ("conversation-1", "C:\\a.png"), ("conversation-1", "/a.png"), ("", "a.png")):
        with pytest.raises(StorageError):
            store.write(conversation, name, b"x")
    assert not (tmp_path / "a.png").exists() and not (tmp_path / "outside").exists()
    assert store.remove_conversation("..") is False and tmp_path.exists()      # 폴더 밖은 지우지도 않는다


def test_tampered_database_paths_never_leak_files_outside_the_folder(client, files_dir, caplog):
    secret = files_dir.parent / "secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")
    session = client.post("/api/sessions", json={"messages": [{"role": "user", "content": "경로 조작"}]}).json()
    for position, path in enumerate(("../secret.txt", str(secret), f"{session['id']}/../../secret.txt")):
        query(client, "INSERT INTO attachments (conversation_id, position, name, mime, kind, file_path) "
                      "VALUES (?, ?, ?, 'text/plain', 'image', ?)", session["id"], position, f"evil-{position}.png", path)
    store = client.app.state.store
    with caplog.at_level(logging.WARNING, logger="docchat.db"):
        for row in attachment_rows(client).values():
            response = client.get(f"/api/attachments/{row['id']}/content")
            assert response.status_code == 404 and "TOP SECRET" not in response.text
            assert client.portal.call(store.load_attachment_data, row["id"]) is None
            assert client.portal.call(store.load_attachment_source, row["id"]) is None
    assert "거부" in caplog.text
    assert secret.read_text(encoding="utf-8") == "TOP SECRET"


def test_missing_file_is_reported_as_not_found(client, mock_llm, files_dir):
    mock_llm.reset(lambda body: "ok")
    data = client.post("/api/chat", json=chat_body(mock_llm, "봐줘", [upload("photo.png", png_bytes(), "image/png")])).json()
    (files_dir / data["conversationId"] / "photo.png").unlink()        # 사용자가 탐색기에서 지운 경우
    assert client.get(data["attachments"][0]["url"]).status_code == 404


# --------------------------------------------------------------------------- 대화 삭제 → 폴더 삭제
def start_conversation(client, mock_llm, name: str) -> str:
    body = chat_body(mock_llm, f"{name} 분석", [upload(name, png_bytes(), "image/png")])
    return client.post("/api/chat", json=body).json()["conversationId"]


def test_deleting_a_conversation_removes_its_folder_and_nothing_else(client, mock_llm, files_dir):
    mock_llm.reset(lambda body: "ok")
    first, second, third = (start_conversation(client, mock_llm, f"{index}.png") for index in range(3))
    unrelated = files_dir / "keep-me-0000"                       # DB에 없는 폴더는 앱의 것이 아니다
    unrelated.mkdir()
    (unrelated / "note.txt").write_text("사용자 파일", encoding="utf-8")
    assert all((files_dir / identifier).is_dir() for identifier in (first, second, third))

    assert client.delete(f"/api/sessions/{first}").json() == {"deleted": True}
    assert not (files_dir / first).exists() and (files_dir / second).is_dir()

    assert client.request("DELETE", "/api/sessions", json={"ids": [second, "keep-me-0000"]}).json() == {"deleted": 1}
    assert not (files_dir / second).exists() and (unrelated / "note.txt").exists()

    assert client.request("DELETE", "/api/sessions", json={"all": True}).json() == {"deleted": 1}
    assert not (files_dir / third).exists() and (unrelated / "note.txt").exists()


def test_folder_delete_failure_is_logged_and_the_conversation_is_still_deleted(client, mock_llm, files_dir,
                                                                              monkeypatch, caplog):
    mock_llm.reset(lambda body: "ok")
    conversation = start_conversation(client, mock_llm, "locked.png")

    def locked(_path, *_args, **_kwargs):
        raise PermissionError("다른 프로세스가 파일을 사용 중입니다")

    monkeypatch.setattr("app.storage.shutil.rmtree", locked)
    with caplog.at_level(logging.WARNING, logger="docchat.storage"):
        assert client.delete(f"/api/sessions/{conversation}").json() == {"deleted": True}
    assert client.get(f"/api/sessions/{conversation}").status_code == 404      # 대화 삭제는 끝까지 진행된다
    assert "대화 폴더를 지우지 못했습니다" in caplog.text
    assert (files_dir / conversation / "locked.png").exists()


# --------------------------------------------------------------------------- 파일 이름
def test_file_names_are_safe_on_windows_and_stay_readable():
    assert safe_filename("도면 A-1024 (최종).pdf") == "도면 A-1024 (최종).pdf"
    assert safe_filename('a<b>c:d"e|f?g*h.png') == "a_b_c_d_e_f_g_h.png"
    assert safe_filename("CON.pdf") == "_CON.pdf" and safe_filename("nul") == "_nul"
    assert safe_filename("  끝에 점과 공백. . ") == "끝에 점과 공백"
    assert safe_filename("") == "file" and safe_filename("...") == "file"
    long_name = safe_filename("가" * 300 + ".pdf")
    assert len(long_name) <= 96 and long_name.endswith(".pdf")


def test_storage_names_tell_the_original_from_the_model_copy():
    pdf = Attachment(name="scan.pdf", mime="application/pdf", kind="pdf", data=b"%PDF")
    page = Attachment(name="scan.pdf · page 12", mime="image/png", kind="image", data=b"x", page_number=12)
    same = Attachment(name="photo.jpg", mime="image/jpeg", kind="image", data=b"x")
    resized = Attachment(name="plan.tif", mime="image/png", kind="image", data=b"copy", source_data=b"original")
    bare = Attachment(name="clipboard", mime="image/png", kind="image", data=b"x")
    assert storage_names(pdf) == ("scan.pdf", None)
    assert storage_names(page) == ("scan.pdf.page-0012.png", None)
    assert storage_names(same) == ("photo.jpg", None)
    assert storage_names(resized) == ("plan.model.png", "plan.tif")
    assert storage_names(bare) == ("clipboard.png", None)
