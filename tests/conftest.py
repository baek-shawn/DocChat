from __future__ import annotations

import base64
import os

# 개발자의 `.env`가 테스트 결과를 바꾸지 않게 한다. app을 가져오기 전에 꺼야 한다.
os.environ["DOCCHAT_ENV_FILE"] = "off"

import pytest
from fastapi.testclient import TestClient

from mock_openai import MockOpenAIServer


@pytest.fixture(autouse=True)
def _clean_ocr_cache():
    from app.pipeline.ocr import OCR_CACHE
    OCR_CACHE.clear()
    yield
    OCR_CACHE.clear()


@pytest.fixture()
def files_dir(tmp_path, monkeypatch):
    """첨부 파일 폴더. 테스트는 임시 폴더(`--basetemp=.pytest_tmp`)만 쓰고 실제 `data/files`는 건드리지 않는다."""
    folder = tmp_path / "files"
    monkeypatch.setenv("DOCCHAT_FILES_DIR", str(folder))
    return folder


@pytest.fixture()
def client(monkeypatch, files_dir):
    """테스트마다 새 메모리 DB를 쓰는 앱(파일 DB 동작은 test_database_file_survives_restart가 확인한다)."""
    monkeypatch.setenv("DOCCHAT_DB_PATH", ":memory:")
    monkeypatch.delenv("DOCCHAT_DEBUG_TRACE", raising=False)
    from app.main import create_app
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def _mock_server():
    server = MockOpenAIServer().start()
    yield server
    server.stop()


@pytest.fixture()
def mock_llm(_mock_server):
    _mock_server.reset(lambda _body: "mock reply")
    yield _mock_server
    _mock_server.reset(lambda _body: "mock reply")


def upload(name: str, data: bytes, mime: str) -> dict:
    return {"name": name, "mime": mime, "size": len(data), "base64": base64.b64encode(data).decode("ascii")}


def chat_body(mock, text: str, attachments: list[dict] | None = None, **extra) -> dict:
    body = {
        "provider": "openaiCompatible", "baseUrl": mock.base_url, "model": "mock-vlm",
        "messages": [{"role": "user", "content": text}], "attachments": attachments or [],
    }
    body.update(extra)
    return body
