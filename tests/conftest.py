from __future__ import annotations

import base64

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
def client(monkeypatch):
    """테스트마다 새 메모리 DB를 쓰는 앱(파일 DB 동작은 test_database_file_survives_restart가 확인한다)."""
    monkeypatch.setenv("DOCCHAT_DB_PATH", ":memory:")
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
