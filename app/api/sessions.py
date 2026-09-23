"""세션(대화) CRUD와 첨부 바이트 제공."""
from __future__ import annotations

from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from ..db import ChatStore, valid_id

router = APIRouter(prefix="/api", tags=["sessions"])


class SessionBody(BaseModel):
    id: str | None = None
    title: str | None = None
    provider: str = ""
    model: str = ""
    messages: list[dict[str, Any]] | None = None


class DeleteBody(BaseModel):
    ids: list[str] = Field(default_factory=list)
    all: bool = False


def _store(request: Request) -> ChatStore:
    return request.app.state.store


def _not_found() -> JSONResponse:
    return JSONResponse({"error": "세션을 찾을 수 없습니다."}, status_code=404)


@router.get("/sessions")
async def list_sessions(request: Request, limit: int = 100) -> dict[str, Any]:
    return {"sessions": await _store(request).list_conversations(limit)}


@router.post("/sessions", status_code=201)
async def create_session(request: Request, body: SessionBody) -> Any:
    store = _store(request)
    identifier = await store.save_conversation(
        conversation_id=body.id, title=body.title, provider=body.provider, model=body.model,
        messages=body.messages if body.messages is not None else [])
    return await store.get_conversation(identifier)


@router.delete("/sessions")
async def delete_sessions(request: Request, body: DeleteBody) -> dict[str, int]:
    store = _store(request)
    return {"deleted": await store.delete_all() if body.all else await store.delete_many(body.ids)}


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str) -> Any:
    conversation = await _store(request).get_conversation(session_id)
    return conversation if conversation is not None else _not_found()


@router.put("/sessions/{session_id}")
async def update_session(request: Request, session_id: str, body: SessionBody) -> Any:
    store = _store(request)
    if not valid_id(session_id):
        return _not_found()
    await store.save_conversation(conversation_id=session_id, title=body.title, provider=body.provider,
                                  model=body.model, messages=body.messages)
    return await store.get_conversation(session_id)


@router.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str) -> dict[str, bool]:
    return {"deleted": await _store(request).delete_conversation(session_id)}


@router.get("/attachments/{attachment_id}/content")
async def attachment_content(request: Request, attachment_id: int, download: bool = False) -> Response:
    """뷰어가 이미지를 URL로 불러 쓴다(채팅 응답에 base64를 싣지 않기 위해)."""
    found = await _store(request).get_attachment_content(attachment_id)
    if found is None:
        return JSONResponse({"error": "첨부를 찾을 수 없습니다."}, status_code=404)
    name, mime, data = found
    disposition = "attachment" if download else "inline"
    return Response(content=data, media_type=mime, headers={
        "Content-Disposition": f"{disposition}; filename*=UTF-8''{quote(name)}",
        "Cache-Control": "private, max-age=3600",
        "X-Content-Type-Options": "nosniff",
    })
