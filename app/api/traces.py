"""턴 트레이스 조회(Step 7, 개발용).

    GET /api/traces/{trace_id}              트레이스 JSON(진행 중인 턴도 그때까지의 기록을 돌려준다)
    GET /api/traces/{trace_id}?download=1   파일로 내려받기(실행 간 비교·실험 기록용)
    GET /api/sessions/{id}/traces           그 대화의 트레이스 목록(id·시각·상태)
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from .. import trace
from ..db import ChatStore

router = APIRouter(prefix="/api", tags=["traces"])


def _store(request: Request) -> ChatStore:
    return request.app.state.store


@router.get("/traces/{trace_id}")
async def get_trace(request: Request, trace_id: str, download: bool = False) -> Any:
    document = await _store(request).get_trace(trace_id)
    if document is None:
        return JSONResponse({"error": "트레이스를 찾을 수 없습니다. 트레이스가 꺼져 있었거나(DOCCHAT_DEBUG_TRACE) "
                                      "대화가 삭제됐을 수 있습니다."}, status_code=404)
    if document.get("status") == "running":
        # 이 프로세스가 지금 진행 중인 턴인가. 아니면 기록만 남고 끝난 것이다(화면은 갱신을 멈춘다).
        document["live"] = trace.is_live(trace_id)
    if not download:
        return document
    body = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8")
    return Response(content=body, media_type="application/json", headers={
        "Content-Disposition": f'attachment; filename="trace-{trace_id}.json"',
        "Cache-Control": "no-store",
    })


@router.get("/sessions/{session_id}/traces")
async def list_traces(request: Request, session_id: str) -> dict[str, Any]:
    return {"traces": await _store(request).list_traces(session_id)}
