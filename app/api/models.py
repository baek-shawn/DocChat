"""모델 목록과 연결 테스트.

API key는 요청 본문(POST)으로만 받는다. 계획서의 `GET /api/models`도 제공하지만,
키를 URL에 실으면 로그·히스토리에 남으므로 GET에서는 키를 `X-Api-Key` 헤더로만 받는다.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..providers import ProviderError, create_provider

router = APIRouter(prefix="/api", tags=["models"])


class ConnectionBody(BaseModel):
    provider: str = "openaiCompatible"
    apiKey: str = ""
    baseUrl: str = ""
    model: str = ""


async def _list_models(provider_name: str, api_key: str, base_url: str) -> list[str]:
    provider = create_provider(provider_name, api_key=api_key, base_url=base_url)
    try:
        return await provider.list_models()
    finally:
        await provider.aclose()


@router.post("/models")
async def post_models(body: ConnectionBody) -> Any:
    try:
        return {"models": await _list_models(body.provider, body.apiKey, body.baseUrl)}
    except ProviderError as error:
        return JSONResponse({"error": str(error)}, status_code=400)


@router.get("/models")
async def get_models(provider: str = "openaiCompatible", baseUrl: str = "",
                     x_api_key: str = Header(default="")) -> Any:
    try:
        return {"models": await _list_models(provider, x_api_key, baseUrl)}
    except ProviderError as error:
        return JSONResponse({"error": str(error)}, status_code=400)


@router.post("/test-connection")
async def test_connection(body: ConnectionBody) -> dict[str, Any]:
    """항상 200으로 {ok, message}를 돌려준다 — 프론트가 성공/실패를 같은 방식으로 표시할 수 있게."""
    try:
        models = await _list_models(body.provider, body.apiKey, body.baseUrl)
    except ProviderError as error:
        return {"ok": False, "message": str(error), "models": []}
    except Exception as error:  # 예상 밖 오류도 연결 실패로 알려 준다
        return {"ok": False, "message": f"연결 확인 중 오류가 발생했습니다: {error}", "models": []}
    if body.model and models and body.model not in models:
        return {"ok": True, "models": models,
                "message": f"연결됨. 다만 '{body.model}'은(는) 이 엔드포인트가 알려 준 모델 {len(models)}개에 없습니다."}
    return {"ok": True, "models": models, "message": f"연결됨. 사용 가능한 모델 {len(models)}개."}
