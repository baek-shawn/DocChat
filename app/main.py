"""FastAPI 앱 조립."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import config, trace
from .api import chat, files, models, sessions, traces
from .db import ChatStore


def _ignore_client_disconnects(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """사용자가 "중지"를 눌러 연결을 끊으면 Windows의 asyncio가 ConnectionResetError 트레이스백을 찍는다.
    실제 오류가 아니므로 이것만 조용히 넘기고 나머지는 기본 처리에 맡긴다."""
    if isinstance(context.get("exception"), (ConnectionResetError, ConnectionAbortedError)):
        return
    loop.default_exception_handler(context)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.get_running_loop().set_exception_handler(_ignore_client_disconnects)
    app.state.store = await ChatStore(config.database_path(), files_dir=config.files_dir()).open()
    # 지난 프로세스가 진행 중인 채 끝난 턴 트레이스는 살아 있을 수 없다 → "중단됨"으로 정리한다(Step 7).
    interrupted = await app.state.store.interrupt_running_traces(trace.interrupt_document)
    if interrupted:
        print(f"진행 중으로 남아 있던 턴 트레이스 {interrupted}개를 '중단됨'으로 정리했습니다.")
    try:
        yield
    finally:
        await app.state.store.close()


def create_app() -> FastAPI:
    app = FastAPI(title="DocChat", description="로컬/클라우드 VLM 문서 분석 챗", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def reject_cross_origin_api(request: Request, call_next):
        """다른 사이트의 스크립트가 localhost API를 호출하지 못하게 한다(세션 삭제·SSRF 악용 방지)."""
        if request.url.path.startswith("/api/"):
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"error": "교차 출처 API 요청은 허용되지 않습니다."}, status_code=403)
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request: Request, error: StarletteHTTPException):
        return JSONResponse({"error": str(error.detail)}, status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, error: RequestValidationError):
        first = error.errors()[0] if error.errors() else {}
        where = ".".join(str(part) for part in first.get("loc", []) if part != "body")
        return JSONResponse({"error": f"요청 형식이 올바르지 않습니다: {where} {first.get('msg', '')}".strip()},
                            status_code=422)

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return {
            "ok": True,
            "limits": {
                "nativeMinChars": config.NATIVE_MIN_CHARS,
                "sparseOverlayChars": config.SPARSE_OVERLAY_CHARS,
                "pdfRenderDpi": config.PDF_RENDER_DPI,
                "maxVisionImageEdge": config.MAX_VISION_IMAGE_EDGE,
                "maxVisionImagePixels": config.MAX_VISION_IMAGE_PIXELS,
                "ocrRetryCount": config.OCR_RETRY_COUNT,
                "ocrConcurrency": config.OCR_CONCURRENCY,
                "groundingRetryCount": config.GROUNDING_RETRY_COUNT,
                "pdfVisualPageLimit": config.pdf_visual_page_limit(),
            },
            # 요청에 imageMode가 없을 때 쓰는 기본값과 지금 적용 중인 타일 설정(설정 화면 표시용)
            "imageMode": config.DEFAULT_IMAGE_MODE,
            "tiling": config.tile_settings(),
            # 답변 호출의 이미지(Step 8): 요청에 answerImageMode가 없을 때의 기본값과 한 호출에 싣는 이미지 수 상한
            "answerImageMode": config.DEFAULT_ANSWER_IMAGE_MODE,
            "maxModelImages": config.MAX_MODEL_IMAGES,
            # 전사·bbox 호출의 폭주 막기(Step 6-0): 호출별 추론 끄기의 기본값과 출력 상한(0 = 상한 없음)
            "vision": {
                "disableThinkingGrounding": config.GROUNDING_DISABLE_THINKING,
                "disableThinkingOcr": config.OCR_DISABLE_THINKING,
                "maxTokens": config.VISION_MAX_TOKENS,
            },
            # 추론 제어(Step 6): 추론을 켠 로컬 호출의 호출 종류별 추론 예산과 반복 감지 기준(설정 화면 표시용)
            "reasoning": config.reasoning_settings(),
            # 추론 수준(Step 6 2차): 요청에 값이 없을 때 호출 종류별로 실어 보내는 값(빈 문자열 = 보내지 않음)
            "reasoningEffort": config.reasoning_effort_settings(),
            # 개발용 턴 트레이스(Step 7)가 켜져 있는지 — 화면이 "과정 보기"를 안내할 때 쓴다
            "debugTrace": config.debug_trace_enabled(),
        }

    for module in (sessions, models, files, chat, traces):
        app.include_router(module.router)

    # API 라우트를 먼저 등록한 뒤 마지막에 정적 파일을 루트에 건다.
    app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")
    return app


app = create_app()
