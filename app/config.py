"""모든 임계값과 환경변수를 한 곳에 모은다.

기본값은 참고 구현(vectra-web `document-pipeline/config.mjs`, `pdf-renderer.mjs`)과 동일하다.
CAD PDF처럼 SHX 폰트 때문에 네이티브 문자 수가 낮게 잡히는 문서를 위해 환경변수로 조정할 수 있다.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"


def _int(name: str, default: int, *, low: int | None = None, high: int | None = None) -> int:
    raw = os.environ.get(name)
    try:
        value = int(float(raw)) if raw not in (None, "") else default
    except ValueError:
        value = default
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


# --------------------------------------------------------------------------- 서버
HOST = os.environ.get("DOCCHAT_HOST", "127.0.0.1")
PORT = _int("DOCCHAT_PORT", 8000, low=1, high=65535)


def database_path() -> Path | str:
    """테스트가 환경변수로 바꿔 끼울 수 있도록 호출 시점에 읽는다. ":memory:"면 메모리 DB."""
    configured = os.environ.get("DOCCHAT_DB_PATH")
    if configured == ":memory:":
        return configured
    return Path(configured or PROJECT_ROOT / "data" / "docchat.sqlite")


# --------------------------------------------------------------------------- §5.1 페이지 판별
NATIVE_MIN_CHARS = _int("DOCCHAT_NATIVE_MIN_CHARS", 24, low=0)
SPARSE_OVERLAY_CHARS = _int("DOCCHAT_SPARSE_OVERLAY_CHARS", 120, low=0)

# --------------------------------------------------------------------------- §5.2 렌더링 캡
PDF_RENDER_DPI = _int("DOCCHAT_PDF_RENDER_DPI", 200, low=36, high=600)
MAX_VISION_IMAGE_EDGE = _int("DOCCHAT_MAX_VISION_IMAGE_EDGE", 3072, low=256)
MAX_VISION_IMAGE_PIXELS = _int("DOCCHAT_MAX_VISION_IMAGE_PIXELS", 8_000_000, low=65_536)

DEFAULT_PDF_VISUAL_PAGES = 60
MAX_PDF_VISUAL_PAGES = 200


def pdf_visual_page_limit() -> int:
    """검사/렌더할 최대 페이지 수. 잘못된 값이면 기본값, 상한은 MAX_PDF_VISUAL_PAGES."""
    raw = os.environ.get("DOCCHAT_MAX_PDF_VISUAL_PAGES")
    try:
        parsed = int(float(raw)) if raw not in (None, "") else 0
    except ValueError:
        parsed = 0
    return min(parsed, MAX_PDF_VISUAL_PAGES) if parsed > 0 else DEFAULT_PDF_VISUAL_PAGES


# --------------------------------------------------------------------------- §5.3 OCR 전사
OCR_RETRY_COUNT = _int("DOCCHAT_OCR_RETRY_COUNT", 3, low=1, high=10)
OCR_CONCURRENCY = _int("DOCCHAT_OCR_CONCURRENCY", 2, low=1, high=16)
OCR_CACHE_LIMIT = 256

# --------------------------------------------------------------------------- §6 bbox 도구
# 계획서: "구조화 JSON이 아니면 최대 2회 재시도" → 첫 시도 + 재시도 2회.
GROUNDING_RETRY_COUNT = _int("DOCCHAT_GROUNDING_RETRY_COUNT", 2, low=0, high=5)
MAX_GROUNDING_REGIONS = 200

# --------------------------------------------------------------------------- 에이전트 루프
MAX_TOOL_STEPS = _int("DOCCHAT_MAX_TOOL_STEPS", 8, low=1, high=32)
REPEATED_TOOL_CALL_LIMIT = 3
MAX_CONTINUATIONS = 8

# --------------------------------------------------------------------------- 업로드 / 컨텍스트 예산
MAX_ATTACHMENTS_PER_MESSAGE = 12
MAX_ATTACHMENT_BASE64_CHARS = 90_000_000
MAX_DOCUMENT_TEXT_CHARS = 8_000_000
MAX_MODEL_IMAGES = 12
MAX_HISTORY_MESSAGES = 30

ALLOWED_IMAGE_MIMES = {
    "image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp", "image/tiff",
}
PDF_MIME = "application/pdf"

DEFAULT_LOCAL_CONTEXT_TOKENS = 8192
CLOUD_CHAR_BUDGET = 600_000
CLOUD_ATTACHMENT_TEXT_BUDGET = 180_000

# 로컬 모델은 CPU 추론이 매우 느릴 수 있어 넉넉히, 클라우드는 짧게.
LOCAL_TIMEOUT_SECONDS = _int("DOCCHAT_LOCAL_TIMEOUT", 3600, low=10)
CLOUD_TIMEOUT_SECONDS = _int("DOCCHAT_CLOUD_TIMEOUT", 180, low=10)

LOCAL_PROVIDERS = {"openaiCompatible"}
CLOUD_PROVIDERS = {"openai", "anthropic", "gemini"}
ALL_PROVIDERS = LOCAL_PROVIDERS | CLOUD_PROVIDERS


def estimate_context_char_budget(context_tokens: int, max_characters: int = 200_000) -> int:
    """로컬 서버는 예산을 넘는 프롬프트를 잘라 주지 않고 HTTP 400으로 거절한다.

    토큰당 약 3.3자(산문+코드 혼합 기준 보수적)로 보고, 35%는 시스템 프롬프트와 응답 몫으로 남긴다.
    """
    budget = int((context_tokens or DEFAULT_LOCAL_CONTEXT_TOKENS) * 3.3 * 0.65)
    return max(4_000, min(max_characters, budget))
