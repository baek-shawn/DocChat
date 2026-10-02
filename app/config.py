"""모든 임계값과 환경변수를 한 곳에 모은다.

기본값은 참고 구현(vectra-web `document-pipeline/config.mjs`, `pdf-renderer.mjs`)과 동일하다.
CAD PDF처럼 SHX 폰트 때문에 네이티브 문자 수가 낮게 잡히는 문서를 위해 환경변수로 조정할 수 있다.

값을 넣는 곳은 두 군데다. 우선순위는 위가 높다.
  1) 프로세스 환경변수 (`$env:DOCCHAT_TILE_SIZE = "1024"`)
  2) 프로젝트 루트의 `.env` 파일 (`DOCCHAT_TILE_SIZE=1024`) — 없으면 건너뛴다. 양식은 `.env.example`
대부분의 값은 이 모듈을 처음 읽을 때(= 서버 시작 시) 정해지므로, 바꾼 뒤에는 서버를 다시 띄워야 한다.
API key는 여기에 두지 않는다 — 요청마다 받아서 쓰고 저장하지 않는다.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
ENV_FILE_DISABLED = ("off", "none", "false", "0")


def parse_env_file(text: str) -> dict[str, str]:
    """`.env` 내용을 {이름: 값}으로 읽는다. `#` 주석, 빈 줄, `export ` 접두사, 따옴표로 감싼 값을 받아 준다."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif value.startswith("#"):
            value = ""                                    # 값 없이 주석만 있는 줄(`NAME=   # 설명`)
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()      # 값 뒤에 붙인 주석
        if name:
            values[name] = value
    return values


def load_env_file(path: Path | str | None = None) -> Path | None:
    """`.env`의 값을 환경변수로 올린다. **이미 설정된 환경변수는 덮어쓰지 않는다**(셸에서 준 값이 우선).

    위치는 `DOCCHAT_ENV_FILE`로 바꿀 수 있고, `off`면 읽지 않는다(테스트가 개발자의 `.env`에 흔들리지 않게).
    읽은 파일의 경로를 돌려준다. 파일이 없거나 껐으면 None.
    """
    configured = os.environ.get("DOCCHAT_ENV_FILE", "")
    if path is None and configured.strip().lower() in ENV_FILE_DISABLED:
        return None
    target = Path(path or configured or PROJECT_ROOT / ".env")
    try:
        text = target.read_text(encoding="utf-8-sig")
    except OSError:
        return None
    for name, value in parse_env_file(text).items():
        os.environ.setdefault(name, value)
    return target


LOADED_ENV_FILE = load_env_file()


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


def _float(name: str, default: float, *, low: float | None = None, high: float | None = None) -> float:
    raw = os.environ.get(name)
    try:
        value = float(raw) if raw not in (None, "") else default
    except ValueError:
        value = default
    if value != value:  # NaN
        value = default
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _flag(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _switch(name: str, default: bool) -> bool:
    """기본값이 있는 켬/끔. 알아볼 수 없는 값이면 기본값."""
    raw = str(os.environ.get(name) or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


# --------------------------------------------------------------------------- 서버
HOST = os.environ.get("DOCCHAT_HOST", "127.0.0.1")
PORT = _int("DOCCHAT_PORT", 8000, low=1, high=65535)


def database_path() -> Path | str:
    """테스트가 환경변수로 바꿔 끼울 수 있도록 호출 시점에 읽는다. ":memory:"면 메모리 DB."""
    configured = os.environ.get("DOCCHAT_DB_PATH")
    if configured == ":memory:":
        return configured
    return Path(configured or PROJECT_ROOT / "data" / "docchat.sqlite")


def files_dir() -> Path | None:
    """첨부 파일 폴더. 기본은 DB 파일 옆의 `files/`(= `data/files`), `DOCCHAT_FILES_DIR`로 바꿀 수 있다.

    DB의 경로 컬럼은 이 폴더 기준 상대 경로라서 DB와 폴더는 짝으로 움직여야 한다.
    메모리 DB인데 폴더를 지정하지 않았으면 None → 저장소가 임시 폴더를 만들어 쓰고 닫을 때 지운다.
    """
    configured = os.environ.get("DOCCHAT_FILES_DIR")
    if configured:
        return Path(configured)
    database = database_path()
    if database == ":memory:":
        return None
    return Path(database).parent / "files"


def debug_trace_enabled() -> bool:
    """개발용 턴 트레이스 스위치(Step 7). 켜면 한 턴의 과정(전처리·전사·모델 호출·도구 실행)을 `turn_traces`에
    기록하고, 타일 모드에서 모델에 보낸 타일을 파일로 남긴다. 꺼져 있으면 아무것도 기록하지 않는다(비용 없음)."""
    return _flag("DOCCHAT_DEBUG_TRACE")


# 트레이스에 남기는 글 한 조각(메시지 내용·응답·도구 결과)의 최대 글자 수. 넘치면 앞·뒤를 남기고 가운데를 잘라 기록한다.
TRACE_TEXT_LIMIT = _int("DOCCHAT_TRACE_TEXT_LIMIT", 40_000, low=500)


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


# --------------------------------------------------------------------------- 이미지 처리 방식 (Step 5 타일링)
# 전체(whole): 쪽/이미지 한 장을 위 한도로 줄여 보낸다 — Step 4까지의 동작 그대로.
# 타일(tile) : 큰 원본을 겹치는 조각으로 나눠 조각마다 따로 보낸다. 전사(OCR)와 bbox 호출에만 적용된다.
# 요청마다 고를 수 있고(`imageMode`), 요청에 없으면 아래 기본값을 쓴다.
IMAGE_MODES = ("whole", "tile")
_configured_mode = str(os.environ.get("DOCCHAT_IMAGE_MODE") or "").strip().lower()
DEFAULT_IMAGE_MODE = _configured_mode if _configured_mode in IMAGE_MODES else "whole"

TILE_SIZE = _int("DOCCHAT_TILE_SIZE", 1536, low=256, high=4096)                 # 타일 한 변의 상한(px)
TILE_OVERLAP = _float("DOCCHAT_TILE_OVERLAP", 0.125, low=0.0, high=0.5)         # 이웃 타일과 겹치는 비율(타일 크기 대비)
TILE_RENDER_DPI = _int("DOCCHAT_TILE_RENDER_DPI", 200, low=36, high=1200)       # PDF 타일 렌더 DPI
TILE_MIN_SOURCE_EDGE = _int("DOCCHAT_TILE_MIN_SOURCE_EDGE", 2048, low=256)      # 원본 긴 변이 이 이하면 전체 모드와 동일
MAX_TILES_PER_IMAGE = _int("DOCCHAT_MAX_TILES_PER_IMAGE", 48, low=1, high=400)  # 넘으면 해상도를 낮춰 타일 수를 맞춘다
# 배경색과 다른 픽셀이 이 비율 이하인 타일은 빈 타일로 보고 모델에 보내지 않는다(1536² 타일에서 약 47픽셀).
TILE_BLANK_PIXEL_RATIO = _float("DOCCHAT_TILE_BLANK_PIXEL_RATIO", 0.00002, low=0.0, high=0.5)
TILE_BLANK_TOLERANCE = _int("DOCCHAT_TILE_BLANK_TOLERANCE", 24, low=0, high=255)   # 배경으로 치는 밝기 차이(0~255)
# 겹침 영역의 중복 줄 제거: 공백을 뺀 글자 수가 이 이상이고 글자(숫자만이 아닌)가 든 줄만 대상이다.
# 짧은 값·숫자만 있는 줄(수량, 치수)은 문서 내용의 정상 반복일 수 있어 절대 지우지 않는다.
TILE_DEDUPE_MIN_CHARS = _int("DOCCHAT_TILE_DEDUPE_MIN_CHARS", 6, low=1)
# 서로 다른 타일에서 나온 같은 종류의 박스가 이만큼 겹치면 같은 대상으로 보고 하나로 합친다.
TILE_BOX_MERGE_IOU = _float("DOCCHAT_TILE_BOX_MERGE_IOU", 0.5, low=0.05, high=1.0)
TILE_BOX_MERGE_CONTAINMENT = _float("DOCCHAT_TILE_BOX_MERGE_CONTAINMENT", 0.8, low=0.05, high=1.0)


def resolve_image_mode(requested: str | None) -> str:
    """요청 값이 비어 있으면 기본값. 모르는 값이면 ValueError(호출부가 사용자 오류로 바꾼다)."""
    value = str(requested or "").strip().lower()
    if not value:
        return DEFAULT_IMAGE_MODE
    if value not in IMAGE_MODES:
        raise ValueError(value)
    return value


# --------------------------------------------------------------------------- 답변(추론) 호출의 이미지 (Step 8)
# 답변 호출에 어떤 이미지를 싣는가. 위의 이미지 처리 방식(전사·bbox 호출의 전체/타일)과는 별개의 축이다.
#   off     : 이미지 없음 — 텍스트(네이티브·전사)와 bbox 도구만
#   uploads : 업로드 이미지는 전체 한 장, PDF 쪽은 싣지 않는다 — 계획서 §5.3·§5.4의 동작(Step 8 이전과 같다)
#   whole   : 업로드 이미지와 PDF의 모든 쪽(네이티브 쪽 포함)을 전체 한 장씩. MAX_MODEL_IMAGES 안에서 쪽 순서로
# 요청마다 고를 수 있고(`answerImageMode`), 요청에 없으면 아래 기본값을 쓴다. 답변 호출은 캐시하지 않으므로 캐시 키는 없다.
ANSWER_IMAGE_MODES = ("off", "uploads", "whole")
_configured_answer_mode = str(os.environ.get("DOCCHAT_ANSWER_IMAGE_MODE") or "").strip().lower()
DEFAULT_ANSWER_IMAGE_MODE = _configured_answer_mode if _configured_answer_mode in ANSWER_IMAGE_MODES else "uploads"


def resolve_answer_image_mode(requested: str | None) -> str:
    """요청 값이 비어 있으면 기본값. 모르는 값이면 ValueError(호출부가 사용자 오류로 바꾼다)."""
    value = str(requested or "").strip().lower()
    if not value:
        return DEFAULT_ANSWER_IMAGE_MODE
    if value not in ANSWER_IMAGE_MODES:
        raise ValueError(value)
    return value


def tile_settings() -> dict[str, float | int]:
    """지금 적용 중인 타일 설정 — 답변 메타데이터, /api/health, 비교 스크립트가 같은 값을 본다."""
    return {
        "tileSize": TILE_SIZE, "overlap": TILE_OVERLAP, "renderDpi": TILE_RENDER_DPI,
        "minSourceEdge": TILE_MIN_SOURCE_EDGE, "maxTiles": MAX_TILES_PER_IMAGE,
    }


def image_mode_variant(mode: str, *, thinking: bool = False, budget: int = 0, effort: str = "") -> str:
    """전사 결과가 어떤 방식·설정으로 만들어졌는지 나타내는 문자열(OCR 캐시 키와 첨부 메타데이터에 쓴다).

    모드나 타일 설정이 바뀌면 값이 달라지므로 이전 결과가 섞여 재사용되지 않는다.
    thinking: 전사 호출의 추론을 끄지 않고 보냈는가(Step 6-0). 추론을 끈 쪽이 기본이라 그때는 아무것도 붙이지 않는다
    — Step 6-0 이전에 전사해 둔 쪽의 기록(`whole`, `tile:…`)과 같은 값이 되어 다시 전사하지 않는다.
    budget: 추론을 켠 전사의 추론 예산(Step 6). 예산을 바꾸면 끊기는 자리가 달라 결과가 달라지므로 키에 넣는다(0이면 안 붙인다).
    effort: 추론을 켠 전사에 실어 보낸 추론 수준(Step 6 2차). 비어 있으면 안 붙인다(1차의 기록과 같은 값).
    """
    suffix = ""
    if thinking:
        suffix = "+thinking" + (f":b{int(budget)}" if budget else "") + (f":e{effort}" if effort else "")
    if mode != "tile":
        return f"whole{suffix}"
    return ":".join(["tile", *(f"{value:g}" for value in (
        TILE_SIZE, TILE_OVERLAP, TILE_RENDER_DPI, TILE_MIN_SOURCE_EDGE, MAX_TILES_PER_IMAGE,
        TILE_BLANK_PIXEL_RATIO, TILE_BLANK_TOLERANCE, TILE_DEDUPE_MIN_CHARS,
    ))]) + suffix


# --------------------------------------------------------------------------- 전사·bbox 호출의 폭주 막기 (Step 6-0)
# 추론형 모델은 타일 하나를 두고 같은 생각을 맴돌다 출력 한도까지 가는 일이 있다(실측: Qwen3.5, 타일 6장 중 2장).
# 전사와 bbox는 "보이는 것을 옮겨 적는" 호출이라 추론을 끄는 쪽이 기본이다. 요청마다 호출 종류별로 고를 수 있다.
# 설정 화면의 "추론 끄기"(모든 호출)가 켜져 있으면 아래 값과 무관하게 모든 호출이 꺼진다.
GROUNDING_DISABLE_THINKING = _switch("DOCCHAT_GROUNDING_DISABLE_THINKING", True)
OCR_DISABLE_THINKING = _switch("DOCCHAT_OCR_DISABLE_THINKING", True)
# 전사·bbox 호출 한 번의 출력 토큰 상한. 추론 토큰도 여기에 포함된다. 0이면 상한을 보내지 않는다.
# 답변 호출에는 적용하지 않는다(답 길이를 미리 알 수 없다).
VISION_MAX_TOKENS = _int("DOCCHAT_VISION_MAX_TOKENS", 4096, low=0, high=1_000_000)


def vision_max_tokens() -> int | None:
    """전사·bbox 호출에 보낼 출력 상한. 상한을 두지 않으면 None."""
    return VISION_MAX_TOKENS if VISION_MAX_TOKENS > 0 else None


def resolve_switch(requested: bool | None, default: bool) -> bool:
    """요청에 값이 없으면(None) 서버 기본값."""
    return default if requested is None else bool(requested)


# --------------------------------------------------------------------------- 추론 제어 (Step 6)
# 추론을 켠 로컬 호출은 스트리밍으로 받으며 추론 토큰을 센다. 예산을 넘거나 같은 줄 묶음이 되풀이되면(반복) 추론을 끊고
# "답만 이어 쓰라"고 다시 보낸다(소프트). 그래도 답이 안 나오면 중단한다(하드). 감지는 추론 부분에만 건다 — 답 부분의
# 반복(표 전사, bbox JSON)은 정상이다. 예산은 호출 종류별로 따로이고 0이면 예산을 두지 않는다(반복 감지는 그대로).
REASONING_BUDGET_ANSWER = _int("DOCCHAT_REASONING_BUDGET_ANSWER", 8000, low=0, high=10_000_000)
REASONING_BUDGET_GROUNDING = _int("DOCCHAT_REASONING_BUDGET_GROUNDING", 4000, low=0, high=10_000_000)
REASONING_BUDGET_OCR = _int("DOCCHAT_REASONING_BUDGET_OCR", 4000, low=0, high=10_000_000)
# 반복 감지: 줄 단위로 최대 REPEAT_LINES줄 묶음이 연달아 REPEAT_COUNT번 같으면 반복이다. 묶음이 REPEAT_MIN_CHARS보다 짧으면
# 반복으로 보지 않는다(`Okay.` 셋은 반복이 아니다). 실측 반복(Qwen3.5): 8줄 묶음 × 40회, 4줄 묶음 순환.
REASONING_REPEAT_LINES = _int("DOCCHAT_REASONING_REPEAT_LINES", 8, low=1, high=64)
REASONING_REPEAT_COUNT = _int("DOCCHAT_REASONING_REPEAT_COUNT", 3, low=2, high=50)
REASONING_REPEAT_MIN_CHARS = _int("DOCCHAT_REASONING_REPEAT_MIN_CHARS", 24, low=1, high=10_000)
# 추론을 끊고 이어 쓰게 한 호출에서 모델이 다시 추론을 시작하면 이만큼만 두고 본다(넘으면 하드 중단).
REASONING_CONTINUATION_ALLOWANCE = 512
REASONING_KINDS = ("answer", "grounding", "ocr")
MAX_REASONING_ACTIONS_IN_META = 60       # 답변 메타데이터에 남기는 "끊긴 호출" 목록의 최대 길이


def reasoning_budget(kind: str) -> int:
    """호출 종류별 추론 예산(토큰). 0이면 예산 없음."""
    return {"answer": REASONING_BUDGET_ANSWER, "grounding": REASONING_BUDGET_GROUNDING,
            "ocr": REASONING_BUDGET_OCR}.get(kind, 0)


def reasoning_settings() -> dict[str, int | dict[str, int]]:
    """지금 적용 중인 추론 제어 설정 — 답변 메타데이터와 /api/health가 같은 값을 본다."""
    return {
        "budget": {kind: reasoning_budget(kind) for kind in REASONING_KINDS},
        "repeatLines": REASONING_REPEAT_LINES, "repeatCount": REASONING_REPEAT_COUNT,
        "repeatMinChars": REASONING_REPEAT_MIN_CHARS,
    }


# --------------------------------------------------------------------------- 추론 수준 (Step 6 2차)
# 모델이 추론의 "정도"를 받는 경우(예: Qwen3.8의 chat_template이 받는 reasoning_effort = low / medium / xhigh)에 실어 보낼 값.
# 호출 종류별로 따로이고 요청마다 고를 수 있다(`reasoningEffortAnswer` 등). 요청에 없으면 아래 값을 쓴다.
# 빈 값이면 보내지 않는다 = 모델의 기본 수준(Qwen3.8은 xhigh — 가장 길게 추론한다).
# 값은 모델마다 달라(Qwen3.8에는 high가 없다) 목록으로 묶지 않고 글자 모양만 검사한다. 서버가 받지 않는 값이면
# 그 턴은 오류로 끝난다(빼고 다시 보내지 않는다 — 요청한 수준과 다른 수준으로 돈 답이 기록에 섞이면 안 된다).
# 추론을 켠 로컬(OpenAI 호환) 호출에만 실린다.
_EFFORT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")


def normalize_reasoning_effort(value: str | None) -> str:
    """추론 수준 값을 다듬는다(앞뒤 공백 제거, 소문자). 빈 값은 그대로 빈 값. 모양이 틀리면 ValueError."""
    text = str(value or "").strip().lower()
    if text and not _EFFORT_PATTERN.match(text):
        raise ValueError(text)
    return text


def _effort(name: str) -> str:
    """환경변수의 추론 수준. 알아볼 수 없는 값이면 빈 값(보내지 않음)."""
    try:
        return normalize_reasoning_effort(os.environ.get(name))
    except ValueError:
        return ""


REASONING_EFFORT_ANSWER = _effort("DOCCHAT_REASONING_EFFORT_ANSWER")
REASONING_EFFORT_GROUNDING = _effort("DOCCHAT_REASONING_EFFORT_GROUNDING")
REASONING_EFFORT_OCR = _effort("DOCCHAT_REASONING_EFFORT_OCR")


def reasoning_effort(kind: str) -> str:
    """호출 종류별 추론 수준의 서버 기본값. 빈 문자열이면 보내지 않는다."""
    return {"answer": REASONING_EFFORT_ANSWER, "grounding": REASONING_EFFORT_GROUNDING,
            "ocr": REASONING_EFFORT_OCR}.get(kind, "")


def resolve_reasoning_effort(requested: str | None, kind: str) -> str:
    """요청에 값이 없으면(None) 서버 기본값, 있으면 그 값(빈 문자열 = 보내지 않음). 모양이 틀리면 ValueError."""
    return reasoning_effort(kind) if requested is None else normalize_reasoning_effort(requested)


def reasoning_effort_settings() -> dict[str, str]:
    """요청에 추론 수준이 없을 때 쓰는 값 — /api/health가 화면에 알려 준다."""
    return {kind: reasoning_effort(kind) for kind in REASONING_KINDS}


# --------------------------------------------------------------------------- §5.3 OCR 전사
OCR_RETRY_COUNT = _int("DOCCHAT_OCR_RETRY_COUNT", 3, low=1, high=10)
# 동시에 나가는 전사 호출 수. 전체 모드에서는 "동시 전사 쪽 수"와 같고, 타일 모드에서는 타일 호출에도 같은 상한을 쓴다.
OCR_CONCURRENCY = _int("DOCCHAT_OCR_CONCURRENCY", 2, low=1, high=16)
OCR_CACHE_LIMIT = _int("DOCCHAT_OCR_CACHE_LIMIT", 256, low=1)

# --------------------------------------------------------------------------- §6 bbox 도구
# 계획서: "구조화 JSON이 아니면 최대 2회 재시도" → 첫 시도 + 재시도 2회.
GROUNDING_RETRY_COUNT = _int("DOCCHAT_GROUNDING_RETRY_COUNT", 2, low=0, high=5)
MAX_GROUNDING_REGIONS = _int("DOCCHAT_MAX_GROUNDING_REGIONS", 200, low=1, high=2000)

# --------------------------------------------------------------------------- 에이전트 루프
MAX_TOOL_STEPS = _int("DOCCHAT_MAX_TOOL_STEPS", 8, low=1, high=32)
REPEATED_TOOL_CALL_LIMIT = _int("DOCCHAT_REPEATED_TOOL_CALL_LIMIT", 3, low=2, high=20)
MAX_CONTINUATIONS = _int("DOCCHAT_MAX_CONTINUATIONS", 8, low=0, high=32)

# --------------------------------------------------------------------------- 업로드 / 컨텍스트 예산
MAX_ATTACHMENTS_PER_MESSAGE = _int("DOCCHAT_MAX_ATTACHMENTS_PER_MESSAGE", 12, low=1, high=100)
MAX_ATTACHMENT_BASE64_CHARS = 90_000_000
MAX_DOCUMENT_TEXT_CHARS = 8_000_000
MAX_MODEL_IMAGES = _int("DOCCHAT_MAX_MODEL_IMAGES", 12, low=1, high=200)          # 답변 호출 한 번에 싣는 이미지 수
MAX_HISTORY_MESSAGES = _int("DOCCHAT_MAX_HISTORY_MESSAGES", 30, low=1, high=500)

ALLOWED_IMAGE_MIMES = {
    "image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp", "image/tiff",
}
PDF_MIME = "application/pdf"

# 요청에 컨텍스트 크기가 없을 때(설정 화면에서 보내면 그 값이 우선)
DEFAULT_LOCAL_CONTEXT_TOKENS = _int("DOCCHAT_DEFAULT_CONTEXT_TOKENS", 8192, low=1024)
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
