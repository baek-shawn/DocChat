"""`.env` 파일로 설정 넣기 — 셸 환경변수가 우선이고, 테스트는 개발자의 `.env`를 읽지 않는다."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from app import config

ROOT = Path(__file__).resolve().parent.parent


def test_env_file_syntax():
    parsed = config.parse_env_file("\n".join([
        "# 주석", "", "DOCCHAT_TILE_SIZE=1024", "  DOCCHAT_TILE_OVERLAP = 0.25  ", "export DOCCHAT_IMAGE_MODE=tile",
        'DOCCHAT_DB_PATH="D:\\data folder\\docchat.sqlite"', "DOCCHAT_FILES_DIR='files #1'",
        "DOCCHAT_DEBUG_TRACE=1   # 값 뒤의 주석", "DOCCHAT_EMPTY=", "no equals sign", "=no name",
        "DOCCHAT_ONLY_COMMENT=      # 값 없이 주석만(비워 두는 설정 — 추론 수준)",
    ]))
    assert parsed == {
        "DOCCHAT_TILE_SIZE": "1024", "DOCCHAT_TILE_OVERLAP": "0.25", "DOCCHAT_IMAGE_MODE": "tile",
        "DOCCHAT_DB_PATH": "D:\\data folder\\docchat.sqlite", "DOCCHAT_FILES_DIR": "files #1",
        "DOCCHAT_DEBUG_TRACE": "1", "DOCCHAT_EMPTY": "", "DOCCHAT_ONLY_COMMENT": "",
    }


def test_shell_variables_win_over_the_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("\ufeffDOCCHAT_TEST_FROM_FILE=file\nDOCCHAT_TEST_FROM_SHELL=file\n", encoding="utf-8")
    monkeypatch.setenv("DOCCHAT_TEST_FROM_SHELL", "shell")
    monkeypatch.delenv("DOCCHAT_TEST_FROM_FILE", raising=False)
    try:
        assert config.load_env_file(path) == path
        assert os.environ["DOCCHAT_TEST_FROM_FILE"] == "file" and os.environ["DOCCHAT_TEST_FROM_SHELL"] == "shell"
    finally:
        os.environ.pop("DOCCHAT_TEST_FROM_FILE", None)
    assert config.load_env_file(tmp_path / "missing.env") is None


def test_tests_never_read_the_developers_env_file():
    assert os.environ["DOCCHAT_ENV_FILE"] == "off" and config.LOADED_ENV_FILE is None


def run_config(env_file: str, expression: str, **extra: str) -> str:
    """새 프로세스에서 config를 읽는다(값은 모듈을 처음 읽을 때 정해지므로)."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("DOCCHAT_")}
    environment.update({"DOCCHAT_ENV_FILE": env_file, **extra})
    code = f"import sys; sys.path.insert(0, r'{ROOT}'); from app import config; print(({expression}))"
    done = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=True)
    return done.stdout.strip()


SHOWN = ("config.TILE_SIZE, config.TILE_OVERLAP, config.DEFAULT_IMAGE_MODE, config.debug_trace_enabled(), "
         "config.MAX_MODEL_IMAGES")


def test_settings_in_the_file_reach_the_running_config(tmp_path):
    path = tmp_path / "experiment.env"
    path.write_text("DOCCHAT_TILE_SIZE=1024\nDOCCHAT_TILE_OVERLAP=0.25\nDOCCHAT_IMAGE_MODE=tile\n"
                    "DOCCHAT_DEBUG_TRACE=1\nDOCCHAT_MAX_MODEL_IMAGES=24\n", encoding="utf-8")
    assert run_config(str(path), SHOWN) == "(1024, 0.25, 'tile', True, 24)"
    assert run_config(str(path), SHOWN, DOCCHAT_TILE_SIZE="2048") == "(2048, 0.25, 'tile', True, 24)"   # 셸이 우선
    assert run_config("off", SHOWN) == "(1536, 0.125, 'whole', False, 12)"


def example_values() -> dict[str, str]:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    return config.parse_env_file("\n".join(
        line.lstrip("# ") for line in lines if re.match(r"^#\s*DOCCHAT_[A-Z_0-9]+=", line)))


def test_example_file_lists_every_setting_config_reads():
    source = (ROOT / "app" / "config.py").read_text(encoding="utf-8")
    known = set(re.findall(r'"(DOCCHAT_[A-Z_]+)"', source)) - {"DOCCHAT_ENV_FILE"}
    assert len(known) > 30 and known - set(example_values()) == set()


def test_example_file_shows_the_real_defaults():
    example = example_values()
    defaults = {
        "DOCCHAT_TILE_SIZE": config.TILE_SIZE, "DOCCHAT_TILE_OVERLAP": config.TILE_OVERLAP,
        "DOCCHAT_TILE_RENDER_DPI": config.TILE_RENDER_DPI, "DOCCHAT_TILE_MIN_SOURCE_EDGE": config.TILE_MIN_SOURCE_EDGE,
        "DOCCHAT_MAX_TILES_PER_IMAGE": config.MAX_TILES_PER_IMAGE,
        "DOCCHAT_TILE_BLANK_PIXEL_RATIO": config.TILE_BLANK_PIXEL_RATIO,
        "DOCCHAT_TILE_BLANK_TOLERANCE": config.TILE_BLANK_TOLERANCE,
        "DOCCHAT_TILE_DEDUPE_MIN_CHARS": config.TILE_DEDUPE_MIN_CHARS,
        "DOCCHAT_TILE_BOX_MERGE_IOU": config.TILE_BOX_MERGE_IOU,
        "DOCCHAT_TILE_BOX_MERGE_CONTAINMENT": config.TILE_BOX_MERGE_CONTAINMENT,
        "DOCCHAT_NATIVE_MIN_CHARS": config.NATIVE_MIN_CHARS, "DOCCHAT_SPARSE_OVERLAY_CHARS": config.SPARSE_OVERLAY_CHARS,
        "DOCCHAT_PDF_RENDER_DPI": config.PDF_RENDER_DPI, "DOCCHAT_MAX_VISION_IMAGE_EDGE": config.MAX_VISION_IMAGE_EDGE,
        "DOCCHAT_MAX_VISION_IMAGE_PIXELS": config.MAX_VISION_IMAGE_PIXELS,
        "DOCCHAT_MAX_PDF_VISUAL_PAGES": config.DEFAULT_PDF_VISUAL_PAGES,
        "DOCCHAT_OCR_RETRY_COUNT": config.OCR_RETRY_COUNT, "DOCCHAT_OCR_CONCURRENCY": config.OCR_CONCURRENCY,
        "DOCCHAT_OCR_CACHE_LIMIT": config.OCR_CACHE_LIMIT, "DOCCHAT_GROUNDING_RETRY_COUNT": config.GROUNDING_RETRY_COUNT,
        "DOCCHAT_MAX_GROUNDING_REGIONS": config.MAX_GROUNDING_REGIONS, "DOCCHAT_MAX_TOOL_STEPS": config.MAX_TOOL_STEPS,
        "DOCCHAT_REPEATED_TOOL_CALL_LIMIT": config.REPEATED_TOOL_CALL_LIMIT,
        "DOCCHAT_MAX_CONTINUATIONS": config.MAX_CONTINUATIONS, "DOCCHAT_MAX_MODEL_IMAGES": config.MAX_MODEL_IMAGES,
        "DOCCHAT_MAX_HISTORY_MESSAGES": config.MAX_HISTORY_MESSAGES,
        "DOCCHAT_MAX_ATTACHMENTS_PER_MESSAGE": config.MAX_ATTACHMENTS_PER_MESSAGE,
        "DOCCHAT_DEFAULT_CONTEXT_TOKENS": config.DEFAULT_LOCAL_CONTEXT_TOKENS,
        "DOCCHAT_LOCAL_TIMEOUT": config.LOCAL_TIMEOUT_SECONDS, "DOCCHAT_CLOUD_TIMEOUT": config.CLOUD_TIMEOUT_SECONDS,
        "DOCCHAT_PORT": config.PORT,
        "DOCCHAT_VISION_MAX_TOKENS": config.VISION_MAX_TOKENS,
        "DOCCHAT_GROUNDING_DISABLE_THINKING": config.GROUNDING_DISABLE_THINKING,
        "DOCCHAT_OCR_DISABLE_THINKING": config.OCR_DISABLE_THINKING,
        "DOCCHAT_REASONING_BUDGET_ANSWER": config.REASONING_BUDGET_ANSWER,
        "DOCCHAT_REASONING_BUDGET_GROUNDING": config.REASONING_BUDGET_GROUNDING,
        "DOCCHAT_REASONING_BUDGET_OCR": config.REASONING_BUDGET_OCR,
        "DOCCHAT_REASONING_REPEAT_LINES": config.REASONING_REPEAT_LINES,
        "DOCCHAT_REASONING_REPEAT_COUNT": config.REASONING_REPEAT_COUNT,
        "DOCCHAT_REASONING_REPEAT_MIN_CHARS": config.REASONING_REPEAT_MIN_CHARS,
        "DOCCHAT_MAX_VIEWED_PAGES": config.MAX_VIEWED_PAGES,
        "DOCCHAT_DRAWING_MIN_RASTER_AREA": config.DRAWING_MIN_RASTER_AREA,
        "DOCCHAT_DRAWING_MIN_VECTOR_OPERATIONS": config.DRAWING_MIN_VECTOR_OPERATIONS,
        "DOCCHAT_ANALYZE_TOOL": config.ANALYZE_TOOL, "DOCCHAT_ANALYZE_PAGES_PER_CALL": config.ANALYZE_PAGES_PER_CALL,
        "DOCCHAT_MAX_ANALYZED_PAGES": config.MAX_ANALYZED_PAGES,
    }
    assert {name: float(example[name]) for name in defaults} == {name: float(value) for name, value in defaults.items()}
    assert example["DOCCHAT_IMAGE_MODE"] == config.DEFAULT_IMAGE_MODE and example["DOCCHAT_HOST"] == config.HOST
    assert example["DOCCHAT_ANSWER_IMAGE_MODE"] == config.DEFAULT_ANSWER_IMAGE_MODE == "uploads"


def test_runaway_guards_can_be_set_from_the_file(tmp_path):
    """Step 6-0: 호출별 추론 끄기의 기본값과 출력 상한. 출력 상한은 화면이 아니라 여기서만 바꾼다."""
    shown = "config.GROUNDING_DISABLE_THINKING, config.OCR_DISABLE_THINKING, config.VISION_MAX_TOKENS, config.vision_max_tokens()"
    assert run_config("off", shown) == "(True, True, 4096, 4096)"
    path = tmp_path / "thinking.env"
    path.write_text("DOCCHAT_GROUNDING_DISABLE_THINKING=0\nDOCCHAT_OCR_DISABLE_THINKING=off\n"
                    "DOCCHAT_VISION_MAX_TOKENS=16000\n", encoding="utf-8")
    assert run_config(str(path), shown) == "(False, False, 16000, 16000)"
    assert run_config(str(path), shown, DOCCHAT_VISION_MAX_TOKENS="0") == "(False, False, 0, None)"      # 0 = 상한 없음
    # 알아볼 수 없는 값은 기본값으로 둔다(오타 하나로 안전장치가 꺼지지 않게)
    assert run_config("off", shown, DOCCHAT_OCR_DISABLE_THINKING="maybe", DOCCHAT_VISION_MAX_TOKENS="many") \
        == "(True, True, 4096, 4096)"
