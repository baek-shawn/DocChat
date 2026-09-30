"""bbox·전사 호출의 폭주 막기(Step 6-0)를 실제 모델로 확인한다.

추론형 모델은 타일 하나를 두고 같은 생각을 맴돌다 출력 한도까지 가는 일이 있다. 이 스크립트는 같은 이미지를
"추론 끔 / 켬", "출력 상한"을 바꿔 가며 앱의 실제 경로로 돌리고, 호출마다 무슨 일이 있었는지 적는다.

    # bbox: 앱의 도구 경로(execute_tool → inspect_visual)를 같은 작업 문장으로 직접 부른다
    uv run python scripts/check_runaway.py --image plan.png --task "Find every door symbol." --conditions A,B,C
    # 전사: 앱의 전사 경로(build_ocr_reader → prepare_visual_ocr_evidence)
    uv run python scripts/check_runaway.py --pdf samples/large_scanned_plan.pdf --expect-file samples/large_plan.expect.txt --conditions D,E
    # 실제 /api/chat 경로로 한 턴("추론 끄기"(모든 호출)는 해제, 호출별 선택은 서버 기본값)
    uv run python scripts/check_runaway.py --image plan.png --question "창호 심볼을 찾아 표시해 줘" --conditions CHAT

조건
    A  bbox 추론 끔 · 상한 4,096      B  bbox 추론 켬 · 상한 4,096      C  bbox 추론 켬 · 상한 16,000
    D  전사 추론 끔 · 상한 4,096      E  전사 추론 켬 · 상한 4,096
    CHAT  /api/chat 한 턴(타일 모드)
    상한은 --limit으로, 되풀이 횟수는 --repeat으로 바꾼다(조건 뒤에 `A:2`처럼 적어도 된다).

대상 모델은 e2e_check.py와 같은 환경변수로 고른다(DOCCHAT_E2E_BASE_URL · DOCCHAT_E2E_MODEL …).
실제 데이터(data/)는 읽기만 한다. CHAT 조건은 임시 DB와 임시 파일 폴더를 쓴다.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")  # Windows 콘솔에서 한글이 깨지지 않게
    except Exception:
        pass

from app import config  # noqa: E402 — `.env`를 먼저 읽는다

CONNECTION = {
    "provider": os.environ.get("DOCCHAT_E2E_PROVIDER", "openaiCompatible"),
    "baseUrl": os.environ.get("DOCCHAT_E2E_BASE_URL", "http://127.0.0.1:11434/v1"),
    "apiKey": os.environ.get("DOCCHAT_E2E_API_KEY", ""),
}
MODEL = os.environ.get("DOCCHAT_E2E_MODEL", "gemma3:latest")
CONTEXT = int(os.environ.get("DOCCHAT_E2E_CONTEXT", "8192"))
MIMES = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff"}
# (종류, 추론 끔, 출력 상한)
CONDITIONS = {"A": ("bbox", True, 4096), "B": ("bbox", False, 4096), "C": ("bbox", False, 16000),
              "D": ("ocr", True, 4096), "E": ("ocr", False, 4096), "CHAT": ("chat", None, None)}
CHAT_TIMEOUT_SECONDS = 600


def upload_of(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"name": path.name, "mime": MIMES[path.suffix.lower()], "size": len(data),
            "base64": base64.b64encode(data).decode("ascii")}


def watch(provider, calls: list[dict[str, Any]], *, grounding: bool) -> None:
    """provider의 모든 호출을 적는다: 어느 이미지(타일)였는지, 어떻게 끝났는지, 토큰·시간.

    기록은 앱의 동작을 바꾸면 안 된다 — 기록하다 난 예외가 앱으로 넘어가면 앱은 그 호출이 실패한 줄 알고 다시 보낸다
    (2026-09-29에 실제로 겪었다). 그래서 기록 부분의 예외는 여기서 삼키고 항목에 남긴다.
    """
    from app.agent.grounding import parse_visual_inspection

    original = provider.analyze

    async def logged(messages, images=None, tools=None, **options):
        started = time.time()
        image = (images or [None])[0]
        entry: dict[str, Any] = {
            "image": getattr(image, "name", ""), "tile": list(image.tile) if image is not None and image.tile else None,
            "disableThinking": bool(options.get("disable_thinking")), "maxTokens": options.get("max_tokens"),
        }
        calls.append(entry)
        try:
            response = await original(messages, images, tools, **options)
        except Exception as error:
            entry.update(seconds=round(time.time() - started, 1), error=str(error)[:300])
            raise
        try:
            entry.update(seconds=round(time.time() - started, 1), finish=response.finish_reason,
                         promptTokens=response.prompt_tokens, completionTokens=response.completion_tokens,
                         textChars=len(response.text), reasoningChars=len(response.reasoning),
                         text=response.text[:4000], reasoningTail=response.reasoning[-600:])
            if grounding:
                parsed = parse_visual_inspection(response.text, image_width=getattr(image, "width", None),
                                                 image_height=getattr(image, "height", None))
                entry.update(structured=parsed.structured, boxes=len(parsed.boxes))
            print(f"      {entry['image'] or '(이미지 없음)':<46} {entry['finish']:<7} {entry['seconds']:>6.1f}s  출력 "
                  f"{entry['completionTokens']}토큰 · 본문 {entry['textChars']}자 · 추론 {entry['reasoningChars']}자", flush=True)
        except Exception as error:
            entry["logError"] = str(error)[:300]
        return response

    provider.analyze = logged


def new_provider():
    from app.providers import create_provider
    # "추론 끄기"(모든 호출)는 해제한다 → 호출별 선택만 작동한다.
    return create_provider(CONNECTION["provider"], model=MODEL, api_key=CONNECTION["apiKey"],
                           base_url=CONNECTION["baseUrl"], disable_thinking=False)


def calls_per_image(calls: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for call in calls:
        grouped.setdefault(call["image"], []).append(call)
    return grouped


def retried_after_limit(calls: list[dict[str, Any]]) -> list[str]:
    """출력 상한에 닿았는데도 같은 이미지를 또 부른 경우(있으면 안 된다)."""
    return [image for image, items in calls_per_image(calls).items()
            if any(item.get("finish") == "length" for item in items) and len(items) > 1]


async def run_bbox(path: Path, task: str, thinking_off: bool, limit: int) -> dict[str, Any]:
    from app.agent.tools import ToolContext, execute_tool
    from app.attachments import sanitize_uploads
    from app.pipeline.preprocess import preprocess_attachments
    from app.providers import ToolCall

    config.VISION_MAX_TOKENS = limit
    attachments = await preprocess_attachments(sanitize_uploads([upload_of(path)]))
    provider, calls = new_provider(), []
    watch(provider, calls, grounding=True)
    context = ToolContext(provider=provider, attachments=attachments, image_mode="tile", disable_thinking=thinking_off)
    started = time.time()
    try:
        raw = await execute_tool(context, ToolCall("inspect_visual", {"name": path.name, "task": task}))
    finally:
        await provider.aclose()
    seconds = time.time() - started
    try:
        output = json.loads(raw)
    except ValueError:
        output = {"error": raw}
    images = calls_per_image(calls)
    stopped = [image for image, items in images.items() if any(item.get("finish") == "length" for item in items)]
    unstructured = [image for image, items in images.items() if not any(item.get("structured") and item.get("finish") != "length"
                                                                        for item in items)]
    return {
        "seconds": round(seconds, 1), "calls": calls, "usage": context.usage.to_public(),
        "tiles": len(images), "stoppedTiles": stopped, "unstructuredTiles": unstructured,
        "retriedAfterLimit": retried_after_limit(calls), "regions": output.get("regions") or [],
        "warning": output.get("warning") or output.get("error") or "",
        # 상한에 닿지 않고 구조화된 답을 낸 호출이 낸 박스 수의 합. 최종 박스 수는 이보다 많을 수 없다(중복 병합으로 줄 수는 있다).
        "boxesFromFinishedCalls": sum(item.get("boxes", 0) for item in calls
                                      if item.get("finish") != "length" and item.get("structured")),
    }


async def run_ocr(path: Path, thinking_off: bool, limit: int, expected: list[str]) -> dict[str, Any]:
    from app.attachments import sanitize_uploads
    from app.pipeline.images import TileSource, VisionUsage
    from app.pipeline.ocr import OcrCache, build_ocr_reader, prepare_visual_ocr_evidence
    from app.pipeline.preprocess import preprocess_attachments
    from compare_tiling import expectation_report

    config.VISION_MAX_TOKENS = limit
    attachments = await preprocess_attachments(sanitize_uploads([upload_of(path)]))
    source = path.read_bytes()
    provider, calls, usage = new_provider(), [], VisionUsage()
    watch(provider, calls, grounding=False)

    async def load(page):
        return TileSource(kind="pdf", data=source, page_number=page.page_number or 1)

    reader = build_ocr_reader(provider, image_mode="tile", load_tile_source=load, usage=usage, disable_thinking=thinking_off)
    started = time.time()
    try:
        output, _ = await prepare_visual_ocr_evidence(
            attachments, read_image=reader, cache=OcrCache(), cache_namespace=provider.cache_namespace,
            image_mode="tile", thinking=not thinking_off)
    finally:
        await provider.aclose()
    text = "\n\n".join(item.text for item in output if item.name.endswith("visual OCR"))
    result = {
        "seconds": round(time.time() - started, 1), "calls": calls, "usage": usage.to_public(),
        "tiles": len(calls_per_image(calls)), "retriedAfterLimit": retried_after_limit(calls),
        "stoppedTiles": [image for image, items in calls_per_image(calls).items()
                         if any(item.get("finish") == "length" for item in items)],
        "failedMarks": text.count("[OCR FAILED"), "incompleteMarks": text.count("[TRANSCRIPTION INCOMPLETE"),
        "text": text,
    }
    if expected:
        result["expectation"] = expectation_report(text, expected)
    return result


def run_chat(path: Path, question: str) -> dict[str, Any]:
    """실제 `/api/chat` 경로. 답변 호출은 추론을 켜고 출력 상한 없이 나간다(그 폭주는 Step 6의 범위)."""
    scratch = Path(tempfile.mkdtemp(prefix="docchat-runaway-"))
    os.environ["DOCCHAT_DB_PATH"] = str(scratch / "check.sqlite")
    os.environ["DOCCHAT_FILES_DIR"] = str(scratch / "files")
    from fastapi.testclient import TestClient

    from app.main import create_app

    body = {**CONNECTION, "model": MODEL, "contextSize": CONTEXT, "imageMode": "tile", "disableThinking": False,
            "messages": [{"role": "user", "content": question}], "attachments": [upload_of(path)]}

    def post() -> dict[str, Any]:
        with TestClient(create_app()) as client:
            started = time.time()
            response = client.post("/api/chat", json=body)
            data = response.json()
            data.update(seconds=round(time.time() - started, 1), status=response.status_code)
            data.pop("attachments", None)
            return data

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(post)
    try:
        return future.result(timeout=CHAT_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        return {"timedOut": True, "seconds": CHAT_TIMEOUT_SECONDS}


def main() -> int:
    parser = argparse.ArgumentParser(description="bbox·전사 호출의 폭주 막기를 실제 모델로 확인한다.")
    parser.add_argument("--image", help="bbox·CHAT 조건에 쓸 이미지")
    parser.add_argument("--task", help="inspect_visual에 넘길 작업 문장(조건 A·B·C)")
    parser.add_argument("--question", help="/api/chat에 보낼 질문(조건 CHAT)")
    parser.add_argument("--pdf", help="전사 조건(D·E)에 쓸 스캔 PDF")
    parser.add_argument("--expect-file", help="전사 글에 있어야 하는 문자열(한 줄에 하나)")
    parser.add_argument("--conditions", default="A,B,C", help="쉼표로 구분. 되풀이 횟수는 A:2처럼 적는다")
    parser.add_argument("--repeat", type=int, default=1, help="횟수를 적지 않은 조건의 되풀이 횟수")
    parser.add_argument("--limit", type=int, help="조건의 출력 상한을 이 값으로 바꾼다(0 = 상한 없음)")
    parser.add_argument("--out", help="결과 폴더(기본 samples/compare/runaway-<시각>)")
    arguments = parser.parse_args()

    plan: list[tuple[str, int]] = []
    for item in arguments.conditions.split(","):
        name, _, count = item.strip().partition(":")
        if name.upper() not in CONDITIONS:
            parser.error(f"모르는 조건입니다: {name}")
        plan.append((name.upper(), int(count) if count else max(1, arguments.repeat)))
    out = Path(arguments.out) if arguments.out else ROOT / "samples" / "compare" / f"runaway-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    expected = []
    if arguments.expect_file:
        expected = [line.strip() for line in Path(arguments.expect_file).read_text(encoding="utf-8").splitlines() if line.strip()]

    print(f"대상: {CONNECTION['provider']} · {CONNECTION['baseUrl']} · {MODEL}")
    print(f"결과 폴더: {out}")
    results: list[dict[str, Any]] = []
    for name, count in plan:
        kind, thinking_off, limit = CONDITIONS[name]
        if limit is not None and arguments.limit is not None:
            limit = arguments.limit
        for attempt in range(1, count + 1):
            label = f"{name} #{attempt}"
            if kind == "chat":
                if not arguments.image or not arguments.question:
                    parser.error("CHAT 조건에는 --image와 --question이 필요합니다.")
                print(f"\n── {label}: /api/chat · 타일 모드 · 추론 끄기(모든 호출) 해제 · 호출별 선택은 서버 기본값", flush=True)
                result = run_chat(Path(arguments.image), arguments.question)
            elif kind == "bbox":
                if not arguments.image or not arguments.task:
                    parser.error("조건 A·B·C에는 --image와 --task가 필요합니다.")
                print(f"\n── {label}: bbox · 추론 {'끔' if thinking_off else '켬'} · 상한 {limit or '없음'}", flush=True)
                result = asyncio.run(run_bbox(Path(arguments.image), arguments.task, thinking_off, limit))
            else:
                if not arguments.pdf:
                    parser.error("조건 D·E에는 --pdf가 필요합니다.")
                print(f"\n── {label}: 전사 · 추론 {'끔' if thinking_off else '켬'} · 상한 {limit or '없음'}", flush=True)
                result = asyncio.run(run_ocr(Path(arguments.pdf), thinking_off, limit, expected))
            result.update(condition=name, attempt=attempt, thinkingDisabled=thinking_off, limit=limit)
            results.append(result)
            summarize(label, kind, result)
            (out / "result.json").write_text(json.dumps({
                "model": MODEL, "baseUrl": CONNECTION["baseUrl"], "image": arguments.image, "pdf": arguments.pdf,
                "task": arguments.task, "question": arguments.question, "results": results,
            }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            if kind == "ocr":
                (out / f"ocr_{name}_{attempt}.txt").write_text(result["text"], encoding="utf-8")
            if result.get("timedOut"):
                print("   시간 안에 끝나지 않아 여기서 멈춥니다.")
                os._exit(1)        # 모델 호출이 아직 돌고 있다 — 기다리지 않고 끝낸다
    print(f"\n호출별 기록은 {out / 'result.json'} 에 있습니다.")
    return 0


def summarize(label: str, kind: str, result: dict[str, Any]) -> None:
    if kind == "chat":
        if result.get("timedOut"):
            print(f"   {label}: {CHAT_TIMEOUT_SECONDS}초 안에 끝나지 않음")
            return
        meta = result.get("meta") or {}
        print(f"   {label}: HTTP {result.get('status')} · {result.get('seconds')}초 · 추론 끔 기록 {meta.get('thinkingDisabled')}"
              f" · 출력 상한 {meta.get('visionMaxTokens')}")
        print(f"   비전 호출: {meta.get('vision')}")
        for artifact in result.get("artifacts") or []:
            print(f"   ▸ {artifact.get('name')}: 박스 {len(artifact.get('boxes') or [])}개 · 작업 문장 {artifact.get('task')!r}")
        print("   답변: " + " ".join(str(result.get("text") or result.get("error") or "").split())[:400])
        return
    usage = result["usage"]
    stops = usage["groundingLengthStops"] + usage["ocrLengthStops"]
    print(f"   {label}: {result['seconds']}초 · 타일 {result['tiles']}장 · 호출 {len(result['calls'])}회 · 상한 도달 {stops}회"
          f" ({', '.join(result['stoppedTiles']) or '없음'})")
    print(f"   상한에 닿은 뒤 다시 보낸 타일: {result['retriedAfterLimit'] or '없음'}")
    if kind == "bbox":
        print(f"   구조화 실패 타일 {len(result['unstructuredTiles'])}/{result['tiles']} · 박스 {len(result['regions'])}개"
              f" (끝까지 답한 호출이 낸 박스 합계 {result['boxesFromFinishedCalls']}개)")
        if result["warning"]:
            print(f"   경고: {result['warning']}")
    else:
        print(f"   전사 {len(result['text']):,}자 · 실패 표식 {result['failedMarks']} · 끊김 표식 {result['incompleteMarks']}")
        report = result.get("expectation")
        if report:
            print(f"   기대 문자열 {report['found']}/{report['expected']} · 무관한 줄 {len(report['unrelated'])}/{report['lines']}"
                  f" · 글자까지 맞는 줄 {report['exact']}/{report['lines']}")
            for line in report["unrelated"][:8]:
                print(f"     무관한 줄: {line[:110]}")


if __name__ == "__main__":
    sys.exit(main())
