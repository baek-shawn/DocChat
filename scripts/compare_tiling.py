"""전체(whole) vs 타일(tile) 비교 — 같은 파일·질문을 두 방식으로 실행해 나란히 보여 준다.

    uv run python scripts/make_samples.py
    # 전사(OCR) 비교: 실제 /api/chat 경로로 질문하고, 도면에 적힌 글자가 얼마나 읽혔는지 센다
    uv run python scripts/compare_tiling.py samples/large_scanned_plan.pdf --question "DWG NO와 REV를 알려 줘" --expect-file samples/large_plan.expect.txt
    # bbox 비교: inspect_visual을 같은 작업 문장으로 직접 부르고, 정답 박스와의 IoU를 잰다
    uv run python scripts/compare_tiling.py samples/large_plan.png --task "Find every red circle." --truth samples/large_plan.truth.json
    # 2026-09-28에 손으로 한 합성 이미지 측정(격자 + 빨간 원 4개 + 파란 사각형 음성 대조)의 자동화
    uv run python scripts/compare_tiling.py --synthetic --task "Find every red circle."

무엇을 재는가
  - 전사 글: 글자 수, 기대 문자열 중 읽힌 개수(--expect / --expect-file), 기대 문자열과 무관한 줄 수, 두 방식의 차이(diff)
  - bbox   : 개수, 정답 박스별 IoU·평균·적중(IoU ≥ 0.5), 정답과 맞지 않는 박스, 음성 대조 위의 박스
  - 비용   : 걸린 시간, 모델 호출 수(전사 / bbox / 답변), 타일 수

대상 모델은 e2e_check.py와 같은 환경변수로 고른다(기본: Ollama의 gemma3:latest).
    DOCCHAT_E2E_PROVIDER · DOCCHAT_E2E_BASE_URL · DOCCHAT_E2E_MODEL · DOCCHAT_E2E_API_KEY · DOCCHAT_E2E_CONTEXT
타일 설정은 서버와 같은 환경변수(DOCCHAT_TILE_SIZE 등)를 따른다.

결과 파일(전사 원문, diff, 박스를 그린 이미지, result.json)은 --out 폴더에 남긴다(기본 samples/compare/…).
실제 데이터(data/)는 건드리지 않는다 — 임시 DB와 임시 파일 폴더를 쓴다.
"""
from __future__ import annotations

import argparse
import base64
import difflib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")  # Windows 콘솔에서 한글이 깨지지 않게
    except Exception:
        pass

from app import config as _config  # noqa: E402,F401 — `.env`를 먼저 읽는다(DOCCHAT_E2E_* 도 거기에 둘 수 있다)

MODES = ("whole", "tile")
MODE_LABELS = {"whole": "전체(whole)", "tile": "타일(tile)"}
CONNECTION = {
    "provider": os.environ.get("DOCCHAT_E2E_PROVIDER", "openaiCompatible"),
    "baseUrl": os.environ.get("DOCCHAT_E2E_BASE_URL", "http://127.0.0.1:11434/v1"),
    "apiKey": os.environ.get("DOCCHAT_E2E_API_KEY", ""),
}
MODEL = os.environ.get("DOCCHAT_E2E_MODEL", "gemma3:latest")
CONTEXT = int(os.environ.get("DOCCHAT_E2E_CONTEXT", "8192"))
MIMES = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff"}
HIT_IOU = 0.5                 # 이 이상 겹치면 "찾았다"
DISTRACTOR_IOU = 0.3          # 음성 대조와 이만큼 겹치면 오탐
RELATED_LINE = 0.6            # 기대 문자열과 이만큼 비슷하면 그 글자를 읽으려던 줄로 본다
_MARKER_LINE = re.compile(r"^\[(?:VISUAL SOURCE|CLASSIFICATION|TILED TRANSCRIPTION|TILE r\d+c\d+|UNCLEAR|OCR FAILED|"
                          r"REPEATED LINE OMITTED|TRANSCRIPTION INCOMPLETE)\b.*\]?$")


# --------------------------------------------------------------------------- 계산 (순수 함수)
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def content_lines(text: str) -> list[str]:
    """전사 글에서 표식 줄([TILE r1c2] 등)과 빈 줄을 뺀, 모델이 실제로 받아쓴 줄."""
    return [line.strip() for line in text.splitlines() if line.strip() and not _MARKER_LINE.match(line.strip())]


def expectation_report(text: str, expected: list[str]) -> dict[str, Any]:
    """기대 문자열 중 몇 개가 전사 글에 있는가, 기대 문자열과 무관한 줄은 몇 개인가."""
    haystack = normalize(text)
    found = [item for item in expected if normalize(item) in haystack]
    lines = content_lines(text)
    targets = [normalize(item) for item in expected]
    unrelated, exact = [], 0
    for line in lines:
        key = normalize(line)
        # 글자까지 맞는 줄: 기대 문자열 그대로이거나, 기대 문자열 여러 개가 한 줄에 이어진 것
        if any(key == target or target in key for target in targets):
            exact += 1
        close = any(target in key or key in target or difflib.SequenceMatcher(None, key, target).ratio() >= RELATED_LINE
                    for target in targets)
        if not close:
            unrelated.append(line)
    # unrelated는 "그림 설명·전혀 다른 글"만 잡는다. 비슷하게 생긴 오독이나 잘린 글자를 지어내 채운 줄
    # (예: `DATE: 2023`)은 잡지 못하므로 exact(글자까지 맞는 줄)를 함께 본다.
    return {"expected": len(expected), "found": len(found), "missing": [item for item in expected if item not in found],
            "lines": len(lines), "exact": exact, "unrelated": unrelated}


def iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    """두 박스(x0, y0, x1, y1)의 IoU."""
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - shared
    return shared / union if union > 0 else 0.0


def corners(box: dict[str, Any]) -> tuple[float, float, float, float]:
    return box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]


def load_truth(path: Path) -> dict[str, Any]:
    """정답 파일: {"image": {"width", "height"}, "targets": [{"label", "bbox": [x0,y0,x1,y1]}], "distractors": [...]}

    bbox는 image 크기 기준 픽셀이다. image가 없으면 0~1 분수로 본다.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    size = raw.get("image") or {}
    width, height = float(size.get("width") or 1), float(size.get("height") or 1)

    def scaled(items: Any) -> list[dict[str, Any]]:
        result = []
        for item in items or []:
            x0, y0, x1, y1 = (float(value) for value in item["bbox"])
            result.append({"label": str(item.get("label") or ""), "box": (x0 / width, y0 / height, x1 / width, y1 / height)})
        return result

    return {"targets": scaled(raw.get("targets")), "distractors": scaled(raw.get("distractors"))}


def score_boxes(boxes: list[dict[str, Any]], truth: dict[str, Any]) -> dict[str, Any]:
    """정답 하나에 예측 하나씩, IoU가 큰 짝부터 맺는다."""
    predicted = [corners(box) for box in boxes]
    pairs = sorted(((iou(target["box"], box), t, p) for t, target in enumerate(truth["targets"])
                    for p, box in enumerate(predicted)), reverse=True)
    best: dict[int, tuple[float, int]] = {}
    used: set[int] = set()
    for value, t, p in pairs:
        if value <= 0 or t in best or p in used:
            continue
        best[t] = (value, p)
        used.add(p)
    per_target = [{"label": target["label"], "iou": round(best.get(t, (0.0, -1))[0], 3)}
                  for t, target in enumerate(truth["targets"])]
    hits = {p for value, p in best.values() if value >= HIT_IOU}
    on_distractor = [p for p, box in enumerate(predicted)
                     if any(iou(item["box"], box) >= DISTRACTOR_IOU for item in truth["distractors"])]
    values = [item["iou"] for item in per_target]
    return {
        "targets": per_target,
        "meanIou": round(sum(values) / len(values), 3) if values else 0.0,
        "hits": sum(1 for value in values if value >= HIT_IOU),
        "unmatched": len(predicted) - len(hits),            # 정답을 맞히지 못한 박스(오탐 또는 크게 빗나간 것)
        "onDistractor": len(on_distractor),                 # 음성 대조 위에 그린 박스
    }


# --------------------------------------------------------------------------- 합성 이미지
def synthetic_image(folder: Path) -> tuple[Path, Path]:
    """격자 위 빨간 원 4개(찾을 대상) + 파란 사각형 3개(음성 대조). 2400 x 1600."""
    from PIL import Image, ImageDraw

    width, height = 2400, 1600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 100):
        draw.line((x, 0, x, height), fill=(215, 215, 215), width=1)
    for y in range(0, height, 100):
        draw.line((0, y, width, y), fill=(215, 215, 215), width=1)
    targets, distractors = [], []
    for index, (cx, cy) in enumerate(((350, 300), (1850, 450), (700, 1250), (2050, 1300)), start=1):
        draw.ellipse((cx - 60, cy - 60, cx + 60, cy + 60), fill=(215, 30, 30))
        targets.append({"label": f"red circle {index}", "bbox": [cx - 60, cy - 60, cx + 60, cy + 60]})
    for index, (x, y) in enumerate(((1100, 250), (300, 850), (1500, 1100)), start=1):
        draw.rectangle((x, y, x + 120, y + 120), fill=(30, 70, 215))
        distractors.append({"label": f"blue square {index}", "bbox": [x, y, x + 120, y + 120]})
    image_path, truth_path = folder / "synthetic.png", folder / "synthetic.truth.json"
    image.save(image_path)
    truth_path.write_text(json.dumps({"image": {"width": width, "height": height}, "targets": targets,
                                      "distractors": distractors}, indent=2), encoding="utf-8")
    return image_path, truth_path


# --------------------------------------------------------------------------- 실행
def upload_of(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    mime = MIMES.get(path.suffix.lower())
    if mime is None:
        raise SystemExit(f"PDF와 이미지만 비교할 수 있습니다: {path.name}")
    return {"name": path.name, "mime": mime, "size": len(data), "base64": base64.b64encode(data).decode("ascii")}


def ask(client, path: Path, question: str, mode: str) -> dict[str, Any]:
    """실제 `/api/chat` 경로로 한 턴을 돌린다(전처리 → 전사 → 답변)."""
    body = {**CONNECTION, "model": MODEL, "contextSize": CONTEXT, "imageMode": mode,
            "messages": [{"role": "user", "content": question}], "attachments": [upload_of(path)]}
    started = time.time()
    response = client.post("/api/chat", json=body)
    data = response.json()
    data["seconds"] = time.time() - started
    data["status"] = response.status_code
    if response.status_code == 200:
        store = client.app.state.store
        attachments = client.portal.call(store.list_attachments, data["conversationId"])
        data["ocrText"] = "\n\n".join(item.text for item in attachments if item.name.endswith("visual OCR"))
    return data


def inspect(client, path: Path, task: str, page: int | None, mode: str) -> dict[str, Any]:
    """`inspect_visual`을 같은 작업 문장으로 직접 부른다.

    답변 모델이 도구를 부를지 말지에 따라 비교가 흔들리지 않게 하기 위해서다. 첨부 준비·저장과 도구 실행은
    앱이 쓰는 함수 그대로다.
    """
    from app.agent.tools import ToolContext, execute_tool
    from app.attachments import sanitize_uploads
    from app.pipeline.preprocess import preprocess_attachments
    from app.providers import ToolCall, create_provider

    session = client.post("/api/sessions", json={"messages": [{"role": "user", "content": f"compare: {task}"}]}).json()
    store = client.app.state.store

    async def run() -> dict[str, Any]:
        attachments = await preprocess_attachments(sanitize_uploads([upload_of(path)]))
        await store.save_attachments(session["id"], attachments)
        provider = create_provider(CONNECTION["provider"], model=MODEL, api_key=CONNECTION["apiKey"],
                                   base_url=CONNECTION["baseUrl"], disable_thinking=True)
        context = ToolContext(provider=provider, attachments=attachments, store=store, conversation_id=session["id"],
                              image_mode=mode)
        arguments: dict[str, Any] = {"name": path.name, "task": task}
        if page:
            arguments["page"] = page
        started = time.time()
        try:
            raw = await execute_tool(context, ToolCall("inspect_visual", arguments))
        finally:
            await provider.aclose()
        seconds = time.time() - started
        artifact = context.artifacts[0] if context.artifacts else {}
        image = await store.load_attachment_data(artifact["attachmentId"]) if artifact.get("attachmentId") else None
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = {"error": raw}
        return {"seconds": seconds, "boxes": artifact.get("boxes") or [], "text": parsed.get("text", ""),
                "warning": parsed.get("warning") or parsed.get("error") or "", "usage": context.usage.to_public(),
                "image": image}

    return client.portal.call(run)


def draw_overlay(image: bytes, boxes: list[dict[str, Any]], truth: dict[str, Any] | None, target: Path) -> None:
    """예측 박스(주황)와 정답(초록)·음성 대조(파랑 점선 대신 얇은 선)를 그려 눈으로 확인할 수 있게 한다."""
    from PIL import Image, ImageDraw

    picture = Image.open(io.BytesIO(image)).convert("RGB")
    picture.thumbnail((2400, 2400))
    draw = ImageDraw.Draw(picture)
    width, height = picture.size
    for group, color, line in (((truth or {}).get("targets", []), (20, 160, 60), 3),
                               ((truth or {}).get("distractors", []), (40, 90, 220), 2)):
        for item in group:
            x0, y0, x1, y1 = item["box"]
            draw.rectangle((x0 * width, y0 * height, x1 * width, y1 * height), outline=color, width=line)
    for box in boxes:
        x0, y0, x1, y1 = corners(box)
        draw.rectangle((x0 * width, y0 * height, x1 * width, y1 * height), outline=(240, 130, 20), width=3)
    picture.save(target)


# --------------------------------------------------------------------------- 출력
def row(label: str, values: list[str]) -> None:
    print(f"  {label:<22}" + "".join(f"{value:<38}" for value in values))


def calls_text(usage: dict[str, int]) -> str:
    text = f"전사 {usage.get('ocrCalls', 0)} · bbox {usage.get('groundingCalls', 0)} · 답변 {usage.get('answerCalls', 0)}"
    stops = usage.get("ocrLengthStops", 0) + usage.get("groundingLengthStops", 0)
    return f"{text} · 상한 도달 {stops}" if stops else text


def tiles_text(usage: dict[str, int]) -> str:
    if not usage.get("tiles"):
        return "-"
    return f"{usage['tiles']}장 (빈 타일 {usage.get('blankTiles', 0)}장 제외)"


def main() -> int:
    parser = argparse.ArgumentParser(description="전체 이미지 방식과 타일 방식을 같은 파일·질문으로 비교한다.")
    parser.add_argument("file", nargs="?", help="PDF 또는 이미지 파일")
    parser.add_argument("--question", help="/api/chat으로 보낼 질문(전사 비교). PDF면 생략해도 기본 질문으로 실행한다")
    parser.add_argument("--task", help="inspect_visual에 넘길 작업 문장(bbox 비교)")
    parser.add_argument("--page", type=int, help="PDF에서 bbox를 잴 쪽(1부터)")
    parser.add_argument("--expect", action="append", default=[], help="전사 글에 있어야 하는 문자열(여러 번 지정 가능)")
    parser.add_argument("--expect-file", help="기대 문자열 파일(한 줄에 하나)")
    parser.add_argument("--truth", help="정답 박스 JSON(있으면 IoU를 계산한다)")
    parser.add_argument("--synthetic", action="store_true", help="합성 이미지(빨간 원 4개 + 파란 사각형)를 만들어 비교한다")
    parser.add_argument("--repeat", type=int, default=1, help="방식마다 되풀이할 횟수(기본 1). 1회 결과는 증거가 아니다")
    parser.add_argument("--out", help="결과 폴더(기본 samples/compare/<이름>-<시각>)")
    arguments = parser.parse_args()
    if not arguments.file and not arguments.synthetic:
        parser.error("파일을 지정하거나 --synthetic을 쓰세요.")

    name = "synthetic" if arguments.synthetic else Path(arguments.file).stem
    out = Path(arguments.out) if arguments.out else ROOT / "samples" / "compare" / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    if arguments.synthetic:
        path, truth_path = synthetic_image(out)
        task = arguments.task or "Find every red circle."
    else:
        path = Path(arguments.file)
        truth_path = Path(arguments.truth) if arguments.truth else None
        task = arguments.task
    if not path.exists():
        print(f"파일이 없습니다: {path}")
        return 2
    question = arguments.question
    if not question and not task and path.suffix.lower() == ".pdf":
        question = "이 문서에 적힌 도면 번호, 개정, 제목을 알려 줘."
    if not question and not task:
        parser.error("--question(전사·답변 비교) 또는 --task(bbox 비교) 중 하나는 필요합니다.")
    truth = load_truth(truth_path) if truth_path else None
    expected = list(arguments.expect)
    if arguments.expect_file:
        expected += [line.strip() for line in Path(arguments.expect_file).read_text(encoding="utf-8").splitlines() if line.strip()]

    scratch = Path(tempfile.mkdtemp(prefix="docchat-compare-"))
    os.environ["DOCCHAT_DB_PATH"] = str(scratch / "compare.sqlite")
    os.environ["DOCCHAT_FILES_DIR"] = str(scratch / "files")
    os.environ["DOCCHAT_DEBUG_TRACE"] = "1"          # 모델에 보낸 타일을 남겨 결과 폴더로 옮긴다
    from fastapi.testclient import TestClient

    from app import config
    from app.main import create_app
    from app.pipeline.ocr import OCR_CACHE

    print(f"대상: {CONNECTION['provider']} · {CONNECTION['baseUrl']} · {MODEL}")
    print(f"파일: {path}   타일 설정: {config.tile_settings()}")
    print(f"결과 폴더: {out}")
    runs: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    failed = False
    with TestClient(create_app()) as client:
        connection = client.post("/api/test-connection", json={**CONNECTION, "model": MODEL}).json()
        if not connection["ok"]:
            print(f"모델 서버에 연결할 수 없습니다: {connection['message']}")
            return 1
        for attempt in range(1, max(1, arguments.repeat) + 1):
            for mode in MODES:
                OCR_CACHE.clear()            # 캐시가 남아 있으면 두 번째 실행부터 호출 수·시간을 비교할 수 없다
                result: dict[str, Any] = {"mode": mode, "attempt": attempt}
                if question:
                    print(f"\n[{attempt}] {MODE_LABELS[mode]} · 질문 실행 중…", flush=True)
                    chat = ask(client, path, question, mode)
                    result["chat"] = chat
                    if chat["status"] != 200:
                        failed = True
                        print(f"    오류({chat['status']}): {chat.get('error')}")
                    else:
                        (out / f"ocr_{mode}_{attempt}.txt").write_text(chat.get("ocrText", ""), encoding="utf-8")
                        (out / f"answer_{mode}_{attempt}.txt").write_text(chat.get("text", ""), encoding="utf-8")
                if task:
                    print(f"[{attempt}] {MODE_LABELS[mode]} · bbox 측정 중…", flush=True)
                    measured = inspect(client, path, task, arguments.page, mode)
                    if measured.get("image"):
                        draw_overlay(measured.pop("image"), measured["boxes"], truth, out / f"boxes_{mode}_{attempt}.png")
                    measured.pop("image", None)
                    result["inspect"] = measured
                runs[mode].append(result)
        tiles = scratch / "files"
        if any(tiles.rglob("tiles")):
            for folder in tiles.rglob("tiles"):
                shutil.copytree(folder, out / "tiles", dirs_exist_ok=True)

    # ------------------------------------------------------------------ 나란히 보기
    for attempt in range(max(1, arguments.repeat)):
        pair = [runs[mode][attempt] for mode in MODES]
        print(f"\n══ 실행 {attempt + 1}/{max(1, arguments.repeat)} " + "═" * 70)
        row("", [MODE_LABELS[mode] for mode in MODES])
        if question and all(item["chat"]["status"] == 200 for item in pair):
            chats = [item["chat"] for item in pair]
            print(f"  질문: {question}")
            row("시간", [f"{chat['seconds']:.1f}s" for chat in chats])
            row("모델 호출", [calls_text(chat["meta"]["vision"]) for chat in chats])
            row("타일", [tiles_text(chat["meta"]["vision"]) for chat in chats])
            row("전사 글자 수", [f"{len(chat.get('ocrText', '')):,}자 · {len(content_lines(chat.get('ocrText', '')))}줄" for chat in chats])
            row("전사 실패 표식", [str(chat.get("ocrText", "").count("[OCR FAILED")) for chat in chats])
            if expected:
                reports = [expectation_report(chat.get("ocrText", ""), expected) for chat in chats]
                for item, report in zip(pair, reports):
                    item["expectation"] = report
                row("기대 문자열", [f"{report['found']}/{report['expected']} 읽음" for report in reports])
                row("글자까지 맞는 줄", [f"{report['exact']}/{report['lines']}줄" for report in reports])
                row("무관한 줄", [f"{len(report['unrelated'])}/{report['lines']}줄" for report in reports])
                for mode, report in zip(MODES, reports):
                    if report["missing"]:
                        print(f"    {MODE_LABELS[mode]}에서 못 읽은 것: {', '.join(report['missing'][:12])}"
                              + (f" … 외 {len(report['missing']) - 12}개" if len(report["missing"]) > 12 else ""))
                    if report["unrelated"]:
                        print(f"    {MODE_LABELS[mode]}의 무관한 줄: " + " | ".join(line[:50] for line in report["unrelated"][:6]))
            difference = list(difflib.unified_diff(content_lines(chats[0].get("ocrText", "")),
                                                   content_lines(chats[1].get("ocrText", "")),
                                                   "whole", "tile", lineterm="", n=0))
            (out / f"ocr_diff_{attempt + 1}.txt").write_text("\n".join(difference), encoding="utf-8")
            row("전사 차이(diff)", [f"{sum(1 for line in difference if line.startswith('-') and not line.startswith('---'))}줄은 전체에만",
                                 f"{sum(1 for line in difference if line.startswith('+') and not line.startswith('+++'))}줄은 타일에만"])
            for mode, chat in zip(MODES, chats):
                print(f"    {MODE_LABELS[mode]} 답변: " + " ".join(str(chat.get("text", "")).split())[:300])
                for artifact in chat.get("artifacts", []):
                    print(f"      ▸ 답변 중 도구가 그린 박스 {len(artifact.get('boxes', []))}개 ({artifact['name']})")
        if task:
            inspections = [item["inspect"] for item in pair]
            print(f"  bbox 작업: {task}")
            row("시간", [f"{item['seconds']:.1f}s" for item in inspections])
            row("모델 호출", [calls_text(item["usage"]) for item in inspections])
            row("타일", [tiles_text(item["usage"]) for item in inspections])
            row("박스 개수", [str(len(item["boxes"])) for item in inspections])
            if truth:
                scores = [score_boxes(item["boxes"], truth) for item in inspections]
                for item, score in zip(pair, scores):
                    item["score"] = score
                row(f"적중(IoU≥{HIT_IOU})", [f"{score['hits']}/{len(truth['targets'])} · 평균 IoU {score['meanIou']:.3f}" for score in scores])
                row("정답 아닌 박스", [str(score["unmatched"]) for score in scores])
                row("음성 대조 위 박스", [str(score["onDistractor"]) for score in scores])
                for index, target in enumerate(truth["targets"]):
                    row(f"  {target['label'][:20]}", [f"IoU {score['targets'][index]['iou']:.3f}" for score in scores])
            for mode, item in zip(MODES, inspections):
                if item["warning"]:
                    print(f"    {MODE_LABELS[mode]} 경고: {item['warning']}")
                for box in item["boxes"][:12]:
                    print(f"    {MODE_LABELS[mode]} [{box.get('type')}] {box.get('label', '')[:30]!r} "
                          f"x={box['x']:.3f} y={box['y']:.3f} w={box['w']:.3f} h={box['h']:.3f}")

    summary = {"model": MODEL, "baseUrl": CONNECTION["baseUrl"], "file": str(path), "question": question, "task": task,
               "tiling": config.tile_settings(), "runs": runs}
    for items in runs.values():
        for item in items:
            item.get("chat", {}).pop("attachments", None)
    (out / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"\n전사 원문·diff·박스 그림·모델에 보낸 타일은 {out} 에 있습니다.")
    if max(1, arguments.repeat) == 1:
        print("주의: 1회 실행 결과입니다. 결론을 내리려면 --repeat로 여러 번 돌려 보세요.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
