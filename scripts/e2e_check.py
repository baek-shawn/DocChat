"""실제 모델로 전체 흐름을 점검한다(서버를 따로 띄울 필요 없이 앱을 프로세스 안에서 돌린다).

    uv run python scripts/make_samples.py          # 먼저 샘플 생성
    uv run python scripts/e2e_check.py             # 기본: Ollama http://127.0.0.1:11434/v1 의 gemma3:latest

환경변수로 대상을 바꿀 수 있다.
    DOCCHAT_E2E_PROVIDER   openaiCompatible(기본) | openai | anthropic | gemini
    DOCCHAT_E2E_BASE_URL   기본 http://127.0.0.1:11434/v1
    DOCCHAT_E2E_MODEL      기본 gemma3:latest
    DOCCHAT_E2E_API_KEY    클라우드 provider일 때
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")  # Windows 콘솔에서 한글이 깨지지 않게
    except Exception:
        pass

SAMPLES = ROOT / "samples"
CONNECTION = {
    "provider": os.environ.get("DOCCHAT_E2E_PROVIDER", "openaiCompatible"),
    "baseUrl": os.environ.get("DOCCHAT_E2E_BASE_URL", "http://127.0.0.1:11434/v1"),
    "apiKey": os.environ.get("DOCCHAT_E2E_API_KEY", ""),
}
MODEL = os.environ.get("DOCCHAT_E2E_MODEL", "gemma3:latest")
MIMES = {".pdf": "application/pdf", ".png": "image/png"}


def upload(name: str) -> dict:
    path = SAMPLES / name
    data = path.read_bytes()
    return {"name": name, "mime": MIMES[path.suffix], "size": len(data), "base64": base64.b64encode(data).decode("ascii")}


def ask(client, text: str, files: list[str] | None = None, conversation_id: str = "", history: list[dict] | None = None) -> dict:
    body = {**CONNECTION, "model": MODEL, "contextSize": 8192, "conversationId": conversation_id,
            "messages": [*(history or []), {"role": "user", "content": text}],
            "attachments": [upload(name) for name in files or []]}
    started = time.time()
    response = client.post("/api/chat", json=body)
    data = response.json()
    data["_seconds"] = time.time() - started
    data["_status"] = response.status_code
    return data


def show(title: str, data: dict) -> None:
    print(f"\n── {title}  [{data['_status']}] {data['_seconds']:.1f}s")
    if "error" in data:
        print(f"   오류: {data['error']}")
        return
    print("   " + str(data.get("text", "")).strip().replace("\n", "\n   ")[:1200])
    for artifact in data.get("artifacts", []):
        print(f"   ▸ 아티팩트 {artifact['name']}: 영역 {len(artifact.get('boxes', []))}개")
        for box in artifact.get("boxes", [])[:8]:
            print(f"       [{box.get('type')}] {box.get('label')!r} x={box['x']:.3f} y={box['y']:.3f} w={box['w']:.3f} h={box['h']:.3f}")


def main() -> int:
    if not (SAMPLES / "native_spec.pdf").exists():
        print("samples/ 가 없습니다. 먼저 `uv run python scripts/make_samples.py`를 실행하세요.")
        return 2
    os.environ["DOCCHAT_DB_PATH"] = str(Path(tempfile.mkdtemp(prefix="docchat-e2e-")) / "e2e.sqlite")
    from fastapi.testclient import TestClient

    from app.main import create_app

    results: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        results.append((name, passed, detail))
        print(f"   {'✔' if passed else '✘'} {name}{' — ' + detail if detail else ''}")

    print(f"대상: {CONNECTION['provider']} · {CONNECTION['baseUrl']} · {MODEL}")
    with TestClient(create_app()) as client:
        connection = client.post("/api/test-connection", json={**CONNECTION, "model": MODEL}).json()
        print(f"\n── 연결 테스트: {connection['message']}")
        check("모델 서버 연결", connection["ok"])
        if not connection["ok"]:
            return 1

        data = ask(client, "한 문장으로 자기소개를 해 줘.")
        show("1) 일반 대화", data)
        check("일반 대화 응답", data["_status"] == 200 and bool(data.get("text")))

        data = ask(client, "이 문서의 도면 번호(DRAWING NO)와 개정(REVISION)을 알려 줘.", ["native_spec.pdf"])
        show("2) 네이티브 텍스트 PDF (이미지 미전송)", data)
        check("네이티브 PDF에서 도면 번호 추출", "PS-2210-A" in data.get("text", ""), "기대값 PS-2210-A")
        check("네이티브 PDF는 페이지 이미지를 만들지 않음", [a["name"] for a in data.get("attachments", [])] == ["native_spec.pdf"])

        data = ask(client, "이 도면의 DWG NO와 REV를 알려 줘.", ["scanned_drawing.pdf"])
        show("3) 스캔 PDF → 비전 전사 → 텍스트만으로 답변", data)
        names = [a["name"] for a in data.get("attachments", [])]
        evidence = next((a for a in data.get("attachments", []) if a["name"].endswith("visual OCR")), None)
        check("시각 OCR 증거 생성", evidence is not None and evidence["parsedCharacters"] > 0, str(names))
        check("스캔 PDF에서 도면 번호 추출", "FA-7731" in data.get("text", ""), "기대값 FA-7731-B")
        scanned_conversation = data.get("conversationId", "")

        data = ask(client, "부품표에 있는 품목을 표로 정리해 줘.", conversation_id=scanned_conversation,
                   history=[{"role": "user", "content": "이 도면의 DWG NO와 REV를 알려 줘."},
                            {"role": "assistant", "content": data.get("text", "")}])
        show("4) 후속 질문 (첨부 재전송 없이 저장된 증거 사용)", data)
        check("후속 턴에서 저장된 OCR 증거 사용", any(token in data.get("text", "") for token in ("FLANGE", "GASKET", "FL-0420")))

        data = ask(client, "승인 도장(APPROVED)과 서명이 어디 있는지 이미지 위에 표시해 줘.", ["sheet_with_stamp.png"])
        show("5) inspect_visual — 분리된 bbox 호출", data)
        artifacts = data.get("artifacts", [])
        check("inspect_visual 아티팩트 생성", bool(artifacts), "모델이 도구를 호출하지 않으면 실패")
        check("bbox 1개 이상 측정", any(a.get("boxes") for a in artifacts))

        # 6) 호출 트리거 — 계획서: "항상 자동으로 부르는 게 아니라 위치 확인 성격일 때만".
        #    불러야 할 때 부르는지만 보면 과잉 호출을 못 잡는다 → 부르면 안 되는 질문(음성 대조)을 함께 본다.
        #    모델 출력은 확률적이라 단발 합격/불합격 대신 비율로 본다. 기준은 실행 전에 고정해 둔 값이다.
        print("\n── 6) 도구 호출 트리거 (같은 이미지, 질문만 바꿈)")
        location = ["도장이 찍힌 위치를 이미지에 표시해 줘.", "부품표(표)가 어디 있는지 영역을 하이라이트해 줘.",
                    "서명 위치를 박스로 보여 줘."]
        plain = ["이 도면의 DWG NO와 REV가 뭐야?", "부품표에 있는 품목 이름을 나열해 줘.", "이 도면을 누가 그렸고 누가 검토했어?"]
        called = {"location": 0, "plain": 0}
        for kind, questions in (("location", location), ("plain", plain)):
            for question in questions:
                reply = ask(client, question, ["sheet_with_stamp.png"])
                used = bool(reply.get("artifacts"))
                called[kind] += used
                print(f"   [{'위치  ' if kind == 'location' else '비위치'}] 도구 {'호출' if used else '미호출'} {reply['_seconds']:5.1f}s  {question}")
                print(f"            ↳ {str(reply.get('text', reply.get('error', ''))).strip().splitlines()[0][:90] if reply.get('text') or reply.get('error') else ''}")
        check("위치 질문에서 도구 호출 (기준 ≥2/3)", called["location"] >= 2, f"{called['location']}/3")
        check("비위치 질문에서 도구 미호출 (기준 오호출 ≤1/3)", called["plain"] <= 1, f"오호출 {called['plain']}/3")

    failed = [name for name, passed, _ in results if not passed]
    print(f"\n결과: {len(results) - len(failed)}/{len(results)} 통과" + (f" · 실패: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
