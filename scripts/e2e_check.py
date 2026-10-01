"""실제 모델로 전체 흐름을 점검한다(서버를 따로 띄울 필요 없이 앱을 프로세스 안에서 돌린다).

    uv run python scripts/make_samples.py          # 먼저 샘플 생성
    uv run python scripts/e2e_check.py             # 기본: Ollama http://127.0.0.1:11434/v1 의 gemma3:latest

턴 트레이스(Step 7)를 켠 채 돌린다(트레이스 항목 T1~T4 포함, STEPS.md "Step 7 실모델 확인 기준").
환경변수로 대상을 바꿀 수 있다.
    DOCCHAT_E2E_PROVIDER   openaiCompatible(기본) | openai | anthropic | gemini
    DOCCHAT_E2E_BASE_URL   기본 http://127.0.0.1:11434/v1
    DOCCHAT_E2E_MODEL      기본 gemma3:latest
    DOCCHAT_E2E_API_KEY    클라우드 provider일 때
"""
from __future__ import annotations

import base64
import json
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

from app import config as _config  # noqa: E402,F401 — `.env`를 먼저 읽는다(DOCCHAT_E2E_* 도 거기에 둘 수 있다)

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


def ask(client, text: str, files: list[str] | None = None, conversation_id: str = "", history: list[dict] | None = None,
        mode: str = "", answer_images: str = "") -> dict:
    body = {**CONNECTION, "model": MODEL, "contextSize": 8192, "conversationId": conversation_id, "imageMode": mode,
            "answerImageMode": answer_images,
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
    scratch = Path(tempfile.mkdtemp(prefix="docchat-e2e-"))
    # `.env`에 실제 위치가 적혀 있어도 점검은 임시 폴더에서만 한다(사용자 데이터를 건드리지 않는다).
    os.environ["DOCCHAT_DB_PATH"] = str(scratch / "e2e.sqlite")
    os.environ["DOCCHAT_FILES_DIR"] = str(scratch / "files")
    # 턴 트레이스(Step 7)를 켠 채 돌린다 — 트레이스가 앱의 동작을 바꾸지 않는지(기존 항목이 그대로 통과하는지)와
    # 실제 모델의 턴이 빠짐없이 기록되는지를 함께 본다.
    os.environ["DOCCHAT_DEBUG_TRACE"] = "1"
    from fastapi.testclient import TestClient

    from app.main import create_app

    results: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        results.append((name, passed, detail))
        print(f"   {'✔' if passed else '✘'} {name}{' — ' + detail if detail else ''}")

    def trace_of(client, data: dict) -> dict | None:
        trace_id = (data.get("meta") or {}).get("traceId")
        response = client.get(f"/api/traces/{trace_id}") if trace_id else None
        return response.json() if response is not None and response.status_code == 200 else None

    def model_events(document: dict, kind: str) -> list[dict]:
        return [event for event in document.get("events", []) if event["kind"] == "model" and event["data"].get("kind") == kind]

    def trace_is_clean(document: dict) -> bool:
        raw = json.dumps(document, ensure_ascii=False)
        starts = [event["startedMs"] for event in document.get("events", [])]
        return "base64," not in raw and (not CONNECTION["apiKey"] or CONNECTION["apiKey"] not in raw) and starts == sorted(starts)

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

        # T1) 턴 트레이스(Step 7): 전처리 판별 → 전사(자식으로 모델 호출) → 답변 호출이 빠짐없이 기록된다.
        document = trace_of(client, data)
        if document is None:
            check("트레이스 T1: 스캔 PDF 턴이 기록됨", False, "meta.traceId 없음 또는 조회 실패")
        else:
            pages = [page for event in document["events"] if event["kind"] == "preprocess"
                     for page in event["data"].get("pages", [])]
            ocr_scopes = [event for event in document["events"] if event["kind"] == "ocr" and event["label"].startswith("전사 · ")]
            ocr_calls = model_events(document, "ocr")
            nested = all(call.get("parent") in {scope["id"] for scope in ocr_scopes} for call in ocr_calls)
            usage = all(call["data"].get("finishReason") and call["data"].get("completionTokens") is not None for call in ocr_calls)
            check("트레이스 T1: 스캔 PDF 턴이 기록됨",
                  document["status"] == "done" and any(page["needsVlm"] and page["classification"] == "scanned-raster" for page in pages)
                  and ocr_calls and nested and usage and model_events(document, "answer"),
                  f"상태 {document['status']} · 전사 호출 {len(ocr_calls)}회(자식 {nested}, 토큰 {usage}) · 답변 호출 {len(model_events(document, 'answer'))}회")
            check("트레이스 T4: base64·API key 없음, 시간순", trace_is_clean(document))

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
        # T2) 도구 이벤트 아래에 위치 확인 호출이 자식으로 있고, 도구 인자·결과가 기록된다(도구를 불렀을 때만 판정).
        document = trace_of(client, data)
        if artifacts and document is not None:
            tools = [event for event in document["events"] if event["kind"] == "tool" and event["data"].get("name") == "inspect_visual"
                     and "arguments" in event["data"]]
            grounding = model_events(document, "grounding")
            check("트레이스 T2: inspect_visual 도구 아래에 위치 확인 호출",
                  tools and grounding and all(call.get("parent") in {tool["id"] for tool in tools} for call in grounding)
                  and all("result" in tool["data"] for tool in tools),
                  f"도구 {len(tools)}건 · 위치 확인 호출 {len(grounding)}회 · 작업 문장 {tools[0]['data']['arguments'].get('task')!r}" if tools else "도구 이벤트 없음")
        elif artifacts:
            check("트레이스 T2: inspect_visual 도구 아래에 위치 확인 호출", False, "트레이스 조회 실패")

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

        # 7~8) 타일 모드(Step 5) — A1 크기 스캔 도면(6622 x 4677px, 작은 글자가 넓게 흩어져 있다).
        #    타일 전용 안내문(TILE_OCR_NOTE, TILE_GROUNDING_NOTE)은 후보 1개씩이고, 아래 기준은 결과를 보기 전에 고정했다.
        #    기준을 못 맞추면 문구를 바꿔 가며 다시 재지 않고 한계로 기록한다(CLAUDE.md 프롬프트 규율).
        if not (SAMPLES / "large_scanned_plan.pdf").exists():
            print("\n── 7~8) 타일 모드: samples/large_scanned_plan.pdf 가 없어 건너뜁니다(make_samples.py를 다시 실행하세요).")
            check("타일 모드 샘플 있음", False, "scripts/make_samples.py 재실행 필요")
        else:
            data = ask(client, "이 도면의 DWG NO와 REV를 알려 줘.", ["large_scanned_plan.pdf"], mode="tile")
            show("7) 타일 모드 전사 → 타일별 전사를 이어 붙인 글로 답변", data)
            vision = data.get("meta", {}).get("vision", {})
            attachments = client.portal.call(client.app.state.store.list_attachments, data.get("conversationId", ""))
            evidence = "\n".join(item.text for item in attachments if item.name.endswith("visual OCR"))
            print("   ▸ " + (evidence.splitlines()[2] if len(evidence.splitlines()) > 2 else "(전사 없음)")[:200])
            check("타일로 나눠 전사", data.get("meta", {}).get("imageMode") == "tile" and vision.get("tiles", 0) > 1,
                  f"타일 {vision.get('tiles')}장 · 전사 호출 {vision.get('ocrCalls')}회")
            check("타일 전사에 도면 번호가 있음", "AR-2044-C" in evidence, "기대값 AR-2044-C")
            unread = evidence.count("[OCR FAILED")
            check("전사 실패 타일 (기준 ≤ 20%)", vision.get("tiles", 0) > 0 and unread <= vision["tiles"] * 0.2,
                  f"{unread}/{vision.get('tiles')}장")
            check("타일 모드 답변에 도면 번호", "AR-2044-C" in data.get("text", ""), "기대값 AR-2044-C")
            # T3) 타일마다 전사 호출이 하나씩 기록되고 타일 위치가 적힌다.
            document = trace_of(client, data)
            if document is None:
                check("트레이스 T3: 타일 전사 호출이 타일 수만큼 기록됨", False, "트레이스 조회 실패")
            else:
                calls = model_events(document, "ocr")
                check("트레이스 T3: 타일 전사 호출이 타일 수만큼 기록됨",
                      len(calls) == vision.get("ocrCalls") and all(call["data"].get("tile") for call in calls),
                      f"기록 {len(calls)}회 vs ocrCalls {vision.get('ocrCalls')} · 타일 표시 {sum(1 for call in calls if call['data'].get('tile'))}회")

            data = ask(client, "소화기 표시(빨간 원)가 어디 있는지 이미지 위에 표시해 줘.", ["large_plan.png"], mode="tile")
            show("8) 타일 모드 inspect_visual — 원본에서 자른 타일마다 분리된 bbox 호출", data)
            vision = data.get("meta", {}).get("vision", {})
            artifacts = data.get("artifacts", [])
            check("타일 모드 inspect_visual 아티팩트 생성", bool(artifacts),
                  f"타일 {vision.get('tiles')}장 · bbox 호출 {vision.get('groundingCalls')}회")
            check("타일 모드 bbox 1개 이상 측정", any(a.get("boxes") for a in artifacts))

        # 9) 답변 호출 이미지(Step 8 1차) — 기준 A1~A4는 STEPS.md "Step 8 1차 실모델 확인 기준"에 결과를 보기 전에 고정했다.
        #    기본 모드(uploads)는 위 1)~8)이 그대로 확인한다(A5).
        def answer_images_of(document: dict | None) -> list[dict]:
            calls = model_events(document, "answer") if document else []
            return calls[0]["data"].get("images", []) if calls else []

        data = ask(client, "이 도면의 DWG NO와 REV를 알려 줘. 그리고 도면에 그려진 형상을 한 줄로 설명해 줘.",
                   ["scanned_drawing.pdf"], answer_images="whole")
        show("9-A1) 답변 호출 이미지 전체 — 스캔 PDF: 전사한 쪽 이미지를 답변 호출에도 실음", data)
        sent = (data.get("meta") or {}).get("answerImages", {})
        document = trace_of(client, data)
        images = answer_images_of(document)
        # 전사는 호출이든 캐시든 답변 전에 끝나 있어야 한다. 같은 쪽을 3)에서 이미 전사했으면 캐시로 온다(첫 실행에서 이 검사가
        # 호출 수만 보고 ✘를 냈다 — 앱은 맞았고 검사가 틀렸다).
        ocr_calls = (data.get("meta") or {}).get("vision", {}).get("ocrCalls", 0)
        ocr_cached = any(event["kind"] == "ocr" and event["label"].startswith("전사 캐시 사용") for event in (document or {}).get("events", []))
        evidence = next((a for a in data.get("attachments", []) if a["name"].endswith("visual OCR")), None)
        check("A1: 전사 후 답변 호출에 쪽 이미지가 실림",
              (ocr_calls >= 1 or ocr_cached) and evidence is not None and evidence["parsedCharacters"] > 0
              and sent.get("sent", 0) >= 1 and any("· page" in image.get("name", "") for image in images),
              f"전사 호출 {ocr_calls}회{' (캐시 사용)' if ocr_cached else ''} · 실은 이미지 {sent.get('sent')}장 · 트레이스 {[image.get('name') for image in images]}")
        check("A1: 전체 모드에서도 도면 번호 추출", "FA-7731" in data.get("text", ""), "기대값 FA-7731-B")

        data = ask(client, "이 문서의 도면 번호(DRAWING NO)를 알려 줘.", ["native_spec.pdf"], answer_images="whole")
        show("9-A2) 답변 호출 이미지 전체 — 네이티브 PDF: 전처리에서 만들지 않던 쪽을 지금 렌더해 실음", data)
        sent = (data.get("meta") or {}).get("answerImages", {})
        names = [a["name"] for a in data.get("attachments", [])]
        check("A2: 네이티브 쪽이 렌더돼 첨부에 추가되고 답변 호출에 실림",
              "native_spec.pdf · page 1" in names and sent.get("names") == ["native_spec.pdf · page 1"], f"{names} · {sent}")
        check("A2: 전체 모드에서도 도면 번호 추출", "PS-2210-A" in data.get("text", ""), "기대값 PS-2210-A")

        data = ask(client, "승인 도장(APPROVED)이 어디 있는지 이미지 위에 표시해 줘.", ["sheet_with_stamp.png"], answer_images="off")
        show("9-A3) 답변 호출 이미지 끔 — 이미지를 싣지 않아도 위치 확인 도구는 그대로", data)
        sent = (data.get("meta") or {}).get("answerImages", {})
        document = trace_of(client, data)
        grounding = model_events(document, "grounding") if document else []
        check("A3: 답변 호출 이미지 0장이어도 bbox 도구 호출(위치 확인 호출에는 이미지 1장)",
              sent.get("sent") == 0 and bool(data.get("artifacts")) and grounding
              and all(len(call["data"].get("images", [])) == 1 for call in grounding),
              f"실은 이미지 {sent.get('sent')}장 · 아티팩트 {len(data.get('artifacts', []))}건 · 위치 확인 호출 {len(grounding)}회")

        data = ask(client, "도면 번호(DRAWING NO)가 적힌 위치를 이미지 위에 표시해 줘.", ["native_spec.pdf"], answer_images="whole")
        show("9-A4) 답변 호출 이미지 전체 + 위치 요청 — [PAGE IMAGES] 줄이 붙어도 도구를 부르는지", data)
        sent = (data.get("meta") or {}).get("answerImages", {})
        check("A4: 쪽 이미지와 [PAGE IMAGES] 줄이 있는 상태에서 위치 요청에 도구 호출",
              sent.get("sent", 0) >= 1 and bool(data.get("artifacts")),
              f"실은 이미지 {sent.get('sent')}장 · 아티팩트 {len(data.get('artifacts', []))}건")

    failed = [name for name, passed, _ in results if not passed]
    print(f"\n결과: {len(results) - len(failed)}/{len(results)} 통과" + (f" · 실패: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
