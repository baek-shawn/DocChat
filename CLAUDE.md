# CLAUDE.md — 작업 지침

이 저장소에서 작업하는 Claude가 **매 세션 시작 시 가장 먼저 읽는** 지침이다.
무엇을 만드는지는 [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md), 어디까지 했는지는 [STEPS.md](STEPS.md)에 있다.

## 1. 프로젝트 한 줄 요약

로컬/클라우드 VLM으로 **PDF·이미지 문서를 분석하는 챗 시스템**.
백엔드는 Python(FastAPI), 프론트는 프레임워크 없는 순수 JS. 참고 구현은 `D:\vectra\vectra-web`(Node).

## 2. 세션 시작 절차

1. `STEPS.md`를 읽고 **현재 단계와 체크박스 상태**를 확인한다.
2. 진행 중인 단계의 "완료 기준"을 다시 읽는다.
3. 작업 후 `STEPS.md` 체크박스와 "진행 기록"을 갱신한다. 계획과 다르게 구현했다면 "계획 대비 변경점" 표에 **이유와 함께** 적는다.
4. 테스트를 돌리고(`uv run pytest`) 결과를 **있는 그대로** 보고한다. 실패를 숨기지 않는다.

## 3. 절대 규칙

### 3.1 vectra는 "참고"만 한다 — 복사 금지
- vectra는 **독점 라이선스**(`D:\vectra\LICENSE`: 무단 복사·재배포·유사 파생 제품 금지)다.
- 따라서 **알고리즘 / 임계값 / 처리 흐름 / API 계약**만 동일하게 맞추고, **코드·프롬프트 문구·CSS·로고는 직접 새로 작성**한다.
- vectra 파일을 복사-붙여넣기 하지 않는다. 주석에 출처를 밝힐 때는 "vectra `pdf-renderer.mjs`의 판별 규칙과 동일" 처럼 **동작**만 언급한다.
- vectra 브랜딩(이름·로고)을 UI에 쓰지 않는다.

### 3.2 계획서에서 확정된 제외 기능은 넣지 않는다
웹검색 / web_fetch / 논문 검색, deepagents·langchain 멀티 서브에이전트, 문서 생성(PDF/DOCX/PPTX)·차트 도구, 정식 SSE·WebSocket, 타일링(구조만 열어둠).
→ 필요해 보여도 **먼저 사용자에게 묻는다.**

### 3.3 구조 원칙
- **단일 tool-calling 루프**만 둔다: 모델 호출 → tool_call 감지 → 실행 → 결과 재주입 → 반복. 프레임워크 금지.
- **bbox는 항상 분리된 별도 호출**(`inspect_visual`)로 받는다. 메인 답변과 한 번에 묶지 않는다.
- 통신은 **동기 HTTP**가 기본. 진행률이 필요할 때만 `StreamingResponse`로 **NDJSON 한 줄씩** 흘린다(`data:` 접두사·`text/event-stream` 쓰지 않음).
- 모델 인터페이스는 하나: `analyze(messages, images=None, tools=None) -> ModelResponse`.
- "페이지 이미지 준비"(`pipeline/images.py`, `pipeline/pdf.py`)와 "VLM에 보낼 이미지 목록 조립"(`assemble_model_images`)을 **분리 유지**한다 — 타일링을 끼워 넣을 자리다.
- API key는 **요청마다 받아서 쓰고 디스크에 저장하지 않는다.** DB·로그에 남기지 않는다.

### 3.4 임계값은 `app/config.py` 한 곳에서만
vectra와 같은 값을 기본으로 두고 환경변수(`DOCCHAT_*`)로 덮어쓸 수 있게 한다. 코드에 매직 넘버를 흩뿌리지 않는다.

| 상수 | 값 | 의미 |
|---|---|---|
| `NATIVE_MIN_CHARS` | 24 | 이 이상이면 네이티브 텍스트 사용 가능 |
| `SPARSE_OVERLAY_CHARS` | 120 | 래스터가 있는데 이 미만이면 VLM 필요 |
| `PDF_RENDER_DPI` | 200 | 페이지 렌더 기준 DPI |
| `MAX_VISION_IMAGE_EDGE` | 3072 | 긴 변 상한(px) |
| `MAX_VISION_IMAGE_PIXELS` | 8,000,000 | 픽셀 수 상한 |
| `OCR_RETRY_COUNT` | 3 | 전사 재시도 |
| `OCR_CONCURRENCY` | 2 | 동시 전사 페이지 수 |
| `GROUNDING_RETRY_COUNT` | 2 | bbox JSON 재시도 |
| `DEFAULT_PDF_VISUAL_PAGES` / `MAX_` | 60 / 200 | 검사 페이지 상한 |

## 4. 기술 스택과 실행

| 영역 | 선택 |
|---|---|
| 런타임 | Python 3.11 (`uv` 가상환경 `.venv`) — 시스템 기본 Python 3.8은 **쓰지 않는다** |
| 서버 | FastAPI + Uvicorn |
| DB | SQLite (`aiosqlite`), 기본 경로 `data/docchat.sqlite` |
| PDF / 이미지 | PyMuPDF — **`import pymupdf`** (`import fitz`는 deprecated 경고가 난다) / Pillow |
| 모델 | `openai` SDK(로컬+OpenAI 겸용), `anthropic`, `google-genai` |
| 테스트 | pytest (+ 내장 mock OpenAI 호환 서버) |

```powershell
uv sync                                  # 의존성 설치(.venv 생성)
uv run python run.py                     # 서버 실행 → http://127.0.0.1:8000
uv run pytest                            # 전체 테스트(모델 불필요, 약 7초)
uv run python scripts/make_samples.py    # samples/ 에 시험용 PDF·이미지 생성
uv run python scripts/e2e_check.py       # 실제 모델(기본 Ollama gemma3)로 종단 점검
```

- 이 PC에는 **Ollama(`http://127.0.0.1:11434/v1`)에 `gemma3:latest`(비전 지원, tool-calling 미지원)** 가 있다 → 종단 점검에 사용. tool-calling 미지원이므로 **JSON 폴백 경로**가 실제로 검증된다. bbox 위치 정확도는 낮다(모델 한계) — 파이프라인 버그로 오해하지 말 것.
- Windows + PowerShell 환경. 셸 명령에 `&&` 대신 `;` 를 쓴다.
- **이 PC 특이사항**
  - `%TEMP%\pytest-of-<user>` 폴더가 접근 거부 상태다 → pytest는 `--basetemp=.pytest_tmp`(pyproject에 설정됨)를 쓴다.
  - D: 드라이브의 fsync가 느리다(SQLite 커밋 1회 ~0.2초) → DB는 `synchronous=NORMAL`, 테스트 fixture는 `:memory:` DB를 쓴다. 파일 DB 동작은 `test_database_file_survives_restart`만 확인한다.
  - 서버 코드를 고치면 **서버를 재시작**해야 반영된다(`--reload`를 주지 않는 한).

## 5. 코드 작성 규칙

- **PyMuPDF는 스레드 안전하지 않다.** 모든 `fitz` 호출은 `pipeline/pdf.py`의 단일 워커(`run_pdf`)를 거친다. 이벤트 루프에서 직접 호출 금지.
- Pillow 등 CPU 작업은 `asyncio.to_thread`로 넘긴다.
- 주석·문서·UI 문구는 **한국어**, 식별자와 모델 프롬프트는 **영어**(프롬프트에 "사용자 언어로 답하라"를 포함).
- 첨부 바이트는 DB에 BLOB으로 저장하고, 프론트에는 **base64를 되돌려 보내지 않는다.** 이미지는 `/api/attachments/{id}/content` URL로 참조한다.
- 새 기능에는 테스트를 같이 쓴다. 네트워크가 필요한 테스트는 `tests/mock_openai.py`(실제 HTTP mock)를 쓰고, 실제 모델이 필요한 확인은 `scripts/e2e_check.py`에 넣는다.
- 예외 메시지는 사용자가 읽는다 → 한국어로, 원인과 다음 행동을 담는다.
- **mock만 통과했다고 끝내지 않는다.** 소형 로컬 VLM은 스키마를 베끼고, 같은 줄을 무한 반복하고, 특수 토큰을 흘리고, 질문 언어를 무시한다. 모델 출력을 다루는 코드를 고치면 `scripts/e2e_check.py`로 실제 응답을 확인하고, 거기서 본 실패 패턴을 그대로 회귀 테스트로 옮긴다(예: `test_transcription_noise_is_removed_but_content_is_untouched`).
- **프롬프트를 고칠 때의 규율** (2026-09-21에 실제로 겪은 일 — 자세한 경위는 STEPS.md "이 과정에서 바로잡은 내 오판")
  - 언어 힌트 한 줄을 추가했더니 mock 테스트는 전부 통과한 채로 **실모델의 bbox 도구 호출이 0%가 됐다.** 프롬프트 변경은 mock으로 검증되지 않는다.
  - 실패하면 문구를 이리저리 바꿔 보기 전에 **변수를 분리한 통제 비교**부터 한다(한 번에 하나만 바꾸고, 조건당 여러 번). 1회 통과는 증거가 아니다(temperature 0.2).
  - **합격 기준과 후보 수는 결과를 보기 전에 고정한다.** "통과할 때까지 변형을 만든다"는 체크에 맞춘 튜닝이다. 기준을 못 맞추면 더 찾지 말고 한계로 기록한다.
  - "불러야 할 때 부르는가"만 재지 말고 **"부르면 안 될 때 안 부르는가"(음성 대조)** 를 같이 잰다. 계획서는 도구를 위치 확인 성격일 때만 부르라고 했다.
  - 수작업 하네스에서 6/6이던 구성이 실제 `/api/chat` 경로에서는 0/4였다. **최종 확인은 항상 `scripts/e2e_check.py`(실제 경로)로.**
  - 원인을 확인하지 않았으면 "확정"이라고 쓰지 않는다. 현상 확인과 원인 확인을 구분해 보고한다.
- 모델이 낸 **도구 호출 JSON으로 보이는 글은 절대 사용자에게 그대로 내보내지 않는다**(복구 → 재요청 → 도구 없이 최종 답 → 안내문).
- 전사 결과를 "정리"할 때는 **잡음만** 지운다. 문서 내용일 수 있는 것(같은 값이 몇 줄 이어지는 표, `<html>` 같은 태그)은 건드리지 않는다.

## 6. 디렉터리 지도

```
app/
  main.py            FastAPI 앱 조립, 정적 파일, Origin 검사
  config.py          모든 임계값/환경변수
  db.py              aiosqlite 세션·메시지·첨부 저장소
  attachments.py     Attachment 모델, 업로드 정제
  api/               sessions / models / files / chat 라우터
  providers/         analyze() 인터페이스와 OpenAI호환·Anthropic·Gemini 구현
  chat_service.py    /api/chat 한 턴의 전체 흐름(전처리→OCR→증거 선택→루프→저장)
  pipeline/          pdf(판별·렌더) / images(캡·이미지 조립) / preprocess / ocr / evidence
  agent/             prompts / tools / grounding(bbox 파싱) / loop(단일 tool-calling 루프)
static/              index.html, styles.css, js/app.js
tests/               pytest (mock_openai.py = OpenAI 호환 mock 서버, pdf_factory.py = 합성 PDF)
scripts/             make_samples.py(샘플 PDF 생성), e2e_check.py(실모델 점검)
data/                SQLite DB (git 제외, 사용자 데이터)
samples/             생성된 시험 문서 (git 제외)
```

import 방향(순환 금지): `config` ← `attachments` ← `pipeline/*` ← `providers/*` ← `agent/*` ← `chat_service` ← `api/*` ← `main`.
`pipeline/ocr.py`만 예외적으로 `agent/prompts`와 `providers/base`를 가져온다. 각 패키지의 `__init__.py`는 비워 둔다(`providers`만 팩토리 제공).

## 7. 하지 말 것

- 계획서에 없는 큰 기능을 임의로 추가하지 않는다(작은 편의 기능은 STEPS.md "변경점"에 기록).
- 테스트를 통과시키려고 임계값이나 단언을 느슨하게 바꾸지 않는다.
- `data/` 안의 DB·업로드 파일을 커밋하거나 지우지 않는다(사용자 데이터).
- 실패한 테스트·미검증 항목을 "완료"로 표시하지 않는다.
