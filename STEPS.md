# STEPS.md — 단계별 개발 계획과 진행 현황

[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)를 실제 작업 단위로 쪼갠 문서다.
각 단계는 **그 단계만으로 실행·테스트가 가능**하도록 나눴다. 작업 지침은 [CLAUDE.md](CLAUDE.md), 실행·테스트 방법은 [README.md](README.md).

범례: `[ ]` 미착수 · `[~]` 진행 중 · `[x]` 완료(테스트 통과 확인)

**현재 상태 (2026-09-21): Step 0~4 완료, Step 5(타일링)는 계획대로 미착수.**
자동 테스트 113개 통과 · 실제 모델(Ollama `gemma3:latest`) 종단 점검 11/11 통과(1회 실행 기준, 아래 "검증 범위의 한계" 참고) · 브라우저 UI 수동 검증 완료.

---

## 분석 요약 — vectra-web에서 무엇을 가져오고 무엇을 버렸는가

| vectra-web 위치 | 역할 | 이 프로젝트에서 |
|---|---|---|
| `server/services/pdf-renderer.mjs` `inspectPage` | 페이지별 네이티브 문자 수·래스터·벡터 연산 집계 → `needsVlm` 판별 | **동일 규칙 이식** → `app/pipeline/pdf.py` (pdf.js → PyMuPDF) |
| 같은 파일 `renderPage` | DPI 200 렌더 + 긴 변 3072 / 800만 픽셀 캡 | **동일 규칙 이식** → `pipeline/pdf.py`, `pipeline/images.py` |
| `document-pipeline/ocr-orchestrator.mjs` | `needsVlm` 페이지만 VLM 전사, 동시 2, SHA-256 캐시, 전사 후 이미지 바이트를 프롬프트 경로에서 제거 | **동일 규칙 이식** → `pipeline/ocr.py` |
| `server.mjs` `readOcrImage` / `isUsableOcrResponse` | 전사 3회 재시도, 거절·잡담 응답 걸러내기 | **동일 규칙 이식** → `pipeline/ocr.py` |
| `document-pipeline/evidence.mjs` | 첨부 묶음(root) 관리, 매니페스트, 프롬프트 예산 내 텍스트 분배·클립 | **동일 규칙 이식** → `pipeline/evidence.py` |
| `imageTools.ts` `inspect_visual` + `server.mjs` `inspectVisual`/`parseVisualInspection` | 분리 호출로 bbox(0~1000) 획득 → 0~1 분수로 변환 | **동일 규칙 이식** → `agent/tools.py`, `agent/grounding.py` |
| `core/src/tools/attachments.ts` | 첨부 텍스트 읽기/검색 도구 | 축소 이식(`read_attachment`, `search_attachments`) |
| `server/services/history.mjs` | SQLite 대화 저장 | 단순화 이식(프로젝트·공유 JSON 제거) → `app/db.py` |
| `public/js/app.js` | `state` + `render()` 구조, 세션 사이드바, 첨부, 뷰어(% 오버레이), 모델 설정 | **구조만 따르고 새로 작성** → `static/js/app.js` |
| deepagents 런타임, 서브에이전트, 웹검색, 문서/차트/이미지 생성, llama.cpp 프로세스 관리, 모델 다운로드, HF 검색, GPU 탐지, SSE, 프로젝트(폴더) | — | **제외** |

> vectra는 독점 라이선스라 **코드·프롬프트·CSS·로고를 복사하지 않았다.** 알고리즘·임계값·흐름만 같고 구현은 전부 새로 썼다.

---

## Step 0 — 분석·문서·환경  ✅

- [x] vectra-web 핵심 로직 분석
- [x] `CLAUDE.md`, `STEPS.md` 작성
- [x] `pyproject.toml` + `uv` 가상환경(Python 3.11) + 의존성 설치
- [x] 디렉터리 골격, `.gitignore`

## Step 1 — FastAPI 스캐폴딩 + SQLite 세션 CRUD + 프론트 골격  ✅

- [x] `app/config.py` — 모든 임계값·환경변수
- [x] `app/db.py` — `conversations` / `messages` / `attachments` 스키마, WAL + `synchronous=NORMAL`, FK cascade
- [x] 세션 API: `GET /api/sessions`, `POST /api/sessions`, `GET|PUT|DELETE /api/sessions/{id}`, `DELETE /api/sessions`(선택/전체)
- [x] 첨부 바이트 제공: `GET /api/attachments/{id}/content`
- [x] `app/main.py` — 정적 파일 서빙, 교차 출처 API 요청 차단, `/api/health`
- [x] 프론트: 레이아웃, 세션 사이드바(목록/새 채팅/불러오기/삭제/선택 삭제/전체 삭제), 채팅 목록·입력창, 테마

검증: `tests/test_sessions_api.py` (9개) — 재시작 후 파일 DB 유지, FK cascade, 정렬, 교차 출처 403 포함.

## Step 2 — 모델 연결  ✅ (Anthropic·Gemini는 라이브 미검증)

- [x] `providers/base.py` — `analyze(messages, images=None, tools=None)` 인터페이스, `ModelResponse`, `ToolCall`
- [x] `providers/openai_compat.py` — `openai` SDK, `base_url`만 바꿔 로컬과 OpenAI 겸용. 네이티브 tools, 컨텍스트 초과 시 2단계 압축 재시도, temperature 거절 모델 대응
- [x] `providers/anthropic_provider.py`, `providers/gemini_provider.py` — 공식 SDK. **요청·응답 변환만 단위 테스트, 실제 API 호출은 key가 없어 검증 못 함**
- [x] `POST /api/models`(+`GET`), `POST /api/test-connection`
- [x] `POST /api/chat` — 동기 JSON / NDJSON 스트리밍
- [x] 프론트: 모델 설정 다이얼로그(provider / API key / base URL / 컨텍스트 크기 / 연결 테스트), 모델 선택

검증: `tests/test_providers.py` (9개), `tests/test_chat_api.py` 일부, Ollama 실호출.

## Step 3 — 문서 전처리 (핵심)  ✅

- [x] §5.1 페이지 판별 `inspect_page` — `native_characters`, `raster_images`, `vector_operations`, `needs_vlm`, 분류 6종
- [x] §5.2 렌더링 캡 — 페이지 렌더·업로드 이미지 공통 스케일 계산, `assemble_model_images`(타일링 자리)
- [x] §5.3 OCR 전사 — 재시도 3, 동시 2, 순서 보존, SHA-256 LRU 캐시, 실패 결과 비캐시, 전사 후 이미지 바이트를 메인 요청에서 제외(표시용만 보관)
- [x] §5.4 일반 이미지 — 전사 없이 그대로 VLM에 전달(캡만 적용)
- [x] `pipeline/evidence.py` — root 이름, 매니페스트, 예산 분배, 클립, OCR 커버리지 미리보기
- [x] `POST /api/attachments/inspect` — 업로드 직후 "N쪽 비전 전사 필요 / 네이티브 텍스트 n자" 표시용
- [x] `/api/chat`에 전처리 연결, 첨부를 DB에 보존해 후속 턴에서도 사용
- [x] 프론트: 드래그&드롭·파일 선택(PDF/이미지만), 첨부 칩, 진행 단계 표시

검증: `tests/test_pdf_pipeline.py` (32개), `tests/test_ocr_evidence.py` (17개), `tests/test_chat_api.py`.

## Step 4 — `inspect_visual` bbox 도구 + 단일 tool-calling 루프 + 오버레이  ✅

- [x] `agent/grounding.py` — grounding 전용 프롬프트, 응답 파싱(코드펜스·산문 속 JSON), 0~1000 → 0~1 분수, 잘못된 박스 제거, 타입 정규화
- [x] `agent/tools.py` — `inspect_visual {name, page?, task}`, `read_attachment`, `search_attachments`; PDF 임의 페이지 온디맨드 렌더
- [x] `agent/loop.py` — 단일 루프, 최대 스텝, 동일 호출 반복 감지, **네이티브 tool-calling 미지원 모델용 JSON 폴백**, `<think>` 제거, 길이 중단 시 이어쓰기
- [x] 거짓 "첨부를 볼 수 없다" 응답 1회 교정 (`chat_service.py`)
- [x] 시스템 프롬프트의 호출 트리거(위치 표시·객체 탐지·표·치수·도장·서명·다이어그램일 때만)
- [x] 프론트: 결과 뷰어(이미지 + `%` 오버레이 `<div>`), 확대/축소·드래그 이동·패널 크기 조절, 영역 목록(타입 색·신뢰도), 라벨 토글

검증: `tests/test_agent.py` (28개), `tests/test_chat_api.py`, Ollama `gemma3`로 실제 bbox 생성(JSON 폴백 경로) 확인.

## Step 5 — (추후) 타일링  ⬜

- [ ] `assemble_model_images`에서 큰 페이지를 겹침 타일로 분할
- [ ] 타일별 OCR 결과 병합 규칙(겹침 영역 중복 제거)
- 자리는 만들어 뒀다: `ModelImage.source_box`, `grounding.map_box_to_source()`(타일 좌표 → 전체 좌표, 테스트 있음), OCR·grounding 모두 "이미지 목록"을 순회하도록 작성돼 있다.

---

## 계획 대비 변경점

| 항목 | 계획서 | 실제 구현 | 이유 |
|---|---|---|---|
| 첨부 저장 | `base64_or_path` | **BLOB 컬럼** + `GET /api/attachments/{id}/content` | 3072px PNG가 수 MB라 base64를 채팅 응답·세션 로드에 실으면 수십 MB가 오간다. 아티팩트는 `attachmentId`로 이미지를 가리킨다 |
| 메시지 테이블 | `artifacts_json` | + `files_json`, `created_at` | 사용자 메시지의 첨부 칩을 세션 재로드 후에도 보여 주기 위해 |
| 모델 목록 | `GET /api/models` | **`POST`가 기본**, `GET`도 제공(키는 `X-Api-Key` 헤더) | API key를 URL에 실으면 로그·히스토리에 남는다 |
| 세션 API | `GET/POST/DELETE`, `GET {id}` | + `PUT {id}`, `DELETE {id}` | 제목 변경·개별 삭제 |
| 대화 저장 주체 | (vectra: 클라이언트가 PUT) | **서버가 `/api/chat` 안에서 저장** | 처리된 첨부(OCR 증거·페이지 이미지)가 서버에 있어 재시작 후에도 후속 질문이 가능해야 한다 |
| 새 업로드 시 이전 첨부 | (vectra: 영어 정규식으로 "compare/forget" 추정) | **항상 병합, 같은 이름은 통째 교체** | 영어 정규식은 한국어 질문에 동작하지 않는다. 결정적 규칙이 예측 가능하다 |
| 일반 이미지 | "vectra와 동일하게 그대로 전달" | 그대로 전달(전사 없음) | 실제 vectra는 업로드 이미지도 OCR하지만, **계획서 §5.4의 명시를 따랐다** |
| 이미지 축소 위치 | (vectra: 브라우저 canvas) | **서버(Pillow)** | 테스트 가능성, EXIF 회전·투명 배경·WebP/BMP 정규화를 한 곳에서 |
| 벡터 연산 수 | path/stroke/fill 연산 개수 | **콘텐츠 스트림의 경로 페인팅 연산자 정규식 집계**(폼 XObject 포함) | `page.get_drawings()`는 선이 수십만 개인 CAD 도면에서 매우 느리다. 분류에는 "0 초과" 여부만 쓰인다 |
| bbox 재시도 | "최대 2회 재시도" | 첫 시도 + 재시도 2회(총 3회), `DOCCHAT_GROUNDING_RETRY_COUNT` | 계획서 문구를 그대로 따름(vectra 코드는 총 2회) |
| bbox 좌표 | 0~1000 | 0~1000 + **방언 보정**(픽셀 좌표, 0~1 분수, `bbox_2d`, Gemini `box_2d`의 y-선행) | VLM마다 학습된 좌표 형식으로 되돌아가는 경우가 흔하다 |
| 마크다운 렌더러 | 제외(필요 시 추가) | **경량 버전 포함**(표·목록·제목·코드·굵게; 구문 강조 없음) | 문서 분석 답변이 거의 항상 표를 포함해 없으면 `\|---\|`가 그대로 보인다. 빼려면 `renderMarkdown` 호출만 `textContent`로 바꾸면 된다 |
| 추가 도구 | `inspect_visual` 등 | + `read_attachment`, `search_attachments` | 프롬프트 예산 때문에 잘린 긴 문서의 가운데를 읽을 방법이 필요 |
| 스트리밍 | 단순 청크 | NDJSON(`application/x-ndjson`), **토큰 단위 스트리밍은 없음**(진행 단계만) | 도구 루프 + JSON 폴백과 토큰 스트리밍을 함께 하면 복잡도가 크다. 추후 개선 후보 |
| vectra의 "나중에 하겠다" 응답 감지 | — | 미이식 | 오탐 위험 대비 이득이 작다 |

### 실제 모델로 돌려 보고 추가한 것 (계획서에 없던 방어 로직)

| 관찰(gemma3 4B) | 조치 |
|---|---|
| 전사 뒤에 `[UNCLEAR]`를 수백 줄 반복(증거 7,349자 중 대부분 잡음) | `ocr.clean_transcription` — `[UNCLEAR]` 연속은 1개로, 같은 줄 12회 초과 반복은 3줄 + 생략 표식. 같은 문서가 7,349자 → 381자 |
| `<start_of_image>` 특수 토큰 누출 | 알려진 특수 토큰만 제거(문서 내용일 수 있는 `<html>` 등은 보존) |
| grounding 스키마의 `"type":"text\|object\|table…"`를 그대로 베낌 | 프롬프트에서 예시와 허용값 목록 분리 + 파서에서 타입 정규화 |
| Ollama의 gemma3는 tools 요청에 HTTP 400 | 자동으로 JSON 폴백 전환(실측 확인) |
| 한국어 질문에 영어로 답함 | 언어 힌트 `[Language: write your final answer in Korean …]`(한글/가나/한자 감지). **단, 붙이는 시점은 루프가 정한다 — 바로 아래 행 참고** |
| **언어 힌트가 JSON 폴백의 도구 호출을 죽임** (통제 비교, 각 6회: 힌트+리마인더 0/6 vs 리마인더만 6/6. 힌트를 리마인더 문장 안에 녹여도 0/6) | 힌트를 메시지에 미리 박지 않고 `run_tool_loop(language_hint=…)`로 넘김. **폴백의 도구 제공 호출에서만 생략**하고, 네이티브 경로·도구 실행 후 최종 답·강제 종료·이어쓰기에는 부착 |
| 폴백에서 시스템 프롬프트의 규약만으로는 도구를 안 부름 | 질문 끝에 짧은 도구 리마인더(`json_tool_reminder`), 도구 결과 뒤에는 "이제 최종 답을 써라"(`AFTER_TOOL_RESULT`) |
| 도구를 부르려다 **괄호가 어긋난 JSON**을 냄: `…"arguments": {…}]} ]}` → 호출 0건으로 오인되어 **날 JSON이 답변으로 노출** | `_salvage_tool_calls`(알려진 도구 이름 뒤의 arguments 객체만 디코드) → 그래도 안 되면 재요청 2회(`RESEND_VALID_TOOL_JSON`) → 도구 없이 최종 답 강제 → 최후에는 안내문. 도구 봉투로 보이는 글은 절대 사용자에게 내보내지 않는다 |
| 첨부 이름을 확장자 없이 넘김(`sheet_with_stamp`) | `_find`: 확장자를 뺀 이름이 **유일할 때만** 허용. `plan.png`/`plan.pdf`처럼 모호하면 추측하지 않고 오류로 정확한 이름을 되묻는다 |

#### 이 과정에서 바로잡은 내 오판 (다음 작업자를 위한 기록)

- 회귀 직후 "원인 확정"이라고 적었지만 그때 확인한 건 *현상*(JSON 대신 산문)뿐이었다. 원인은 그 뒤의 통제 비교로 비로소 뒷받침됐다.
- "소형 모델은 마지막 줄을 가장 강하게 따른다"는 설명은 **틀렸다.** 리마인더가 힌트보다 뒤(메시지 맨 끝)에 있었는데도 졌다.
- 첫 A/B의 "A = 원래 구성" 라벨은 부정확했다(나중에 추가한 문장이 든 현재 시스템 프롬프트를 썼다). 그래서 **최초 9/9 통과가 실력이었는지 운(temperature 0.2, 1회)이었는지는 끝내 가려내지 못했다.**
- 수작업 A/B에서 6/6이던 구성이 실제 `/api/chat` 경로에서는 0/4였다(모델이 코드펜스 없이 깨진 JSON을 냄). **수작업 하네스 ≠ 실제 경로** — 프롬프트 변경은 반드시 `scripts/e2e_check.py`로 확인한다.
- e2e가 "불러야 할 때 부르는지"만 보고 "부르면 안 될 때 안 부르는지"는 보지 않았다 → **음성 대조** 3문항 추가. 합격 기준(위치 ≥2/3 호출, 비위치 오호출 ≤1/3)은 결과를 보기 전에 고정했고, 이후 바꾸지 않았다.

---

## 알려진 한계

- **JSON 폴백 모델에서는 답변 언어가 흔들린다.** 첨부가 있는 턴(=도구가 제공되는 턴)에 모델이 도구 없이 바로 답하면 언어 힌트가 없어 영어로 나올 수 있다(실측: `H. Park drew the diagram…`). 도구 호출을 살리기 위한 의도된 대가다. 네이티브 tool-calling 모델은 해당 없음(단, 그 경로에서 힌트가 무해한지는 **미실측** — 이 PC엔 gemma3뿐).
- **검증 범위의 한계.** 도구 트리거는 이미지 1장 · 4B 모델 1개 · 질문 7개 · **1회 실행**으로 확인했다(temperature 0.2). "기준 통과"이지 신뢰성의 증명이 아니다. 다른 모델·문서에서는 `scripts/e2e_check.py`로 다시 재 볼 것.
- **추론(thinking) 모델**: 기본으로 thinking을 끈다(vLLM·llama.cpp). 켜면 답이 매우 느려지고, 서버에 reasoning parser가 없으면 생각 내용이 `Thinking Process:`로 본문에 섞여 나온다(vLLM은 `--reasoning-parser qwen3`로 분리 가능). 토큰 스트리밍이 없어 긴 생성은 멈춘 것처럼 보인다.
- **bbox 정확도는 모델 의존.** `gemma3:4b`는 위치가 크게 어긋난다(파이프라인·오버레이는 정상). Qwen-VL 계열·Gemini·Claude 권장.
- 타일링이 없어 A0급 도면은 3072px로 줄면서 작은 글자가 뭉개진다.
- Anthropic / Gemini provider는 라이브 호출 미검증.
- 토큰 단위 스트리밍 없음(느린 로컬 모델에서는 "답변을 생성하는 중…"에서 오래 머문다).
- 대화에서 특정 첨부만 빼는 UI는 없다(새 채팅을 시작하거나 같은 이름으로 다시 올려 교체).
- 서버에 인증이 없다 → `127.0.0.1` 바인딩 전제.

## 진행 기록

| 날짜 | 한 일 | 테스트 |
|---|---|---|
| 2026-09-21 | Step 0~4 구현, README/CLAUDE/STEPS 작성, 샘플 생성기·종단 점검 스크립트 | `uv run pytest` **109 passed (7s)** · `scripts/e2e_check.py` **9/9** (Ollama gemma3:latest) · 브라우저에서 업로드→전사→표 답변, bbox 뷰어, 중지(취소) 확인 |
| 2026-09-22 | 사용자 보고 2건: ① vLLM `Qwen3.5-35B-A3B`가 "답변을 생성하는 중…"에서 멈춤 → 서버에 직접 재현: 추론 모델이 thinking으로 수천 토큰을 쓰고(시스템 프롬프트 포함 시 90초 초과), 토큰 스트리밍이 없어 멈춘 것처럼 보임. `chat_template_kwargs.enable_thinking=false`면 0.3초 → 로컬 provider에 "추론 끄기" 옵션(기본 켬, 필드를 거절하는 서버면 빼고 재시도) ② GPT 모델이 "저는 ChatGPT입니다" → 시스템 프롬프트에 DocChat 정체성 + 모델명 | `uv run pytest` **115 passed** · 실제 경로로 vLLM 확인: 인사 1.2초, 이름=DocChat, 스캔 PDF 4.0초(FA-7731-B), bbox 3.6초(네이티브 tool-calling, 영역 2개) · Ollama gemma3도 정상. **OpenAI 쪽 정체성 수정은 key가 없어 미실측** |
| 2026-09-21 | 언어 힌트 도입 후 실모델에서 **bbox 도구 호출 회귀** 발견 → 통제 비교로 원인 분리 → 힌트 부착 시점을 루프로 이동, 깨진 도구 JSON 복구/재요청/비노출, 확장자 없는 첨부 이름 허용, e2e에 음성 대조 추가 | `uv run pytest` **113 passed** · `e2e_check.py` **11/11** (위치 4/4 호출, 비위치 오호출 0/3, 1회 실행) |
| 2026-09-21 | 검증 중 발견해 고친 결함: JPEG 불필요 재인코딩(EXIF 판정), 5000px→3071px 부동소수 오차, 모델 목록 연결 실패 메시지 미번역, 같은 ms 갱신 시 세션 정렬 뒤집힘, D: 드라이브 fsync 지연(커밋 182ms→1ms), 상단 버튼 줄바꿈, 중지 시 Windows asyncio 트레이스백 | 각 항목 회귀 테스트 추가 |
