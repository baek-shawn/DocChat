# DocChat — 로컬/클라우드 VLM 문서 분석 챗

PDF·이미지를 올리고 질문하면, **텍스트가 충분한 쪽은 그대로 읽고 부족한 쪽만 비전 모델로 전사**해서 답합니다.
"위치를 표시해 줘" 같은 요청에는 **분리된 bbox 호출(`inspect_visual`)** 로 영역을 재서 이미지 위에 오버레이합니다.

- 설계: [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) · 단계/진행 현황: [STEPS.md](STEPS.md) · 작업 지침: [CLAUDE.md](CLAUDE.md)

## 1. 준비

필요한 것: [uv](https://docs.astral.sh/uv/) (Python 3.11은 uv가 알아서 씁니다), 그리고 모델 하나.

```powershell
uv sync
```

모델은 둘 중 하나면 됩니다.

| 방식 | 준비 |
|---|---|
| **로컬 (권장: 시험용)** | [Ollama](https://ollama.com) 실행 후 비전 모델 받기: `ollama pull gemma3` → 주소 `http://127.0.0.1:11434/v1` |
| 로컬 (기타) | llama.cpp `llama-server --mmproj …`(`:8080/v1`), LM Studio(`:1234/v1`), vLLM(`:8000/v1`) 등 OpenAI 호환 서버 |
| 클라우드 | OpenAI / Anthropic / Gemini API key |

> bbox 품질은 모델에 크게 좌우됩니다. `gemma3:4b`는 위치 정확도가 낮습니다. 좌표가 중요하면 Qwen2.5-VL / Qwen3-VL 계열이나 Gemini·Claude를 쓰세요.

## 2. 실행

```powershell
uv run python run.py
```

브라우저에서 <http://127.0.0.1:8000> 을 엽니다.

1. 오른쪽 위 **⚙ 모델 설정** → 제공자 선택 → (로컬이면) 서버 주소 입력 → **연결 테스트** → 모델 이름 선택 → **저장**
2. 입력창에 PDF/이미지를 끌어다 놓습니다. 첨부 칩에 `3쪽 · 2쪽은 비전 전사 필요` 처럼 **어떤 경로로 읽힐지** 바로 표시됩니다.
3. 질문을 보냅니다. 진행 단계(페이지 판별 → 전사 n/N → 답변 생성)가 실시간으로 보입니다.
4. "도장 위치를 표시해 줘"처럼 위치 확인이 필요한 요청을 하면 오른쪽 뷰어에 이미지와 영역이 뜹니다(휠 확대, 드래그 이동, 왼쪽 경계 드래그로 너비 조절).

옵션: `uv run python run.py --port 9000 --reload`

### 환경변수

| 변수 | 기본 | 의미 |
|---|---|---|
| `DOCCHAT_HOST` / `DOCCHAT_PORT` | `127.0.0.1` / `8000` | 바인딩 주소(인증이 없으니 localhost 권장) |
| `DOCCHAT_DB_PATH` | `data/docchat.sqlite` | 대화·첨부 저장 위치 |
| `DOCCHAT_NATIVE_MIN_CHARS` | `24` | 이 글자 수 이상이면 네이티브 텍스트 사용 |
| `DOCCHAT_SPARSE_OVERLAY_CHARS` | `120` | 래스터가 있는데 이 미만이면 비전 전사 |
| `DOCCHAT_PDF_RENDER_DPI` | `200` | 페이지 렌더 DPI |
| `DOCCHAT_MAX_VISION_IMAGE_EDGE` / `_PIXELS` | `3072` / `8000000` | 이미지 크기 상한 |
| `DOCCHAT_OCR_RETRY_COUNT` / `DOCCHAT_OCR_CONCURRENCY` | `3` / `2` | 전사 재시도 / 동시 처리 |
| `DOCCHAT_GROUNDING_RETRY_COUNT` | `2` | bbox JSON 재시도 |
| `DOCCHAT_MAX_PDF_VISUAL_PAGES` | `60` (최대 200) | 검사할 최대 페이지 |
| `DOCCHAT_MAX_TOOL_STEPS` | `8` | 한 턴의 최대 도구 호출 횟수 |

> **CAD PDF 팁**: SHX 폰트로 그린 글자는 텍스트 객체가 아니라 선이라서 `native characters`가 낮게 잡히고 비전 전사로 넘어갑니다. 반대로 글자는 적지만 네이티브 텍스트로 충분한 도면이라면 `DOCCHAT_NATIVE_MIN_CHARS`를 낮추세요.

## 3. 테스트

### 3.1 자동 테스트 (모델 불필요, 약 10초, 113개)

```powershell
uv run pytest
```

내장 **mock OpenAI 호환 서버**를 실제 HTTP로 호출하며 전체 흐름을 검증합니다.

| 파일 | 검증 내용 |
|---|---|
| `tests/test_sessions_api.py` | 세션 CRUD, 제목 생성, 일괄/전체 삭제, 교차 출처 차단, 정적 파일 |
| `tests/test_pdf_pipeline.py` | §5.1 분류 6종과 경계값(24/120), needs_vlm 페이지만 렌더, §5.2 캡(3072 / 8M), 업로드 이미지 정규화 |
| `tests/test_ocr_evidence.py` | §5.3 순서 보존·동시 2·SHA-256 캐시·실패 비캐시·재시도 3회, 증거 예산 분배 |
| `tests/test_agent.py` | bbox 파싱(0~1000→분수, 잘못된 박스 제거), 단일 루프, JSON 폴백, 반복 호출 차단, 도구 |
| `tests/test_providers.py` | provider 팩토리, OpenAI/Anthropic/Gemini 요청·응답 변환 |
| `tests/test_chat_api.py` | "네이티브 PDF는 이미지 미전송", "스캔 PDF는 전사 후 이미지 미전송", "일반 이미지는 직접 전송", bbox 분리 호출, 후속 턴 첨부 유지, NDJSON 스트리밍 |

특정 테스트만: `uv run pytest tests/test_pdf_pipeline.py -k classification -v`

### 3.2 실제 모델 종단 점검

```powershell
uv run python scripts/make_samples.py     # samples/ 에 시험용 PDF·이미지 생성
uv run python scripts/e2e_check.py        # 기본: Ollama의 gemma3:latest
```

다른 모델로: `$env:DOCCHAT_E2E_BASE_URL="http://127.0.0.1:8080/v1"; $env:DOCCHAT_E2E_MODEL="qwen2.5-vl"; uv run python scripts/e2e_check.py`
클라우드로: `$env:DOCCHAT_E2E_PROVIDER="gemini"; $env:DOCCHAT_E2E_API_KEY="…"; $env:DOCCHAT_E2E_MODEL="gemini-2.5-flash"`

### 3.3 브라우저에서 손으로 확인

`samples/`의 파일로 아래를 해 보세요.

| 파일 | 기대 동작 |
|---|---|
| `native_spec.pdf` | 칩에 "네이티브 텍스트 n자". 전사 단계 없이 바로 답변. "도면 번호 알려 줘" → `PS-2210-A` |
| `scanned_drawing.pdf` | 칩에 "1쪽은 비전 전사 필요". 진행 로그에 "페이지 전사 중… (1/1)". → `FA-7731-B` |
| `mixed_3pages.pdf` | 2·3쪽만 전사(1쪽은 네이티브) |
| `sheet_with_stamp.png` | "승인 도장 위치를 표시해 줘" → 오른쪽 뷰어에 박스 오버레이 |

### 3.4 API 직접 호출

```powershell
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/test-connection -H "Content-Type: application/json" -d '{\"provider\":\"openaiCompatible\",\"baseUrl\":\"http://127.0.0.1:11434/v1\"}'
```

대화형 API 문서: <http://127.0.0.1:8000/docs>

## 4. API 요약

| 메서드 · 경로 | 설명 |
|---|---|
| `POST /api/chat` | 한 턴 실행. `stream:false`(기본)면 JSON 한 번, `stream:true`면 NDJSON(`progress`… `final`) |
| `GET /api/sessions` · `POST /api/sessions` · `DELETE /api/sessions` | 목록 · 생성 · 선택/전체 삭제(`{ids}` 또는 `{all:true}`) |
| `GET` · `PUT` · `DELETE /api/sessions/{id}` | 불러오기 · 수정 · 삭제 |
| `POST /api/models` (`GET`도 가능) | 모델 목록. API key는 본문/`X-Api-Key` 헤더로만 |
| `POST /api/test-connection` | 항상 200 + `{ok, message, models}` |
| `POST /api/attachments/inspect` | 업로드 미리 검사(페이지별 분류) |
| `GET /api/attachments/{id}/content` | 저장된 첨부 바이트(뷰어 이미지) |

## 5. 알아 둘 점

- **API key는 저장하지 않습니다.** 브라우저 sessionStorage에만 있고 서버는 요청 때만 씁니다.
- 대화·첨부는 `data/docchat.sqlite`에 저장됩니다. 지우려면 UI의 "전체 세션 삭제" 또는 파일 삭제.
- Anthropic / Gemini provider는 요청·응답 변환까지만 단위 테스트했고 **실제 API 호출은 key가 없어 검증하지 못했습니다.**
- 타일링은 아직 없습니다. A0 도면은 3072px로 줄어 작은 글자가 뭉개질 수 있습니다(STEPS.md Step 5).
- **네이티브 tool-calling을 지원하지 않는 모델**(예: Ollama의 gemma3)은 JSON 폴백으로 도구를 부릅니다. 이 경우 첨부가 있는 대화에서 **답변이 가끔 영어로 나올 수 있습니다** — 언어 지시를 넣으면 이 모델이 도구 호출을 건너뛰는 것이 실측돼서, 도구 호출을 살리는 쪽을 택했습니다(STEPS.md "알려진 한계").
- 도구 호출 트리거는 **gemma3 4B · 이미지 1장 · 질문 7개 · 1회 실행**으로만 확인했습니다. 다른 모델을 쓸 때는 `scripts/e2e_check.py`로 직접 재 보세요(위치 질문에서 부르는지 + 비위치 질문에서 안 부르는지를 함께 봅니다).
