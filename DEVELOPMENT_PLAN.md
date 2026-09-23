# 개발 계획 — Local/Cloud VLM 문서 분석 챗 시스템

vectra-web(`D:\vectra\vectra-web`)의 핵심 로직을 최대한 그대로 가져오되, 불필요한 기능(웹검색, 멀티 서브에이전트, 문서 생성, SSE/WebSocket 등)은 제거하고 백엔드를 Python으로 재구현한다.

## 1. 전체 구조

```
[Browser — JS 프론트 (vectra public/js/app.js 기반 수정)]
        │  HTTP (JSON, 필요시 청크 스트리밍)
        ▼
[Python 백엔드 — FastAPI]
  ├─ 세션/히스토리 API        → SQLite
  ├─ 모델 프록시              → Local(OpenAI-호환) / OpenAI / Anthropic / Gemini
  ├─ 문서 전처리 파이프라인    → PDF/이미지 판별 + 렌더링 + OCR
  └─ 단일 tool-calling 루프   → inspect_visual(bbox) 등 on-demand 툴
```

- **WebSocket / 정식 SSE는 사용하지 않는다.** 기본은 동기 HTTP 요청-응답. 긴 처리(다중 페이지 PDF) 진행률 표시가 필요하면 `StreamingResponse`로 단순 청크만 흘려보낸다.
- **deepagents/langchain류 멀티 서브에이전트 프레임워크는 사용하지 않는다.** 모델 호출 → tool_call 감지 → 실행 → 결과 재주입 → 반복하는 **단일 tool-calling 루프**만 둔다.

## 2. 프론트엔드 (JS)

vectra `app.js` 구조(프레임워크 없는 순수 JS, `state` 객체 + `render()`)를 그대로 이식하고 아래만 남긴다.

| 기능 | 비고 |
|---|---|
| 채팅 UI (메시지 목록/입력창/전송) | vectra 그대로 |
| 세션 사이드바 (목록/생성/불러오기/삭제) | vectra 그대로 |
| 파일 첨부 (drag&drop + 파일선택, base64 인코딩) | PDF/이미지만 허용 |
| 결과 뷰어 (이미지 + bbox 오버레이) | vectra의 `%` 기반 오버레이 `<div>` 패턴 재사용 |
| 모델 설정 (provider 선택 / API key / connection test) | vectra 그대로 |

**제외**: 문서 생성 다운로드 버튼, 웹검색 결과 표시, subagent 진행 표시 UI, 차트 뷰어, 마크다운 렌더러(필요 시에만 추가).

## 3. 세션 / DB (Python)

- FastAPI + `aiosqlite`
- 스키마 (vectra `history.mjs` 단순화):
  - `conversations(id, title, provider, model, created_at, updated_at)`
  - `messages(id, conversation_id, position, role, content, artifacts_json)`
  - `attachments(id, conversation_id, position, name, mime, text_content, base64_or_path)`
- API: `POST /api/chat`, `GET/POST/DELETE /api/sessions`, `GET /api/sessions/{id}`

## 4. 모델 연결 (Python)

- 단일 인터페이스: `analyze(messages, images=None, tools=None) -> response`
- **Local / OpenAI-호환 (llama.cpp, Ollama, vLLM 등)**: `openai` SDK를 `base_url`만 바꿔서 사용 → 모든 로컬 런타임을 하나의 구현으로 커버
- **Anthropic / Gemini**: 필요 시점에 각 공식 SDK 추가
- `GET /api/models`, `POST /api/test-connection`

## 5. 문서 전처리 — vectra 로직 그대로 이식 (핵심)

### 5.1 PDF 페이지 판별 (vectra `pdf-renderer.mjs`의 `inspectPage` 그대로)

라이브러리는 **PyMuPDF(`fitz`)** 로 대체하되, 판별 알고리즘과 임계값은 **동일하게** 유지한다.

```python
# 페이지별로 계산
native_characters = len([c for c in native_text if c.isalnum()])   # 실제 텍스트 객체(Tj/TJ)에서 추출된 문자 수
raster_images     = <페이지 내 비트맵 이미지 연산 개수>
vector_operations = <페이지 내 path/stroke/fill 연산 개수>

usable_native  = native_characters >= 24
sparse_overlay = raster_images > 0 and native_characters < 120
needs_vlm      = (not usable_native) or sparse_overlay
```

- `needs_vlm == False` → **이미지 렌더링/전송 없이 네이티브 텍스트만 사용** (vectra와 동일하게, 문서에 텍스트가 충분하면 이미지는 아예 만들지도 보내지도 않는다)
- `needs_vlm == True` → 해당 페이지만 이미지로 렌더링해서 비전 처리 대상이 됨

> CAD PDF는 SHX 폰트 등으로 인해 `native_characters`가 낮게 나와 `needs_vlm=True`가 되는 경우가 많다는 점을 감안해 임계값 튜닝 여지를 config로 남겨둔다.

### 5.2 렌더링 캡 (vectra `config.mjs` 값 그대로)

```python
MAX_VISION_IMAGE_EDGE   = 3072   # px
MAX_VISION_IMAGE_PIXELS = 8_000_000
PDF_RENDER_DPI          = 200
```

비율 유지하며 이 한도 이하로 다운스케일. **타일링은 이번 단계에서 구현하지 않음** — 단, 나중에 타일 분할을 끼워 넣기 쉽도록 "페이지 이미지 준비" 함수와 "VLM에 보낼 이미지 목록 조립" 함수를 분리해서 설계한다.

### 5.3 OCR (vision 처리, vectra `ocr-orchestrator.mjs` 그대로)

`needs_vlm=True`인 페이지만, 메인 답변 생성 전에 **선택된 VLM 자신에게 별도의 전사(transcription) 호출**을 한 번 보낸다.

```python
OCR_RETRY_COUNT  = 3   # 거절/이상 응답 시 재시도 횟수
OCR_CONCURRENCY  = 2   # 동시 처리 페이지 수
```

- 전사 전용 시스템 프롬프트: "있는 그대로만 받아써라, 안 읽히면 [UNCLEAR]로 표기, 요약/해석/추론 금지"
- 전사가 끝나면 해당 페이지의 원본 이미지 바이트는 **메인 답변 생성 요청에서 제외**하고(표시용으로만 보관), 전사된 텍스트만 컨텍스트에 포함 — vectra와 동일한 2단계 구조
- SHA-256 캐싱으로 동일 이미지 재전사 방지

### 5.4 일반 이미지 업로드 (PDF 아닌 png/jpg)

전사 사전 처리 없이 **그대로 VLM에 이미지로 전달** (vectra와 동일).

## 6. bbox 툴 — vectra `inspect_visual`처럼 분리된 별도 호출로 구현

**메인 분석 응답과 한 번에 묶어서 bbox까지 받으려 하지 말 것** (VLM 테스트 결과 안정적으로 안 나옴이 확인됨). vectra와 동일하게 **완전히 분리된 tool 호출**로 구현한다.

- tool 이름: `inspect_visual` (또는 동등한 이름)
- 입력: `{ name, page?, task }` — 어떤 이미지/페이지를, 뭘 찾을지
- 실행 시:
  1. 대상 이미지 **한 장만** 떼어냄 (OCR 단계에서 보관해둔 표시용 이미지 바이트 재사용 가능)
  2. grounding 전용 시스템 프롬프트로 **별도 호출**, temperature 0:
     > "완전한 이미지 전체만 보고, 요청한 스키마의 JSON만 반환하라. 0~1000 정수 좌표로 모든 bbox를 측정하라. 사전 지식으로 추정하거나 안 보이는 내용을 지어내지 마라."
  3. 응답 스키마: `{"text": "...", "regions": [{"type": "text|object|table|dimension|stamp|signature|diagram|other", "label": "...", "bbox": [x1,y1,x2,y2], "confidence": 0.0}]}`
  4. 구조화 JSON이 아니면 최대 2회 재시도
- 좌표계: 0~1000 정규화 (원본 이미지 기준)
- 프론트: 정규화 좌표를 `%` 로 변환해 이미지 위에 오버레이 `<div>`로 표시

**호출 트리거**: 에이전트가 항상 자동으로 부르는 게 아니라, 유저 요청이 "위치 표시/객체 탐지/표·치수·도장·서명·다이어그램 확인"처럼 **시각적 위치 확인이 필요한 성격일 때만** 시스템 프롬프트로 유도한다 (vectra와 동일한 트리거 기준).

## 7. 제외 기능 (확정)

- 웹검색 / web_fetch / 학술논문 검색
- deepagents/langchain 멀티 서브에이전트 (planner/researcher/coder/tester/reviewer/security/documentation)
- 문서 생성(PDF/DOCX/PPTX 출력), 차트 생성 도구
- SSE 정식 스펙, WebSocket
- 타일링 (구조만 확장 가능하게 남겨두고 이번 단계는 제외)

## 8. 기술 스택 요약

| 영역 | 선택 |
|---|---|
| 웹 서버 | FastAPI + Uvicorn |
| DB | SQLite (`aiosqlite`) |
| PDF | PyMuPDF (`fitz`) |
| 이미지 처리 | Pillow |
| 모델 연결 | `openai` SDK(로컬+OpenAI 겸용), 필요시 `anthropic`/`google-genai` |
| 프론트 | vectra `app.js` 기반 수정 (프레임워크 없는 순수 JS) |
| 통신 | 동기 HTTP (+선택적 청크 스트리밍), SSE/WebSocket 불필요 |

## 9. 개발 순서

1. FastAPI 스캐폴딩 + SQLite 세션 CRUD + vectra 프론트 이식
2. 로컬(OpenAI-호환) 모델 연결 → 이후 클라우드 provider 추가
3. PDF/이미지 전처리: PyMuPDF 페이지 판별(§5.1) + 렌더링 캡(§5.2) + OCR 전사(§5.3) 이식
4. `inspect_visual` 스타일 bbox tool-calling 루프(§6) + 프론트 오버레이 렌더링
5. (추후) 타일링 확장
