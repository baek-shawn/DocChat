# CLAUDE.md — 작업 지침

이 저장소에서 작업하는 Claude가 **매 세션 시작 시 가장 먼저 읽는** 지침이다.
무엇을 만드는지는 [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md), 어디까지 했는지는 [STEPS.md](STEPS.md)에 있다.
왜 그 방향인지(실측에서 본 것, 그 원인에 대한 이해, 아직 만들지 않은 구상)는 [IDEAS.md](IDEAS.md)에 있다 — 로드맵·도구 설계를 논의할 때 먼저 읽고, 논의에서 나온 이해·구상은 실측 / 이해 / 구상을 구분해 거기에 적는다.

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
웹검색 / web_fetch / 논문 검색, deepagents·langchain 멀티 서브에이전트, 문서 생성(PDF/DOCX/PPTX)·차트 도구, 정식 SSE·WebSocket.
→ 필요해 보여도 **먼저 사용자에게 묻는다.**
- 타일링은 원래 제외 항목이었으나 **2026-09-29 사용자 결정으로 해제**됐다(STEPS.md Step 5, 구현됨). 계획서에 없던 항목은 STEPS.md의 Step·Experiments에 적힌 범위까지만 한다.
  - 타일은 **전사(OCR)와 bbox 호출에만** 적용한다. 답변(추론) 호출에 **어떤 이미지를 실을지**는 별개의 축 `answerImageMode`(끔 / 업로드 이미지만 / 전체, Step 8 1차 · 자동 = 업로드 이미지만 + 보기 도구, Step 10 · 자동에서 따로 보기 도구는 요청 옵션 `analyzeTool`, Step 10 2차)로 고른다. 답변 호출의 이미지와 따로 보기 호출의 이미지는 어느 모드든 전체 한 장이다 — 타일 계열(타일 / 타일+전체 / 개요+확대)은 Step 8 나머지의 범위다.
  - 타일 경계에서 잘린 대상을 하나로 잇는 것(분할 박스 병합)은 범위 밖이다 — 겹침 영역의 **중복**만 합친다.

### 3.3 구조 원칙
- **단일 tool-calling 루프**만 둔다: 모델 호출 → tool_call 감지 → 실행 → 결과 재주입 → 반복. 프레임워크 금지.
- **bbox는 항상 분리된 별도 호출**(`inspect_visual`)로 받는다. 메인 답변과 한 번에 묶지 않는다. 그림을 읽어 답하는 길은 둘이고 **어느 쪽을 쓸지는 모델이 고른다**(Step 10): **같이 보기**(보기 도구 `view_page`)는 쪽 이미지를 답변 모델 자신의 다음 호출에 붙여 직접 보게 하고, **따로 보기**(`analyze_pages`, Step 10 2차)는 쪽마다 별도 VLM 호출로 모델이 넘긴 질문의 답 **글**을 받아 답변 모델이 종합한다(비교는 같이 보기, 쪽마다 독립이거나 후보가 많아 훑어야 하면 따로 보기, 따로 본 글로 애매하면 다시 같이 보기). 앱이 쪽을 골라 주거나 자동으로 훑지 않는다.
- 통신은 **동기 HTTP**가 기본. 진행률이 필요할 때만 `StreamingResponse`로 **NDJSON 한 줄씩** 흘린다(`data:` 접두사·`text/event-stream` 쓰지 않음).
- 모델 인터페이스는 하나: `analyze(messages, images=None, tools=None) -> ModelResponse`. 호출마다 달라지는 선택(`temperature`, `disable_thinking`, `max_tokens`, `reasoning_budget`, `on_reasoning`, `reasoning_effort`)은 키워드 인자로만 받는다. 서버↔모델 구간의 스트리밍(추론을 켠 로컬 호출만)은 provider 안에서 끝나고 호출 지점은 완성된 `ModelResponse`만 본다.
- "페이지 이미지 준비"(`pipeline/images.py`, `pipeline/pdf.py`)와 "VLM에 보낼 이미지 목록 조립"(`assemble_model_images`)을 **분리 유지**한다 — 전체/타일 모드가 갈리는 곳은 `assemble_model_images` 한 곳이다.
- **비교 실험을 위한 모드는 요청마다 고를 수 있게** 둔다(예: 전처리 전체/타일, 추론 호출 이미지 끔/전체/개요+확대). 기본값은 `config.py`, 각 답변에 어떤 모드로 처리했는지 기록하고, 캐시 키에 모드를 포함한다(모드를 바꿨는데 이전 결과가 재사용되면 비교가 무의미해진다).
- API key는 **요청마다 받아서 쓰고 디스크에 저장하지 않는다.** DB·로그에 남기지 않는다.

### 3.4 임계값은 `app/config.py` 한 곳에서만
설정값은 환경변수 또는 프로젝트 루트의 `.env`로 넣는다(셸 환경변수가 우선, `.env`는 git 제외). **새 설정을 추가하면 `.env.example`에도 기본값과 함께 적는다** — 빠뜨리면 `tests/test_env_file.py`가 실패한다. API key는 `.env`에 두지 않는다. 테스트는 `.env`를 읽지 않는다(`conftest.py`가 끈다).

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

타일링(Step 5) 설정 — vectra에 없는 기능이라 기본값은 이 프로젝트가 정했다. 바꾸면 `image_mode_variant()`(OCR 캐시 키)가 함께 바뀐다.

| 상수 | 값 | 의미 |
|---|---|---|
| `DEFAULT_IMAGE_MODE` | `whole` | 요청에 `imageMode`가 없을 때(`DOCCHAT_IMAGE_MODE`) |
| `TILE_SIZE` | 1536 | 타일 한 변의 상한(px) |
| `TILE_OVERLAP` | 0.125 | 이웃 타일과 겹치는 비율 |
| `TILE_RENDER_DPI` | 200 | PDF 타일 렌더 DPI(스캔 쪽은 박힌 이미지 해상도가 상한) |
| `TILE_MIN_SOURCE_EDGE` | 2048 | 원본 긴 변이 이 이하면 전체 모드와 동일 |
| `MAX_TILES_PER_IMAGE` | 48 | 넘으면 해상도를 낮춰 타일 수를 맞춘다 |
| `TILE_BLANK_PIXEL_RATIO` / `TILE_BLANK_TOLERANCE` | 0.00002 / 24 | 내용이 없는 타일 판정(보내지 않는다) |
| `TILE_DEDUPE_MIN_CHARS` | 6 | 겹침 중복으로 지울 수 있는 줄의 최소 길이 |
| `TILE_BOX_MERGE_IOU` / `_CONTAINMENT` | 0.5 / 0.8 | 서로 다른 타일의 박스를 같은 대상으로 보는 기준 |

전사·bbox 호출의 폭주 막기(Step 6-0) — 역시 이 프로젝트가 정한 값이다.

| 상수 | 값 | 의미 |
|---|---|---|
| `VISION_MAX_TOKENS` | 4096 | 전사·bbox 호출 한 번의 출력 토큰 상한(추론 포함). 0이면 보내지 않는다. **화면에 넣지 않는다**(`.env`로만) |
| `GROUNDING_DISABLE_THINKING` / `OCR_DISABLE_THINKING` | True / True | 요청에 값이 없을 때 bbox / 전사 호출의 추론을 끌지 |

- 출력 상한에 닿은(`finish_reason=length`) 전사·bbox 호출은 **다시 보내지 않는다.** bbox는 끊긴 글·미완성 JSON을 결과로 쓰지 않는다. 전사는 읽은 데까지 남기되 끊긴 자리를 표시하고, 끊긴 글이 추론일 수 있으면(`ocr.cut_off_transcription`) 버린다.
- 추론 끄기와 출력 상한은 **로컬(OpenAI 호환) provider에만** 보낸다. 답변 호출에는 출력 상한을 붙이지 않는다(Step 6의 범위).
- 전사 호출의 추론 여부는 `image_mode_variant(mode, thinking=…)`로 OCR 캐시 키와 쪽 기록에 들어간다.

추론 제어(Step 6 1차) — 추론을 **켠** 로컬 호출만 스트리밍으로 받으며 추론 부분을 지켜본다(`providers/reasoning.py`, `openai_compat.py`). 추론을 끈 호출(기본값의 bbox·전사)은 비스트리밍 경로 그대로다.

| 상수 | 값 | 의미 |
|---|---|---|
| `REASONING_BUDGET_ANSWER` / `_GROUNDING` / `_OCR` | 8,000 / 4,000 / 4,000 | 호출 종류별 추론 토큰 예산(0 = 없음). 추론을 켠 bbox·전사 호출의 `max_tokens` = `VISION_MAX_TOKENS` + 예산 |
| `REASONING_REPEAT_LINES` / `_COUNT` / `_MIN_CHARS` | 8 / 3 / 24 | 최대 8줄 묶음이 연달아 3번 같으면 반복(묶음이 24자 미만이면 제외) |
| `REASONING_CONTINUATION_ALLOWANCE` | 512 | 이어 쓰기 호출에서 모델이 다시 추론하면 이만큼만 두고 하드 중단 |

- 감지(예산 초과·반복)는 **추론 부분에만** 건다. 답 부분의 반복(표 전사, bbox JSON)은 정상이다 → `max_tokens`가 맡는다.
- 소프트: 스트림을 끊고 쓴 추론 + 종료 문장 + `</think>`를 assistant 메시지로 넣어 **같은 요청을 이어 쓰기**(`continue_final_message`, 스트리밍)로 다시 보낸다. 하드: 이어 쓰기도 걸리거나 빈 답이면 `finish_reason="reasoning_runaway"` — 답변은 안내문, 전사는 `[OCR FAILED …]`, bbox는 박스 없음 + 경고. **어느 쪽도 다시 보내지 않는다**(Step 6-0 규칙).
- 호출 지점은 `analyze(..., reasoning_budget=config.reasoning_budget(kind), on_reasoning=…)`만 넘긴다. 감지·조치 코드를 호출 지점에 두지 않는다. 가짜 provider의 `analyze`는 이 두 키워드 인자를 받아야 한다(2차에서 `reasoning_effort`가 더해져 셋 — 아래).
- 추론을 켠 전사의 캐시 키·쪽 기록에 예산이 들어간다(`whole+thinking:b4000`). 예산을 바꾸면 다시 전사한다.
- 진행 문구 중 1초마다 갱신되는 것("추론 중… n토큰")은 `progress(message, live=True)`로 보내 화면이 같은 줄을 바꿔 쓰고, 트레이스에는 적지 않는다(호출 이벤트가 토큰 수를 갖는다).

추론 수준(Step 6 2차-a) — 모델의 채팅 템플릿이 받는 `reasoning_effort`를 호출 종류별로 싣는다. 요청의 `reasoningEffortAnswer` / `reasoningEffortGrounding` / `reasoningEffortOcr`, 없으면 아래 기본값.

| 상수 | 값 | 의미 |
|---|---|---|
| `REASONING_EFFORT_ANSWER` / `_GROUNDING` / `_OCR` | 빈 값 | 요청에 값이 없을 때 실어 보낼 추론 수준. 빈 값 = 보내지 않는다(모델 기본 수준 — Qwen3.8은 `xhigh`) (`DOCCHAT_REASONING_EFFORT_*`) |

- 값은 모델마다 다르다(Qwen3.8: `low`/`medium`/`xhigh`, `high` 없음) → **목록으로 묶지 않고 모양만 검사한다**(`config.normalize_reasoning_effort`). 화면은 고른 값을 모델 이름별로 저장한다.
- **추론을 켠 로컬 호출에만** `chat_template_kwargs.reasoning_effort`로 실린다. 추론을 끈 호출의 요청은 이전과 같아야 한다. 이어 쓰기(소프트) 요청도 같은 값을 싣는다. 요청 최상위 `reasoning_effort`와 클라우드 provider는 연결하지 않았다(확인할 서버·키가 없다).
- 어느 호출에 실을지는 `chat_service.plan_effort` **한 곳**에서 정한다(추론을 끄는 호출 종류·수준을 보내지 않는 provider는 비운다). 호출 지점은 `analyze(..., reasoning_effort=…)`만 넘긴다. 가짜 provider의 `analyze`는 `reasoning_budget`·`on_reasoning`·`reasoning_effort` 세 키워드 인자를 받아야 한다.
- **서버가 값을 받지 않으면 `ReasoningEffortError`로 턴을 끝낸다. 값을 빼고 다시 보내지 않는다** — 빼고 보내면 모델 기본 수준으로 돌고 답에는 요청한 수준이 적혀 실험 기록이 틀린다. 이 오류는 전사 재시도(`ocr._transcribe`), 도구 오류(`loop`), 타일의 일부 실패(`tools._inspect_tiles`)에 묻히면 안 된다 — 넓게 잡는 `except`를 새로 넣을 때 이 예외를 먼저 다시 던진다.
- 원인이 수준인지는 **서버의 오류 문구로 판단하지 않는다**. 거절된 뒤에만 1토큰 확인 요청 둘(수준을 실은 것 / 뺀 것)을 보내 가린다(`openai_compat._effort_is_refused`). 이 판정은 "도구 미지원 → JSON 폴백"(400이면 무엇이든 해당)보다 먼저다.
- "받았다 ≠ 적용됐다": 수준이 없는 모델(Qwen3.5)은 값을 거절하지 않고 무시한다. 답변의 `meta.reasoningEffort`는 "실어 보냈고 거절되지 않았다"는 뜻이다. 효과는 추론 토큰 수로 본다.
- 추론을 켠 전사의 캐시 키·쪽 기록에 수준이 들어간다(`whole+thinking:b4000:elow`). 수준이 없으면 1차의 기록과 같은 값이다.

답변(추론) 호출의 이미지(Step 8 1차) — 요청의 `answerImageMode`, 없으면 아래 기본값.

| 상수 | 값 | 의미 |
|---|---|---|
| `DEFAULT_ANSWER_IMAGE_MODE` | `uploads` | `off`(이미지 없음) / `uploads`(업로드 이미지만 — 계획서 §5.3·§5.4, Step 8 이전과 같은 동작) / `whole`(업로드 이미지 + PDF 모든 쪽) / `auto`(업로드 이미지만 + 보기 도구, Step 10 · 따로 보기 도구는 요청 옵션, Step 10 2차) (`DOCCHAT_ANSWER_IMAGE_MODE`) |
| `MAX_MODEL_IMAGES` | 12 | 답변 호출 한 번에 싣는 이미지 수 상한. `whole`에서 넘치면 쪽 순서로 앞에서부터 |

- 어떤 첨부를 실을지는 `evidence.attachment_context_for_prompt(answer_images=…)` **한 곳**에서 정한다. 아직 렌더하지 않은 PDF 쪽은 자리표시(`pending_page_image`)로 나오고 `chat_service._render_pending_pages`가 실을 것만 그려 첨부로 저장한다(bbox 도구의 즉석 렌더와 같은 `preprocess.render_page_attachment`).
- 쪽 이미지(`send_to_model=False`)는 어느 모드에서도 스스로 답변 호출에 실리지 않는다 — `send_to_model=True`는 업로드 이미지에만 있다. 렌더해 둔 쪽이 기본 모드로 새면 안 된다(`tests/test_answer_images.py`).
- 쪽 이미지가 실릴 때만 질문 끝에 `[PAGE IMAGES: …]` 한 줄이 붙는다. 기본 모드의 프롬프트는 바꾸지 않는다. 이미지 토큰은 예산에 넣지 않는다(모델마다 달라 추정하지 않음).

필요할 때만 그림을 보기(Step 10) — 답변 이미지 모드 `auto`에서만 보기 도구 `view_page`를 내놓는다.

| 상수 | 값 | 의미 |
|---|---|---|
| `MAX_VIEWED_PAGES` | 12 | 보기 도구로 한 턴에 모을 수 있는 쪽 수. `MAX_MODEL_IMAGES`(전체 모드가 처음부터 싣는 수)와 **별개** (`DOCCHAT_MAX_VIEWED_PAGES`) |
| `DRAWING_MIN_RASTER_AREA` / `DRAWING_MIN_VECTOR_OPERATIONS` | 0.02 / 100 | 매니페스트에 "그림이 있는 쪽"으로 적는 기준 — 래스터가 쪽 넓이에서 차지하는 비율(개수가 아니다: 로고 0.001~0.002, 본문 그림 0.03 이상) / 벡터 경로 연산 수(표 테두리 1~44, 데이터시트 66, CAD 도면 수천). `vector-outlines`·`scanned-raster` 쪽은 무조건 그림 |

- **역할 분담**: `view_page`는 답을 내려면 그림을 봐야 할 때(형상·배치·개수·어느 대상의 치수인지·여러 쪽 비교), `inspect_visual`(bbox)은 **보여 달라고 할 때만**("표시해줘", "박스로", "시각화해줘"). "어디 있어?"처럼 말로 답하면 되는 질문은 보기 도구다. 두 도구의 문구를 바꾸는 것은 프롬프트 변경이다(통제 비교, STEPS.md Step 10 실험 ②).
- 보기 도구는 **한 번에 한 쪽**을 받고 모델 호출을 하지 않는다. 요청한 쪽은 `ToolContext.viewed`에 쌓이고, 루프가 **다음 답변 호출부터** `images` 뒤에 붙여 보낸다(`run_tool_loop(viewed_images=…)`). 이미지는 provider마다 **원래 질문(닻)** 에 붙으므로 그 질문 끝에 `[IMAGES ATTACHED TO THIS MESSAGE, in this order - …]` 한 줄을 호출마다 다시 만들어 붙인다(`prompts.attached_images_note`). `analyze()`는 바꾸지 않는다. 5단계 "첨부를 볼 수 없다" 재요청에도 같은 이미지를 보낸다.
- 상한을 넘으면 그 요청은 실행하지 않고 "더 볼 수 없다, 지금 보는 쪽으로 답하라"를 돌려준다(`view_refusals`에 세어 `meta.viewedPages.refused`). 같은 쪽·이미 실린 업로드 이미지를 다시 요청하면 더하지 않고 자리만 알려 준다. 쪽을 하나씩 부르면 `MAX_TOOL_STEPS`(8)에 먼저 걸린다 — 한 응답에 여러 호출을 담을 수 있다고 도구 설명에 적어 두었다.
- 이전 턴에서 본 쪽은 다음 턴에 자동으로 실리지 않는다(모델이 다시 요청하면 렌더해 둔 첨부를 다시 쓴다). 보기 도구로 그린 쪽은 bbox 도구처럼 첨부로 저장된다(`_resolve_visual_surface` 공유).
- **판단 재료**: `auto`에서만 매니페스트에 **파일 검사 결과** 한 줄을 덧붙인다(`evidence.drawing_cue`, v2 2026-10-07): `file check (embedded images and vector strokes in the file, not a visual inspection): 37 of 53 pages contain them - …; 16 pages have none detected - …; pages 4, 37, 40, 51 have no text layer, so their text is only a transcription of the visible labels. What is drawn … is NOT in the text of any page`. 전부 스캔본이면 그렇게, 아무것도 안 잡혀도 "none detected - drawings stored in another way would not be detected"라고 적는다(검사는 파일 안의 이미지·선 명령을 **센** 것이라 보안 설정으로 글만 뽑히는 CAD PDF는 놓친다 — 그 한계도 사실로 보여 주고 전부 훑을지는 모델이 정한다). **도구를 가리키지 않는다**(도구 문단이 맡는다). v1("drawings on pages 4, 37, 40, 51 (transcription …); 1, 6-15 … (native text …) … call view_page")은 Qwen3.5가 앞 묶음만 그림 쪽으로 읽어 33쪽을 후보에서 빠뜨렸다(STEPS.md "매니페스트 v2"). 재료는 `PageAnalysis.to_public()`(래스터 면적 비율 포함)을 PDF 첨부의 메타데이터 `pageAnalysis`에 남긴 것이고, 그게 없는 옛 첨부는 본문의 `[PAGE ANALYSIS]` 줄에서 읽는다(면적을 몰라 개수로 대신). 다른 모드의 매니페스트·시스템 프롬프트는 Step 8까지와 같아야 한다(실험 ①의 비교 기준 — `tests/test_view_tool.py`가 지킨다).
- 메타 `viewedPages{names, limit, refused}`는 `auto`에서만 적는다(한 쪽도 안 봤어도 — "안 봤다"도 결과다). 화면은 `그림 확인: a.pdf 2쪽, 3쪽` / `그림 확인 없음`.

따로 보기(Step 10 2차) — `auto`에서 요청 옵션 `analyzeTool`(없으면 아래 기본값)이 켜져 있을 때만 `analyze_pages`를 더 내놓는다. 모드는 늘리지 않았다 — 끄면 1차의 자동 모드(같이 보기만)와 도구 목록·시스템 프롬프트가 같아야 한다(실험 ③의 비교 기준 — `tests/test_analyze_tool.py`가 문단 하나 차이로 지킨다).

| 상수 | 값 | 의미 |
|---|---|---|
| `ANALYZE_TOOL` | True | 요청에 `analyzeTool`이 없을 때 따로 보기 도구를 내놓을지 (`DOCCHAT_ANALYZE_TOOL`) |
| `ANALYZE_PAGES_PER_CALL` | 10 | 한 호출이 받는 쪽 수(끊어 보기 단위). 넘치면 **앞에서부터 보고** 다음 범위를 알려 준다 (`DOCCHAT_ANALYZE_PAGES_PER_CALL`) |
| `MAX_ANALYZED_PAGES` | 60 | 한 턴에 따로 보는 총 쪽 수 — **시간 상한**이다(쪽마다 호출이 실제로 돈다). `MAX_VIEWED_PAGES`(답변 호출의 입력 크기 상한)와 성격이 다르다. 전사 상한과 같은 선 (`DOCCHAT_MAX_ANALYZED_PAGES`) |

- 도구는 첨부 이름 · 쪽(`"7"` / `"1-10"` / `"2,5,7-9"`, 이미지 첨부·한 쪽짜리 PDF면 생략) · **질문**(결과를 가장 크게 좌우한다 — 도구 설명에 "구체적으로")을 받아, 쪽마다 `prompts.PAGE_ANALYSIS_SYSTEM_PROMPT`(옮겨 적기가 아니라 질문에 답하기, 보이는 것만, 없으면 없다고)로 **별도 호출**을 돌리고 `[page n] 답`으로 모아 **글만** 돌려준다. 답변 호출에 이미지는 붙지 않는다. 추론 끄기·출력 상한·추론 예산·수준은 **bbox 호출과 같은 축**(`grounding`), temperature 0, 동시 `OCR_CONCURRENCY`개, 트레이스에서는 도구 이벤트의 자식 "따로 보기 호출 · a.pdf · page n"(`TracedProvider._kind`의 `analysis`).
- 재시도는 없다(답의 모양을 검사할 기준이 없다). 상한에 닿은 호출은 읽은 데까지 남기고(추론일 수 있으면 버림, `ocr.cut_off_answer`를 전사와 공유), 추론 중단·쪽별 예외는 그 쪽의 글로 표시한다 — **다시 보내지 않는다**(Step 6-0). 전부 실패하면 도구 오류. 같은 턴·같은 쪽·같은 질문은 다시 묻지 않고(`ToolContext.analysis_cache`) 상한에도 세지 않는다. 턴 상한에 닿으면 남은 만큼만 보고 알려 주고, 0이면 실행하지 않고 거절한다(`analysis_refusals`).
- 쪽을 고르는 것도, 같이 보기 / 따로 보기 / 끊어서 전부 훑기를 고르는 것도 **모델**이다. 시스템 프롬프트 문단(켰을 때만)의 핵심 두 문장: "좁혀지지 않거나 전체를 봐야 답이 되면 후보 일부로 단정하지 말고 끊어서 전부 훑어라", "따로 본 글만으로 애매하면 `view_page`로 그 쪽을 직접 보고 같이 판단하라". 전사 호출에 쪽 종류 판정을 섞지 않는다(사용자 결정). 업로드 이미지는 그대로 처음부터 실린다.
- 메타 `analyzedPages{enabled, names, calls, limit, refused}`는 `auto`에서 항상 적는다(`enabled`가 실험 조건이다). `vision.analysisCalls`·`analysisLengthStops`·`analysisReasoningForced`·`analysisReasoningStops`. 화면은 `따로 보기: a.pdf 1-10쪽 (10쪽, 호출 1번)` / `따로 보기 없음` / `따로 보기 꺼짐`. `/api/health.analyze`.

개발용 턴 트레이스(Step 7) — `DOCCHAT_DEBUG_TRACE=1`일 때만 기록한다(`config.debug_trace_enabled()`, 요청마다 읽는다).

| 상수 | 값 | 의미 |
|---|---|---|
| `TRACE_TEXT_LIMIT` | 40,000 | 트레이스에 남기는 글 한 조각(메시지·응답·도구 결과)의 최대 글자 수. 넘치면 앞·뒤를 남기고 자른다 |

- 기록 지점은 `trace.note()`(한 줄)와 `trace.scope()`(시작·끝이 있는 구간, 그 안의 모델 호출은 자식이 된다)만 부른다. 트레이스가 없으면 아무 일도 하지 않으므로 **기록 지점을 조건문으로 감싸지 않는다.** 데이터 키는 `kind`·`label`과 겹쳐도 된다(위치 전용 인자).
- 모델 호출은 `providers/traced.py`의 `TracedProvider`가 기록한다(`chat_service`가 트레이스를 켠 턴에만 감싼다). 새 호출 지점에 따로 기록 코드를 넣지 않는다.
- **기록이 앱의 동작을 바꾸면 안 된다.** `TurnTrace`의 메서드는 예외를 삼키고, 안쪽 provider의 예외는 그대로 다시 던진다. 새 기록 지점을 넣을 때도 예외를 낼 수 있는 계산은 기록 인자 안에서 하지 않는다.
- 이미지는 첨부 ID·타일 위치로만 가리킨다(base64 저장 금지). API key는 어디에도 적지 않는다(`TracedProvider`는 key를 갖지 않는다).
- 답변의 `meta.traceId`가 트레이스를 가리킨다. 메시지 id는 저장할 때마다 새로 매겨져 키로 쓸 수 없다.

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
uv run pytest                            # 전체 테스트(모델 불필요, 약 30초)
uv run python scripts/make_samples.py    # samples/ 에 시험용 PDF·이미지 생성
uv run python scripts/e2e_check.py       # 실제 모델(기본 Ollama gemma3)로 종단 점검(전체 모드 11항목 + 타일 모드 6항목)
uv run python scripts/compare_tiling.py samples/large_scanned_plan.pdf --expect-file samples/large_plan.expect.txt --repeat 3
                                         # 전체 vs 타일 비교(전사 글·bbox IoU·시간·호출 수). 사용법은 README 3.3
uv run python scripts/check_runaway.py --image plan.png --task "Find every door symbol." --conditions A:2,B:2,C
                                         # 추론형 모델로 bbox·전사 호출의 폭주 확인(호출별 종료 사유·토큰·시간). 사용법은 README 3.4
```

- 한 턴이 왜 그렇게 답했는지·왜 느렸는지는 `.env`에 `DOCCHAT_DEBUG_TRACE=1`을 켜고 답변의 **"과정 보기"**(또는 `GET /api/traces/{id}`)로 본다(Step 7). 재현 스크립트를 따로 짜기 전에 이것부터 본다. `docchat-scratch`는 켠 채 뜬다.
- 추론 모델의 폭주(예산·반복)는 `scripts/check_runaway.py`로 실제 서버에서 확인한다(README 3.4). 조건 `SOFT`/`FULL`이 답변 호출의 소프트 조치, `B`/`E`가 bbox·전사의 조치를 적는다. `LEVELS`는 추론 수준이 모델에 닿는지를 입력 토큰 수로 본다(출력 1토큰), `--effort`는 다른 조건의 추론을 켠 호출에 수준을 싣는다. 추론 모델은 사용자 vLLM(Qwen3.5 · Qwen3.8)뿐이라 **요청받았을 때만** 보낸다.

- 브라우저로 UI를 확인할 때는 `.claude/launch.json`의 **`docchat-scratch`**(포트 8765, DB는 `%TEMP%\docchat-scratch`)를 쓴다. 기본 `docchat` 구성은 사용자의 실제 `data/`를 쓴다.
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
- 첨부 바이트 저장: **새 첨부는 `data/files/{대화ID}/` 아래 파일**로 쓰고 DB에는 `data/files` 기준 **상대 경로만** 둔다(`app/storage.py`, Step 5-0). 원칙:
  - 기존 BLOB은 옮기지 않고 그대로 읽는다("경로가 있으면 파일, 없으면 BLOB"). 새로 BLOB에 쓰지 않는다.
  - 경로는 항상 `data/files` 안으로 정규화해 검사한다(`..`·절대 경로로 폴더 밖 접근 금지). 파일을 읽고 쓰는 코드는 `FileStore`를 거친다 — 경로를 직접 이어 붙여 `open()`하지 않는다.
  - 대화를 지우면 그 대화 폴더도 함께 지운다. 그 외에는 `data/` 파일을 지우지 않는다 → **파일은 덮어쓰지 않는다**(같은 이름이면 ` (2)`를 붙여 새로 쓴다).
  - 업로드 이미지는 **원본을 보관**한다(타일링 재료). 3072px로 줄인 사본으로 덮어쓰지 않는다 — `Attachment.data`는 사본, `source_data`/`source_path`가 원본이다.
  - 디스크 I/O는 `asyncio.to_thread`로 넘긴다.
- 프론트에는 **base64를 되돌려 보내지 않는다.** 이미지는 `/api/attachments/{id}/content` URL로 참조한다.
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
- **측정 도구가 앱의 동작을 바꾸면 안 된다.** 2026-09-29에 확인 스크립트의 기록용 코드가 예외를 던져, 앱이 멀쩡한 타일 3장을 "실패"로 보고 3번씩 다시 보냈다(측정 1회를 버렸다). provider를 감싸 기록할 때는 기록 부분의 예외를 삼키고, 호출 수가 예상과 다르면 앱보다 도구를 먼저 의심한다.
- 모델이 낸 **도구 호출 JSON으로 보이는 글은 절대 사용자에게 그대로 내보내지 않는다**(복구 → 재요청 → 도구 없이 최종 답 → 안내문).
- 전사 결과를 "정리"할 때는 **잡음만** 지운다. 문서 내용일 수 있는 것(같은 값이 몇 줄 이어지는 표, `<html>` 같은 태그)은 건드리지 않는다.

## 6. 디렉터리 지도

```
app/
  main.py            FastAPI 앱 조립, 정적 파일, Origin 검사
  config.py          모든 임계값/환경변수
  trace.py           개발용 턴 트레이스(Step 7) — 이벤트 기록·컨텍스트 변수·DB 저장 시점. config만 가져온다
  db.py              aiosqlite 세션·메시지·첨부·턴 트레이스 저장소(첨부 바이트는 storage를 통해 파일로)
  storage.py         FileStore — data/files 아래 파일 쓰기·읽기, 경로 검사, 대화 폴더 삭제
  attachments.py     Attachment 모델, 업로드 정제
  api/               sessions / models / files / chat / traces 라우터
  providers/         analyze() 인터페이스와 OpenAI호환(추론 수준 싣기·거절 판별 포함)·Anthropic·Gemini 구현, reasoning(추론 예산·반복 감지, Step 6),
                     traced(트레이스용 겉싸개)
  chat_service.py    /api/chat 한 턴의 전체 흐름(전처리→OCR→증거 선택→루프→저장)
  pipeline/          geometry(크기 한도·타일 분할 계산) / pdf(판별·렌더·타일 렌더) / images(업로드 준비·타일 자르기·이미지 조립)
                     / preprocess(업로드 펼치기, 쪽 즉석 렌더) / ocr(전사·타일 전사 병합) / evidence(증거 예산·답변 호출 이미지 선택)
  agent/             prompts / tools(bbox·보기·따로 보기·읽기·검색 도구) / grounding(bbox 파싱·타일 박스 병합) / loop(단일 tool-calling 루프, 본 쪽 싣기)
static/              index.html, styles.css, js/app.js
tests/               pytest (mock_openai.py = OpenAI 호환 mock 서버, pdf_factory.py = 합성 PDF·큰 도면)
scripts/             make_samples.py(샘플 생성), e2e_check.py(실모델 점검), compare_tiling.py(전체 vs 타일 비교),
                     check_runaway.py(추론형 모델의 bbox·전사 폭주 확인)
data/                SQLite DB (git 제외, 사용자 데이터)
  files/             대화별 첨부 파일 — 원본·모델 전송용 사본·페이지 렌더·(트레이스 켬) 모델에 보낸 타일
samples/             생성된 시험 문서, compare/ 아래에 비교 결과 (git 제외)
```

import 방향(순환 금지): `config` ← `attachments` ← `storage` ← `db`, `config` ← `attachments` ← `pipeline/*` ← `providers/*` ← `agent/*` ← `chat_service` ← `api/*` ← `main`.
`trace`는 `config` 바로 옆이다(`config` ← `trace`): `pipeline/*`·`agent/*`·`providers/traced`·`chat_service`가 가져오되, `trace`는 그 어느 것도 가져오지 않는다(이미지·provider 객체는 속성 이름으로만 읽는다).
`pipeline` 안에서는 `geometry` ← `pdf` ← `images` 순이다(`images`가 타일 렌더를 부르므로 `pdf`는 `images`를 가져오지 않는다).
`pipeline/ocr.py`만 예외적으로 `agent/prompts`와 `providers/base`를 가져온다. 각 패키지의 `__init__.py`는 비워 둔다(`providers`만 팩토리 제공).

## 7. git

브랜치·커밋·태그 규칙은 [GIT_WORKFLOW.md](GIT_WORKFLOW.md)에 있다. 커밋과 push는 **사용자가 직접** 한다 — Claude는 요청받았을 때만 하고, 그 전에는 커밋 메시지 초안만 제안한다.

## 8. 하지 말 것

- 계획서에 없는 큰 기능을 임의로 추가하지 않는다(작은 편의 기능은 STEPS.md "변경점"에 기록).
- 테스트를 통과시키려고 임계값이나 단언을 느슨하게 바꾸지 않는다.
- `data/` 안의 DB·업로드 파일을 커밋하거나 지우지 않는다(사용자 데이터).
- 실패한 테스트·미검증 항목을 "완료"로 표시하지 않는다.
