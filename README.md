# DocChat — 로컬/클라우드 VLM 문서 분석 챗

PDF·이미지를 올리고 질문하면, **텍스트가 충분한 쪽은 그대로 읽고 부족한 쪽만 비전 모델로 전사**해서 답합니다.
"위치를 표시해 줘" 같은 요청에는 **분리된 bbox 호출(`inspect_visual`)** 로 영역을 재서 이미지 위에 오버레이합니다.
큰 도면은 **전체 한 장**으로 보낼지 **겹치는 타일**로 나눠 보낼지 요청마다 고를 수 있어, 같은 문서로 두 방식을 비교할 수 있습니다.

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
5. 큰 도면이면 **⚙ 모델 설정 → 이미지 처리**를 `타일`로 바꿔 같은 질문을 다시 해 보세요. 답변 아래에 `이미지 처리: 타일 · 타일 20장 · 전사 호출 20회 · 95.3초`처럼 어떤 방식으로 처리했는지 표시됩니다.

옵션: `uv run python run.py --port 9000 --reload`

### 이미지 처리 방식: 전체 / 타일

| | 전체(`whole`, 기본) | 타일(`tile`) |
|---|---|---|
| 모델에 보내는 것 | 쪽·이미지 한 장(긴 변 3072px·800만 픽셀로 축소) | 원본을 겹치는 조각으로 나눠 조각마다 한 장(조각당 최대 1536px) |
| A1 도면(200 DPI)의 글자 | 6622px → 3072px로 줄어 작은 글자가 뭉개짐 | 줄이지 않음 |
| 호출 수 | 쪽당 1회 | 쪽당 타일 수만큼(A1 20회, A0 35회). 내용이 없는 타일은 보내지 않음 |
| 적용 대상 | — | **전사(OCR)와 bbox(`inspect_visual`) 호출만.** 답변 호출의 이미지는 어느 방식이든 전체 한 장 |

- PDF는 쪽 전체를 고해상도로 만든 뒤 자르지 않고 **타일 영역만 바로 렌더**합니다. 스캔 PDF는 박힌 이미지의 원래 해상도까지만 올립니다.
- 업로드 이미지는 보관해 둔 **원본**에서 자릅니다(모델 전송용 3072px 사본이 아니라).
- 원본의 긴 변이 2048px 이하면 타일로 골라도 전체와 똑같이 처리합니다.
- 같은 대화에서 방식을 바꾸면 이미 전사한 쪽도 **새 방식으로 다시 전사**합니다(이전 결과가 섞이면 비교가 무의미하므로).
- 어느 쪽이 나은지는 문서와 모델에 따라 다릅니다. 본문이 쪽 너비를 가득 채우는 문서는 타일 경계에서 줄이 잘려 오히려 나빠질 수 있습니다 → `scripts/compare_tiling.py`로 재 보세요(3.3절).

### 추론(thinking) 끄기 — 호출 종류별로

⚙ 모델 설정(로컬 서버일 때)에 세 가지 선택이 있습니다.

| 선택 | 기본 | 적용되는 호출 |
|---|---|---|
| 추론 끄기 — 모든 호출 | 켜짐 | 답변 · 위치 확인(bbox) · 전사(OCR) 전부 |
| 위치 확인(bbox) 호출은 추론 끄기 | 켜짐 | `inspect_visual` 도구 안의 호출만 |
| 전사(OCR) 호출은 추론 끄기 | 켜짐 | 스캔 쪽을 받아쓰는 호출만 |

- "모든 호출"이 켜져 있으면 아래 둘은 적용되지 않습니다(전부 꺼짐). **답변만 추론을 쓰게 하려면** "모든 호출"을 끄고 아래 둘은 켜 둡니다.
- 위치 확인·전사 호출에는 **출력 상한**이 붙습니다(기본 4,096토큰, 추론 토큰 포함). 상한은 화면이 아니라 `.env`의 `DOCCHAT_VISION_MAX_TOKENS`로 바꿉니다. 이 둘에 추론을 켜려면 8,000 이상을 권합니다 — 추론만으로 타일당 1,700~7,000토큰을 씁니다(Qwen3.5 실측).
- 상한에 닿아 끊긴 호출은 **다시 보내지 않습니다.** 위치 확인은 그 타일을 재지 못한 것으로 남기고(끊긴 글·미완성 JSON은 쓰지 않음), 전사는 읽은 데까지 남기고 끊긴 자리에 `[TRANSCRIPTION INCOMPLETE …]`를 적습니다. 끊긴 글이 추론일 수 있으면 버립니다.
- 답변 아래에 `위치 확인 호출 6회 (추론 끄지 않음, 출력 상한 4,096토큰 도달 4회)`처럼 표시됩니다.
- 같은 대화에서 전사의 추론 선택을 바꾸면 이미 전사한 쪽을 **다시 전사**합니다(비교가 섞이지 않게).
- 추론 끄기와 출력 상한은 로컬(OpenAI 호환) 서버에만 보냅니다. 클라우드 provider에는 적용되지 않습니다.
- 답변 호출에는 출력 상한이 없습니다. 답변 호출의 추론이 길어지는 문제는 아직 남아 있습니다(STEPS.md Step 6).

### 턴 과정 보기 — 개발용 트레이스

도면을 왜 잘/못 읽었는지, 어떤 정보를 어디서 가져왔는지, 왜 느렸는지를 **한 턴 단위로** 볼 수 있습니다. `.env`에 `DOCCHAT_DEBUG_TRACE=1`을 적고 서버를 다시 띄우면 켜집니다(기본은 꺼짐, 꺼져 있으면 아무것도 기록하지 않습니다).

- 답변마다 **⏱ 과정 보기** 버튼이 생깁니다. 누르면 타임라인이 열립니다: 입력 → 전처리(쪽마다 판별 결과·글자 수·임계값) → 전사(쪽·타일마다 시도·시간·원문) → 증거 조립(실은 텍스트·잘림·예산·보낸 이미지) → 모델 호출마다(보낸 메시지, 제공한 도구, 응답 원문, 추론 글, 도구 호출, 종료 사유, 토큰 수, 시간) → 도구 실행마다(인자·결과, 그 안의 비전 호출) → 최종 답.
- **진행 중인 턴도 열립니다.** 모델 호출·도구 실행은 시작할 때 먼저 기록되므로, 끝나지 않은 호출이 "진행 중 · 경과 n초"로 보입니다(어느 타일의 어느 호출이 멈춰 있는지). 중지하면 그 호출에 "취소"와 사유가 남습니다. 서버가 다시 시작돼 끊긴 턴은 "중단됨"으로 정리됩니다. 중지·오류로 답변이 저장되지 않은 턴은 질문 아래에 "과정 보기 (취소된 턴)"으로 붙습니다.
- 행을 누르면 자세히 보입니다. 보낸 이미지는 첨부 ID로 가리켜 "보기"로 뷰어에서 열 수 있습니다(타일은 쪽·이미지 전체가 열립니다. 타일 파일 자체는 `data/files/{대화ID}/tiles/`).
- **JSON 내려받기**로 실행 간 비교·실험 기록에 쓸 수 있습니다(`GET /api/traces/{id}?download=1`).
- 기록은 `turn_traces` 테이블에 따로 저장되고(대화를 지우면 함께 지워짐), 답변의 `meta.traceId`가 가리킵니다. API key와 이미지 바이트는 기록하지 않습니다. 글은 조각당 `DOCCHAT_TRACE_TEXT_LIMIT`(기본 40,000자)까지만 남깁니다.
- 브라우저 확인용 서버(`docchat-scratch`)는 트레이스를 켠 채 뜹니다.

### 설정: `.env` 파일 또는 환경변수

`.env.example`을 `.env`로 복사한 뒤, 바꿀 줄의 `#`를 지우고 값을 고칩니다. 서버를 다시 띄우면 적용됩니다.

```powershell
Copy-Item .env.example .env
```

- 셸에서 준 환경변수(`$env:DOCCHAT_TILE_SIZE = "1024"`)가 `.env`보다 **우선**합니다. 한 번만 다른 값으로 돌려 볼 때 편합니다.
- 서버를 띄우면 첫머리에 어떤 설정 파일을 읽었는지 표시됩니다. 다른 파일을 쓰려면 `DOCCHAT_ENV_FILE`에 경로를, 읽지 않으려면 `off`를 줍니다.
- `e2e_check.py`, `compare_tiling.py`도 같은 `.env`를 읽습니다(`DOCCHAT_E2E_*`로 대상 모델 지정). 두 스크립트는 `.env`에 실제 DB 위치가 적혀 있어도 임시 폴더만 씁니다.
- **API key는 `.env`에 넣지 않습니다.** 설정 화면에서 입력합니다.
- 전체 목록과 기본값은 `.env.example`에 있습니다. 아래는 자주 쓰는 것입니다.

| 변수 | 기본 | 의미 |
|---|---|---|
| `DOCCHAT_HOST` / `DOCCHAT_PORT` | `127.0.0.1` / `8000` | 바인딩 주소(인증이 없으니 localhost 권장) |
| `DOCCHAT_DB_PATH` | `data/docchat.sqlite` | 대화·메시지 저장 위치 |
| `DOCCHAT_FILES_DIR` | DB 파일 옆의 `files/` (= `data/files`) | 첨부 파일(PDF·이미지 원본·페이지 렌더) 저장 폴더. DB와 짝으로 옮겨야 합니다 |
| `DOCCHAT_IMAGE_MODE` | `whole` | 요청에 `imageMode`가 없을 때의 이미지 처리 방식(`whole` / `tile`) |
| `DOCCHAT_TILE_SIZE` | `1536` | 타일 한 변의 상한(px) |
| `DOCCHAT_TILE_OVERLAP` | `0.125` | 이웃 타일과 겹치는 비율(타일 크기 대비) |
| `DOCCHAT_TILE_RENDER_DPI` | `200` | PDF 타일 렌더 DPI(스캔 쪽은 박힌 이미지 해상도가 상한) |
| `DOCCHAT_TILE_MIN_SOURCE_EDGE` | `2048` | 원본 긴 변이 이 이하면 타일로 나누지 않음 |
| `DOCCHAT_MAX_TILES_PER_IMAGE` | `48` | 한 장의 타일 수 상한(넘으면 해상도를 낮춰 맞춤) |
| `DOCCHAT_DEBUG_TRACE` | 꺼짐 | `1`이면 턴 과정을 기록해 답변마다 "과정 보기"(위 절), 타일 모드에서는 모델에 보낸 타일도 `data/files/{대화ID}/tiles/`에 남김 |
| `DOCCHAT_TRACE_TEXT_LIMIT` | `40000` | 트레이스에 남기는 글 한 조각(메시지·응답·도구 결과)의 최대 글자 수 |
| `DOCCHAT_NATIVE_MIN_CHARS` | `24` | 이 글자 수 이상이면 네이티브 텍스트 사용 |
| `DOCCHAT_SPARSE_OVERLAY_CHARS` | `120` | 래스터가 있는데 이 미만이면 비전 전사 |
| `DOCCHAT_PDF_RENDER_DPI` | `200` | 페이지 렌더 DPI |
| `DOCCHAT_MAX_VISION_IMAGE_EDGE` / `_PIXELS` | `3072` / `8000000` | 이미지 크기 상한 |
| `DOCCHAT_OCR_RETRY_COUNT` / `DOCCHAT_OCR_CONCURRENCY` | `3` / `2` | 전사 재시도 / 동시 처리 |
| `DOCCHAT_GROUNDING_RETRY_COUNT` | `2` | bbox JSON 재시도 |
| `DOCCHAT_VISION_MAX_TOKENS` | `4096` | 위치 확인·전사 호출 한 번의 출력 토큰 상한(추론 포함). `0`이면 상한 없음. 닿은 호출은 다시 보내지 않음 |
| `DOCCHAT_GROUNDING_DISABLE_THINKING` / `DOCCHAT_OCR_DISABLE_THINKING` | `1` / `1` | 요청에 값이 없을 때 위치 확인 / 전사 호출의 추론을 끌지(설정 화면에서 고르면 그 값이 우선) |
| `DOCCHAT_MAX_PDF_VISUAL_PAGES` | `60` (최대 200) | 검사할 최대 페이지 |
| `DOCCHAT_MAX_TOOL_STEPS` | `8` | 한 턴의 최대 도구 호출 횟수 |

> **CAD PDF 팁**: SHX 폰트로 그린 글자는 텍스트 객체가 아니라 선이라서 `native characters`가 낮게 잡히고 비전 전사로 넘어갑니다. 반대로 글자는 적지만 네이티브 텍스트로 충분한 도면이라면 `DOCCHAT_NATIVE_MIN_CHARS`를 낮추세요.

## 3. 테스트

### 3.1 자동 테스트 (모델 불필요, 약 35초, 267개)

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
| `tests/test_file_storage.py` | 첨부를 파일로 저장·DB에는 상대 경로만, 업로드 원본 보관, 재시작 후 읽기, 기존 BLOB 폴백, 경로 탈출 거부, 대화 삭제 시 폴더 삭제 |
| `tests/test_tiling.py` | 타일 분할 좌표·겹침, PDF 타일 = 전체 렌더의 같은 영역, 스캔 해상도 상한, 빈 타일 건너뛰기, 타일 좌표 → 전체 좌표, 중복 박스·중복 줄 병합(표의 반복은 보존), 캐시 키 분리, 요청별 모드·모드 전환 시 재전사 |
| `tests/test_compare_script.py` | 비교 스크립트의 IoU·정답 짝짓기·기대 문자열 대조 |
| `tests/test_runaway_guards.py` | 호출별 추론 끄기·출력 상한이 요청에 실리는지, 상한에 닿은 호출은 다시 보내지 않는지, 끊긴 추론 글·미완성 JSON을 쓰지 않는지, 전사의 추론 선택을 바꾸면 다시 전사하는지 |
| `tests/test_env_file.py` | `.env` 읽기, 셸 환경변수 우선, `.env.example`이 모든 설정과 실제 기본값을 담고 있는지 |
| `tests/test_trace.py` | 턴 트레이스: 꺼져 있으면 아무것도 기록하지 않음, 한 턴의 입력→전처리→전사→증거→모델 호출→도구→답변 기록, 도구 아래 자식 호출, 진행 중인 턴의 `running` 호출과 취소 시 `cancelled`, API key·base64 미기록, 대화 삭제 시 함께 삭제 |

특정 테스트만: `uv run pytest tests/test_pdf_pipeline.py -k classification -v`

### 3.2 실제 모델 종단 점검

```powershell
uv run python scripts/make_samples.py     # samples/ 에 시험용 PDF·이미지 생성
uv run python scripts/e2e_check.py        # 기본: Ollama의 gemma3:latest
```

트레이스를 켠 채 돌아가며(`DOCCHAT_DEBUG_TRACE=1`) 기존 항목에 더해 트레이스 항목 T1~T4를 확인합니다(STEPS.md "Step 7 실모델 확인 기준").

다른 모델로: `$env:DOCCHAT_E2E_BASE_URL="http://127.0.0.1:8080/v1"; $env:DOCCHAT_E2E_MODEL="qwen2.5-vl"; uv run python scripts/e2e_check.py`
클라우드로: `$env:DOCCHAT_E2E_PROVIDER="gemini"; $env:DOCCHAT_E2E_API_KEY="…"; $env:DOCCHAT_E2E_MODEL="gemini-2.5-flash"`

### 3.3 전체 vs 타일 비교

같은 파일·질문을 두 방식으로 실행해 전사 글·bbox·시간·호출 수를 나란히 보여 줍니다. 모델은 3.2와 같은 환경변수로 고릅니다.

```powershell
# 전사 비교: 실제 /api/chat 경로로 질문하고, 도면에 적힌 글자 중 몇 개가 읽혔는지 셉니다
uv run python scripts/compare_tiling.py samples/large_scanned_plan.pdf --question "DWG NO와 REV를 알려 줘" --expect-file samples/large_plan.expect.txt
# bbox 비교: 정답 박스(JSON)를 주면 IoU를 계산합니다
uv run python scripts/compare_tiling.py samples/large_plan.png --task "Find every red circle." --truth samples/large_plan.truth.json
# 합성 이미지(격자 + 빨간 원 4개 + 파란 사각형 음성 대조)를 만들어 비교
uv run python scripts/compare_tiling.py --synthetic
```

- `--repeat 3`처럼 여러 번 돌리세요. 1회 결과는 증거가 아닙니다.
- 내 도면으로 잴 때: `--expect "A-1024" --expect "3600"`처럼 도면에 있어야 할 글자를 주거나, 정답 박스 파일(`{"image": {"width", "height"}, "targets": [{"label", "bbox": [x0,y0,x1,y1]}], "distractors": [...]}`, 픽셀 좌표)을 `--truth`로 줍니다.
- 결과 폴더(`samples/compare/…`)에 전사 원문, diff, 박스를 그린 이미지, **모델에 실제로 보낸 타일**이 남습니다.
- 실제 데이터(`data/`)는 건드리지 않습니다(임시 DB·임시 폴더 사용).

### 3.4 추론 폭주 확인 (추론형 모델)

추론형 모델이 위치 확인·전사 호출에서 출력 상한까지 가는지, 그때 앱이 다시 보내지 않고 끝내는지를 호출 단위로 기록합니다.

```powershell
$env:DOCCHAT_E2E_BASE_URL="http://<서버>/v1"; $env:DOCCHAT_E2E_MODEL="<모델>"
# bbox: A = 추론 끔·상한 4096, B = 추론 켬·상한 4096, C = 추론 켬·상한 16000
uv run python scripts/check_runaway.py --image plan.png --task "Find every door symbol." --conditions A:2,B:2,C
# 전사: D = 추론 끔, E = 추론 켬
uv run python scripts/check_runaway.py --pdf samples/large_scanned_plan.pdf --expect-file samples/large_plan.expect.txt --conditions D,E
# 실제 /api/chat 경로로 한 턴(타일 모드, "추론 끄기 — 모든 호출" 해제)
uv run python scripts/check_runaway.py --image plan.png --question "문 심볼을 찾아 표시해 줘" --conditions CHAT
```

- 호출마다 타일·종료 사유(`stop`/`length`)·출력 토큰·시간이 찍히고, `result.json`에 본문과 추론의 끝부분이 남습니다.
- `--limit 1000`처럼 상한을 바꿔 상한 도달을 일부러 만들 수 있습니다.
- 같은 조건이어도 어느 타일이 상한에 닿는지는 실행마다 달라집니다(동시에 나가는 호출의 영향).

### 3.5 브라우저에서 손으로 확인

`samples/`의 파일로 아래를 해 보세요.

| 파일 | 기대 동작 |
|---|---|
| `native_spec.pdf` | 칩에 "네이티브 텍스트 n자". 전사 단계 없이 바로 답변. "도면 번호 알려 줘" → `PS-2210-A` |
| `scanned_drawing.pdf` | 칩에 "1쪽은 비전 전사 필요". 진행 로그에 "페이지 전사 중… (1/1)". → `FA-7731-B` |
| `mixed_3pages.pdf` | 2·3쪽만 전사(1쪽은 네이티브) |
| `sheet_with_stamp.png` | "승인 도장 위치를 표시해 줘" → 오른쪽 뷰어에 박스 오버레이 |
| `large_scanned_plan.pdf` | 이미지 처리를 `타일`로 → 진행 로그에 "타일 전사 중… (n/20)", 답변 아래에 "이미지 처리: 타일 · 타일 20장…" |
| `large_plan.png` | 이미지 처리를 `타일`로 → "소화기 표시(빨간 원) 위치를 표시해 줘" |

### 3.6 API 직접 호출

```powershell
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/test-connection -H "Content-Type: application/json" -d '{\"provider\":\"openaiCompatible\",\"baseUrl\":\"http://127.0.0.1:11434/v1\"}'
```

대화형 API 문서: <http://127.0.0.1:8000/docs>

## 4. API 요약

| 메서드 · 경로 | 설명 |
|---|---|
| `POST /api/chat` | 한 턴 실행. `stream:false`(기본)면 JSON 한 번, `stream:true`면 NDJSON(`conversation` → `progress`… → `final`). `imageMode`: `whole`/`tile`(비우면 서버 기본값). `disableThinking`(모든 호출) · `disableThinkingGrounding` · `disableThinkingOcr`(비우면 서버 기본값). 응답의 `meta`에 처리 방식·비전 호출 수·출력 상한에 닿은 호출 수·호출별 추론 끔 여부(`thinkingDisabled`)·걸린 시간·(트레이스 켬) `traceId`. `conversation` 이벤트에도 `traceId`가 실려 진행 중에 조회할 수 있다 |
| `GET /api/health` | 임계값(`limits`), 기본 이미지 처리 방식(`imageMode`), 타일 설정(`tiling`), 호출별 추론 끄기의 기본값과 출력 상한(`vision`), 트레이스 켬 여부(`debugTrace`) |
| `GET /api/traces/{id}` | 턴 트레이스 JSON(진행 중이면 그때까지의 기록). `?download=1`이면 파일로 |
| `GET /api/sessions/{id}/traces` | 그 대화의 트레이스 목록(id·시각·상태) |
| `GET /api/sessions` · `POST /api/sessions` · `DELETE /api/sessions` | 목록 · 생성 · 선택/전체 삭제(`{ids}` 또는 `{all:true}`) |
| `GET` · `PUT` · `DELETE /api/sessions/{id}` | 불러오기 · 수정 · 삭제 |
| `POST /api/models` (`GET`도 가능) | 모델 목록. API key는 본문/`X-Api-Key` 헤더로만 |
| `POST /api/test-connection` | 항상 200 + `{ok, message, models}` |
| `POST /api/attachments/inspect` | 업로드 미리 검사(페이지별 분류) |
| `GET /api/attachments/{id}/content` | 저장된 첨부 바이트(뷰어 이미지) |

## 5. 알아 둘 점

- **API key는 저장하지 않습니다.** 브라우저 sessionStorage에만 있고 서버는 요청 때만 씁니다.
- 대화는 `data/docchat.sqlite`에, 첨부 파일은 `data/files/{대화ID}/`에 저장됩니다(탐색기로 바로 열어 볼 수 있습니다). 예: `scan.pdf`, `scan.pdf.page-0002.png`(페이지 렌더), `plan.tif`(업로드 원본) + `plan.model.png`(모델 전송용 3072px 사본).
  - 세션을 지우면 그 대화 폴더도 함께 지워집니다. 같은 이름으로 다시 올려도 이전 파일은 덮어쓰지 않고 `scan (2).pdf`로 따로 씁니다.
  - 이전 버전에서 DB 안(BLOB)에 저장된 첨부는 옮기지 않고 그대로 읽습니다.
- Anthropic / Gemini provider는 요청·응답 변환까지만 단위 테스트했고 **실제 API 호출은 key가 없어 검증하지 못했습니다.**
- 타일 모드는 호출 수가 타일 수만큼 늘어납니다(A1 도면 한 쪽 = 20회). 느린 로컬 모델에서는 몇 분이 걸립니다.
- 타일 모드의 한계: **타일 경계에 걸린 긴 글자 줄과 표는 잘립니다**(잘린 부분을 모델이 지어내 채운 사례도 있음). 겹침 영역의 짧은 글자·숫자는 두 번 나올 수 있고, 없는 것을 그리는 모델은 타일 수만큼 틀린 박스를 더 그립니다. 자세한 내용과 실측은 STEPS.md "알려진 한계"·Step 5.
- 타일의 효과는 지금까지 **gemma3 · 합성 도면 1장**으로만 쟀습니다(전사: 읽은 글자 3/47 → 37/47, bbox: 두 방식 모두 0/4로 판단 불가). 쓰는 모델과 도면으로 직접 재 보세요(3.3절).
- **네이티브 tool-calling을 지원하지 않는 모델**(예: Ollama의 gemma3)은 JSON 폴백으로 도구를 부릅니다. 이 경우 첨부가 있는 대화에서 **답변이 가끔 영어로 나올 수 있습니다** — 언어 지시를 넣으면 이 모델이 도구 호출을 건너뛰는 것이 실측돼서, 도구 호출을 살리는 쪽을 택했습니다(STEPS.md "알려진 한계").
- 도구 호출 트리거는 **gemma3 4B · 이미지 1장 · 질문 7개 · 1회 실행**으로만 확인했습니다. 다른 모델을 쓸 때는 `scripts/e2e_check.py`로 직접 재 보세요(위치 질문에서 부르는지 + 비위치 질문에서 안 부르는지를 함께 봅니다).
