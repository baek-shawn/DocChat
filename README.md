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
| 적용 대상 | — | **전사(OCR)와 bbox(`inspect_visual`) 호출만.** 답변 호출의 이미지는 어느 방식이든 전체 한 장(무엇을 실을지는 아래 "답변 호출 이미지"가 따로 정함) |

- PDF는 쪽 전체를 고해상도로 만든 뒤 자르지 않고 **타일 영역만 바로 렌더**합니다. 스캔 PDF는 박힌 이미지의 원래 해상도까지만 올립니다.
- 업로드 이미지는 보관해 둔 **원본**에서 자릅니다(모델 전송용 3072px 사본이 아니라).
- 원본의 긴 변이 2048px 이하면 타일로 골라도 전체와 똑같이 처리합니다.
- 같은 대화에서 방식을 바꾸면 이미 전사한 쪽도 **새 방식으로 다시 전사**합니다(이전 결과가 섞이면 비교가 무의미하므로).
- 어느 쪽이 나은지는 문서와 모델에 따라 다릅니다. 본문이 쪽 너비를 가득 채우는 문서는 타일 경계에서 줄이 잘려 오히려 나빠질 수 있습니다 → `scripts/compare_tiling.py`로 재 보세요(3.3절).

### 답변(추론) 호출 이미지: 끔 / 업로드 이미지만 / 전체 / 자동

답변을 만드는 호출이 **그림을 볼 수 있게 할지**를 요청마다 고릅니다(⚙ 모델 설정 → "답변(추론) 호출 이미지"). 위의 전체/타일과는 별개의 설정입니다.

| | 끔(`off`) | 업로드 이미지만(`uploads`, 기본) | 전체(`whole`) | 자동(`auto`) |
|---|---|---|---|---|
| 업로드 이미지 | 싣지 않음 | 전체 한 장 | 전체 한 장 | 전체 한 장 |
| PDF 쪽 | 싣지 않음 | 싣지 않음(텍스트만) | **모든 쪽**을 한 장씩 — 글자가 충분해 전사하지 않은 쪽도 | **모델이 보기 도구(`view_page`)로 요청한 쪽만**, 한 번에 한 쪽씩 |
| 답변 모델이 보는 것 | 네이티브 텍스트 + 전사 글 + 위치 확인 도구 | 위 + 업로드 이미지 | 위 + 쪽 그림(형상·심볼·배치) | 위 + 모델이 필요하다고 판단한 쪽 그림 |

- 기본값은 지금까지의 동작입니다(계획서 §5.3·§5.4). "전체"는 그림에만 있는 것(벽·창 형상, 무엇의 치수인지, 심볼 개수)을 답하게 하려는 실험용 모드입니다 — 효과는 모델·도면으로 직접 재 보세요.
- **"자동"(Step 10)**: 모델에게 보기 도구 `view_page`를 내놓습니다. 모델이 "답을 내려면 그림을 봐야 한다"고 판단하면(형상·배치·개수·어느 대상의 치수인지·여러 쪽 비교) 쪽을 한 장씩 요청하고, 요청한 쪽은 **그 턴 안에서 쌓여 다음 호출부터 모두** 실립니다(여러 쪽 비교는 차례로 모아 한 호출에서 나란히 봅니다). 한 턴에 최대 `DOCCHAT_MAX_VIEWED_PAGES`(기본 12)장 — 넘으면 붙이지 않고 "더 볼 수 없다"고 알려 줍니다. 이전 턴에서 본 쪽은 다음 턴에 자동으로 실리지 않지만 모델이 다시 요청할 수 있습니다(렌더해 둔 쪽을 다시 씁니다). 판단을 돕기 위해 매니페스트에 **그림이 있는 쪽**과 그 쪽의 글이 무엇을 담는지(전사한 라벨뿐인지, 네이티브 글인지)를 적습니다 — "그림 있음"은 래스터 개수가 아니라 쪽에서 차지하는 면적 비율(`DOCCHAT_DRAWING_MIN_RASTER_AREA`)과 벡터 연산 수(`DOCCHAT_DRAWING_MIN_VECTOR_OPERATIONS`)로 판정합니다. 답변 아래에 `그림 확인: spec.pdf 2쪽, 3쪽`(또는 `그림 확인 없음`)이 붙고 `meta.viewedPages`에 남습니다. 끔·업로드만·전체의 프롬프트는 바뀌지 않았습니다(비교 기준).
- 위치 확인(bbox) 도구 `inspect_visual`은 어느 방식이든 **보여 달라고 할 때만**("표시해줘", "박스로", "시각화해줘") 씁니다 — Step 10에서 문구를 좁혔습니다. 그림을 읽어 답하는 용도는 보기 도구의 몫입니다("어디 있어?"처럼 말로 답하면 되는 질문도 보기 도구).
- "전체"에서 전처리 때 렌더하지 않은 쪽은 답변 직전에 렌더해 첨부로 저장합니다(뷰어에서 열리고, 다음 턴에 다시 그리지 않습니다). 한 번에 최대 `DOCCHAT_MAX_MODEL_IMAGES`(기본 12)장, 넘치면 **쪽 순서로 앞에서부터** 싣고 뺀 수를 답변 아래에 적습니다. 실을 쪽만 렌더합니다.
- 쪽 이미지가 실릴 때는 질문 끝에 `[PAGE IMAGES: 1: a.pdf · page 1; …]` 한 줄이 붙어 모델이 몇 번째 이미지가 어느 쪽인지 알 수 있습니다. 기본 모드의 프롬프트는 바뀌지 않습니다.
- 비용: 이미지는 도구 루프의 **호출마다** 다시 갑니다(3072px 도면 한 장 ≈ Qwen3.5 입력 6,500토큰). 이미지 토큰은 예산 계산에 넣지 않으므로 컨텍스트가 작은 로컬 모델(8192)은 "전체"에서 넘쳐 오류가 날 수 있습니다.
- 답변 아래에 `답변 호출 이미지: 전체 2장 (scan.pdf · page 1, scan.pdf · page 2)`처럼 표시되고, 답변의 `meta.answerImageMode`·`meta.answerImages`에 남습니다. 트레이스(아래)에서는 실제로 보낸 이미지를 첨부 ID로 볼 수 있습니다.

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
- 답변 호출에는 출력 상한이 없습니다. 답변 호출의 추론이 길어지는 문제는 아래 "추론 폭주 막기"가 맡습니다.

### 추론 폭주 막기 — 추론 예산과 반복 감지 (Step 6)

추론형 모델(Qwen3.5 등)은 "안녕"에도 수천 토큰을 생각하고, 가끔 같은 생각을 맴돌며 끝나지 않습니다(실측: 80,000토큰, 20분). **추론을 켠** 로컬 호출(답변·위치 확인·전사 모두)은 서버↔모델 구간을 스트리밍으로 받으며 **추론 부분만** 지켜봅니다.

| 감지 | 조치 |
|---|---|
| 추론 토큰이 예산을 넘음 — 답변 8,000 · 위치 확인 4,000 · 전사 4,000(`.env`의 `DOCCHAT_REASONING_BUDGET_*`, 0이면 없음) | **소프트**: 스트림을 끊고, 쓴 추론 뒤에 "충분히 생각했으니 답한다"와 `</think>`를 붙여 같은 요청을 **이어 쓰기**로 다시 보냅니다. 답은 잘리지 않고 추론만 잘립니다 |
| 추론이 같은 줄 묶음(최대 8줄)을 연달아 3번 되풀이함(`DOCCHAT_REASONING_REPEAT_*`) | 같은 소프트 조치 — 예산을 다 쓸 때까지 기다리지 않습니다 |
| 이어 쓰기에서도 추론이 넘치거나 반복, 빈 답, 서버가 이어 쓰기를 거절 | **하드**: 그 호출을 끝냅니다. 답변은 안내문("추론이 끝나지 않아 답변을 받지 못했습니다…"), 전사는 `[OCR FAILED …]`, 위치 확인은 박스 없음 + 경고. 어느 쪽도 **다시 보내지 않습니다** |
| 추론만 하다 출력 한도(`finish_reason=length`)에 닿음 | 추론 글을 답으로 내보내지 않고 안내문 |

- 생성 중에는 `추론 중… 1,234토큰 · 12초`가 같은 줄에서 갱신되고(타일이면 타일 이름 포함), 조치가 있으면 `추론 예산(…)을 넘어 추론을 끊고 답으로 넘기는 중…`이 보입니다.
- 답변 아래에는 **어느 타일이 왜 끊겼는지**와 턴의 추론 토큰 합계가 남습니다: `위치 확인 호출 20회 (추론 끄지 않음, 추론을 끊고 답으로 넘김 3회 — 반복: r1c1 · 예산: r4c4, r4c5) · 추론 22,522토큰`. 반복으로 끊긴 타일의 결과는 한 번 의심해 볼 만합니다(모델이 헷갈려 맴돈 자리).
- "과정 보기" 맨 위 요약에 입력·추론 토큰 합계, `추론 끊음 3건 (반복 1 · 예산 2)`, 가장 오래 걸린 호출 3개가 나오고, 끊긴 호출 행은 색 띠와 함께 `추론 끊음(반복) · “되풀이된 첫 줄”`로 표시됩니다. 호출마다 추론 예산·토큰·시간·조치 칩이 있습니다.
- 이 표시는 **화면에만** 있습니다. 답을 낸 타일이 추론을 끊긴 채 답했다는 사실은 도구 결과(답변 모델이 읽는 글)에 넣지 않습니다 — 넣으면 답변 모델이 "덜 됐다"고 보고 도구를 다시 부릅니다.
- 추론을 끈 호출(기본값의 위치 확인·전사)은 스트리밍하지 않고 이전과 똑같이 동작합니다. 추론을 켠 위치 확인·전사 호출의 출력 상한은 `DOCCHAT_VISION_MAX_TOKENS` + 그 호출의 추론 예산입니다(상한 = 예산 + 출력 몫).
- 전사의 추론 예산은 OCR 캐시 키에 들어가므로, 추론을 켠 채 예산을 바꾸면 그 쪽은 다시 전사합니다.
- 감지는 답 부분에는 걸지 않습니다 — 표 전사나 bbox JSON처럼 비슷한 줄이 이어지는 출력은 정상입니다.
- 이어 쓰기 요청 형식(`continue_final_message`)은 vLLM에서 확인했습니다. Ollama·llama.cpp처럼 그 필드를 모르는 서버는 assistant 메시지를 이력으로 보고 새로 답할 수도, 빈 답을 낼 수도 있습니다(빈 답이면 하드로 끝납니다, 미실측). 클라우드 provider에는 적용되지 않습니다.
- 실제 서버에서 확인하려면 `scripts/check_runaway.py`(3.4절)의 `SOFT`/`FULL` 조건을 씁니다.

### 추론 수준 — `reasoning_effort` (Step 6 2차)

추론을 켜고 끄는 것 말고 **얼마나 길게 생각할지**를 받는 모델이 있습니다(예: Qwen3.8-27B의 채팅 템플릿은 `low` / `medium` / `xhigh`를 받고, 안 보내면 가장 긴 `xhigh`). ⚙ 모델 설정의 "추론 수준"에서 **호출 종류별로**(답변 · 위치 확인 · 전사) 적습니다.

- **비우면 보내지 않습니다**(모델의 기본 수준 — 지금까지의 동작). 값은 모델마다 달라 목록에서 고르지 않고 글자로 적습니다(영문 소문자·숫자·`-`·`_`, 24자까지).
- **추론을 켠 로컬 호출에만** 실립니다(`chat_template_kwargs.reasoning_effort`). 그 호출의 추론이 꺼져 있으면 칸이 잠기고, 요청은 이전과 똑같이 나갑니다. 클라우드 provider에는 연결돼 있지 않습니다.
- 화면에서 고른 값은 **모델 이름별로** 브라우저에 저장됩니다 — 모델을 바꾸면 그 모델에서 고른 값이 따라옵니다. 한 번도 고르지 않은 모델은 서버 기본값(`.env`의 `DOCCHAT_REASONING_EFFORT_ANSWER` / `_GROUNDING` / `_OCR`, 기본은 빈 값)을 씁니다.
- **서버가 받지 않는 값이면 그 턴은 오류로 끝납니다**(예: Qwen3.8에 `high`). 값을 빼고 다시 보내지 않습니다 — 그러면 모델 기본 수준으로 돌아 오래 걸리고, 답에는 요청한 수준이 적혀 비교 기록이 틀어집니다. 오류 문구에 서버가 한 말이 그대로 나옵니다. 이미 끝난 전사는 저장돼 있으므로 값을 고쳐 다시 보내면 됩니다.
- **"받았다"와 "적용됐다"는 다릅니다.** 수준이 없는 모델(Qwen3.5 등)은 값을 거절하지 않고 무시합니다. 효과는 "과정 보기"의 추론 토큰 수로 확인하세요.
- 답변 아래에 `답변 호출 1회 (추론 끄지 않음, 추론 수준 low)`처럼 표시되고, 답변의 `meta.reasoningEffort`(실어 보낸 호출 종류와 값)와 트레이스의 호출별 `추론 수준` 칩에 남습니다.
- 전사의 수준은 OCR 캐시 키에 들어가므로, 추론을 켠 전사의 수준을 바꾸면 그 쪽은 다시 전사합니다.
- 추론 예산·반복 감지(위)는 수준과 무관하게 그대로 작동합니다. 수준은 "모델에게 짧게 생각하라고 부탁"하는 것이고, 예산은 "넘으면 끊는" 것입니다.
- 실제 서버에서 값이 닿는지는 `scripts/check_runaway.py --conditions LEVELS`(3.4절)로 봅니다.

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
| `DOCCHAT_ANSWER_IMAGE_MODE` | `uploads` | 요청에 `answerImageMode`가 없을 때 답변 호출에 실을 이미지(`off` / `uploads` / `whole` / `auto`) |
| `DOCCHAT_MAX_MODEL_IMAGES` | `12` | 답변 호출 한 번에 싣는 이미지 수 상한(업로드 이미지 + `whole`의 PDF 쪽) |
| `DOCCHAT_MAX_VIEWED_PAGES` | `12` | `auto`에서 보기 도구로 한 턴에 모을 수 있는 쪽 수(모은 쪽은 그 뒤 호출마다 다시 실림) |
| `DOCCHAT_DRAWING_MIN_RASTER_AREA` / `DOCCHAT_DRAWING_MIN_VECTOR_OPERATIONS` | `0.02` / `100` | `auto`의 매니페스트에 "그림이 있는 쪽"으로 적는 기준(래스터 면적 비율 / 벡터 경로 연산 수) |
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
| `DOCCHAT_REASONING_BUDGET_ANSWER` / `_GROUNDING` / `_OCR` | `8000` / `4000` / `4000` | 추론을 켠 호출의 추론 토큰 예산. 넘으면 추론을 끊고 답만 이어 쓰게 함(`0`이면 예산 없음) |
| `DOCCHAT_REASONING_REPEAT_LINES` / `_COUNT` / `_MIN_CHARS` | `8` / `3` / `24` | 추론 반복 감지: 최대 8줄 묶음이 연달아 3번 같으면 반복(묶음이 24자 미만이면 제외) |
| `DOCCHAT_MAX_PDF_VISUAL_PAGES` | `60` (최대 200) | 검사할 최대 페이지 |
| `DOCCHAT_MAX_TOOL_STEPS` | `8` | 한 턴의 최대 도구 호출 횟수 |

> **CAD PDF 팁**: SHX 폰트로 그린 글자는 텍스트 객체가 아니라 선이라서 `native characters`가 낮게 잡히고 비전 전사로 넘어갑니다. 반대로 글자는 적지만 네이티브 텍스트로 충분한 도면이라면 `DOCCHAT_NATIVE_MIN_CHARS`를 낮추세요.

## 3. 테스트

### 3.1 자동 테스트 (모델 불필요, 약 40초, 292개)

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
| `tests/test_answer_images.py` | 답변 호출 이미지(Step 8): 기본값은 이전 동작 그대로, 전체 모드는 전사 뒤 쪽 이미지를 쪽 순서로 싣고 네이티브 쪽을 렌더해 저장, 상한 안에서만 렌더, 루프의 매 호출에 실림, 끔 모드에서도 bbox 도구 동작, 렌더한 쪽이 기본 모드로 새지 않음, 트레이스 기록 |
| `tests/test_reasoning_control.py` | 추론 제어(Step 6): 실측 반복 표본을 3번째 순환에서 잡고 정상 추론·짧은 줄은 오인하지 않음, 조각·글자 수로 토큰 세기, 본문 속 `<think>` 가르기, 추론을 켠 로컬 호출만 스트리밍, 도구 호출 델타 조립, 예산 초과·반복 → 이어 쓰기 요청 형식(`</think>` 접두 + `continue_final_message`), 이어 쓰기 실패 → 하드 중단(반복·빈 답·거절), 상한 = 예산 + 출력 몫, 전사·bbox 호출의 실패 처리와 재시도 없음, `/api/chat`의 메타·live 진행·안내문, 트레이스 기록 |
| `tests/test_view_tool.py` | 보기 도구(Step 10): 자동 모드에서만 `view_page`와 매니페스트의 "그림이 있는 쪽"(다른 모드의 프롬프트는 그대로), 본 쪽이 다음 호출부터 원래 질문에 붙고 쌓임, 상한·중복·이미 실린 업로드 이미지, 후속 턴은 자동으로 싣지 않되 다시 요청 가능(다시 그리지 않음), bbox 호출과 섞이지 않음, JSON 폴백 경로, 트레이스, 래스터 면적·벡터 수로 그림 판정, 옛 첨부의 폴백, `meta.viewedPages` |
| `tests/test_reasoning_effort.py` | 추론 수준(Step 6 2차): 값의 모양 검사, 추론을 켠 로컬 호출에만 `chat_template_kwargs.reasoning_effort`가 실리는지(추론을 끈 호출·클라우드는 이전과 같은 요청), 이어 쓰기 요청도 같은 수준, 서버가 받지 않는 값 → 오류(빼고 다시 보내지 않음, 같은 값을 다시 묻지 않음), 다른 원인의 거절을 수준 탓으로 돌리지 않음(1토큰 확인 요청), 전사 3회 재시도·도구 오류·타일 일부 실패에 묻히지 않음, 호출 종류별 값과 `meta.reasoningEffort`, 수준을 바꾸면 다시 전사, 트레이스 기록 |

특정 테스트만: `uv run pytest tests/test_pdf_pipeline.py -k classification -v`

### 3.2 실제 모델 종단 점검

```powershell
uv run python scripts/make_samples.py     # samples/ 에 시험용 PDF·이미지 생성
uv run python scripts/e2e_check.py        # 기본: Ollama의 gemma3:latest
```

트레이스를 켠 채 돌아가며(`DOCCHAT_DEBUG_TRACE=1`) 기존 항목에 더해 트레이스 항목 T1~T4(STEPS.md "Step 7 실모델 확인 기준"), 답변 호출 이미지 항목 A1~A4(STEPS.md "Step 8 1차 실모델 확인 기준"), 보기 도구 항목 V1~V5(STEPS.md "Step 10 실모델 확인 기준" — 자동 모드에서 그림 필요 / 글로 충분 / 시각화 질문별 도구 선택)를 확인합니다.

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
# 추론 제어(Step 6): 답변 호출의 추론 예산이 걸리는지 — 첨부 없는 짧은 질문을 예산 500으로 3번, 예산 없이 1번
uv run python scripts/check_runaway.py --question "17 곱하기 23은? 숫자만 답해 줘." --conditions SOFT:3,FULL --budget 500
# 실제 /api/chat 경로로 한 턴(타일 모드, "추론 끄기 — 모든 호출" 해제)
uv run python scripts/check_runaway.py --image plan.png --question "문 심볼을 찾아 표시해 줘" --conditions CHAT
# 추론 수준(Step 6 2차): 값이 모델에 닿는지(수준별 입력 토큰 수, 출력 1토큰이라 몇 초)와 받지 않는 값이 오류로 끝나는지
uv run python scripts/check_runaway.py --conditions LEVELS --levels low,medium,xhigh --invalid high
# 추론 수준을 실어 한 턴 — 수준별 추론 토큰·시간 비교(--effort는 B·C·E·CHAT·SOFT·FULL의 추론을 켠 호출에 실린다)
uv run python scripts/check_runaway.py --question "17 곱하기 23은? 숫자만 답해 줘." --conditions FULL:2 --effort low
```

- `LEVELS`는 같은 짧은 질문을 수준만 바꿔 보내 **입력 토큰 수**를 적습니다. 템플릿이 수준에 따라 지시문을 넣는 모델이면 수가 달라지고(Qwen3.8: `medium` < `low`·`xhigh`, 미지정 = `xhigh`), 전부 같으면 그 모델은 값을 무시하는 것입니다.
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
| `POST /api/chat` | 한 턴 실행. `stream:false`(기본)면 JSON 한 번, `stream:true`면 NDJSON(`conversation` → `progress`… → `final`; `progress`에 `live: true`가 있으면 "추론 중… n토큰"처럼 같은 줄을 갱신하는 문구). `imageMode`: `whole`/`tile`(비우면 서버 기본값). `answerImageMode`: `off`/`uploads`/`whole`/`auto`(답변 호출에 실을 이미지, 비우면 서버 기본값. `auto`는 보기 도구로 모델이 쪽을 요청). `disableThinking`(모든 호출) · `disableThinkingGrounding` · `disableThinkingOcr`(비우면 서버 기본값). `reasoningEffortAnswer` · `reasoningEffortGrounding` · `reasoningEffortOcr`(호출 종류별 추론 수준. 없으면 서버 기본값, `""`이면 보내지 않음, 추론을 켠 로컬 호출에만 실림 — 서버가 받지 않는 값이면 400). 응답의 `meta`에 처리 방식·답변 호출 이미지(`answerImageMode`, `answerImages{sent, candidates, names}`)·비전 호출 수·출력 상한에 닿은 호출 수·호출별 추론 끔 여부(`thinkingDisabled`)·실어 보낸 추론 수준(`reasoningEffort`)·(자동 모드) 보기 도구로 본 쪽(`viewedPages{names, limit, refused}`)·걸린 시간·(트레이스 켬) `traceId`. `conversation` 이벤트에도 `traceId`가 실려 진행 중에 조회할 수 있다 |
| `GET /api/health` | 임계값(`limits`), 기본 이미지 처리 방식(`imageMode`), 타일 설정(`tiling`), 답변 호출 이미지의 기본값과 상한(`answerImageMode`, `maxModelImages`), 보기 도구의 상한과 그림 판정 기준(`view`), 호출별 추론 끄기의 기본값과 출력 상한(`vision`), 추론 예산·반복 기준(`reasoning`), 추론 수준의 서버 기본값(`reasoningEffort`), 트레이스 켬 여부(`debugTrace`) |
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
