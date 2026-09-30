# GIT_WORKFLOW.md — 브랜치·커밋 규칙

혼자 개발하는 저장소다. 규칙은 "기록을 나중에 다시 읽을 수 있게" 하는 데 목적이 있고, 그 이상으로 복잡하게 하지 않는다.
원격은 `origin` = https://github.com/baek-shawn/DocChat

## 1. 브랜치

| 브랜치 | 규칙 |
|---|---|
| `main` | 항상 `uv run pytest`가 통과하는 상태만 둔다. 직접 커밋하지 않고 PR로만 들어온다 |
| `feature/<step>-<내용>` | Step 하나에 브랜치 하나. `main`에서 만든다 |

브랜치 이름 예:

```
feature/step7-trace
feature/step8-answer-image-mode
feature/step6-answer-thinking-guard
fix/grounding-cutoff-json          # Step과 무관한 버그 수정
docs/steps-known-limits            # 문서만 고칠 때
```

- 병합된 브랜치는 **지운다**(로컬·GitHub 모두). 되돌아갈 시점은 브랜치가 아니라 **태그**로 남긴다.
- Step 하나가 너무 크면(예: Step 8의 1차·2차) `feature/step8-1-off-whole`처럼 나눠도 된다. 나눈 단위마다 PR 하나.

## 2. 한 Step의 흐름

```bash
git switch main
git pull origin main
git switch -c feature/step7-trace
```

작업 중에는 **테스트가 통과하는 지점마다** 커밋한다. Step 완료 기준(STEPS.md)을 채우면:

```bash
uv run pytest                        # 통과 확인
git push -u origin feature/step7-trace
```

GitHub에서 PR을 만든다.

- base `main` ← compare `feature/...`
- 본문에 STEPS.md의 완료 기준 체크 결과, 실모델 확인 결과(어떤 모델·몇 회·결과), 계획 대비 변경점을 적는다.
- 병합 방식은 **"Create a merge commit"**. Squash는 쓰지 않는다(Step 안의 커밋 기록이 사라진다).
- 병합 후 **"Delete branch"**.

로컬을 맞추고 태그를 찍는다.

```bash
git switch main
git pull origin main
git tag step-7
git push origin step-7
git branch -d feature/step7-trace
git fetch --prune
```

## 3. 커밋 메시지

형식: `종류: 무엇을 했는지 (Step 번호)`

| 종류 | 쓰는 경우 |
|---|---|
| `feat` | 새 기능 |
| `fix` | 버그 수정 |
| `test` | 테스트만 추가·수정 |
| `docs` | STEPS.md, README, CLAUDE.md 등 문서만 |
| `refactor` | 동작이 같은 구조 변경 |
| `chore` | 설정, 의존성, `.gitignore`, 스크립트 |

규칙:

- 첫 줄은 제목, 50자 안팎. 둘째 줄은 **반드시 비운다.**
- 본문에는 **왜** 그렇게 했는지를 적는다. 무엇을 했는지는 diff에 있다.
- 테스트 결과를 마지막 줄에 적는다: `테스트: uv run pytest 243 passed`
- **프롬프트를 바꾼 커밋은 따로 뗀다.** 본문에 실모델 확인 결과를 적는다(이 프로젝트에서 프롬프트 한 줄이 도구 호출률을 0%로 만든 적이 있다 — CLAUDE.md §5).
- 임계값(`config.py` 기본값)을 바꾼 커밋도 따로 떼고, 바꾼 근거(측정 결과)를 적는다.

예:

```
feat: 전사·bbox 호출의 폭주 막기 (Step 6-0)

추론형 모델(Qwen3.5)이 전사 한 번에 3만 토큰을 쓰는 경우가 있어
호출별로 추론을 끄고 출력 상한을 둔다. 상한에 닿은 호출은 다시
보내지 않는다 — 같은 입력이면 같은 자리에서 또 끊길 가능성이 높고,
재시도가 폭주의 원인이 되기 때문.

- bbox/전사의 추론 끄기를 따로 옵션으로(기본 둘 다 끔)
- DOCCHAT_VISION_MAX_TOKENS(기본 4096), .env로만 설정
- scripts/check_runaway.py 추가

테스트: uv run pytest 243 passed
```

## 4. 태그

Step을 `main`에 병합할 때마다 `step-<번호>` 태그를 찍는다. 비교 실험에서 "Step 6-0 시점의 코드"를 다시 돌릴 때 쓴다.

```bash
git tag                              # 목록
git switch --detach step-6-0         # 그 시점 코드로 이동(읽기 전용 작업)
git switch main                      # 돌아오기
```

## 5. 되돌리기

| 상황 | 명령 |
|---|---|
| 아직 push 안 한 마지막 커밋 메시지 수정 | `git commit --amend` |
| 아직 push 안 한 마지막 커밋 취소(변경은 남김) | `git reset --soft HEAD~1` |
| 이미 `main`에 병합된 Step을 취소 | `git revert -m 1 <병합 커밋 해시>` → 새 커밋으로 되돌린다 |
| 파일 하나를 특정 시점 내용으로 | `git restore --source=step-6-0 -- app/config.py` |

push된 기록은 `reset`·`force push`로 지우지 않는다. `revert`로 새 커밋을 얹는다.

## 6. 커밋에 넣지 않는 것

`.gitignore`에 이미 있다. `git add -A` 전에 `git status`로 아래가 목록에 **없는지** 본다.

- `.env` — 설정. `.env.example`만 커밋한다.
- `data/` — 사용자 데이터(DB·첨부 파일)
- `samples/` — 생성된 시험 문서와 비교 결과
- `.venv/`, `.pytest_tmp/`
- API key는 어디에도 쓰지 않는다. 사용자 vLLM 서버 주소도 문서에 적지 않는다(`.env`·명령줄 인자로만).

올린 뒤에 확인하려면:

```bash
git grep -n "sk-" HEAD
git grep -n "182.162" HEAD
```

## 7. Windows cmd에서 주의할 것

- `-m` 메시지는 **큰따옴표**로 감싼다. 작은따옴표는 cmd에서 인용 부호가 아니다.
- 여러 문단은 `-m`을 여러 번 쓰거나, `git commit`만 쳐서 편집기에서 쓴다. 편집기를 메모장으로: `git config --global core.editor notepad`
- 편집기에서는 `#`으로 시작하는 줄 **위에** 메시지를 쓴다. `#` 줄은 커밋에 들어가지 않는다.
- `git log`에서 한글이 깨져 보이면 `chcp 65001` (표시 문제일 뿐 커밋은 정상).
- `LF will be replaced by CRLF` 경고는 무시해도 된다.

## 8. 기록

| 날짜 | 태그 | 내용 |
|---|---|---|
| 2026-09-21 | — | `68abc96` Initial commit: Step 0~4 |
| 2026-09-30 | `step-6-0` | `c72d862` Step 5-0, 5, 6-0 (한 커밋으로 묶음 — 세 Step의 변경이 같은 파일에 섞여 있어 나누면 중간 커밋의 테스트가 깨짐) |
