"""추론(thinking) 지켜보기(Step 6) — 스트리밍으로 받는 추론 조각을 세고, 예산 초과와 반복을 잡는다.

provider가 추론을 켠 호출을 스트리밍으로 받으면서 추론 조각마다 `ReasoningMonitor.feed()`를 부른다.
  - 예산: 추론 토큰 수가 예산을 넘으면 "budget". 토큰 수는 조각 수와 글자 수/4 중 큰 값이다 — vLLM은 조각 하나가
    토큰 하나지만(실측 평균 3.8자), 여러 토큰을 한 조각으로 보내는 서버에서도 너무 늦게 잡지 않도록 글자 수로도 센다.
  - 반복: 줄 단위로 최대 N줄 묶음이 연달아 M번 같으면 "repeat". 짧은 묶음(`Okay.`)은 제외한다.
감지는 **추론 부분에만** 건다. 답 부분의 반복(표 전사, bbox JSON의 비슷한 항목)은 정상이다.

서버에 추론 파서가 없으면 추론이 본문에 `<think>…</think>`로 섞여 온다 — `InlineThinkSplitter`가 조각을 추론과 본문으로 가른다.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

from .. import config

# 끊은 추론 뒤에 붙여 "이제 답한다"로 넘기는 문장(모델에게 보이는 글이라 영어).
FORCED_END_NOTE = "I have spent enough time thinking. I will give the final answer now."

_THINK_OPEN = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)
_TAG_PREFIXES = ("<", "</", "<t", "<th", "<thi", "<thin", "<think", "</t", "</th", "</thi", "</thin", "</think")


@dataclass
class ReasoningProgress:
    """진행 표시용. stage: "reasoning"(추론 중) | "forced"(추론을 끊고 답으로 넘김) | "runaway"(하드 중단)"""
    stage: str
    tokens: int
    chars: int
    seconds: float
    image: str = ""           # 이 호출에 실린 첫 이미지 이름(타일이면 타일 이름) — 어느 호출인지 알 수 있게
    reason: str = ""          # forced/runaway일 때 "budget" | "repeat"
    cycle: str = ""           # 반복으로 잡혔을 때 되풀이된 묶음의 첫 줄


OnReasoning = Callable[[ReasoningProgress], None]


def describe_reasoning_progress(info: ReasoningProgress) -> str:
    """화면 진행 문구(한국어)."""
    where = f" · {info.image}" if info.image else ""
    if info.stage == "forced":
        why = "같은 내용을 반복하고 있어" if info.reason == "repeat" else f"추론 예산({info.tokens:,}토큰)을 넘어"
        return f"{why} 추론을 끊고 답으로 넘기는 중…{where}"
    if info.stage == "runaway":
        return f"추론이 끝나지 않아 중단했습니다{where}"
    return f"추론 중… {info.tokens:,}토큰 · {info.seconds:.0f}초{where}"


# --------------------------------------------------------------------------- 반복 감지
class RepetitionDetector:
    """줄 단위 반복 감지. 완성된 줄(개행이 온 줄)만 본다. 공백 차이는 무시하고 빈 줄은 건너뛴다."""

    def __init__(self, *, max_lines: int | None = None, count: int | None = None, min_chars: int | None = None):
        self.max_lines = max_lines or config.REASONING_REPEAT_LINES
        self.count = count or config.REASONING_REPEAT_COUNT
        self.min_chars = min_chars or config.REASONING_REPEAT_MIN_CHARS
        self._lines: list[str] = []
        self._pending = ""
        self.cycle: list[str] = []

    def feed(self, piece: str) -> bool:
        """조각을 넣는다. 반복이 확인되면 True(그 뒤로는 계속 True)."""
        if self.cycle:
            return True
        self._pending += piece
        while "\n" in self._pending:
            line, _, self._pending = self._pending.partition("\n")
            key = " ".join(line.split())
            if not key:
                continue
            self._lines.append(key)
            if self._repeats_at_end():
                return True
        return False

    def _repeats_at_end(self) -> bool:
        lines = self._lines
        for length in range(1, self.max_lines + 1):
            if length * self.count > len(lines):
                break
            block = lines[-length:]
            if sum(len(line) for line in block) < self.min_chars:
                continue
            if all(lines[-length * (index + 1): len(lines) - length * index] == block for index in range(1, self.count)):
                self.cycle = block
                # 더는 쌓아 둘 필요가 없다
                del lines[:-length]
                return True
        # 메모리를 묶음 길이 × 횟수 이상으로 쌓아 두지 않는다
        keep = self.max_lines * self.count + 1
        if len(lines) > keep * 2:
            del lines[:-keep]
        return False


# --------------------------------------------------------------------------- 추론 조각 세기
@dataclass
class ReasoningMonitor:
    """한 호출의 추론을 지켜본다. feed()가 "" | "budget" | "repeat"를 돌려준다(한 번 걸리면 그 값을 유지)."""
    budget: int = 0
    image: str = ""
    on_progress: OnReasoning | None = None
    detector: RepetitionDetector = field(default_factory=RepetitionDetector)
    chunks: int = 0
    chars: int = 0
    verdict: str = ""
    started: float = field(default_factory=time.monotonic)
    last_at: float = 0.0          # 마지막 추론 조각이 온 시각 — 추론이 끝난 뒤의 답 생성 시간은 추론 시간에 넣지 않는다
    _reported: float = 0.0

    @property
    def tokens(self) -> int:
        return max(self.chunks, self.chars // 4)

    @property
    def seconds(self) -> float:
        return (self.last_at or time.monotonic()) - self.started

    def feed(self, piece: str) -> str:
        if self.verdict:
            return self.verdict
        if not piece:
            return ""
        self.chunks += 1
        self.chars += len(piece)
        self.last_at = time.monotonic()
        if self.detector.feed(piece):
            self.verdict = "repeat"
        elif self.budget and self.tokens > self.budget:
            self.verdict = "budget"
        if self.verdict:
            return self.verdict
        self._report()
        return ""

    def progress(self, stage: str = "reasoning", reason: str = "") -> ReasoningProgress:
        return ReasoningProgress(stage=stage, tokens=self.tokens, chars=self.chars, seconds=self.seconds,
                                 image=self.image, reason=reason or self.verdict,
                                 cycle=self.detector.cycle[0] if self.detector.cycle else "")

    def _report(self, *, force: bool = False) -> None:
        """1초에 한 번만 진행을 알린다(조각마다 알리면 수천 번이다). 알리는 쪽의 예외는 호출을 멈추게 하지 않는다."""
        if self.on_progress is None:
            return
        now = time.monotonic()
        if not force and now - self._reported < 1.0:
            return
        self._reported = now
        try:
            self.on_progress(self.progress())
        except Exception:
            pass

    def notify(self, stage: str, reason: str = "") -> None:
        if self.on_progress is None:
            return
        try:
            self.on_progress(self.progress(stage, reason))
        except Exception:
            pass


# --------------------------------------------------------------------------- 본문 속 <think> 가르기
class InlineThinkSplitter:
    """추론 파서가 없는 서버: 본문 조각에서 `<think>…</think>` 안을 추론으로, 밖을 본문으로 가른다.

    태그가 조각 경계에서 잘려 올 수 있어(`<thi` + `nk>`) 태그의 앞부분일 수 있는 꼬리는 다음 조각까지 들고 있는다.
    """

    def __init__(self) -> None:
        self.in_think = False
        self._buffer = ""
        self.saw_think = False

    def feed(self, piece: str) -> tuple[str, str]:
        """(추론 조각, 본문 조각)"""
        self._buffer += piece
        reasoning, content = [], []
        while self._buffer:
            if self.in_think:
                match = _THINK_CLOSE.search(self._buffer)
                if match:
                    reasoning.append(self._buffer[:match.start()])
                    self._buffer = self._buffer[match.end():]
                    self.in_think = False
                    continue
                hold = self._tail_that_may_be_a_tag()
                reasoning.append(self._buffer[:len(self._buffer) - hold])
                self._buffer = self._buffer[len(self._buffer) - hold:]
                break
            match = _THINK_OPEN.search(self._buffer)
            if match:
                content.append(self._buffer[:match.start()])
                self._buffer = self._buffer[match.end():]
                self.in_think = True
                self.saw_think = True
                continue
            hold = self._tail_that_may_be_a_tag()
            content.append(self._buffer[:len(self._buffer) - hold])
            self._buffer = self._buffer[len(self._buffer) - hold:]
            break
        return "".join(reasoning), "".join(content)

    def flush(self) -> tuple[str, str]:
        """스트림이 끝났다. 들고 있던 꼬리를 지금 상태대로 내보낸다."""
        rest, self._buffer = self._buffer, ""
        return (rest, "") if self.in_think else ("", rest)

    def _tail_that_may_be_a_tag(self) -> int:
        """버퍼 끝이 `<think>`/`</think>`의 앞부분일 수 있으면 그 길이(다음 조각을 기다린다)."""
        for length in range(min(7, len(self._buffer)), 0, -1):
            if self._buffer[-length:].lower() in _TAG_PREFIXES:
                return length
        return 0
