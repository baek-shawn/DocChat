"""개발용 턴 트레이스(Step 7) — 한 턴(요청 하나)의 과정을 시간순 이벤트로 기록한다.

무엇을 위한 것인가: 도면을 왜 잘/못 읽었는지, 어떤 정보를 어디서 가져왔는지, 왜 빠르거나 느렸는지를 **한 턴 단위로** 본다.
기존 대화 데이터와 분리돼 있어(`turn_traces` 테이블) 개발이 끝나면 지울 수 있다.

기록 방식
  - `DOCCHAT_DEBUG_TRACE=1`일 때만 `chat_service`가 `TurnTrace`를 만들어 이 모듈의 컨텍스트 변수에 건다.
    그 밖의 코드는 `note()`/`scope()`만 부른다 — 트레이스가 없으면 아무 일도 하지 않는다(비용 없음).
  - 모델 호출·도구 실행은 **시작할 때 먼저 기록**하고 끝나면 결과를 채운다. 끝나지 않은 항목은 `running`으로 남아
    화면에서 "진행 중 · 경과 n초"로 보인다. 요청이 취소되거나 예외로 끝나면 남은 항목에 그 사유를 적는다.
  - 이벤트가 생길 때마다 잠시 모았다가(`FLUSH_DELAY`) DB에 쓴다. 턴이 끝나기 전에도 저장되므로 서버가 내려가도
    그때까지의 기록은 남는다.
  - 이미지는 첨부 ID·타일 위치로만 가리킨다(base64를 다시 저장하지 않는다). API key는 어디에도 적지 않는다.
  - 기록 코드의 예외는 여기서 삼킨다. **측정 도구가 앱의 동작을 바꾸면 안 된다**(CLAUDE.md §5).

이 모듈은 `config`만 가져온다(import 방향: `config ← trace ← 나머지`). 모델 호출을 감싸는 `TracedProvider`는
provider 기반 클래스가 필요해 `providers/traced.py`에 있다.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

from . import config

logger = logging.getLogger("docchat.trace")

TRACE_VERSION = 1
FLUSH_DELAY = 0.25          # 이벤트를 모아 DB에 쓰기까지 기다리는 시간(초)
# 이벤트 종류 — 화면의 아이콘·색과 짝이다. 새 종류를 더하면 static/js/app.js의 TRACE_KINDS도 함께 고친다.
KINDS = ("input", "preprocess", "ocr", "evidence", "model", "tool", "loop", "cleanup", "progress", "files", "answer")
STATUSES = ("running", "done", "failed", "cancelled", "interrupted")
CANCELLED_REASON = "요청이 취소됐습니다."
INTERRUPTED_REASON = "서버가 다시 시작돼 기록이 끊겼습니다. 이 턴이 어떻게 끝났는지는 알 수 없습니다."

Writer = Callable[[str, str, str, str, int], Awaitable[None]]     # (trace_id, conversation_id, status, json, created_at)

_CURRENT: ContextVar["TurnTrace | None"] = ContextVar("docchat_trace", default=None)
_PARENT: ContextVar[int | None] = ContextVar("docchat_trace_parent", default=None)
# 이 프로세스에서 지금 진행 중인 턴. DB의 `running`만으로는 "정말 진행 중"과 "끝났는데 기록만 남음"을 가를 수 없다
# → API가 여기 있는지(`live`)를 함께 알려 준다.
LIVE: dict[str, "TurnTrace"] = {}


def is_live(trace_id: str) -> bool:
    return trace_id in LIVE


def interrupt_document(document: dict[str, Any]) -> dict[str, Any]:
    """서버가 다시 시작된 뒤 `running`으로 남은 기록을 `interrupted`로 정리한다(사유를 적고, 열린 이벤트도 닫는다)."""
    document = dict(document)
    document["status"] = "interrupted"
    document["reason"] = INTERRUPTED_REASON
    ended = document.get("elapsedMs") or 0
    for event in document.get("events") or []:
        if event.get("status") == "running":
            event["status"] = "interrupted"
            event["endedMs"] = max(ended, event.get("startedMs") or 0)
            event["elapsedMs"] = event["endedMs"] - (event.get("startedMs") or 0)
            event.setdefault("data", {}).setdefault("reason", INTERRUPTED_REASON)
    return document


def clip(text: Any, limit: int | None = None) -> str:
    """기록용 글 자르기 — 앞·뒤를 남기고 가운데를 뺀다(원문의 시작과 끝을 함께 봐야 하는 경우가 많다)."""
    value = str(text if text is not None else "")
    limit = limit or config.TRACE_TEXT_LIMIT
    if len(value) <= limit:
        return value
    half = max(1, (limit - 60) // 2)
    return f"{value[:half]}\n…[{len(value) - half * 2:,}자 생략]…\n{value[-half:]}"


def describe_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """모델에 보낸 내부 메시지 목록을 기록용으로 옮긴다(내용은 잘라서, provider 원본 객체는 빼고)."""
    output: list[dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        item: dict[str, Any] = {"role": str(message.get("role") or ""), "content": clip(message.get("content") or ""),
                                "chars": len(str(message.get("content") or ""))}
        calls = message.get("tool_calls") or []
        if calls:
            item["toolCalls"] = [describe_tool_call(call) for call in calls]
        if message.get("role") == "tool":
            item["name"] = str(message.get("name") or "")
            item["toolCallId"] = str(message.get("tool_call_id") or "")
        if message.get("images_anchor"):
            item["imagesAnchor"] = True
        output.append(item)
    return output


def describe_tool_call(call: Any) -> dict[str, Any]:
    """`ToolCall`(이름·인자·id)이든 같은 모양의 dict든 받아 준다."""
    if isinstance(call, dict):
        name, arguments, identifier = call.get("name"), call.get("arguments"), call.get("id")
    else:
        name, arguments, identifier = getattr(call, "name", ""), getattr(call, "arguments", {}), getattr(call, "id", "")
    return {"id": str(identifier or ""), "name": str(name or ""), "arguments": arguments if isinstance(arguments, dict) else {}}


def describe_tools(tools: list[Any] | None) -> list[str]:
    return [str(getattr(tool, "name", "") or "") for tool in tools or []]


@dataclass
class TraceEvent:
    id: int
    kind: str
    label: str
    started_ms: int
    ended_ms: int | None = None
    status: str = "running"
    parent: int | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_public(self) -> dict[str, Any]:
        item: dict[str, Any] = {"id": self.id, "kind": self.kind, "label": self.label, "status": self.status,
                                "startedMs": self.started_ms, "endedMs": self.ended_ms}
        if self.ended_ms is not None:
            item["elapsedMs"] = self.ended_ms - self.started_ms
        if self.parent is not None:
            item["parent"] = self.parent
        if self.data:
            item["data"] = self.data
        return item


class Span:
    """`scope()`가 돌려주는 손잡이. 본문에서 `span.set(...)`으로 결과를 채운다. 트레이스가 없으면 아무것도 하지 않는다."""

    def __init__(self, turn: "TurnTrace | None", event: TraceEvent | None):
        self._turn, self._event = turn, event
        self.status: str | None = None      # 본문이 정한 종료 상태(없으면 done/failed/cancelled를 scope가 정한다)

    @property
    def event_id(self) -> int | None:
        return self._event.id if self._event is not None else None

    def set(self, **data: Any) -> None:
        if self._turn is not None and self._event is not None:
            self._turn.update(self._event, **data)


class TurnTrace:
    """한 턴의 기록. 이벤트는 이벤트 루프 스레드에서만 더해진다(동기 메서드, 잠금 없음)."""

    def __init__(self, conversation_id: str, writer: Writer | None = None, *, trace_id: str | None = None):
        self.id = trace_id or str(uuid.uuid4())
        self.conversation_id = conversation_id
        self.created_at = int(time.time() * 1000)
        self.updated_at = self.created_at
        self.status = "running"
        self.reason = ""
        self.events: list[TraceEvent] = []
        self._origin = time.monotonic()
        self._writer = writer
        self._dirty = False
        self._closed = False
        self._flush_task: asyncio.Task[None] | None = None
        self._flush_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._token: Any = None
        self._attachments: dict[str, dict[str, Any]] = {}     # 이미지 이름 → {id, width, height}
        self._counts: dict[str, int] = {}

    # ------------------------------------------------------------------ 활성화
    def activate(self) -> None:
        """이 태스크(와 여기서 만든 하위 태스크)의 `note()`/`scope()`가 이 트레이스에 기록되게 한다."""
        self._token = _CURRENT.set(self)
        LIVE[self.id] = self

    def deactivate(self) -> None:
        LIVE.pop(self.id, None)          # close()를 못 거쳤어도 "진행 중"으로 남기지 않는다
        if self._token is not None:
            try:
                _CURRENT.reset(self._token)
            except ValueError:      # 다른 컨텍스트에서 풀려도 트레이스 자체는 멀쩡하다
                _CURRENT.set(None)
            self._token = None

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._origin) * 1000)

    def count(self, key: str) -> int:
        """같은 종류의 이벤트에 번호를 붙일 때 쓴다(답변 호출 #2)."""
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def find(self, event_id: int | None) -> TraceEvent | None:
        if event_id is None or event_id < 1 or event_id > len(self.events):
            return None
        event = self.events[event_id - 1]
        return event if event.id == event_id else None

    def parent(self) -> TraceEvent | None:
        """지금 열려 있는 `scope()`의 이벤트(모델 호출이 어떤 단계 안에서 나갔는지 알 때 쓴다)."""
        return self.find(_PARENT.get())

    # ------------------------------------------------------------------ 이벤트
    def start(self, kind: str, label: str, /, *, parent: int | None = None, **data: Any) -> TraceEvent | None:
        try:
            event = TraceEvent(id=len(self.events) + 1, kind=kind if kind in KINDS else "progress", label=str(label),
                               started_ms=self.elapsed_ms(), parent=parent if parent is not None else _PARENT.get(),
                               data={key: value for key, value in data.items() if value is not None})
            self.events.append(event)
            self._touch()
            return event
        except Exception:       # 기록이 앱을 멈추게 하면 안 된다
            logger.debug("트레이스 이벤트를 만들지 못했습니다", exc_info=True)
            return None

    def update(self, event: TraceEvent | None, **data: Any) -> None:
        if event is None:
            return
        try:
            event.data.update({key: value for key, value in data.items() if value is not None})
            self._touch()
        except Exception:
            logger.debug("트레이스 이벤트를 갱신하지 못했습니다", exc_info=True)

    def finish(self, event: TraceEvent | None, status: str = "done", **data: Any) -> None:
        if event is None or event.ended_ms is not None:
            return
        try:
            event.ended_ms = self.elapsed_ms()
            event.status = status if status in STATUSES else "done"
            event.data.update({key: value for key, value in data.items() if value is not None})
            self._touch()
        except Exception:
            logger.debug("트레이스 이벤트를 끝내지 못했습니다", exc_info=True)

    def note(self, kind: str, label: str, /, **data: Any) -> TraceEvent | None:
        """시작과 끝이 같은 한 줄 기록."""
        event = self.start(kind, label, **data)
        if event is not None:
            event.ended_ms = event.started_ms
            event.status = "done"
        return event

    # ------------------------------------------------------------------ 첨부·이미지
    def register_attachments(self, attachments: list[Any]) -> None:
        """이미지 이름 → 첨부 ID. 모델 호출에 실린 이미지를 첨부 ID로 가리키기 위해서다."""
        for item in attachments:
            name = getattr(item, "name", None)
            if not name:
                continue
            self._attachments[str(name)] = {"id": getattr(item, "id", None), "width": getattr(item, "width", None),
                                            "height": getattr(item, "height", None)}

    def describe_image(self, image: Any) -> dict[str, Any]:
        """`ModelImage`(이름·mime·바이트·타일 위치)를 첨부 ID 참조로 옮긴다. 바이트 자체는 기록하지 않는다."""
        name = str(getattr(image, "name", "") or "")
        data = getattr(image, "data", b"") or b""
        item: dict[str, Any] = {"name": name, "mime": str(getattr(image, "mime", "") or ""), "bytes": len(data)}
        tile = getattr(image, "tile", None)
        root = name.split(" · tile ", 1)[0] if tile else name
        known = self._attachments.get(root)
        if known is not None and known.get("id") is not None:
            item["attachmentId"] = known["id"]
        width = getattr(image, "width", None) or (known or {}).get("width")
        height = getattr(image, "height", None) or (known or {}).get("height")
        if width and height:
            item["width"], item["height"] = width, height
        if tile:
            item["tile"] = list(tile)
            box = getattr(image, "source_box", None)
            if box:
                item["sourceBox"] = [round(float(value), 4) for value in box]
        return item

    def describe_images(self, images: list[Any] | None) -> list[dict[str, Any]]:
        return [self.describe_image(image) for image in images or []]

    # ------------------------------------------------------------------ 저장
    def to_document(self) -> dict[str, Any]:
        return {
            "version": TRACE_VERSION, "id": self.id, "conversationId": self.conversation_id,
            "createdAt": self.created_at, "updatedAt": self.updated_at, "status": self.status, "reason": self.reason,
            "elapsedMs": self.elapsed_ms(), "events": [event.to_public() for event in self.events],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_document(), ensure_ascii=False, default=str)

    def _touch(self) -> None:
        self._dirty = True
        if self._writer is None or self._closed or (self._flush_task is not None and not self._flush_task.done()):
            return
        try:
            self._flush_task = asyncio.get_running_loop().create_task(self._flush_later())
        except RuntimeError:        # 이벤트 루프 밖(테스트의 동기 호출) — close()에서 쓴다
            self._flush_task = None

    async def _flush_later(self) -> None:
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=FLUSH_DELAY)
        except asyncio.TimeoutError:
            pass
        await self.flush()

    async def flush(self) -> None:
        """지금까지의 기록을 DB에 쓴다. 실패해도 답변은 계속된다.

        저장은 한 번에 하나씩이고 JSON은 쓰기 직전에 만든다 → 나중 저장이 늘 더 새 상태다. (예전에는 먼저 시작된 저장이
        DB 잠금에 밀려 늦게 끝나면서 취소 기록을 `running`으로 되돌린 일이 있었다.)
        """
        if self._writer is None or not self._dirty:
            return
        async with self._flush_lock:
            if not self._dirty:
                return
            self._dirty = False
            self.updated_at = int(time.time() * 1000)
            try:
                await self._writer(self.id, self.conversation_id, self.status, self.to_json(), self.created_at)
            except asyncio.CancelledError:
                self._dirty = True
                raise
            except Exception as error:
                logger.warning("턴 트레이스를 저장하지 못했습니다: %s", error)

    async def close(self, status: str = "done", reason: str = "") -> None:
        """턴의 끝. 아직 running인 이벤트에 같은 상태·사유를 적고 마지막으로 저장한다.

        취소된 요청 안에서 불려도 마지막 저장은 끝까지 간다: 저장을 별도 태스크로 띄우고 `shield`로 기다린다.
        (anyio 취소 범위 안에서는 `await`마다 취소가 다시 던져져, 그냥 기다리면 저장이 끊긴다.)
        """
        if self._closed:
            return
        self._closed = True
        LIVE.pop(self.id, None)
        self.status = status if status in STATUSES else "done"
        self.reason = str(reason or "")
        now = self.elapsed_ms()
        for event in self.events:
            if event.ended_ms is None:
                event.ended_ms, event.status = now, self.status
                if self.reason and "reason" not in event.data:
                    event.data["reason"] = self.reason
        self._dirty = True
        self._wake.set()
        if self._writer is None:
            return
        final = asyncio.ensure_future(self.flush())
        try:
            await asyncio.shield(final)
        except asyncio.CancelledError:
            pass            # 요청은 취소됐지만 저장 태스크는 계속된다(잠금 순서상 마지막에 쓴다)


# --------------------------------------------------------------------------- 기록 지점이 부르는 것(트레이스가 없으면 무시)
def current() -> TurnTrace | None:
    return _CURRENT.get()


def note(kind: str, label: str, /, **data: Any) -> TraceEvent | None:
    turn = _CURRENT.get()
    return turn.note(kind, label, **data) if turn is not None else None


def start(kind: str, label: str, /, **data: Any) -> TraceEvent | None:
    turn = _CURRENT.get()
    return turn.start(kind, label, **data) if turn is not None else None


def finish(event: TraceEvent | None, status: str = "done", **data: Any) -> None:
    turn = _CURRENT.get()
    if turn is not None:
        turn.finish(event, status, **data)


@asynccontextmanager
async def scope(kind: str, label: str, /, **data: Any) -> AsyncIterator[Span]:
    """시작·끝이 있는 구간. 안에서 일어나는 기록(모델 호출 등)은 이 이벤트의 자식이 된다.

    정상으로 나가면 done, 예외면 failed(오류 문구 기록), 취소면 cancelled. 본문이 `span.status`를 정하면 그 값을 쓴다.
    """
    turn = _CURRENT.get()
    if turn is None:
        yield Span(None, None)
        return
    event = turn.start(kind, label, **data)
    span = Span(turn, event)
    token = _PARENT.set(event.id) if event is not None else None
    try:
        yield span
    except asyncio.CancelledError:
        turn.finish(event, "cancelled", reason=CANCELLED_REASON)
        raise
    except Exception as error:
        turn.finish(event, "failed", error=f"{type(error).__name__}: {error}"[:2000])
        raise
    else:
        turn.finish(event, span.status or "done")
    finally:
        if token is not None:
            _PARENT.reset(token)
