"""SQLite(aiosqlite) 저장소 — 대화, 메시지, 첨부.

참고 구현(vectra-web `history.mjs`)을 단순화했다: 프로젝트/공유 JSON 미러는 없다.
API key는 어떤 테이블에도 저장하지 않는다.

첨부 바이트(Step 5-0): 새 첨부는 `data/files/{대화ID}/` 아래 **파일**로 쓰고 DB에는 상대 경로만 둔다.
그 전에 저장된 첨부는 `data` BLOB 컬럼에 그대로 있고, 옮기지 않는다 → 읽을 때 "경로가 있으면 파일, 없으면 BLOB".
어느 쪽이든 프론트로 base64를 되돌려 보내지 않고 `/api/attachments/{id}/content` URL로 참조한다.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

import aiosqlite

from . import config
from .attachments import Attachment
from .storage import FileStore, StorageError, extension_for, safe_filename

logger = logging.getLogger("docchat.db")

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  provider TEXT NOT NULL DEFAULT '',
  model TEXT NOT NULL DEFAULT '',
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  position INTEGER NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  artifacts_json TEXT NOT NULL DEFAULT '[]',
  files_json TEXT NOT NULL DEFAULT '[]',
  created_at INTEGER NOT NULL,
  meta_json TEXT NOT NULL DEFAULT '{}',
  UNIQUE(conversation_id, position)
);
CREATE TABLE IF NOT EXISTS attachments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  position INTEGER NOT NULL,
  name TEXT NOT NULL,
  mime TEXT NOT NULL,
  kind TEXT NOT NULL,
  size INTEGER NOT NULL DEFAULT 0,
  text_content TEXT NOT NULL DEFAULT '',
  data BLOB,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  file_path TEXT,
  source_path TEXT,
  UNIQUE(conversation_id, name)
);
CREATE TABLE IF NOT EXISTS turn_traces (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'running',
  events_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, position);
CREATE INDEX IF NOT EXISTS idx_attachments_conversation ON attachments(conversation_id, position);
CREATE INDEX IF NOT EXISTS idx_turn_traces_conversation ON turn_traces(conversation_id, created_at);
"""

# 이미 만들어진 DB에는 CREATE TABLE IF NOT EXISTS가 컬럼을 더해 주지 않는다 → 없는 컬럼만 덧붙인다.
_ADDED_COLUMNS = (
    ("attachments", "file_path", "TEXT"),            # 본 파일(모델 전송·뷰어용)의 상대 경로
    ("attachments", "source_path", "TEXT"),          # 업로드 이미지 원본의 상대 경로(사본과 다를 때만)
    ("messages", "meta_json", "TEXT NOT NULL DEFAULT '{}'"),   # 답변을 어떤 모드로 처리했는지
)

_ID_PATTERN = re.compile(r"^[a-zA-Z0-9-]{8,80}$")
_BOX_KEYS = ("x", "y", "w", "h")
_PAGE_IMAGE_NAME = re.compile(r"^(?P<root>.*) · page (?P<number>\d+)$")
_META_COUNTERS = ("ocrCalls", "groundingCalls", "answerCalls", "tiledImages", "tiles", "blankTiles",
                  "ocrLengthStops", "groundingLengthStops",
                  "answerReasoningForced", "groundingReasoningForced", "ocrReasoningForced",
                  "answerReasoningStops", "groundingReasoningStops", "ocrReasoningStops", "reasoningTokens")
_META_REASONING_SETTINGS = ("repeatLines", "repeatCount", "repeatMinChars")
_META_TILE_SETTINGS = ("tileSize", "overlap", "renderDpi", "minSourceEdge", "maxTiles")
_META_THINKING_CALLS = ("answer", "grounding", "ocr")


def valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_ID_PATTERN.match(value))


def now_ms() -> int:
    return int(time.time() * 1000)


def clean_title(value: Any) -> str:
    title = re.sub(r"\s+", " ", str(value or "")).strip()
    return (title or "새 채팅")[:80]


def _clean_field(value: Any, length: int) -> str:
    return str(value or "").strip()[:length]


def sanitize_artifacts(value: Any) -> list[dict[str, Any]]:
    """아티팩트는 이미지 바이트 대신 attachmentId로 이미지를 가리킨다."""
    if not isinstance(value, list):
        return []
    result = []
    for artifact in value[:12]:
        if not isinstance(artifact, dict):
            continue
        item: dict[str, Any] = {
            "name": _clean_field(artifact.get("name") or "artifact", 240),
            "mime": _clean_field(artifact.get("mime") or "application/octet-stream", 160),
        }
        if artifact.get("view") == "image":
            item["view"] = "image"
        if artifact.get("title"):
            item["title"] = _clean_field(artifact["title"], 240)
        if artifact.get("task"):
            item["task"] = _clean_field(artifact["task"], 500)
        if artifact.get("text"):
            item["text"] = str(artifact["text"])[:20_000]
        if isinstance(artifact.get("attachmentId"), int):
            item["attachmentId"] = artifact["attachmentId"]
        boxes = []
        for box in (artifact.get("boxes") or [])[:200] if isinstance(artifact.get("boxes"), list) else []:
            if not isinstance(box, dict):
                continue
            try:
                clean = {key: float(box[key]) for key in _BOX_KEYS}
            except (KeyError, TypeError, ValueError):
                continue
            if box.get("label"):
                clean["label"] = _clean_field(box["label"], 80)
            if box.get("type"):
                clean["type"] = _clean_field(box["type"], 24)
            if isinstance(box.get("confidence"), (int, float)):
                clean["confidence"] = max(0.0, min(1.0, float(box["confidence"])))
            boxes.append(clean)
        if boxes:
            item["boxes"] = boxes
        result.append(item)
    return result


def sanitize_files(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result = []
    for file in value[:24]:
        if not isinstance(file, dict):
            continue
        item: dict[str, Any] = {
            "name": _clean_field(file.get("name") or "attachment", 240),
            "kind": _clean_field(file.get("kind") or "binary", 24),
            "size": max(0, int(file.get("size") or 0)) if isinstance(file.get("size"), (int, float)) else 0,
        }
        if isinstance(file.get("attachmentId"), int):
            item["attachmentId"] = file["attachmentId"]
        if file.get("mime"):
            item["mime"] = _clean_field(file["mime"], 160)
        result.append(item)
    return result


def _bounded_number(value: Any, high: float) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        return None
    return max(0, min(high, value))


def sanitize_meta(value: Any) -> dict[str, Any]:
    """답변 메타데이터: 어떤 이미지 처리 모드로, 비전 호출을 몇 번 써서 만든 답인지. 아는 항목만 남긴다."""
    if not isinstance(value, dict):
        return {}
    meta: dict[str, Any] = {}
    if value.get("imageMode") in config.IMAGE_MODES:
        meta["imageMode"] = value["imageMode"]
    for key, names, high in (("tiling", _META_TILE_SETTINGS, 1_000_000), ("vision", _META_COUNTERS, 1_000_000_000)):
        source = value.get(key)
        if isinstance(source, dict):
            numbers = {name: _bounded_number(source.get(name), high) for name in names}
            numbers = {name: number for name, number in numbers.items() if number is not None}
            if numbers:
                meta[key] = numbers
    elapsed = _bounded_number(value.get("elapsedMs"), 86_400_000)
    if elapsed is not None:
        meta["elapsedMs"] = int(elapsed)
    # 호출 종류별로 추론을 끄고 보냈는지, 전사·bbox 호출의 출력 상한(Step 6-0)
    thinking = value.get("thinkingDisabled")
    if isinstance(thinking, dict):
        flags = {name: thinking[name] for name in _META_THINKING_CALLS if isinstance(thinking.get(name), bool)}
        if flags:
            meta["thinkingDisabled"] = flags
    limit = _bounded_number(value.get("visionMaxTokens"), 1_000_000)
    if limit:
        meta["visionMaxTokens"] = int(limit)
    # 추론 제어 설정(Step 6): 호출 종류별 추론 예산과 반복 감지 기준
    reasoning = value.get("reasoning")
    if isinstance(reasoning, dict):
        budget = reasoning.get("budget")
        budgets = {kind: _bounded_number((budget or {}).get(kind), 10_000_000) for kind in config.REASONING_KINDS} \
            if isinstance(budget, dict) else {}
        settings = {name: _bounded_number(reasoning.get(name), 10_000) for name in _META_REASONING_SETTINGS}
        if all(number is not None for number in budgets.values()) and all(number is not None for number in settings.values()):
            meta["reasoning"] = {"budget": {kind: int(number) for kind, number in budgets.items()},
                                 **{name: int(number) for name, number in settings.items()}}
    actions = value.get("reasoningActions")
    if isinstance(actions, list):
        kept = [{"kind": item["kind"], "image": str(item.get("image") or "")[:300],
                 "reason": str(item.get("reason") or "")[:40], "stopped": bool(item.get("stopped"))}
                for item in actions[:config.MAX_REASONING_ACTIONS_IN_META]
                if isinstance(item, dict) and item.get("kind") in config.REASONING_KINDS]
        if kept:
            meta["reasoningActions"] = kept
    # 답변 호출에 실은 이미지(Step 8): 모드, 실은 수·이름, 모드상 실을 수 있었던 수
    if value.get("answerImageMode") in config.ANSWER_IMAGE_MODES:
        meta["answerImageMode"] = value["answerImageMode"]
    images = value.get("answerImages")
    if isinstance(images, dict):
        counts = {name: _bounded_number(images.get(name), 1_000_000) for name in ("sent", "candidates")}
        if all(number is not None for number in counts.values()):
            names = images.get("names")
            meta["answerImages"] = {
                **{name: int(number) for name, number in counts.items()},
                "names": [str(item)[:300] for item in names[:200] if isinstance(item, str)] if isinstance(names, list) else [],
            }
    # 이 답을 만든 턴의 트레이스(Step 7). 트레이스를 켠 턴에만 있다.
    if valid_id(value.get("traceId")):
        meta["traceId"] = value["traceId"]
    return meta


def sanitize_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result = []
    for message in value[:500]:
        if not isinstance(message, dict):
            continue
        created = message.get("createdAt")
        is_assistant = message.get("role") == "assistant"
        result.append({
            "role": "assistant" if is_assistant else "user",
            "content": str(message.get("content") or "")[:4_000_000],
            "artifacts": sanitize_artifacts(message.get("artifacts")),
            "files": sanitize_files(message.get("files")),
            "meta": sanitize_meta(message.get("meta")) if is_assistant else {},
            "createdAt": int(created) if isinstance(created, (int, float)) and created > 0 else now_ms(),
        })
    return result


def derive_title(messages: Iterable[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user" and str(message.get("content") or "").strip():
            return clean_title(message["content"])
    return "새 채팅"


def _json_list(value: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _json_dict(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def storage_names(attachment: Attachment) -> tuple[str, str | None]:
    """(본 파일 이름, 원본 파일 이름 또는 None). 탐색기에서 바로 알아볼 수 있는 이름을 쓴다.

      scan.pdf                 → scan.pdf
      scan.pdf · page 2        → scan.pdf.page-0002.png
      plan.tif (줄여서 전달)     → plan.model.png  +  plan.tif(원본)
    """
    extension = extension_for(attachment.mime)
    page = _PAGE_IMAGE_NAME.match(attachment.name)
    if page:
        return f"{safe_filename(page['root'])}.page-{int(page['number']):04d}.{extension}", None
    name = safe_filename(attachment.name)
    stem, dot, _ = name.rpartition(".")
    if not dot:
        stem, name = name, f"{name}.{extension}"
    if attachment.source_data is None:
        return name, None
    return f"{stem}.model.{extension}", name


class ChatStore:
    """단일 연결을 공유한다. 여러 문장으로 이뤄진 쓰기는 `_write_lock`으로 묶어 트랜잭션이 섞이지 않게 한다."""

    def __init__(self, path: Path | str, files_dir: Path | str | None = None):
        self.in_memory = str(path) == ":memory:"   # 테스트용. 연결 하나를 공유하므로 메모리 DB도 그대로 동작한다.
        self.path = Path(path)
        # 메모리 DB는 닫으면 사라진다 → 폴더를 따로 지정하지 않았으면 파일도 임시 폴더에 두고 함께 지운다.
        self._temporary_files = files_dir is None and self.in_memory
        self._files_dir = Path(files_dir) if files_dir is not None else (None if self.in_memory else self.path.parent / "files")
        self._files: FileStore | None = None
        self._db: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()
        self._last_stamp = 0

    def _next_stamp(self) -> int:
        """같은 밀리초 안의 연속 쓰기에서도 '마지막 수정 순' 정렬이 뒤집히지 않도록 단조 증가시킨다."""
        self._last_stamp = max(now_ms(), self._last_stamp + 1)
        return self._last_stamp

    async def open(self) -> "ChatStore":
        if self.in_memory:
            self._db = await aiosqlite.connect(":memory:")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        # WAL에서는 NORMAL이 표준 권장값이다: 손상 위험 없이 커밋마다의 fsync를 없앤다.
        # (이 PC의 D: 드라이브에서 커밋 1회 182ms → 1ms. 정전 시 마지막 커밋 몇 개가 사라질 수 있을 뿐이다.)
        await self._db.execute("PRAGMA synchronous = NORMAL")
        await self._db.executescript(_SCHEMA)
        for table, column, definition in _ADDED_COLUMNS:
            cursor = await self._db.execute(f"PRAGMA table_info({table})")
            if column not in {row["name"] for row in await cursor.fetchall()}:
                await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.commit()
        if self._temporary_files:
            self._files_dir = Path(tempfile.mkdtemp(prefix="docchat-files-"))
        self._files = FileStore(self._files_dir)
        return self

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None
        if self._temporary_files and self._files_dir is not None:
            shutil.rmtree(self._files_dir, ignore_errors=True)
            self._files_dir = None
        self._files = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("데이터베이스가 열려 있지 않습니다.")
        return self._db

    @property
    def files(self) -> FileStore:
        if self._files is None:
            raise RuntimeError("첨부 파일 저장소가 열려 있지 않습니다.")
        return self._files

    # ------------------------------------------------------------------ 대화
    async def list_conversations(self, limit: Any = 100) -> list[dict[str, Any]]:
        try:
            bounded = max(1, min(500, int(limit)))
        except (TypeError, ValueError):
            bounded = 100
        cursor = await self.db.execute(
            """
            SELECT c.id, c.title, c.provider, c.model, c.created_at AS createdAt, c.updated_at AS updatedAt,
                   (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS messageCount,
                   (SELECT COUNT(*) FROM attachments a WHERE a.conversation_id = c.id AND a.kind != 'document'
                           AND a.name NOT LIKE '% · page %') AS fileCount
            FROM conversations c ORDER BY c.updated_at DESC, c.rowid DESC LIMIT ?
            """,
            (bounded,),
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def conversation_exists(self, conversation_id: str) -> bool:
        cursor = await self.db.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,))
        return await cursor.fetchone() is not None

    async def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        if not valid_id(conversation_id):
            return None
        cursor = await self.db.execute(
            "SELECT id, title, provider, model, created_at AS createdAt, updated_at AS updatedAt "
            "FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        conversation = dict(row)
        cursor = await self.db.execute(
            "SELECT role, content, artifacts_json, files_json, meta_json, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY position",
            (conversation_id,),
        )
        conversation["messages"] = [
            {
                "role": item["role"],
                "content": item["content"],
                "artifacts": _json_list(item["artifacts_json"]),
                "files": _json_list(item["files_json"]),
                "meta": _json_dict(item["meta_json"]),
                "createdAt": item["created_at"],
            }
            for item in await cursor.fetchall()
        ]
        attachments = await self.list_attachments(conversation_id)
        conversation["attachments"] = [attachment.to_public() for attachment in attachments]
        return conversation

    async def save_conversation(
        self,
        *,
        conversation_id: str | None = None,
        title: str | None = None,
        provider: str = "",
        model: str = "",
        messages: Any = None,
    ) -> str:
        """대화를 upsert하고, messages가 주어지면 메시지 목록 전체를 교체한다. 첨부는 건드리지 않는다."""
        identifier = conversation_id if valid_id(conversation_id) else str(uuid.uuid4())
        clean_messages = sanitize_messages(messages) if messages is not None else None
        async with self._write_lock:
            stamp = self._next_stamp()
            cursor = await self.db.execute("SELECT title, created_at FROM conversations WHERE id = ?", (identifier,))
            existing = await cursor.fetchone()
            if title and str(title).strip():
                final_title = clean_title(title)
            elif existing is not None and existing["title"] != "새 채팅":
                final_title = existing["title"]
            else:
                final_title = derive_title(clean_messages or [])
            try:
                await self.db.execute(
                    """
                    INSERT INTO conversations (id, title, provider, model, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET title = excluded.title, provider = excluded.provider,
                      model = excluded.model, updated_at = excluded.updated_at
                    """,
                    (identifier, final_title, _clean_field(provider, 80), _clean_field(model, 240),
                     existing["created_at"] if existing is not None else stamp, stamp),
                )
                if clean_messages is not None:
                    await self.db.execute("DELETE FROM messages WHERE conversation_id = ?", (identifier,))
                    await self.db.executemany(
                        "INSERT INTO messages (conversation_id, position, role, content, artifacts_json, files_json, "
                        "meta_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            (identifier, position, message["role"], message["content"],
                             json.dumps(message["artifacts"], ensure_ascii=False),
                             json.dumps(message["files"], ensure_ascii=False),
                             json.dumps(message["meta"], ensure_ascii=False), message["createdAt"])
                            for position, message in enumerate(clean_messages)
                        ],
                    )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise
        return identifier

    async def delete_conversation(self, conversation_id: str) -> bool:
        if not valid_id(conversation_id):
            return False
        async with self._write_lock:
            cursor = await self.db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            await self.db.commit()
            deleted = cursor.rowcount > 0
        if deleted:
            # 폴더를 못 지워도(파일이 열려 있는 등) 대화 삭제는 끝난 것으로 본다. 실패는 저장소가 로그로 남긴다.
            await asyncio.to_thread(self.files.remove_conversation, conversation_id)
        return deleted

    async def delete_many(self, ids: Iterable[Any]) -> int:
        count = 0
        for identifier in dict.fromkeys(ids):
            if valid_id(identifier) and await self.delete_conversation(identifier):
                count += 1
        return count

    async def delete_all(self) -> int:
        async with self._write_lock:
            cursor = await self.db.execute("SELECT id FROM conversations")
            identifiers = [row["id"] for row in await cursor.fetchall()]
            cursor = await self.db.execute("DELETE FROM conversations")
            await self.db.commit()
            deleted = cursor.rowcount
        # DB에 있던 대화의 폴더만 지운다. 저장 폴더에 있는 그 밖의 것은 건드리지 않는다.
        for identifier in identifiers:
            await asyncio.to_thread(self.files.remove_conversation, identifier)
        return deleted

    # ------------------------------------------------------------------ 첨부
    async def list_attachments(self, conversation_id: str) -> list[Attachment]:
        """메타데이터와 텍스트만 읽는다. 바이트는 필요할 때 `load_attachment_data`로 가져온다."""
        cursor = await self.db.execute(
            "SELECT id, name, mime, kind, size, text_content, metadata_json, file_path, source_path, "
            "(data IS NOT NULL OR file_path IS NOT NULL) AS has_data "
            "FROM attachments WHERE conversation_id = ? ORDER BY position, id",
            (conversation_id,),
        )
        return [
            Attachment(
                id=row["id"], name=row["name"], mime=row["mime"], kind=row["kind"], size=row["size"],
                text=row["text_content"], has_data=bool(row["has_data"]),
                file_path=row["file_path"], source_path=row["source_path"],
            ).apply_metadata(_json_dict(row["metadata_json"]))
            for row in await cursor.fetchall()
        ]

    async def _read_file(self, relative: str, attachment_id: int) -> bytes | None:
        try:
            return await asyncio.to_thread(self.files.read, relative)
        except StorageError as error:
            # DB 값이 저장 폴더 밖을 가리킨다 → 읽지 않는다.
            logger.warning("첨부 %s의 경로를 거부했습니다: %s", attachment_id, error)
            return None

    async def load_attachment_data(self, attachment_id: int) -> bytes | None:
        """본 파일(모델 전송·뷰어용)의 바이트. 경로가 있으면 파일, 없으면 예전 방식의 BLOB."""
        cursor = await self.db.execute("SELECT data, file_path FROM attachments WHERE id = ?", (attachment_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        if row["file_path"]:
            return await self._read_file(row["file_path"], attachment_id)
        return bytes(row["data"]) if row["data"] is not None else None

    async def load_attachment_source(self, attachment_id: int) -> tuple[bytes, str] | None:
        """업로드 원본의 (바이트, MIME). 원본을 따로 두지 않은 첨부(사본과 같거나 예전 BLOB)는 본 파일을 돌려준다."""
        cursor = await self.db.execute(
            "SELECT mime, metadata_json, source_path FROM attachments WHERE id = ?", (attachment_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        if row["source_path"]:
            data = await self._read_file(row["source_path"], attachment_id)
            if data is not None:
                return data, str(_json_dict(row["metadata_json"]).get("sourceMime") or row["mime"])
        data = await self.load_attachment_data(attachment_id)
        return (data, row["mime"]) if data is not None else None

    async def get_attachment_content(self, attachment_id: int) -> tuple[str, str, bytes] | None:
        cursor = await self.db.execute("SELECT name, mime FROM attachments WHERE id = ?", (attachment_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        data = await self.load_attachment_data(attachment_id)
        return (row["name"], row["mime"], data) if data is not None else None

    async def _write_files(self, conversation_id: str, attachment: Attachment, written: list[str]) -> None:
        """새 첨부의 바이트를 파일로 쓴다. 업로드 이미지는 원본과 모델 전송용 사본을 따로 둔다."""
        primary, source = storage_names(attachment)
        if source is not None and attachment.source_data is not None:
            attachment.source_path = await asyncio.to_thread(
                self.files.write, conversation_id, source, attachment.source_data)
            written.append(attachment.source_path)
        attachment.file_path = await asyncio.to_thread(self.files.write, conversation_id, primary, attachment.data)
        written.append(attachment.file_path)

    async def save_attachments(self, conversation_id: str, attachments: list[Attachment]) -> None:
        """목록 순서대로 position을 매긴다. 새 항목은 바이트를 파일로 쓰고 경로와 함께 INSERT, 기존 항목은 텍스트·메타만 UPDATE."""
        written: list[str] = []
        async with self._write_lock:
            try:
                for position, attachment in enumerate(attachments):
                    if attachment.id is None and attachment.data is not None:
                        await self._write_files(conversation_id, attachment, written)
                    metadata = json.dumps(attachment.metadata(), ensure_ascii=False)
                    if attachment.id is None:
                        cursor = await self.db.execute(
                            """
                            INSERT INTO attachments (conversation_id, position, name, mime, kind, size, text_content,
                                                     data, metadata_json, file_path, source_path)
                            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                            ON CONFLICT(conversation_id, name) DO UPDATE SET position = excluded.position,
                              mime = excluded.mime, kind = excluded.kind, size = excluded.size,
                              text_content = excluded.text_content, data = NULL,
                              metadata_json = excluded.metadata_json, file_path = excluded.file_path,
                              source_path = excluded.source_path
                            RETURNING id
                            """,
                            (conversation_id, position, attachment.name, attachment.mime, attachment.kind,
                             attachment.size, attachment.text or "", metadata, attachment.file_path,
                             attachment.source_path),
                        )
                        row = await cursor.fetchone()
                        attachment.id = row["id"]
                        attachment.has_data = attachment.file_path is not None
                        attachment.source_data = None      # 원본은 디스크에 있다. 큰 바이트를 메모리에 붙들지 않는다.
                    else:
                        await self.db.execute(
                            "UPDATE attachments SET position = ?, mime = ?, kind = ?, size = ?, text_content = ?, metadata_json = ? "
                            "WHERE id = ? AND conversation_id = ?",
                            (position, attachment.mime, attachment.kind, attachment.size,
                             attachment.text or "", metadata, attachment.id, conversation_id),
                        )
                await self.db.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?", (self._next_stamp(), conversation_id)
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                # DB에 기록되지 못한 파일은 아무도 가리키지 않는다 → 이번 호출이 쓴 것만 되돌린다.
                for relative in written:
                    await asyncio.to_thread(self.files.discard, relative)
                raise

    async def delete_attachments(self, conversation_id: str, attachment_ids: Iterable[int]) -> None:
        identifiers = [identifier for identifier in attachment_ids if isinstance(identifier, int)]
        if not identifiers:
            return
        async with self._write_lock:
            await self.db.executemany(
                "DELETE FROM attachments WHERE id = ? AND conversation_id = ?",
                [(identifier, conversation_id) for identifier in identifiers],
            )
            await self.db.commit()

    # ------------------------------------------------------------------ 턴 트레이스(Step 7, 개발용)
    async def save_trace(self, trace_id: str, conversation_id: str, status: str, document_json: str,
                         created_at: int) -> None:
        """턴이 진행되는 동안 여러 번 덮어쓴다(같은 id). 대화를 지우면 함께 지워진다(FK CASCADE)."""
        if not valid_id(trace_id) or not valid_id(conversation_id):
            return
        async with self._write_lock:
            await self.db.execute(
                """
                INSERT INTO turn_traces (id, conversation_id, created_at, updated_at, status, events_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at, status = excluded.status,
                  events_json = excluded.events_json
                """,
                (trace_id, conversation_id, int(created_at), now_ms(), _clean_field(status, 24), document_json),
            )
            await self.db.commit()

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        if not valid_id(trace_id):
            return None
        cursor = await self.db.execute("SELECT events_json FROM turn_traces WHERE id = ?", (trace_id,))
        row = await cursor.fetchone()
        return _json_dict(row["events_json"]) if row is not None else None

    async def interrupt_running_traces(self, rewrite: Callable[[dict[str, Any]], dict[str, Any]]) -> int:
        """서버 시작 때: 지난 프로세스가 `running`으로 남긴 트레이스를 정리한다(살아 있을 수 없다). 정리한 수를 돌려준다."""
        cursor = await self.db.execute("SELECT id, events_json FROM turn_traces WHERE status = 'running'")
        rows = await cursor.fetchall()
        if not rows:
            return 0
        async with self._write_lock:
            for row in rows:
                document = rewrite(_json_dict(row["events_json"]))
                await self.db.execute(
                    "UPDATE turn_traces SET status = ?, updated_at = ?, events_json = ? WHERE id = ?",
                    (str(document.get("status") or "interrupted"), now_ms(),
                     json.dumps(document, ensure_ascii=False, default=str), row["id"]),
                )
            await self.db.commit()
        return len(rows)

    async def list_traces(self, conversation_id: str) -> list[dict[str, Any]]:
        if not valid_id(conversation_id):
            return []
        cursor = await self.db.execute(
            "SELECT id, created_at AS createdAt, updated_at AS updatedAt, status FROM turn_traces "
            "WHERE conversation_id = ? ORDER BY created_at, rowid", (conversation_id,))
        return [dict(row) for row in await cursor.fetchall()]
