"""SQLite(aiosqlite) 저장소 — 대화, 메시지, 첨부.

참고 구현(vectra-web `history.mjs`)을 단순화했다: 프로젝트/공유 JSON 미러는 없고,
첨부 바이트는 base64 문자열이 아니라 BLOB으로 저장한다(프론트로 되돌려 보내지 않기 위해).
API key는 어떤 테이블에도 저장하지 않는다.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import aiosqlite

from .attachments import Attachment

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
  UNIQUE(conversation_id, name)
);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, position);
CREATE INDEX IF NOT EXISTS idx_attachments_conversation ON attachments(conversation_id, position);
"""

_ID_PATTERN = re.compile(r"^[a-zA-Z0-9-]{8,80}$")
_BOX_KEYS = ("x", "y", "w", "h")


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


def sanitize_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result = []
    for message in value[:500]:
        if not isinstance(message, dict):
            continue
        created = message.get("createdAt")
        result.append({
            "role": "assistant" if message.get("role") == "assistant" else "user",
            "content": str(message.get("content") or "")[:4_000_000],
            "artifacts": sanitize_artifacts(message.get("artifacts")),
            "files": sanitize_files(message.get("files")),
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


class ChatStore:
    """단일 연결을 공유한다. 여러 문장으로 이뤄진 쓰기는 `_write_lock`으로 묶어 트랜잭션이 섞이지 않게 한다."""

    def __init__(self, path: Path | str):
        self.in_memory = str(path) == ":memory:"   # 테스트용. 연결 하나를 공유하므로 메모리 DB도 그대로 동작한다.
        self.path = Path(path)
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
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("데이터베이스가 열려 있지 않습니다.")
        return self._db

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
            "SELECT role, content, artifacts_json, files_json, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY position",
            (conversation_id,),
        )
        conversation["messages"] = [
            {
                "role": item["role"],
                "content": item["content"],
                "artifacts": _json_list(item["artifacts_json"]),
                "files": _json_list(item["files_json"]),
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
                        "INSERT INTO messages (conversation_id, position, role, content, artifacts_json, files_json, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [
                            (identifier, position, message["role"], message["content"],
                             json.dumps(message["artifacts"], ensure_ascii=False),
                             json.dumps(message["files"], ensure_ascii=False), message["createdAt"])
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
            return cursor.rowcount > 0

    async def delete_many(self, ids: Iterable[Any]) -> int:
        count = 0
        for identifier in dict.fromkeys(ids):
            if valid_id(identifier) and await self.delete_conversation(identifier):
                count += 1
        return count

    async def delete_all(self) -> int:
        async with self._write_lock:
            cursor = await self.db.execute("DELETE FROM conversations")
            await self.db.commit()
            return cursor.rowcount

    # ------------------------------------------------------------------ 첨부
    async def list_attachments(self, conversation_id: str) -> list[Attachment]:
        """메타데이터와 텍스트만 읽는다. 바이트는 필요할 때 `load_attachment_data`로 가져온다."""
        cursor = await self.db.execute(
            "SELECT id, name, mime, kind, size, text_content, metadata_json, data IS NOT NULL AS has_data "
            "FROM attachments WHERE conversation_id = ? ORDER BY position, id",
            (conversation_id,),
        )
        return [
            Attachment(
                id=row["id"], name=row["name"], mime=row["mime"], kind=row["kind"], size=row["size"],
                text=row["text_content"], has_data=bool(row["has_data"]),
            ).apply_metadata(_json_dict(row["metadata_json"]))
            for row in await cursor.fetchall()
        ]

    async def load_attachment_data(self, attachment_id: int) -> bytes | None:
        cursor = await self.db.execute("SELECT data FROM attachments WHERE id = ?", (attachment_id,))
        row = await cursor.fetchone()
        return bytes(row["data"]) if row is not None and row["data"] is not None else None

    async def get_attachment_content(self, attachment_id: int) -> tuple[str, str, bytes] | None:
        cursor = await self.db.execute("SELECT name, mime, data FROM attachments WHERE id = ?", (attachment_id,))
        row = await cursor.fetchone()
        if row is None or row["data"] is None:
            return None
        return row["name"], row["mime"], bytes(row["data"])

    async def save_attachments(self, conversation_id: str, attachments: list[Attachment]) -> None:
        """목록 순서대로 position을 매긴다. 새 항목은 바이트와 함께 INSERT, 기존 항목은 텍스트·메타만 UPDATE."""
        async with self._write_lock:
            try:
                for position, attachment in enumerate(attachments):
                    metadata = json.dumps(attachment.metadata(), ensure_ascii=False)
                    if attachment.id is None:
                        cursor = await self.db.execute(
                            """
                            INSERT INTO attachments (conversation_id, position, name, mime, kind, size, text_content, data, metadata_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(conversation_id, name) DO UPDATE SET position = excluded.position,
                              mime = excluded.mime, kind = excluded.kind, size = excluded.size,
                              text_content = excluded.text_content, data = excluded.data,
                              metadata_json = excluded.metadata_json
                            RETURNING id
                            """,
                            (conversation_id, position, attachment.name, attachment.mime, attachment.kind,
                             attachment.size, attachment.text or "", attachment.data, metadata),
                        )
                        row = await cursor.fetchone()
                        attachment.id = row["id"]
                        attachment.has_data = attachment.data is not None
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
