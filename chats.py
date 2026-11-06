"""Saved chats: one Cassandra row per chat, in the ``chats`` table.

Every query is scoped to the owner (the signed-in user's lower-cased email),
which is the partition key, so a user can only ever reach their own chats.

Writes are UPDATEs of named columns rather than whole-row INSERTs, so the app
saving a chat and the summary thread storing its summary don't overwrite each
other's columns.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

TITLE_CHARS = 50

# Columns read for the sidebar list; ``messages`` is left out on purpose.
_LIST_COLUMNS = "chat_id, title, created_at, updated_at, pinned, message_count"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def make_title(question: str) -> str:
    """First question on one line, cut to about TITLE_CHARS characters."""
    text = " ".join(question.split())
    if len(text) <= TITLE_CHARS:
        return text
    cut = text[:TITLE_CHARS].rsplit(" ", 1)[0] or text[:TITLE_CHARS]
    return cut.rstrip(",.;:") + "…"


@dataclass
class Chat:
    owner: str
    chat_id: uuid.UUID = field(default_factory=uuid.uuid1)
    title: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    pinned: bool = False
    messages: list[dict] = field(default_factory=list)
    # Last real prompt size reported by Ollama.
    prompt_tokens: int = 0
    # Index of the first message sent to the model, and the summary placed in
    # front of it (frozen until the window moves, so the prompt stays a
    # continuation of the previous one even after a newer summary is made).
    window_start: int = 0
    window_summary: str = ""
    # Idle summary of the whole chat, covering messages[:summary_upto].
    summary: str = ""
    summary_upto: int = 0

    @property
    def message_count(self) -> int:
        return len(self.messages)


@dataclass(frozen=True)
class ChatInfo:
    """A row of the sidebar list."""

    chat_id: uuid.UUID
    title: str
    created_at: datetime | None
    updated_at: datetime | None
    pinned: bool
    message_count: int


@dataclass(frozen=True)
class IdleChat:
    """A chat the summary thread looks at."""

    owner: str
    chat_id: uuid.UUID
    updated_at: datetime | None
    message_count: int
    summary_upto: int


class CassandraChatStore:
    def __init__(self, session, keyspace: str, *, keep: int, pinned_max: int):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", keyspace):
            raise ValueError(f"invalid keyspace name: {keyspace!r}")
        self.session = session
        self.ks = keyspace
        self.keep = keep
        self.pinned_max = pinned_max

    def ensure_schema(self) -> None:
        """Assumes the keyspace exists (CassandraUserStore creates it)."""
        self.session.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.ks}.chats (
                owner          text,
                chat_id        timeuuid,
                title          text,
                created_at     timestamp,
                updated_at     timestamp,
                pinned         boolean,
                messages       text,
                message_count  int,
                prompt_tokens  int,
                window_start   int,
                window_summary text,
                summary        text,
                summary_upto   int,
                PRIMARY KEY (owner, chat_id)
            )""")
        t = f"{self.ks}.chats"
        self._save = self.session.prepare(
            f"UPDATE {t} SET title = ?, created_at = ?, updated_at = ?, messages = ?, "
            "message_count = ?, prompt_tokens = ?, window_start = ?, window_summary = ? "
            "WHERE owner = ? AND chat_id = ?"
        )
        self._list = self.session.prepare(f"SELECT {_LIST_COLUMNS} FROM {t} WHERE owner = ?")
        self._load = self.session.prepare(f"SELECT * FROM {t} WHERE owner = ? AND chat_id = ?")
        self._delete = self.session.prepare(f"DELETE FROM {t} WHERE owner = ? AND chat_id = ?")
        self._delete_all = self.session.prepare(f"DELETE FROM {t} WHERE owner = ?")
        self._pin = self.session.prepare(
            f"UPDATE {t} SET pinned = ? WHERE owner = ? AND chat_id = ? IF EXISTS"
        )
        self._set_summary = self.session.prepare(
            f"UPDATE {t} SET summary = ?, summary_upto = ? WHERE owner = ? AND chat_id = ? IF EXISTS"
        )

    # -- the signed-in user's chats ------------------------------------------

    def save(self, chat: Chat) -> list[uuid.UUID]:
        """Store the chat as most recently used, then apply retention.

        Returns the ids of chats dropped by retention. An empty chat is never
        saved.
        """
        if not chat.messages:
            return []
        now = _now()
        chat.created_at = chat.created_at or now
        chat.updated_at = now
        if not chat.title:
            chat.title = make_title(chat.messages[0]["content"])
        self.session.execute(self._save, (
            chat.title, chat.created_at, chat.updated_at,
            json.dumps(chat.messages, ensure_ascii=False), chat.message_count,
            chat.prompt_tokens, chat.window_start, chat.window_summary,
            chat.owner, chat.chat_id,
        ))
        return self.enforce_retention(chat.owner)

    def list_for(self, owner: str) -> list[ChatInfo]:
        """Pinned chats first, then the rest; most recently used first in each."""
        infos = [
            ChatInfo(r.chat_id, r.title or "Untitled", r.created_at, r.updated_at,
                     bool(r.pinned), r.message_count or 0)
            for r in self.session.execute(self._list, (owner,))
        ]
        return sorted(infos, key=lambda c: (not c.pinned, -_ts(c.updated_at)))

    def load(self, owner: str, chat_id: uuid.UUID) -> Chat | None:
        row = self.session.execute(self._load, (owner, chat_id)).one()
        if row is None:
            return None
        return Chat(
            owner=row.owner,
            chat_id=row.chat_id,
            title=row.title or "",
            created_at=row.created_at,
            updated_at=row.updated_at,
            pinned=bool(row.pinned),
            messages=json.loads(row.messages or "[]"),
            prompt_tokens=row.prompt_tokens or 0,
            window_start=row.window_start or 0,
            window_summary=row.window_summary or "",
            summary=row.summary or "",
            summary_upto=row.summary_upto or 0,
        )

    def delete(self, owner: str, chat_id: uuid.UUID) -> None:
        self.session.execute(self._delete, (owner, chat_id))

    def delete_all(self, owner: str) -> None:
        self.session.execute(self._delete_all, (owner,))

    def set_pinned(self, owner: str, chat_id: uuid.UUID, pinned: bool) -> bool:
        """Pin or unpin. Pinning fails (returns False) at the pinned_max cap.

        Unpinning doesn't drop anything straight away: the chat counts towards
        ``keep`` again and retention applies at the next save.
        """
        if pinned and sum(c.pinned for c in self.list_for(owner)) >= self.pinned_max:
            return False
        return self.session.execute(self._pin, (pinned, owner, chat_id)).was_applied

    def enforce_retention(self, owner: str) -> list[uuid.UUID]:
        """Keep every pinned chat and the ``keep`` most recently used others."""
        unpinned = [c for c in self.list_for(owner) if not c.pinned]
        dropped = [c.chat_id for c in unpinned[self.keep:]]
        for chat_id in dropped:
            self.delete(owner, chat_id)
        return dropped

    # -- summary thread ------------------------------------------------------

    def scan_for_summary(self) -> list[IdleChat]:
        """Every chat's bookkeeping columns, across all users.

        A full table scan, but a small one: at most keep + pinned_max rows per
        user, without the messages.
        """
        rows = self.session.execute(
            f"SELECT owner, chat_id, updated_at, message_count, summary_upto FROM {self.ks}.chats"
        )
        return [IdleChat(r.owner, r.chat_id, r.updated_at, r.message_count or 0, r.summary_upto or 0)
                for r in rows]

    def set_summary(self, owner: str, chat_id: uuid.UUID, summary: str, upto: int) -> bool:
        """Store a summary unless the chat was deleted meanwhile (no resurrection)."""
        return self.session.execute(self._set_summary, (summary, upto, owner, chat_id)).was_applied


def _ts(value: datetime | None) -> float:
    if value is None:
        return 0.0
    if value.tzinfo is None:  # the driver returns naive UTC datetimes
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()
