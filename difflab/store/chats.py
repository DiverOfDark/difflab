"""Chats with the VLM, as rows."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

USER, ASSISTANT, TOOL = "user", "assistant", "tool"
QUEUED, WRITING, DONE, STOPPED, FAILED = "queued", "writing", "done", "stopped", "failed"
PENDING = (QUEUED, WRITING)
ASKING = "asking"  # a tool call that waits for the person to allow it


@dataclass(frozen=True)
class Chat:
    id: int
    title: str
    system: str
    created_at: float
    updated_at: float
    production_id: int | None
    summary: str
    summary_through: int
    compact_next: bool

    @classmethod
    def of(cls, row: sqlite3.Row) -> Chat:
        return cls(
            row["id"],
            row["title"],
            row["system"],
            row["created_at"],
            row["updated_at"],
            row["production_id"],
            row["summary"],
            row["summary_through"],
            bool(row["compact_next"]),
        )


@dataclass(frozen=True)
class Message:
    id: int
    chat_id: int
    role: str
    text: str
    thinking: str
    media: list[str]
    state: str
    error: str
    run_id: str | None
    created_at: float
    calls: list[dict[str, Any]]
    tokens: int

    @classmethod
    def of(cls, row: sqlite3.Row) -> Message:
        return cls(
            id=row["id"],
            chat_id=row["chat_id"],
            role=row["role"],
            text=row["text"],
            thinking=row["thinking"],
            media=list(json.loads(row["media"])),
            state=row["state"],
            error=row["error"],
            run_id=row["run_id"],
            created_at=row["created_at"],
            calls=list(json.loads(row["calls"])),
            tokens=row["tokens"],
        )

    def data(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "text": self.text,
            "thinking": self.thinking,
            "media": self.media,
            "state": self.state,
            "error": self.error,
            "run": self.run_id,
            "created": self.created_at,
            "calls": self.calls,
            "tokens": self.tokens,
        }


def add_chat(connection: sqlite3.Connection, title: str = "", system: str = "") -> Chat:
    now = time.time()
    cursor = connection.execute(
        "INSERT INTO chats (title, system, created_at, updated_at) VALUES (?, ?, ?, ?)", (title, system, now, now)
    )
    made = chat(connection, int(cursor.lastrowid or 0))
    assert made is not None
    return made


def chat(connection: sqlite3.Connection, chat_id: int) -> Chat | None:
    row = connection.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
    return None if row is None else Chat.of(row)


def chats(connection: sqlite3.Connection) -> list[Chat]:
    return [Chat.of(row) for row in connection.execute("SELECT * FROM chats ORDER BY updated_at DESC")]


def update_chat(connection: sqlite3.Connection, chat_id: int, **fields: str | int | None) -> None:
    for name, value in fields.items():
        if name not in ("title", "system", "production_id", "summary", "summary_through", "compact_next"):
            raise ValueError(f"a chat has no {name}")
        connection.execute(f"UPDATE chats SET {name} = ? WHERE id = ?", (value, chat_id))


def touch(connection: sqlite3.Connection, chat_id: int) -> None:
    connection.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (time.time(), chat_id))


def delete_chat(connection: sqlite3.Connection, chat_id: int) -> None:
    connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def add_message(
    connection: sqlite3.Connection,
    chat_id: int,
    role: str,
    text: str = "",
    media: list[str] | None = None,
    state: str = DONE,
    calls: list[dict[str, Any]] | None = None,
) -> Message:
    cursor = connection.execute(
        "INSERT INTO chat_messages (chat_id, role, text, media, state, created_at, calls) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            chat_id,
            role,
            text,
            json.dumps(media or [], ensure_ascii=False),
            state,
            time.time(),
            json.dumps(calls or [], ensure_ascii=False),
        ),
    )
    touch(connection, chat_id)
    made = message(connection, int(cursor.lastrowid or 0))
    assert made is not None
    return made


def message(connection: sqlite3.Connection, message_id: int) -> Message | None:
    row = connection.execute("SELECT * FROM chat_messages WHERE id = ?", (message_id,)).fetchone()
    return None if row is None else Message.of(row)


def messages(connection: sqlite3.Connection, chat_id: int) -> list[Message]:
    rows = connection.execute("SELECT * FROM chat_messages WHERE chat_id = ? ORDER BY id", (chat_id,))
    return [Message.of(row) for row in rows]


def write(connection: sqlite3.Connection, message_id: int, **fields: Any) -> None:
    for name, value in fields.items():
        if name not in ("text", "thinking", "state", "error", "run_id", "media", "calls", "tokens"):
            raise ValueError(f"a message has no {name}")
        if name in ("media", "calls"):
            value = json.dumps(value, ensure_ascii=False)
        connection.execute(f"UPDATE chat_messages SET {name} = ? WHERE id = ?", (value, message_id))


def delete_after(connection: sqlite3.Connection, chat_id: int, message_id: int) -> None:
    """The message and everything said after it, and a summary that covered any of them."""
    connection.execute("DELETE FROM chat_messages WHERE chat_id = ? AND id >= ?", (chat_id, message_id))
    connection.execute(
        "UPDATE chats SET summary = '', summary_through = 0 WHERE id = ? AND summary_through >= ?",
        (chat_id, message_id),
    )


def following(connection: sqlite3.Connection) -> list[Message]:
    """Every tool call waiting on a run, in any chat."""
    rows = connection.execute(
        "SELECT * FROM chat_messages WHERE role = ? AND state = ? AND run_id IS NOT NULL ORDER BY id", (TOOL, WRITING)
    )
    return [Message.of(row) for row in rows]
