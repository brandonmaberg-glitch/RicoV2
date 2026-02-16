from __future__ import annotations

import json
import logging
import sqlite3
import struct
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

LOGGER = logging.getLogger(__name__)


def utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def serialize_embedding(vec: list[float]) -> bytes:
    """Serialize embedding vectors as compact float32 blob."""
    if not vec:
        return b""
    return struct.pack(f"<{len(vec)}f", *[float(x) for x in vec])


def deserialize_embedding(blob: bytes | str | None) -> list[float]:
    """Deserialize embedding from float32 blob or JSON string."""
    if blob is None:
        return []
    if isinstance(blob, str):
        return [float(x) for x in json.loads(blob)]
    if len(blob) == 0:
        return []
    count = len(blob) // 4
    return [float(x) for x in struct.unpack(f"<{count}f", blob)]


class Database:
    """SQLite database wrapper with schema bootstrap and safe transactions."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._bootstrap()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except sqlite3.OperationalError as exc:
            LOGGER.exception("SQLite operational error")
            conn.rollback()
            raise RuntimeError("database operation failed") from exc
        finally:
            conn.close()

    def _bootstrap(self) -> None:
        with self.tx() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    ts TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS short_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding BLOB,
                    importance INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    source_turn_id INTEGER
                );

                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding BLOB,
                    importance INTEGER NOT NULL,
                    frequency INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at TEXT,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    source_turn_id INTEGER
                );

                CREATE TABLE IF NOT EXISTS memory_usage (
                    memory_id INTEGER NOT NULL,
                    memory_type TEXT NOT NULL,
                    hits INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(memory_id, memory_type)
                );

                CREATE TABLE IF NOT EXISTS summary_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    summary_text TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    content,
                    memory_type UNINDEXED,
                    memory_id UNINDEXED
                );
                """
            )
