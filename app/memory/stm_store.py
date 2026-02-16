from __future__ import annotations

from datetime import datetime, timedelta

from app.memory.db import Database, deserialize_embedding, serialize_embedding, utc_now
from app.memory.models import MemoryItem


class ShortTermMemoryStore:
    """CRUD for STM records plus expiry sweep."""

    def __init__(self, db: Database, ttl_hours: int = 48):
        self.db = db
        self.ttl_hours = ttl_hours

    def add(
        self,
        content: str,
        embedding: list[float],
        importance: int,
        source_turn_id: int | None,
        created_at: datetime | None = None,
    ) -> int:
        created = created_at or utc_now()
        expires = created + timedelta(hours=self.ttl_hours)
        with self.db.tx() as conn:
            cur = conn.execute(
                """
                INSERT INTO short_term_memory(content, embedding, importance, created_at, expires_at, source_turn_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    content,
                    serialize_embedding(embedding),
                    importance,
                    created.isoformat(),
                    expires.isoformat(),
                    source_turn_id,
                ),
            )
            memory_id = int(cur.lastrowid)
            conn.execute(
                "INSERT INTO memory_fts(content, memory_type, memory_id) VALUES (?, 'stm', ?)",
                (content, memory_id),
            )
            return memory_id

    def sweep_expired(self, now: datetime | None = None) -> int:
        cutoff = (now or utc_now()).isoformat()
        with self.db.tx() as conn:
            ids = [
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM short_term_memory WHERE expires_at <= ?", (cutoff,)
                ).fetchall()
            ]
            if ids:
                conn.executemany(
                    "DELETE FROM memory_fts WHERE memory_type='stm' AND memory_id=?",
                    [(item_id,) for item_id in ids],
                )
            cur = conn.execute("DELETE FROM short_term_memory WHERE expires_at <= ?", (cutoff,))
            return int(cur.rowcount)

    def active(self, limit: int = 50) -> list[MemoryItem]:
        with self.db.tx() as conn:
            rows = conn.execute(
                """
                SELECT id, content, embedding, importance, created_at, expires_at, source_turn_id
                FROM short_term_memory
                WHERE expires_at > ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (utc_now().isoformat(), limit),
            ).fetchall()
        return [
            MemoryItem(
                id=row["id"],
                memory_type="stm",
                content=row["content"],
                embedding=deserialize_embedding(row["embedding"]),
                importance=row["importance"],
                created_at=datetime.fromisoformat(row["created_at"]),
                expires_at=datetime.fromisoformat(row["expires_at"]),
                source_turn_id=row["source_turn_id"],
            )
            for row in rows
        ]
