from __future__ import annotations

from datetime import datetime

from app.memory.db import Database, deserialize_embedding, serialize_embedding, utc_now
from app.memory.models import MemoryItem


class LongTermMemoryStore:
    """CRUD for long-term memory and access tracking."""

    def __init__(self, db: Database):
        self.db = db

    def add(
        self,
        content: str,
        embedding: list[float],
        importance: int,
        source_turn_id: int | None,
        pinned: bool = False,
    ) -> int:
        with self.db.tx() as conn:
            cur = conn.execute(
                """
                INSERT INTO long_term_memory(content, embedding, importance, frequency, last_accessed_at, pinned, source_turn_id)
                VALUES (?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    content,
                    serialize_embedding(embedding),
                    importance,
                    utc_now().isoformat(),
                    int(pinned),
                    source_turn_id,
                ),
            )
            memory_id = int(cur.lastrowid)
            conn.execute(
                "INSERT INTO memory_fts(content, memory_type, memory_id) VALUES (?, 'ltm', ?)",
                (content, memory_id),
            )
            return memory_id

    def touch(self, memory_id: int) -> None:
        now = utc_now().isoformat()
        with self.db.tx() as conn:
            conn.execute(
                "UPDATE long_term_memory SET frequency = frequency + 1, last_accessed_at = ? WHERE id = ?",
                (now, memory_id),
            )

    def list_all(self, limit: int = 200) -> list[MemoryItem]:
        with self.db.tx() as conn:
            rows = conn.execute(
                """
                SELECT id, content, embedding, importance, frequency, last_accessed_at, pinned, source_turn_id
                FROM long_term_memory
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        now = utc_now()
        return [
            MemoryItem(
                id=row["id"],
                memory_type="ltm",
                content=row["content"],
                embedding=deserialize_embedding(row["embedding"]),
                importance=row["importance"],
                created_at=now,
                frequency=row["frequency"],
                last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]) if row["last_accessed_at"] else None,
                pinned=bool(row["pinned"]),
                source_turn_id=row["source_turn_id"],
            )
            for row in rows
        ]
