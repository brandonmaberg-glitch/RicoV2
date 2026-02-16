from __future__ import annotations

from app.memory.db import Database


class FTSHelper:
    """FTS helper for keyword candidate lookup."""

    def __init__(self, db: Database):
        self.db = db

    def query(self, query_text: str, memory_type: str, limit: int = 20) -> list[int]:
        if not query_text.strip():
            return []
        with self.db.tx() as conn:
            rows = conn.execute(
                """
                SELECT memory_id
                FROM memory_fts
                WHERE memory_fts MATCH ? AND memory_type = ?
                LIMIT ?
                """,
                (query_text, memory_type, limit),
            ).fetchall()
        return [int(row["memory_id"]) for row in rows]
