from __future__ import annotations

from datetime import datetime

from app.memory.db import Database, utc_now
from app.memory.models import ConversationMessage


class ConversationLogStore:
    """Store and query conversation turns."""

    def __init__(self, db: Database):
        self.db = db

    def add(self, role: str, content: str, ts: datetime | None = None) -> int:
        timestamp = (ts or utc_now()).isoformat()
        with self.db.tx() as conn:
            cur = conn.execute(
                "INSERT INTO conversation_log(role, content, ts) VALUES (?, ?, ?)",
                (role, content, timestamp),
            )
            return int(cur.lastrowid)

    def last(self, limit: int = 8) -> list[ConversationMessage]:
        with self.db.tx() as conn:
            rows = conn.execute(
                "SELECT id, role, content, ts FROM conversation_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            ConversationMessage(
                id=row["id"],
                role=row["role"],
                content=row["content"],
                ts=datetime.fromisoformat(row["ts"]),
            )
            for row in reversed(rows)
        ]
