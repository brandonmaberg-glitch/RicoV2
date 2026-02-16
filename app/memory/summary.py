from __future__ import annotations

from app.memory.db import Database, utc_now


class RollingSummary:
    """Keeps a compact rolling summary in SQLite."""

    def __init__(self, db: Database, use_llm_summary: bool = False, llm_client=None, update_every: int = 4):
        self.db = db
        self.use_llm_summary = use_llm_summary
        self.llm_client = llm_client
        self.update_every = update_every
        self.turn_counter = 0

    def get(self) -> str:
        with self.db.tx() as conn:
            row = conn.execute("SELECT summary_text FROM summary_state WHERE id = 1").fetchone()
            return row["summary_text"] if row else ""

    def maybe_update(self, recent_turns: list[dict[str, str]]) -> None:
        self.turn_counter += 1
        if self.turn_counter % self.update_every != 0:
            return

        old = self.get()
        if self.use_llm_summary and self.llm_client is not None:
            joined = "\n".join(f"{t['role']}: {t['content']}" for t in recent_turns)
            prompt = (
                "Update the running conversation summary in <= 8 bullet points. "
                f"Existing summary:\n{old}\n\nRecent turns:\n{joined}"
            )
            summary = self.llm_client.complete(prompt)
        else:
            snippets = [f"{t['role']}: {t['content']}" for t in recent_turns[-8:]]
            summary = " | ".join(snippets)[-1200:]

        with self.db.tx() as conn:
            conn.execute(
                """
                INSERT INTO summary_state(id, summary_text, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    updated_at = excluded.updated_at
                """,
                (summary, utc_now().isoformat()),
            )
