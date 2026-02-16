from __future__ import annotations

from app.memory.classifier.llm_judge import LlmJudge
from app.memory.classifier.router import ClassificationRouter
from app.memory.db import Database
from app.memory.embeddings import OllamaEmbeddingsProvider, SbertEmbeddingsProvider
from app.memory.log_store import ConversationLogStore
from app.memory.ltm_store import LongTermMemoryStore
from app.memory.retrieval.fts import FTSHelper
from app.memory.retrieval.retriever import MemoryRetriever
from app.memory.stm_store import ShortTermMemoryStore
from app.memory.summary import RollingSummary


class MemoryService:
    """Main facade for local logging, memory storage, retrieval, and summary."""

    def __init__(self, cfg, llm_client):
        self.cfg = cfg
        self.db = Database(cfg.memory_db_path)
        self.log_store = ConversationLogStore(self.db)
        self.stm_store = ShortTermMemoryStore(self.db, ttl_hours=cfg.memory_stm_ttl_hours)
        self.ltm_store = LongTermMemoryStore(self.db)
        self.fts = FTSHelper(self.db)

        if cfg.embeddings_backend == "ollama":
            self.embeddings = OllamaEmbeddingsProvider(cfg.ollama_base_url, cfg.embeddings_model)
        else:
            self.embeddings = SbertEmbeddingsProvider(cfg.sbert_model_name)

        judge = LlmJudge(llm_client) if cfg.memory_use_llm_judge else None
        self.classifier = ClassificationRouter(cfg.memory_use_llm_judge, judge)
        self.retriever = MemoryRetriever(
            self.stm_store,
            self.ltm_store,
            self.fts,
            self.embeddings,
            cfg.memory_top_n,
            cfg.memory_scoring_weights,
        )
        self.summary = RollingSummary(
            self.db,
            use_llm_summary=cfg.memory_use_llm_summary,
            llm_client=llm_client,
            update_every=cfg.memory_summary_update_every,
        )

    def ingest_user_message(self, text: str) -> int:
        """Log user turn and persist STM/LTM based on classifier label."""
        turn_id = self.log_store.add("user", text)
        result = self.classifier.classify(text)
        if result.label in {"stm_task", "stm_thread"}:
            self.stm_store.add(result.memory_text, self.embeddings.embed(result.memory_text), result.importance, turn_id)
        elif result.label in {"ltm_fact", "ltm_preference", "ltm_profile", "pin"}:
            self.ltm_store.add(
                result.memory_text,
                self.embeddings.embed(result.memory_text),
                result.importance,
                turn_id,
                pinned=result.label == "pin",
            )
        return turn_id

    def ingest_assistant_message(self, text: str) -> int:
        """Log assistant turn and update rolling summary."""
        turn_id = self.log_store.add("assistant", text)
        recent = [{"role": t.role, "content": t.content} for t in self.log_store.last(40)]
        self.summary.maybe_update(recent)
        return turn_id

    def build_context_for_prompt(self, user_text: str) -> str:
        """Build context block with summary, relevant memories, and recent turns."""
        top = self.retriever.retrieve(user_text)
        lines = []
        if self.summary.get().strip():
            lines.append("Rolling summary:\n" + self.summary.get().strip())

        if top:
            mem_lines = [f"- [{item.memory_type.upper()}#{item.id}] {item.content}" for item, _ in top]
            lines.append("Relevant memories:\n" + "\n".join(mem_lines))
            for item, _ in top:
                if item.memory_type == "ltm":
                    self.ltm_store.touch(item.id)

        recent_turns = self.log_store.last(8)
        if recent_turns:
            recent_lines = [f"{m.role}: {m.content}" for m in recent_turns]
            lines.append("Recent turns:\n" + "\n".join(recent_lines))

        return "\n\n".join(lines)

    def reset(self) -> None:
        """Clear all memory tables while keeping schema."""
        with self.db.tx() as conn:
            conn.executescript(
                """
                DELETE FROM conversation_log;
                DELETE FROM short_term_memory;
                DELETE FROM long_term_memory;
                DELETE FROM memory_usage;
                DELETE FROM memory_fts;
                DELETE FROM summary_state;
                """
            )

    def debug_top_memories(self, query: str) -> list[str]:
        """Return human-readable top memories for a query."""
        top = self.retriever.retrieve(query)
        return [f"{item.memory_type}:{item.id} score={score:.3f} :: {item.content}" for item, score in top]
