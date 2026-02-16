from __future__ import annotations

from datetime import datetime, timezone

from app.memory.db import utc_now
from app.memory.ltm_store import LongTermMemoryStore
from app.memory.models import MemoryItem
from app.memory.retrieval.fts import FTSHelper
from app.memory.retrieval.rerank import score_memory
from app.memory.stm_store import ShortTermMemoryStore


class MemoryRetriever:
    """Hybrid candidate retrieval (semantic + keyword) and reranking."""

    def __init__(
        self,
        stm_store: ShortTermMemoryStore,
        ltm_store: LongTermMemoryStore,
        fts: FTSHelper,
        embeddings,
        top_n: int,
        weights: dict[str, float],
    ):
        self.stm_store = stm_store
        self.ltm_store = ltm_store
        self.fts = fts
        self.embeddings = embeddings
        self.top_n = top_n
        self.weights = weights

    def retrieve(self, user_query: str) -> list[tuple[MemoryItem, float]]:
        query_embedding = self.embeddings.embed(user_query)
        self.stm_store.sweep_expired()

        stm_items = self.stm_store.active(limit=60)
        ltm_items = self.ltm_store.list_all(limit=200)

        stm_keyword_ids = set(self.fts.query(user_query, "stm", limit=25))
        ltm_keyword_ids = set(self.fts.query(user_query, "ltm", limit=25))

        candidates: list[MemoryItem] = []
        now = utc_now()
        for item in stm_items:
            is_keyword = item.id in stm_keyword_ids
            is_recent = (now - item.created_at).total_seconds() < 48 * 3600
            if is_keyword or is_recent:
                candidates.append(item)

        for item in ltm_items:
            if item.id in ltm_keyword_ids or item.pinned:
                candidates.append(item)
            else:
                candidates.append(item)

        unique: dict[tuple[str, int], MemoryItem] = {(item.memory_type, item.id): item for item in candidates}
        scored = [
            (item, score_memory(query_embedding, item, self.weights))
            for item in unique.values()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.top_n]
