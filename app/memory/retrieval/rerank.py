from __future__ import annotations

import math
from datetime import datetime, timezone

from app.memory.models import MemoryItem


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def recency_decay(created_at: datetime, half_life_hours: float) -> float:
    now = datetime.now(timezone.utc)
    age_hours = max(0.0, (now - created_at).total_seconds() / 3600.0)
    return math.exp(-age_hours / half_life_hours)


def score_memory(
    query_embedding: list[float],
    item: MemoryItem,
    weights: dict[str, float],
    stm_half_life_hours: float = 12.0,
    ltm_half_life_hours: float = 168.0,
    pin_bonus: float = 0.4,
) -> float:
    """Compute weighted memory score for context retrieval."""
    similarity = cosine_similarity(query_embedding, item.embedding)
    importance_norm = (item.importance - 1) / 4
    half_life = stm_half_life_hours if item.memory_type == "stm" else ltm_half_life_hours
    recency = recency_decay(item.created_at, half_life)
    frequency_norm = min(1.0, math.log1p(item.frequency) / math.log(11))
    score = (
        weights["similarity"] * similarity
        + weights["importance"] * importance_norm
        + weights["recency"] * recency
        + weights["frequency"] * frequency_norm
    )
    if item.pinned:
        score += pin_bonus
    return score
