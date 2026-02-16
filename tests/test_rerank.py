from datetime import datetime, timedelta, timezone

from app.memory.models import MemoryItem
from app.memory.retrieval.rerank import score_memory


WEIGHTS = {"similarity": 0.55, "importance": 0.20, "recency": 0.15, "frequency": 0.10}


def make_item(mid, mtype, emb, importance, age_hours, freq=0, pinned=False):
    return MemoryItem(
        id=mid,
        memory_type=mtype,
        content=f"m{mid}",
        embedding=emb,
        importance=importance,
        created_at=datetime.now(timezone.utc) - timedelta(hours=age_hours),
        frequency=freq,
        pinned=pinned,
    )


def test_pinned_gets_bonus():
    query = [1.0, 0.0]
    plain = make_item(1, "ltm", [1.0, 0.0], 3, age_hours=24, pinned=False)
    pinned = make_item(2, "ltm", [1.0, 0.0], 3, age_hours=24, pinned=True)
    assert score_memory(query, pinned, WEIGHTS) > score_memory(query, plain, WEIGHTS)


def test_recency_helps_stm():
    query = [1.0, 0.0]
    recent = make_item(1, "stm", [0.7, 0.3], 3, age_hours=1)
    stale = make_item(2, "stm", [0.7, 0.3], 3, age_hours=36)
    assert score_memory(query, recent, WEIGHTS) > score_memory(query, stale, WEIGHTS)


def test_similarity_dominates_when_other_factors_close():
    query = [1.0, 0.0]
    hi = make_item(1, "ltm", [0.99, 0.01], 3, age_hours=24, freq=1)
    lo = make_item(2, "ltm", [0.1, 0.9], 3, age_hours=24, freq=1)
    assert score_memory(query, hi, WEIGHTS) > score_memory(query, lo, WEIGHTS)
