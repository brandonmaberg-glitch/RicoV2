from datetime import timedelta

from app.memory.db import Database, utc_now
from app.memory.stm_store import ShortTermMemoryStore


def test_stm_ttl_sweep_expires_items(tmp_path):
    db = Database(str(tmp_path / "mem.db"))
    store = ShortTermMemoryStore(db, ttl_hours=48)

    old = utc_now() - timedelta(hours=49)
    fresh = utc_now() - timedelta(hours=1)
    store.add("old", [0.1, 0.2], 3, source_turn_id=1, created_at=old)
    store.add("fresh", [0.1, 0.2], 3, source_turn_id=2, created_at=fresh)

    removed = store.sweep_expired(now=utc_now())
    assert removed == 1

    active = store.active(limit=10)
    assert len(active) == 1
    assert active[0].content == "fresh"
