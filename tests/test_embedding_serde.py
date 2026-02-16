from app.memory.db import deserialize_embedding, serialize_embedding


def test_embedding_round_trip_precision():
    original = [0.123, -4.56, 7.89]
    blob = serialize_embedding(original)
    restored = deserialize_embedding(blob)
    assert len(restored) == len(original)
    for a, b in zip(original, restored):
        assert abs(a - b) < 1e-5
