from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.memory.embeddings.base import EmbeddingsProvider


class SbertEmbeddingsProvider(EmbeddingsProvider):
    """SBERT all-MiniLM-L6-v2 embedding provider."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.astype("float32").tolist()
