from __future__ import annotations

import requests

from app.memory.embeddings.base import EmbeddingsProvider


class OllamaEmbeddingsProvider(EmbeddingsProvider):
    """Ollama embedding provider using local /api/embeddings endpoint."""

    def __init__(self, base_url: str, model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        return [float(x) for x in response.json().get("embedding", [])]
