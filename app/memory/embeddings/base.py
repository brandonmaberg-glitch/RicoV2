from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingsProvider(ABC):
    """Embedding provider interface for swappable local backends."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Create a dense vector for the given text."""
