from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

MemoryType = Literal["stm", "ltm"]
ClassifierLabel = Literal[
    "none",
    "stm_task",
    "stm_thread",
    "ltm_fact",
    "ltm_preference",
    "ltm_profile",
    "pin",
]


@dataclass(slots=True)
class ConversationMessage:
    id: int
    role: str
    content: str
    ts: datetime


@dataclass(slots=True)
class MemoryItem:
    id: int
    memory_type: MemoryType
    content: str
    embedding: list[float]
    importance: int
    created_at: datetime
    expires_at: datetime | None = None
    frequency: int = 0
    last_accessed_at: datetime | None = None
    pinned: bool = False
    source_turn_id: int | None = None


@dataclass(slots=True)
class ClassificationResult:
    label: ClassifierLabel
    importance: int
    memory_text: str
    uncertain: bool = False
