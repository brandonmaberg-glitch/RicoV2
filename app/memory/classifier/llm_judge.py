from __future__ import annotations

import json

from app.memory.models import ClassificationResult


class LlmJudge:
    """Optional local LLM classifier with strict JSON output."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def classify(self, text: str) -> ClassificationResult:
        prompt = (
            "Classify the user message for memory storage. Return ONLY JSON with "
            '{"label":"none|stm_task|stm_thread|ltm_fact|ltm_preference|ltm_profile|pin",'
            '"importance":1-5,"memory_text":"..."}. '
            f"Message: {text}"
        )
        raw = self.llm_client.complete(prompt)
        payload = json.loads(raw)
        return ClassificationResult(
            label=payload.get("label", "none"),
            importance=max(1, min(5, int(payload.get("importance", 2)))),
            memory_text=payload.get("memory_text", text),
            uncertain=False,
        )
