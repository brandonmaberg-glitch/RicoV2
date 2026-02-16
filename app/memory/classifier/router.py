from __future__ import annotations

from app.memory.classifier.rules import classify_rules


class ClassificationRouter:
    """Rules-first classifier with optional LLM fallback."""

    def __init__(self, use_llm_judge: bool = False, llm_judge=None):
        self.use_llm_judge = use_llm_judge
        self.llm_judge = llm_judge

    def classify(self, text: str):
        result = classify_rules(text)
        if result.uncertain and self.use_llm_judge and self.llm_judge is not None:
            return self.llm_judge.classify(text)
        return result
