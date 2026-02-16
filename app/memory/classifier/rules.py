from __future__ import annotations

import re

from app.memory.models import ClassificationResult


REMEMBER_PATTERNS = re.compile(r"\b(remember this|don't forget|from now on)\b", re.IGNORECASE)
TIME_PATTERNS = re.compile(r"\b(tomorrow|later|this week|tonight|next week)\b", re.IGNORECASE)
PROFILE_PATTERNS = re.compile(r"\b(i have|my partner|i work|i own|my job|my family)\b", re.IGNORECASE)
PREFERENCE_PATTERNS = re.compile(r"\b(i prefer|i hate|i like|i love|i dislike)\b", re.IGNORECASE)


def classify_rules(text: str) -> ClassificationResult:
    """Deterministic first-pass memory classifier."""
    cleaned = text.strip()
    if not cleaned:
        return ClassificationResult("none", 1, "", uncertain=False)

    if REMEMBER_PATTERNS.search(cleaned):
        label = "pin" if "from now on" in cleaned.lower() else "ltm_fact"
        return ClassificationResult(label, 4, cleaned, uncertain=False)

    if PREFERENCE_PATTERNS.search(cleaned):
        return ClassificationResult("ltm_preference", 4, cleaned, uncertain=False)

    if TIME_PATTERNS.search(cleaned):
        return ClassificationResult("stm_task", 3, cleaned, uncertain=False)

    if PROFILE_PATTERNS.search(cleaned):
        return ClassificationResult("ltm_profile", 3, cleaned, uncertain=True)

    if len(cleaned.split()) <= 3:
        return ClassificationResult("none", 1, cleaned, uncertain=False)

    return ClassificationResult("none", 2, cleaned, uncertain=True)
