"""Offline finance-lexicon sentiment model.

A zero-dependency, deterministic ``SentimentModel`` used as the default for tests
and demos and as the low-compute fallback. Scores each text by the balance of
bullish vs bearish finance terms; ``score = (pos - neg) / (pos + neg)`` in
[-1, 1], or 0 when no lexicon term appears. It is the dependency-free default
sentiment backend; the hosted Gemini backend is the higher-capability alternative.
"""
from __future__ import annotations

import re
from typing import List, Sequence

__all__ = ["HeuristicSentimentModel"]

_POSITIVE = {
    "beat", "beats", "surge", "surges", "growth", "record", "strong", "bullish",
    "upgrade", "upgraded", "profit", "profits", "gain", "gains", "rally", "rallies",
    "outperform", "高", "soar", "soars", "jump", "jumps", "rise", "rises", "boost",
    "optimistic", "expansion", "tops", "all-time",
}
_NEGATIVE = {
    "miss", "misses", "plunge", "plunges", "loss", "losses", "weak", "bearish",
    "downgrade", "downgraded", "decline", "declines", "fall", "falls", "lawsuit",
    "cut", "cuts", "slump", "slumps", "drop", "drops", "warning", "fraud",
    "bankruptcy", "recall", "probe", "selloff", "tumble", "tumbles",
}

_WORD_RE = re.compile(r"[a-z\-']+")


class HeuristicSentimentModel:
    """Finance-lexicon sentiment scorer (deterministic, CPU, no deps)."""

    name = "heuristic-lexicon"

    def score_texts(self, texts: Sequence[str]) -> List[float]:
        return [self._score_one(t) for t in texts]

    def _score_one(self, text: str) -> float:
        words = _WORD_RE.findall(text.lower())
        pos = sum(1 for w in words if w in _POSITIVE)
        neg = sum(1 for w in words if w in _NEGATIVE)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total
