"""Sentiment agent: financial texts + an SLM backend -> typed Signal.

Backend-agnostic: it consumes any object satisfying the ``SentimentModel``
protocol (heuristic lexicon or a hosted Gemini backend). Confidence rises with
both the magnitude of the mean sentiment and the *coverage* (fraction of texts
carrying a non-neutral signal), so a strong read from many sources is trusted
more than one outlier.
"""
from __future__ import annotations

from typing import Sequence

from agentic_finance.agents_v2.schemas import Signal
from agentic_finance.slm.base import SentimentModel

__all__ = ["sentiment_signal"]


def sentiment_signal(
    texts: Sequence[str],
    model: SentimentModel,
    long_threshold: float = 0.10,
) -> Signal:
    """Aggregate per-text sentiment into a directional Signal."""
    if not texts:
        return Signal("sentiment", "flat", 0.0, 0.0, rationale="no texts available")

    scores = model.score_texts(texts)
    if not scores:
        return Signal("sentiment", "flat", 0.0, 0.0, rationale="backend returned no scores")

    mean = sum(scores) / len(scores)
    coverage = sum(1 for s in scores if s != 0.0) / len(scores)

    if mean > long_threshold:
        direction = "long"
    elif mean < -long_threshold:
        direction = "short"
    else:
        return Signal(
            "sentiment", "flat", min(1.0, abs(mean)), max(0.0, 0.3 * coverage),
            rationale=f"net sentiment {mean:+.2f} within neutral band",
        )

    strength = min(1.0, abs(mean))
    confidence = min(0.95, 0.20 + 0.50 * abs(mean) + 0.30 * coverage)
    rationale = (
        f"{direction} on mean sentiment {mean:+.2f} across {len(texts)} texts "
        f"(coverage {coverage:.0%}, backend={model.name})."
    )
    evidence = tuple(f"{s:+.2f}: {t[:80]}" for s, t in zip(scores, texts))[:5]
    return Signal("sentiment", direction, strength, confidence,
                  rationale=rationale, evidence=evidence)
