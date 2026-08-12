"""Typed message contract for the agent layer.

A ``Signal`` is what every specialist agent emits — a structured, validated view
rather than free-form prose. Typed signals keep token usage low for small
models, are directly auditable, and feed both the fusion layer and the SHAP
feature vector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

__all__ = ["Signal", "DIRECTIONS"]

DIRECTIONS = ("long", "short", "flat")


@dataclass(frozen=True)
class Signal:
    """A single specialist agent's view (immutable, validated).

    - ``direction`` ∈ {long, short, flat}
    - ``strength``  ∈ [0, 1]: magnitude/conviction of the directional view
    - ``confidence``∈ [0, 1]: how reliable the agent considers this view
    - ``rationale`` / ``evidence``: human + machine auditable justification
    """

    source: str
    direction: str
    strength: float
    confidence: float
    rationale: str = ""
    evidence: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {self.direction!r}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {self.strength}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def signed_strength(self) -> float:
        """Strength signed by direction: +strength (long), -strength (short), 0 (flat)."""
        if self.direction == "long":
            return self.strength
        if self.direction == "short":
            return -self.strength
        return 0.0
