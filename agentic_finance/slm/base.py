"""Protocols for SLM backends.

Agents depend on these structural types, not concrete classes, so any backend
(heuristic lexicon, a hosted endpoint) is interchangeable.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, runtime_checkable

__all__ = ["SentimentModel", "ChatModel"]


@runtime_checkable
class SentimentModel(Protocol):
    """Maps financial texts to sentiment scores in [-1, 1] (negative..positive)."""

    name: str

    def score_texts(self, texts: Sequence[str]) -> List[float]:
        ...


@runtime_checkable
class ChatModel(Protocol):
    """A generative text model (the reasoning tier).

    The single interchange point between Gemini and the local Ollama + SLMs: any
    agent that needs free-form generation depends on this protocol, never a
    concrete provider.
    """

    name: str

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        ...
