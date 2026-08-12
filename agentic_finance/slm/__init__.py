"""Small-Language-Model (SLM) serving backends for the agent layer.

Backends are accessed behind small protocols so agent logic is testable without
a model download, and real models drop in for live runs:

- ``base``      : the ``SentimentModel`` protocol.
- ``heuristic`` : offline finance-lexicon sentiment (CPU, zero deps) — default
  for tests/demos and the low-compute fallback.
- ``gemini``    : hosted Gemini sentiment backend.

Generative SLMs (Qwen2.5-1.5B/3B, Fin-R1) are served via Ollama; that client is
added alongside these as the reasoning tier.
"""

from agentic_finance.slm.config import (
    get_chat_model,
    get_sentiment_model,
    sentiment_backend,
    slm_provider,
)
from agentic_finance.slm.heuristic import HeuristicSentimentModel

__all__ = [
    "HeuristicSentimentModel",
    "slm_provider",
    "get_chat_model",
    "sentiment_backend",
    "get_sentiment_model",
]

