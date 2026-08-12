"""Env-driven selection of the SLM provider.

One switch decides whether the generative tier (and, optionally, sentiment) runs
on **Gemini** (hosted) or the local **Ollama + SLMs**. Nothing else in the code
hard-codes a provider.

Env vars
--------
- ``AGENTIC_SLM_PROVIDER``      : ``gemini`` | ``ollama``. If unset, auto-detect:
  ``gemini`` when ``GOOGLE_API_KEY`` is present, else ``ollama``.
- ``AGENTIC_GEMINI_MODEL``      : default ``gemini-2.5-flash``.
- ``AGENTIC_OLLAMA_MODEL``      : default ``qwen2.5:3b``.
- ``OLLAMA_BASE_URL``           : default ``http://localhost:11434``.
- ``AGENTIC_SENTIMENT_BACKEND`` : ``heuristic`` | ``gemini``
  (default ``heuristic`` — offline, zero-dep).
"""
from __future__ import annotations

import os
from typing import Optional

from agentic_finance.slm.base import ChatModel, SentimentModel
from agentic_finance.slm.gemini import GeminiChatModel, GeminiSentimentModel
from agentic_finance.slm.heuristic import HeuristicSentimentModel
from agentic_finance.slm.ollama import OllamaChatModel

__all__ = ["slm_provider", "get_chat_model", "sentiment_backend", "get_sentiment_model"]

_GEMINI = "gemini"
_OLLAMA = "ollama"


def slm_provider() -> str:
    """Resolve the generative provider from env (with auto-detect)."""
    explicit = os.environ.get("AGENTIC_SLM_PROVIDER", "").strip().lower()
    if explicit in (_GEMINI, _OLLAMA):
        return explicit
    return _GEMINI if os.environ.get("GOOGLE_API_KEY") else _OLLAMA


def get_chat_model(provider: Optional[str] = None) -> ChatModel:
    """Construct the configured generative chat model (lazy client)."""
    provider = (provider or slm_provider()).strip().lower()
    if provider == _GEMINI:
        return GeminiChatModel(model=os.environ.get("AGENTIC_GEMINI_MODEL", "gemini-2.5-flash"))
    if provider == _OLLAMA:
        return OllamaChatModel()  # reads AGENTIC_OLLAMA_MODEL / OLLAMA_BASE_URL
    raise ValueError(f"unknown SLM provider {provider!r}; use 'gemini' or 'ollama'")


def sentiment_backend() -> str:
    """Resolve the sentiment backend from env (default heuristic)."""
    return os.environ.get("AGENTIC_SENTIMENT_BACKEND", "heuristic").strip().lower()


def get_sentiment_model(backend: Optional[str] = None) -> SentimentModel:
    """Construct the configured sentiment model (lazy client where applicable)."""
    backend = (backend or sentiment_backend()).strip().lower()
    if backend == "heuristic":
        return HeuristicSentimentModel()
    if backend == _GEMINI:
        return GeminiSentimentModel(model=os.environ.get("AGENTIC_GEMINI_MODEL", "gemini-2.5-flash"))
    raise ValueError(f"unknown sentiment backend {backend!r}; use heuristic|gemini")
