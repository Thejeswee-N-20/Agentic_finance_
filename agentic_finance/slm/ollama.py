"""Ollama backend (local generative SLMs).

The local alternative to Gemini, selected via ``AGENTIC_SLM_PROVIDER=ollama``.
Talks to the Ollama HTTP API; the connection happens lazily on first
``complete`` so construction needs nothing running. Defaults read from
``AGENTIC_OLLAMA_MODEL`` and ``OLLAMA_BASE_URL``.
"""
from __future__ import annotations

import os
from typing import Optional

__all__ = ["OllamaChatModel"]

_DEFAULT_MODEL = "qwen2.5:3b"
_DEFAULT_BASE = "http://localhost:11434"


def _normalize_base(url: str) -> str:
    url = url.rstrip("/")
    # Accept an OpenAI-style ".../v1" base and strip it for the native API.
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url


class OllamaChatModel:
    """Generative chat via a local Ollama server (lazy HTTP).

    Decoding is greedy by default (``temperature=0``) so that the local tier is
    exactly reproducible — the framework's remedy for the run-to-run variance
    of sampled LLM output (see RESULTS_RAG_EVAL.md). Override per-call with
    ``AGENTIC_OLLAMA_TEMPERATURE`` or the constructor argument.
    """

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 timeout: float = 300.0, temperature: Optional[float] = None):
        self.model = model or os.environ.get("AGENTIC_OLLAMA_MODEL", _DEFAULT_MODEL)
        self.base_url = _normalize_base(base_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE))
        if temperature is None:
            temperature = float(os.environ.get("AGENTIC_OLLAMA_TEMPERATURE", "0"))
        self.temperature = temperature
        self.name = f"ollama:{self.model}"
        self._timeout = timeout

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        import requests

        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "options": {"temperature": self.temperature}}
        if system:
            payload["system"] = system
        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        # Reasoning models (e.g. Fin-R1) may stream their chain-of-thought into
        # a separate "thinking" field; the answer remains in "response".
        return data.get("response", "") or ""
