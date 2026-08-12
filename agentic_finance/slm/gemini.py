"""Gemini backends (generative chat + sentiment).

The hosted alternative to the local Ollama tier, selected via
``AGENTIC_SLM_PROVIDER=gemini``. The ``langchain_google_genai`` client is built
**lazily on first use**, so constructing these objects needs no API key or
network — only ``complete`` / ``score_texts`` do. The key comes from
``GOOGLE_API_KEY`` (auto-loaded from ``.env``).
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Sequence

__all__ = ["GeminiChatModel", "GeminiSentimentModel"]

_DEFAULT_MODEL = "gemini-2.5-flash"


def _build_llm(model: str, api_key: Optional[str], temperature: float):
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Gemini backend requires 'langchain-google-genai' (a project dependency). "
            "Run `pip install .` in the repo venv."
        ) from exc
    kwargs = {"model": model, "temperature": temperature}
    if api_key:
        kwargs["google_api_key"] = api_key
    return ChatGoogleGenerativeAI(**kwargs)


class GeminiChatModel:
    """Generative chat via Gemini (lazy client)."""

    def __init__(self, model: str = _DEFAULT_MODEL, api_key: Optional[str] = None,
                 temperature: float = 0.2):
        self.model = model
        self.name = f"gemini:{model}"
        self._api_key = api_key
        self._temperature = temperature
        self._llm = None

    def _client(self):
        if self._llm is None:
            self._llm = _build_llm(self.model, self._api_key, self._temperature)
        return self._llm

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append(("system", system))
        messages.append(("human", prompt))
        resp = self._client().invoke(messages)
        return resp.content if isinstance(resp.content, str) else str(resp.content)


class GeminiSentimentModel:
    """Sentiment scoring via Gemini, satisfying the ``SentimentModel`` protocol.

    Prompts the model to return a JSON array of per-text scores in [-1, 1]. Falls
    back to 0.0 for any text it cannot parse, so a malformed reply never crashes
    the agent.
    """

    name = "gemini-sentiment"

    def __init__(self, model: str = _DEFAULT_MODEL, api_key: Optional[str] = None):
        self._chat = GeminiChatModel(model=model, api_key=api_key, temperature=0.0)
        self.name = f"gemini-sentiment:{model}"

    def score_texts(self, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(texts))
        prompt = (
            "You are a financial sentiment classifier. For each numbered headline, "
            "output a sentiment score in [-1, 1] (negative to positive). Respond with "
            "ONLY a JSON array of floats in the same order, e.g. [0.8, -0.4].\n\n"
            f"{numbered}"
        )
        try:
            raw = self._chat.complete(prompt)
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            scores = json.loads(match.group(0)) if match else []
            out = [max(-1.0, min(1.0, float(s))) for s in scores]
        except Exception:
            out = []
        # Pad/truncate to len(texts), defaulting unparsed entries to neutral.
        out = (out + [0.0] * len(texts))[: len(texts)]
        return out
