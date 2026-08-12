"""LLM reasoning agent: free-form context + a ChatModel -> typed Signal.

Provider-agnostic (Gemini or local Ollama, chosen by config) — it depends only on
the ``ChatModel`` protocol. The model is prompted for a small JSON object and the
reply is parsed defensively into a validated ``Signal``; any parse/validation
failure degrades to a flat, zero-confidence signal rather than raising, so a bad
model response can never break the pipeline.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from agentic_finance.agents_v2.schemas import DIRECTIONS, Signal
from agentic_finance.slm.base import ChatModel

__all__ = ["llm_signal", "SYSTEM_PROMPT"]

SYSTEM_PROMPT = (
    "You are a disciplined financial analyst. Using ONLY the provided context "
    "(do not rely on outside or prior knowledge of the asset), judge whether the "
    "news contains a MATERIAL, recent, directional catalyst — e.g. an earnings "
    "surprise, guidance change, major contract/product, or regulatory or macro "
    "shock. Take a 'long' or 'short' view ONLY when such a catalyst is clearly and "
    "unambiguously present; if the news is routine, mixed, or stale, return "
    "'flat'. Be conservative: when in doubt, prefer 'flat'. Respond with ONLY a "
    "JSON object: "
    '{"direction":"long|short|flat","strength":0..1,"confidence":0..1,'
    '"rationale":"one sentence citing the catalyst or noting its absence"}. '
    "No extra text."
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _flat(reason: str, source: str) -> Signal:
    return Signal(source=source, direction="flat", strength=0.0, confidence=0.0, rationale=reason)


_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)


def llm_signal(context: str, model: ChatModel, source: str = "llm") -> Signal:
    """Ask ``model`` for a directional view on ``context`` and parse a Signal."""
    prompt = f"Context:\n{context}\n\nReturn the JSON object now."
    try:
        raw = model.complete(prompt, system=SYSTEM_PROMPT)
    except Exception as exc:  # network / provider failure -> safe flat
        return _flat(f"model error: {exc}", source)

    # Reasoning models (Fin-R1 / R1-style) wrap chain-of-thought in <think>
    # tags, which may themselves contain braces; strip them before extracting
    # the JSON answer. An unterminated <think> (truncated output) is stripped
    # to the end.
    cleaned = _THINK_BLOCK.sub("", raw or "")
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return _flat("no JSON object in model response", source)
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return _flat("unparseable JSON in model response", source)

    direction = str(data.get("direction", "")).strip().lower()
    if direction not in DIRECTIONS:
        return _flat(f"invalid/absent direction {direction!r}", source)
    if direction == "flat":
        return _flat(str(data.get("rationale", "model returned flat")), source)

    try:
        strength = _clamp01(data.get("strength", 0.0))
        confidence = _clamp01(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        return _flat("non-numeric strength/confidence", source)

    rationale = str(data.get("rationale", ""))[:300]
    return Signal(source=source, direction=direction, strength=strength,
                  confidence=confidence, rationale=rationale)
