"""Unit tests for the LLM reasoning agent (provider-agnostic).

Exercised against a deterministic stub ChatModel so no Gemini/Ollama is needed.
The agent must turn free-form model output into a validated typed Signal and
never crash on malformed responses.
"""
from typing import Optional

import pytest

from agentic_finance.agents_v2.llm_agent import llm_signal
from agentic_finance.agents_v2.schemas import Signal

pytestmark = pytest.mark.unit


class StubChat:
    name = "stub"

    def __init__(self, response: str):
        self._response = response
        self.last_prompt: Optional[str] = None

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        self.last_prompt = prompt
        return self._response


def test_parses_clean_json_signal():
    chat = StubChat('{"direction": "long", "strength": 0.7, "confidence": 0.8, "rationale": "AI demand"}')
    s = llm_signal("NVDA beats earnings", chat, source="news")
    assert isinstance(s, Signal)
    assert s.source == "news"
    assert s.direction == "long"
    assert s.strength == pytest.approx(0.7)
    assert s.confidence == pytest.approx(0.8)
    assert "AI demand" in s.rationale


def test_parses_json_embedded_in_prose():
    chat = StubChat('Here is my view:\n{"direction":"short","strength":0.5,"confidence":0.6}\nThanks!')
    s = llm_signal("weak guidance", chat)
    assert s.direction == "short"
    assert s.strength == pytest.approx(0.5)


def test_malformed_response_is_flat_zero_confidence():
    s = llm_signal("ctx", StubChat("I cannot help with that."))
    assert s.direction == "flat"
    assert s.confidence == 0.0


def test_invalid_direction_falls_back_to_flat():
    s = llm_signal("ctx", StubChat('{"direction":"up","strength":0.9,"confidence":0.9}'))
    assert s.direction == "flat"


def test_out_of_range_values_are_clamped():
    chat = StubChat('{"direction":"long","strength":5,"confidence":-2}')
    s = llm_signal("ctx", chat)
    assert 0.0 <= s.strength <= 1.0
    assert 0.0 <= s.confidence <= 1.0


def test_context_is_included_in_prompt():
    chat = StubChat('{"direction":"flat","strength":0,"confidence":0}')
    llm_signal("UNIQUE_TOKEN_XYZ headline", chat)
    assert "UNIQUE_TOKEN_XYZ" in (chat.last_prompt or "")


def test_think_block_is_stripped_before_json_parse():
    """Reasoning models (Fin-R1) wrap CoT in <think> tags that may contain braces."""
    chat = StubChat(
        '<think>The catalyst {earnings beat} is material... {"draft": 1}</think>\n'
        '{"direction":"long","strength":0.8,"confidence":0.9,"rationale":"earnings beat"}'
    )
    s = llm_signal("ctx", chat)
    assert s.direction == "long"
    assert s.strength == pytest.approx(0.8)
    assert s.rationale == "earnings beat"


def test_unterminated_think_block_degrades_to_flat():
    """Truncated reasoning output (no answer after <think>) must not crash."""
    s = llm_signal("ctx", StubChat('<think>hmm {"direction":"long"} but wait'))
    assert s.direction == "flat"
    assert s.confidence == 0.0
