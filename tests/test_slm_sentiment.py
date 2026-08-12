"""Unit tests for the SLM sentiment backend(s) and the sentiment agent.

The agent logic is tested against a deterministic stub backend (no model
download), and the offline heuristic lexicon backend is tested directly.
"""
from typing import List, Sequence

import pytest

from agentic_finance.slm.heuristic import HeuristicSentimentModel
from agentic_finance.agents_v2.sentiment_agent import sentiment_signal
from agentic_finance.agents_v2.schemas import Signal

pytestmark = pytest.mark.unit


class _StubModel:
    """Returns preset per-text scores so we can isolate agent logic."""
    name = "stub"

    def __init__(self, scores: List[float]):
        self._scores = scores

    def score_texts(self, texts: Sequence[str]) -> List[float]:
        return list(self._scores[: len(texts)])


# --- heuristic backend ----------------------------------------------------

def test_heuristic_scores_bullish_positive():
    m = HeuristicSentimentModel()
    (score,) = m.score_texts(["Company posts record profit and strong growth, beats estimates"])
    assert score > 0


def test_heuristic_scores_bearish_negative():
    m = HeuristicSentimentModel()
    (score,) = m.score_texts(["Shares plunge on weak guidance, lawsuit and surprise loss"])
    assert score < 0


def test_heuristic_neutral_is_zero():
    m = HeuristicSentimentModel()
    (score,) = m.score_texts(["The company held its annual meeting on Tuesday"])
    assert score == pytest.approx(0.0)


def test_heuristic_score_in_range():
    m = HeuristicSentimentModel()
    for s in m.score_texts(["surge rally beat", "loss plunge weak", "neutral text"]):
        assert -1.0 <= s <= 1.0


# --- sentiment agent ------------------------------------------------------

def test_agent_bullish_scores_go_long():
    s = sentiment_signal(["a", "b", "c"], _StubModel([0.8, 0.6, 0.9]))
    assert isinstance(s, Signal)
    assert s.source == "sentiment"
    assert s.direction == "long"
    assert s.confidence > 0.5


def test_agent_bearish_scores_go_short():
    s = sentiment_signal(["a", "b"], _StubModel([-0.7, -0.5]))
    assert s.direction == "short"


def test_agent_empty_texts_is_flat_zero_confidence():
    s = sentiment_signal([], _StubModel([]))
    assert s.direction == "flat"
    assert s.confidence == 0.0


def test_agent_mixed_scores_are_flat_or_low_confidence():
    s = sentiment_signal(["a", "b"], _StubModel([0.6, -0.6]))
    assert s.direction == "flat" or s.confidence < 0.4


def test_agent_with_heuristic_backend_end_to_end():
    headlines = [
        "Record profit, strong growth, analyst upgrade",
        "Shares rally to all-time high after earnings beat",
    ]
    s = sentiment_signal(headlines, HeuristicSentimentModel())
    assert s.direction == "long"
