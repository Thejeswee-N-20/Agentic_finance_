"""Unit tests for the agent pipeline producer (agents -> fusion -> PreDecision)."""
import pytest

from agentic_finance.agents_v2.pipeline import agent_direction_signal, technical_provider
from agentic_finance.risk_engine.schemas import PreDecision

pytestmark = pytest.mark.unit


def _uptrend(n=120, start=100.0, step=0.005):
    return [start * (1 + step) ** i for i in range(n)]


def _downtrend(n=120, start=200.0, step=0.005):
    return [start * (1 - step) ** i for i in range(n)]


def test_pipeline_returns_predecision():
    sig = agent_direction_signal([technical_provider()])
    out = sig(_uptrend())
    assert isinstance(out, PreDecision)


def test_pipeline_long_in_uptrend():
    sig = agent_direction_signal([technical_provider()])
    assert sig(_uptrend()).direction == "long"


def test_pipeline_flat_in_downtrend_long_only():
    # The technical agent is long-only, so a downtrend yields no position.
    sig = agent_direction_signal([technical_provider()])
    assert sig(_downtrend()).direction == "flat"


def test_pipeline_flat_on_insufficient_history():
    sig = agent_direction_signal([technical_provider()])
    assert sig([100.0, 101.0, 102.0]).direction == "flat"
