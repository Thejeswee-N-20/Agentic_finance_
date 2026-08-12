"""Unit tests for the technical agent.

Fuses several indicators (trend, momentum, MACD, RSI, volatility) into a single
typed Signal — a genuine, point-in-time replacement for the placeholder 20-day
momentum rule. Deterministic; no model required.
"""
import pytest

from agentic_finance.agents_v2.schemas import Signal
from agentic_finance.agents_v2.technical_agent import technical_signal

pytestmark = pytest.mark.unit


def _uptrend(n=120, start=100.0, step=0.005):
    return [start * (1 + step) ** i for i in range(n)]


def _downtrend(n=120, start=200.0, step=0.005):
    return [start * (1 - step) ** i for i in range(n)]


def test_returns_signal():
    s = technical_signal(_uptrend())
    assert isinstance(s, Signal)
    assert s.source == "technical"


def test_uptrend_is_long_with_decent_confidence():
    s = technical_signal(_uptrend())
    assert s.direction == "long"
    assert s.confidence > 0.6
    assert s.strength > 0.0
    assert s.evidence  # carries indicator evidence


def test_downtrend_is_flat_long_only():
    # Long-only agent: negative momentum yields no position, not a short.
    s = technical_signal(_downtrend())
    assert s.direction == "flat"
    assert s.confidence == 0.0


def test_insufficient_history_is_flat_zero_confidence():
    s = technical_signal([100.0, 101.0, 102.0])
    assert s.direction == "flat"
    assert s.confidence == 0.0


def test_constant_prices_are_flat():
    s = technical_signal([100.0] * 80)
    assert s.direction == "flat"


def test_confidence_and_strength_in_range():
    for prices in (_uptrend(), _downtrend(), [100.0] * 80):
        s = technical_signal(prices)
        assert 0.0 <= s.strength <= 1.0
        assert 0.0 <= s.confidence <= 1.0


def test_conflicting_components_reduce_confidence_vs_clean_trend():
    # A clean uptrend (all components agree) should be more confident than a
    # series that just turned up after a long decline (mixed components).
    clean = technical_signal(_uptrend())
    mixed_prices = _downtrend(100) + _uptrend(25, start=_downtrend(100)[-1], step=0.02)
    mixed = technical_signal(mixed_prices)
    assert clean.confidence >= mixed.confidence
