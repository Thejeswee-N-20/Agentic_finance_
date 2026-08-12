"""Unit tests for the signal-fusion layer.

Fuses typed agent Signals into a risk-engine ``PreDecision``. This replaces
TradingAgents' bull/bear prose debate with a confidence-weighted vote.
"""
import pytest

from agentic_finance.agents_v2.schemas import Signal
from agentic_finance.agents_v2.fusion import fuse_signals
from agentic_finance.risk_engine.schemas import PreDecision

pytestmark = pytest.mark.unit


def test_returns_predecision():
    out = fuse_signals([Signal("technical", "long", 0.7, 0.8)])
    assert isinstance(out, PreDecision)
    assert out.direction == "long"
    assert out.confidence > 0


def test_empty_is_flat_zero_confidence():
    out = fuse_signals([])
    assert out.direction == "flat"
    assert out.confidence == 0.0


def test_agreeing_long_signals_go_long():
    out = fuse_signals([
        Signal("technical", "long", 0.7, 0.8),
        Signal("sentiment", "long", 0.5, 0.9),
    ])
    assert out.direction == "long"
    assert out.confidence > 0.5


def test_equal_conflict_is_flat():
    out = fuse_signals([
        Signal("technical", "long", 0.7, 0.8),
        Signal("sentiment", "short", 0.7, 0.8),
    ])
    assert out.direction == "flat"


def test_strong_long_outweighs_weak_short():
    out = fuse_signals([
        Signal("technical", "long", 0.8, 0.9),
        Signal("sentiment", "short", 0.3, 0.5),
    ])
    assert out.direction == "long"


def test_flat_signals_do_not_force_direction():
    out = fuse_signals([
        Signal("technical", "flat", 0.0, 0.4),
        Signal("sentiment", "flat", 0.0, 0.4),
    ])
    assert out.direction == "flat"


def test_confidence_in_unit_range():
    out = fuse_signals([
        Signal("technical", "long", 1.0, 1.0),
        Signal("sentiment", "long", 1.0, 1.0),
    ])
    assert 0.0 <= out.confidence <= 1.0


def test_source_weights_can_override_balance():
    signals = [
        Signal("technical", "long", 0.6, 0.8),
        Signal("sentiment", "short", 0.6, 0.8),
    ]
    # Up-weight sentiment so the tie breaks short.
    out = fuse_signals(signals, weights={"sentiment": 3.0, "technical": 1.0})
    assert out.direction == "short"
