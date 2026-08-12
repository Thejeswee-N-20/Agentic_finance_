"""Unit tests for the risk_manager policy node.

The policy converts a directional pre-decision + market/portfolio features into
an enforced, risk-budgeted order (or a veto). This is the quantitative
replacement for TradingAgents' prose risk debate.
"""
import dataclasses

import pytest

from agentic_finance.risk_engine.schemas import (
    PreDecision,
    MarketRiskInputs,
    PortfolioState,
    RiskBudget,
    RiskedOrder,
)
from agentic_finance.risk_engine.risk_manager import assess_and_size

pytestmark = pytest.mark.unit


def _returns():
    # 30 modest daily returns, mildly volatile, slight positive drift.
    base = [0.01, -0.012, 0.008, -0.006, 0.015, -0.009, 0.004, -0.011,
            0.013, -0.007, 0.006, -0.005, 0.009, -0.014, 0.011, -0.004,
            0.007, -0.010, 0.012, -0.008, 0.005, -0.013, 0.010, -0.006,
            0.008, -0.009, 0.014, -0.007, 0.006, -0.011]
    return base


def _budget(**overrides):
    base = dict(
        target_vol=0.15,
        max_position=0.25,
        cvar_limit=0.10,
        atr_multiplier=2.0,
        max_drawdown=0.20,
        kelly_fraction=0.5,
        min_confidence=0.55,
        confidence_level=0.95,
        periods_per_year=252,
    )
    base.update(overrides)
    return RiskBudget(**base)


def _market(price=100.0, atr=2.0):
    return MarketRiskInputs(price=price, atr=atr, returns=tuple(_returns()))


def _portfolio(equity=100.0, peak=100.0, remaining=1.0):
    return PortfolioState(equity=equity, peak_equity=peak, remaining_budget=remaining)


# --- veto paths -----------------------------------------------------------

def test_drawdown_breach_vetoes_to_hold():
    order = assess_and_size(
        PreDecision(direction="long", confidence=0.9),
        _market(),
        _portfolio(equity=70.0, peak=100.0),  # 30% drawdown > 20% limit
        _budget(),
    )
    assert order.action == "hold"
    assert order.size == 0.0
    assert order.vetoed is True
    assert order.binding_constraint == "max_drawdown"


def test_low_confidence_holds():
    order = assess_and_size(
        PreDecision(direction="long", confidence=0.40),  # below 0.55
        _market(),
        _portfolio(),
        _budget(),
    )
    assert order.action == "hold"
    assert order.size == 0.0
    assert order.binding_constraint == "low_confidence"


def test_flat_direction_holds():
    order = assess_and_size(
        PreDecision(direction="flat", confidence=0.9),
        _market(),
        _portfolio(),
        _budget(),
    )
    assert order.action == "hold"
    assert order.size == 0.0


# --- normal long ----------------------------------------------------------

def test_long_signal_produces_sized_buy_with_stop_below_price():
    order = assess_and_size(
        PreDecision(direction="long", confidence=0.8),
        _market(price=100.0, atr=2.0),
        _portfolio(),
        _budget(),
    )
    assert order.action == "buy"
    assert 0.0 < order.size <= 0.25  # within max_position
    assert order.stop_price == pytest.approx(96.0)  # 100 - 2*2
    assert order.stop_price < 100.0
    assert order.var > 0 and order.cvar >= order.var
    assert order.vetoed is False


def test_short_signal_produces_sell_with_stop_above_price():
    order = assess_and_size(
        PreDecision(direction="short", confidence=0.8),
        _market(price=100.0, atr=2.0),
        _portfolio(),
        _budget(),
    )
    assert order.action == "sell"
    assert order.stop_price == pytest.approx(104.0)  # 100 + 2*2


# --- risk caps bind -------------------------------------------------------

def test_size_never_exceeds_max_position():
    order = assess_and_size(
        PreDecision(direction="long", confidence=1.0),
        _market(),
        _portfolio(),
        _budget(max_position=0.10),
    )
    assert order.size <= 0.10 + 1e-9


def test_remaining_budget_caps_size():
    order = assess_and_size(
        PreDecision(direction="long", confidence=1.0),
        _market(),
        _portfolio(remaining=0.05),
        _budget(max_position=0.50),
    )
    assert order.size <= 0.05 + 1e-9


def test_tight_cvar_limit_reduces_size_and_is_binding():
    loose = assess_and_size(
        PreDecision(direction="long", confidence=0.9), _market(), _portfolio(),
        _budget(cvar_limit=0.50),
    )
    tight = assess_and_size(
        PreDecision(direction="long", confidence=0.9), _market(), _portfolio(),
        _budget(cvar_limit=0.005),  # very tight -> must scale down
    )
    assert tight.size < loose.size
    assert tight.binding_constraint == "cvar_limit"


# --- immutability ---------------------------------------------------------

def test_risked_order_is_frozen():
    order = assess_and_size(
        PreDecision(direction="long", confidence=0.8), _market(), _portfolio(), _budget(),
    )
    assert isinstance(order, RiskedOrder)
    with pytest.raises(dataclasses.FrozenInstanceError):
        order.size = 0.99  # type: ignore[misc]
