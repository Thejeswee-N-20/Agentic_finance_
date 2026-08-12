"""Unit tests for the risk-budgeted backtest.

This is the headline integration: a directional signal is converted, bar by bar,
into an enforced sized position by the quantitative risk engine, with portfolio
equity threaded through so the max-drawdown circuit breaker actually engages.
"""
import pytest

from agentic_finance.backtest.risk_budgeted import (
    run_risk_budgeted_backtest,
    constant_long,
    RiskBudgetedResult,
)
from agentic_finance.backtest.engine import BacktestResult
from agentic_finance.risk_engine.schemas import PreDecision, RiskBudget

pytestmark = pytest.mark.unit


def _budget(**ov):
    base = dict(max_drawdown=0.20, max_position=0.25, min_confidence=0.55)
    base.update(ov)
    return RiskBudget(**base)


def test_returns_composed_result_with_orders():
    prices = [100.0 + i for i in range(40)]  # gentle uptrend
    res = run_risk_budgeted_backtest(prices, constant_long(0.8), _budget(), warmup=15)
    assert isinstance(res, RiskBudgetedResult)
    assert isinstance(res.performance, BacktestResult)
    # one order per realized period
    assert len(res.orders) == len(res.performance.returns)


def test_high_confidence_long_takes_buy_positions_in_uptrend():
    prices = [100.0 * (1.01 ** i) for i in range(40)]
    res = run_risk_budgeted_backtest(prices, constant_long(0.9), _budget(), warmup=15)
    actions = {o.action for o in res.orders}
    assert "buy" in actions
    assert res.performance.cumulative_return > 0


def test_low_confidence_signal_stays_flat():
    prices = [100.0 * (1.01 ** i) for i in range(40)]
    res = run_risk_budgeted_backtest(prices, constant_long(0.40), _budget(), warmup=15)
    assert all(o.action == "hold" for o in res.orders)
    assert res.performance.cumulative_return == pytest.approx(0.0)


def test_drawdown_breaker_engages_on_crash_and_caps_losses():
    # Rise to build a peak, then crash hard. A buy-and-hold long would suffer the
    # full crash; the breaker should trip and flatten the position partway down.
    prices = [100.0 + i for i in range(20)]            # 100 -> 119
    prices += [119.0 * (0.85 ** k) for k in range(1, 15)]  # steep multi-bar crash
    budget = _budget(max_drawdown=0.05, max_position=1.0)  # tight breaker, full size allowed
    res = run_risk_budgeted_backtest(prices, constant_long(0.9), budget, warmup=15)

    bindings = [o.binding_constraint for o in res.orders]
    assert "max_drawdown" in bindings
    # Once tripped, the breaker holds (size 0) for the remaining crash bars.
    first_trip = bindings.index("max_drawdown")
    assert all(res.orders[j].action == "hold" for j in range(first_trip, len(res.orders)))


def test_orders_carry_var_cvar_and_stop_for_buys():
    prices = [100.0 * (1.005 ** i) for i in range(40)]
    res = run_risk_budgeted_backtest(prices, constant_long(0.8), _budget(), warmup=20)
    buys = [o for o in res.orders if o.action == "buy"]
    assert buys, "expected at least one buy"
    o = buys[0]
    assert o.var is not None and o.cvar is not None
    assert o.stop_price > 0
