"""Unit tests for the multi-regime / vol-matched evaluation harness.

The harness runs any set of strategies (as uniform return-series producers) over
a price window and reports risk-adjusted metrics PLUS a volatility-matched
return, so a risk-managed strategy can be compared to an unconstrained
benchmark at *equal risk* rather than at face-value raw return.
"""
import math

import pytest

from agentic_finance.backtest.evaluation import (
    volatility_scale,
    realized_vol,
    StrategyResult,
    evaluate_window,
    aggregate,
    baseline_producer,
    risk_budgeted_producer,
)
from agentic_finance.backtest.baselines import buy_and_hold
from agentic_finance.backtest.risk_budgeted import constant_long
from agentic_finance.risk_engine.schemas import RiskBudget

pytestmark = pytest.mark.unit


def _uptrend(n=140, start=100.0, step=0.004):
    return [start * (1 + step) ** i for i in range(n)]


def _noisy_uptrend(n=140, start=100.0, step=0.004):
    # drift + deterministic oscillation so the series has real volatility
    # (vol-matching is undefined on a perfectly smooth / zero-vol series).
    return [start * (1 + step) ** i * (1 + 0.03 * math.sin(i * 1.3)) for i in range(n)]


# --- volatility_scale -----------------------------------------------------

def test_scale_to_own_vol_is_identity():
    returns = [0.01, -0.012, 0.008, -0.006, 0.015, -0.009]
    target = realized_vol(returns)
    scaled = volatility_scale(returns, target)
    assert scaled == pytest.approx(returns)


def test_scaling_doubles_when_target_is_double():
    returns = [0.01, -0.012, 0.008, -0.006, 0.015, -0.009]
    target = 2 * realized_vol(returns)
    scaled = volatility_scale(returns, target)
    assert scaled == pytest.approx([2 * r for r in returns])


def test_scaled_series_has_target_realized_vol():
    returns = [0.01, -0.012, 0.008, -0.006, 0.015, -0.009, 0.004, -0.011]
    target = 0.25
    scaled = volatility_scale(returns, target)
    assert realized_vol(scaled) == pytest.approx(target, rel=1e-6)


def test_scale_zero_vol_series_returns_zeros_without_error():
    returns = [0.0, 0.0, 0.0]
    assert volatility_scale(returns, 0.20) == pytest.approx([0.0, 0.0, 0.0])


# --- evaluate_window ------------------------------------------------------

def _producers():
    budget = RiskBudget(max_position=0.25, min_confidence=0.55)
    return {
        "Buy & Hold": baseline_producer(buy_and_hold, warmup=20),
        "Risk-budgeted": risk_budgeted_producer(constant_long(0.8), budget, warmup=20),
    }


def test_evaluate_window_returns_one_result_per_strategy():
    results = evaluate_window(_uptrend(), _producers(), regime="bull")
    assert len(results) == 2
    assert all(isinstance(r, StrategyResult) for r in results)
    assert {r.name for r in results} == {"Buy & Hold", "Risk-budgeted"}


def test_results_carry_regime_and_all_metrics():
    results = evaluate_window(_uptrend(), _producers(), regime="bull")
    r = results[0]
    assert r.regime == "bull"
    for field in ("cum_return", "vol_matched_return", "sharpe", "sortino",
                  "max_drawdown", "calmar"):
        assert hasattr(r, field)
        assert math.isfinite(getattr(r, field)) or getattr(r, field) in (math.inf, -math.inf)


def test_buy_and_hold_positive_in_uptrend():
    results = evaluate_window(_uptrend(), _producers(), regime="bull")
    bh = next(r for r in results if r.name == "Buy & Hold")
    assert bh.cum_return > 0


def test_vol_matched_lifts_low_exposure_strategy_return():
    # The risk-budgeted strategy runs at low exposure (realized vol < target),
    # so scaling to equal risk should lift its return above the raw figure.
    results = evaluate_window(_noisy_uptrend(), _producers(), regime="bull", target_vol=0.30)
    rb = next(r for r in results if r.name == "Risk-budgeted")
    assert rb.cum_return > 0
    assert rb.vol_matched_return > rb.cum_return


# --- aggregate ------------------------------------------------------------

def test_aggregate_averages_metric_across_windows_by_strategy():
    r1 = evaluate_window(_uptrend(), _producers(), regime="bull")
    r2 = evaluate_window(_uptrend(160, step=0.003), _producers(), regime="bull")
    agg = aggregate(r1 + r2, metric="sharpe")
    assert set(agg.keys()) == {"Buy & Hold", "Risk-budgeted"}
    # average of the two per-window Sharpes for Buy & Hold
    bh_vals = [r.sharpe for r in (r1 + r2) if r.name == "Buy & Hold"]
    assert agg["Buy & Hold"] == pytest.approx(sum(bh_vals) / len(bh_vals))
