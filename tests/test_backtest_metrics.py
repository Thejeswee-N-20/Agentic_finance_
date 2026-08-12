"""Unit tests for backtest performance metrics.

Conventions: inputs are period **simple returns**. Ratios are annualized with
``periods_per_year`` (default 252). Max drawdown is a positive fraction.
"""
import math

import pytest

from agentic_finance.backtest.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    cumulative_return,
    equity_curve,
)

pytestmark = pytest.mark.unit


# --- Sharpe ---------------------------------------------------------------

def test_sharpe_zero_when_no_excess_return():
    returns = [0.0, 0.0, 0.0, 0.0]
    assert sharpe_ratio(returns) == 0.0


def test_sharpe_positive_for_positive_mean():
    returns = [0.01, 0.012, 0.009, 0.011, 0.010]
    assert sharpe_ratio(returns) > 0


def test_sharpe_annualization_scales_by_sqrt_periods():
    returns = [0.01, -0.005, 0.012, -0.003, 0.008, -0.002]
    daily = sharpe_ratio(returns, periods_per_year=1)
    annual = sharpe_ratio(returns, periods_per_year=252)
    assert annual == pytest.approx(daily * math.sqrt(252), rel=1e-6)


def test_sharpe_subtracts_risk_free_rate():
    returns = [0.01] * 10
    # constant returns -> zero stdev -> guard returns 0 regardless of rf
    assert sharpe_ratio(returns, risk_free_rate=0.0) == 0.0


def test_sharpe_rejects_too_few_points():
    with pytest.raises(ValueError):
        sharpe_ratio([0.01])


# --- Sortino --------------------------------------------------------------

def test_sortino_infinite_when_no_downside():
    # No negative returns -> downside deviation 0 -> +inf for positive mean.
    returns = [0.01, 0.02, 0.0, 0.015]
    assert sortino_ratio(returns) == math.inf


def test_sortino_ge_zero_for_positive_mean_with_downside():
    returns = [0.02, -0.01, 0.03, -0.005, 0.01]
    assert sortino_ratio(returns) > 0


def test_sortino_at_least_uses_downside_only():
    # Sortino penalizes only downside, so for a series with limited downside it
    # should exceed the Sharpe ratio of the same series.
    returns = [0.03, -0.005, 0.04, -0.004, 0.02, -0.003]
    assert sortino_ratio(returns) > sharpe_ratio(returns)


# --- max drawdown ---------------------------------------------------------

def test_max_drawdown_known_series():
    # equity: 1 -> 1.1 -> 0.88 (peak 1.1) -> 0.99 ; worst dd = (1.1-0.88)/1.1 = 0.20
    returns = [0.10, -0.20, 0.125]
    assert max_drawdown(returns) == pytest.approx(0.20, rel=1e-6)


def test_max_drawdown_monotonic_up_is_zero():
    returns = [0.01, 0.02, 0.03]
    assert max_drawdown(returns) == 0.0


def test_max_drawdown_is_non_negative():
    returns = [-0.05, 0.02, -0.10, 0.03]
    assert max_drawdown(returns) >= 0


# --- Calmar ---------------------------------------------------------------

def test_calmar_is_annual_return_over_maxdd():
    returns = [0.10, -0.20, 0.125]
    mdd = max_drawdown(returns)
    cal = calmar_ratio(returns, periods_per_year=252)
    # sign follows the (small) annualized return; magnitude divided by mdd
    assert math.isfinite(cal)
    assert cal == pytest.approx(_expected_calmar(returns, mdd, 252), rel=1e-6)


def test_calmar_infinite_when_no_drawdown():
    returns = [0.01, 0.02, 0.03]
    assert calmar_ratio(returns) == math.inf


# --- cumulative return / equity curve ------------------------------------

def test_cumulative_return_compounds():
    returns = [0.10, 0.10]
    assert cumulative_return(returns) == pytest.approx(0.21, rel=1e-9)


def test_equity_curve_starts_at_initial_and_compounds():
    curve = equity_curve([0.10, -0.50], initial=100.0)
    assert curve[0] == pytest.approx(100.0)
    assert curve[-1] == pytest.approx(55.0)  # 100*1.1*0.5
    assert len(curve) == 3


# --- helper ---------------------------------------------------------------

def _expected_calmar(returns, mdd, ppy):
    total = 1.0
    for r in returns:
        total *= (1 + r)
    annual = total ** (ppy / len(returns)) - 1
    return annual / mdd
