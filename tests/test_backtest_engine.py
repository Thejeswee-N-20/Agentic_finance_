"""Unit tests for the point-in-time backtest engine.

The engine must be leakage-safe: the weight applied to period t may depend only
on prices through t (the decision close), and is multiplied by the realized
t -> t+1 return.
"""
import dataclasses

import pytest

from agentic_finance.backtest.engine import run_backtest, BacktestResult
from agentic_finance.backtest.baselines import buy_and_hold

pytestmark = pytest.mark.unit


def test_buy_and_hold_matches_underlying_return():
    prices = [100.0, 110.0, 121.0]  # +10% each step
    result = run_backtest(prices, buy_and_hold, warmup=0)
    # Two periods of +10% compounded = +21%.
    assert result.cumulative_return == pytest.approx(0.21, rel=1e-9)
    assert result.returns == pytest.approx([0.10, 0.10])


def test_result_is_frozen_and_has_metrics():
    prices = [100.0, 110.0, 99.0, 105.0]
    result = run_backtest(prices, buy_and_hold, warmup=0)
    assert isinstance(result, BacktestResult)
    assert hasattr(result, "sharpe") and hasattr(result, "max_drawdown")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.sharpe = 1.0  # type: ignore[misc]


def test_warmup_skips_initial_periods():
    prices = [100.0, 110.0, 121.0, 133.1]
    result = run_backtest(prices, buy_and_hold, warmup=1)
    # warmup=1 -> first usable decision close is index 1; 2 realized returns.
    assert len(result.returns) == 2


def test_no_lookahead_weight_depends_only_on_past():
    # A spy strategy records the longest price index it was shown. It must never
    # exceed the index of the bar whose NEXT return it is paid — i.e. it cannot
    # see the bar that determines its own period return.
    seen_lengths = []

    def spy(prices):
        seen_lengths.append(len(prices))
        return 1.0

    prices = [10.0, 11.0, 12.0, 13.0, 14.0]
    run_backtest(prices, spy, warmup=0)
    # For a series of length N there are N-1 realized returns; the decision for
    # period i (return prices[i+1]/prices[i]) is shown prices[:i+1] (length i+1).
    # So the max window length shown is N-1, never N (the last bar's future).
    assert max(seen_lengths) == len(prices) - 1


def test_flat_weight_yields_zero_returns():
    prices = [100.0, 50.0, 200.0]
    result = run_backtest(prices, lambda p: 0.0, warmup=0)
    assert result.returns == pytest.approx([0.0, 0.0])
    assert result.cumulative_return == pytest.approx(0.0)


def test_partial_weight_scales_return():
    prices = [100.0, 110.0]  # +10%
    result = run_backtest(prices, lambda p: 0.5, warmup=0)
    assert result.returns == pytest.approx([0.05])  # half exposure


def test_rejects_too_short_series():
    with pytest.raises(ValueError):
        run_backtest([100.0], buy_and_hold, warmup=0)
