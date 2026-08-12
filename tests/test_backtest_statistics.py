"""Tests for statistical-rigor metrics: bootstrap CI + deflated Sharpe (offline)."""
from __future__ import annotations

import math
import random

from agentic_finance.backtest.metrics import sharpe_ratio
from agentic_finance.backtest.statistics import (
    bootstrap_confidence_interval,
    deflated_sharpe_ratio,
)


def _steady_returns(n=252, mu=0.001, vol=0.01, seed=7):
    rng = random.Random(seed)
    return [rng.gauss(mu, vol) for _ in range(n)]


def _noise_returns(n=252, seed=11):
    rng = random.Random(seed)
    return [rng.gauss(0.0, 0.02) for _ in range(n)]


class TestBootstrapCI:
    def test_ci_brackets_point_estimate(self):
        rets = _steady_returns()
        point = sharpe_ratio(rets)
        lo, hi = bootstrap_confidence_interval(rets, sharpe_ratio, n_boot=300, seed=0)
        assert lo <= point <= hi
        assert lo < hi

    def test_deterministic_with_seed(self):
        rets = _steady_returns()
        a = bootstrap_confidence_interval(rets, sharpe_ratio, n_boot=200, seed=42)
        b = bootstrap_confidence_interval(rets, sharpe_ratio, n_boot=200, seed=42)
        assert a == b

    def test_wider_interval_at_higher_confidence(self):
        rets = _steady_returns()
        lo90, hi90 = bootstrap_confidence_interval(rets, sharpe_ratio, n_boot=300,
                                                   ci=0.90, seed=1)
        lo99, hi99 = bootstrap_confidence_interval(rets, sharpe_ratio, n_boot=300,
                                                   ci=0.99, seed=1)
        assert (hi99 - lo99) > (hi90 - lo90)

    def test_short_series_is_safe(self):
        lo, hi = bootstrap_confidence_interval([0.01, -0.01], sharpe_ratio,
                                               n_boot=50, seed=0)
        assert math.isfinite(lo) and math.isfinite(hi)

    def test_works_with_other_metrics(self):
        from agentic_finance.backtest.metrics import max_drawdown
        rets = _steady_returns()
        lo, hi = bootstrap_confidence_interval(rets, max_drawdown, n_boot=100, seed=0)
        assert 0.0 <= lo <= hi <= 1.0


class TestDeflatedSharpe:
    def test_probability_bounds(self):
        for rets in (_steady_returns(), _noise_returns()):
            dsr = deflated_sharpe_ratio(rets, n_trials=5)
            assert 0.0 <= dsr <= 1.0

    def test_more_trials_deflate_more(self):
        rets = _steady_returns()
        assert deflated_sharpe_ratio(rets, n_trials=100) <= \
               deflated_sharpe_ratio(rets, n_trials=2)

    def test_strong_signal_beats_noise(self):
        strong = _steady_returns(mu=0.003, vol=0.01)
        noise = _noise_returns()
        assert deflated_sharpe_ratio(strong, n_trials=5) > \
               deflated_sharpe_ratio(noise, n_trials=5)

    def test_short_or_degenerate_series_is_zero(self):
        assert deflated_sharpe_ratio([0.01], n_trials=5) == 0.0
        assert deflated_sharpe_ratio([0.0] * 100, n_trials=5) == 0.0

    def test_single_trial_no_selection_penalty_high_for_strong(self):
        strong = _steady_returns(mu=0.004, vol=0.01)
        assert deflated_sharpe_ratio(strong, n_trials=1) > 0.9
