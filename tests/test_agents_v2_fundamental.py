"""Tests for the fundamental agent: metrics -> typed Signal.

Offline by design: scoring is exercised on synthetic ``FundamentalMetrics``;
the yfinance fetcher is tested against a stubbed ``Ticker`` object.
"""
from __future__ import annotations

import pytest

from agentic_finance.agents_v2.fundamental_agent import (
    FundamentalMetrics,
    fetch_fundamentals,
    fundamental_signal,
)


def _strong() -> FundamentalMetrics:
    """Healthy large-cap: growing, profitable, cheapening, low leverage."""
    return FundamentalMetrics(
        trailing_pe=30.0, forward_pe=24.0, revenue_growth=0.25,
        profit_margin=0.30, return_on_equity=0.40, debt_to_equity=40.0,
    )


def _weak() -> FundamentalMetrics:
    """Shrinking, loss-making, richening multiple, heavy leverage."""
    return FundamentalMetrics(
        trailing_pe=20.0, forward_pe=35.0, revenue_growth=-0.10,
        profit_margin=-0.05, return_on_equity=-0.08, debt_to_equity=250.0,
    )


class TestFundamentalSignal:
    def test_strong_fundamentals_yield_long(self):
        sig = fundamental_signal(_strong())
        assert sig.source == "fundamental"
        assert sig.direction == "long"
        assert sig.strength > 0.0
        assert sig.confidence > 0.5

    def test_weak_fundamentals_yield_short(self):
        sig = fundamental_signal(_weak())
        assert sig.direction == "short"
        assert sig.strength > 0.0

    def test_empty_metrics_yield_flat_zero_confidence(self):
        sig = fundamental_signal(FundamentalMetrics())
        assert sig.direction == "flat"
        assert sig.strength == 0.0
        assert sig.confidence == 0.0
        assert "no fundamental data" in sig.rationale

    def test_partial_data_lowers_confidence(self):
        partial = FundamentalMetrics(revenue_growth=0.25, profit_margin=0.30)
        full_conf = fundamental_signal(_strong()).confidence
        partial_conf = fundamental_signal(partial).confidence
        assert 0.0 < partial_conf < full_conf

    def test_high_leverage_lowers_confidence(self):
        low_lev = fundamental_signal(_strong())
        high_lev = fundamental_signal(
            FundamentalMetrics(
                trailing_pe=30.0, forward_pe=24.0, revenue_growth=0.25,
                profit_margin=0.30, return_on_equity=0.40, debt_to_equity=300.0,
            )
        )
        assert high_lev.confidence < low_lev.confidence

    def test_mixed_signals_yield_flat(self):
        mixed = FundamentalMetrics(
            trailing_pe=30.0, forward_pe=30.0,   # valuation: neutral
            revenue_growth=0.25,                  # growth: +1
            profit_margin=-0.05,                  # profitability: -1
        )
        sig = fundamental_signal(mixed)
        assert sig.direction == "flat"

    def test_evidence_carries_metric_values(self):
        sig = fundamental_signal(_strong())
        joined = " ".join(sig.evidence)
        assert "revenue_growth" in joined
        assert "profit_margin" in joined

    def test_signal_bounds_are_valid(self):
        for metrics in (_strong(), _weak(), FundamentalMetrics()):
            sig = fundamental_signal(metrics)
            assert 0.0 <= sig.strength <= 1.0
            assert 0.0 <= sig.confidence <= 1.0

    def test_metrics_are_immutable(self):
        m = _strong()
        with pytest.raises(Exception):
            m.trailing_pe = 10.0  # type: ignore[misc]


class TestFetchFundamentals:
    def test_parses_yfinance_info_dict(self, monkeypatch):
        class _StubTicker:
            def __init__(self, ticker):
                self.info = {
                    "trailingPE": 28.5, "forwardPE": 22.1,
                    "revenueGrowth": 0.18, "profitMargins": 0.26,
                    "returnOnEquity": 0.35, "debtToEquity": 45.2,
                }

        import agentic_finance.agents_v2.fundamental_agent as fa
        monkeypatch.setattr(fa.yf, "Ticker", _StubTicker)
        m = fetch_fundamentals("NVDA")
        assert m.trailing_pe == 28.5
        assert m.revenue_growth == 0.18
        assert m.debt_to_equity == 45.2

    def test_fetch_failure_returns_empty_metrics(self, monkeypatch):
        class _BoomTicker:
            def __init__(self, ticker):
                raise RuntimeError("network down")

        import agentic_finance.agents_v2.fundamental_agent as fa
        monkeypatch.setattr(fa.yf, "Ticker", _BoomTicker)
        m = fetch_fundamentals("NVDA")
        assert m == FundamentalMetrics()

    def test_missing_keys_map_to_none(self, monkeypatch):
        class _SparseTicker:
            def __init__(self, ticker):
                self.info = {"trailingPE": 30.0}

        import agentic_finance.agents_v2.fundamental_agent as fa
        monkeypatch.setattr(fa.yf, "Ticker", _SparseTicker)
        m = fetch_fundamentals("NVDA")
        assert m.trailing_pe == 30.0
        assert m.revenue_growth is None
