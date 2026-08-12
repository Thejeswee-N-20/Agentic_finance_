"""Tests for portfolio-level constraints (exposure, sector, correlation)."""
from __future__ import annotations

import math
import random

from agentic_finance.risk_engine.portfolio import (
    Holding,
    PortfolioLimits,
    apply_portfolio_constraints,
    pairwise_correlation,
)
from agentic_finance.risk_engine.schemas import RiskedOrder


def _order(size: float, action: str = "buy") -> RiskedOrder:
    return RiskedOrder(action=action, size=size, stop_price=95.0, var=0.03,
                       cvar=0.05, binding_constraint="kelly", reason="test")


LIMITS = PortfolioLimits(max_gross_exposure=1.0, max_per_sector=0.40,
                         max_correlated_exposure=0.50, correlation_threshold=0.70)


class TestPairwiseCorrelation:
    def test_perfectly_correlated(self):
        a = [0.01, -0.02, 0.03, 0.01, -0.01]
        assert pairwise_correlation(a, a) > 0.999

    def test_anti_correlated(self):
        a = [0.01, -0.02, 0.03, 0.01, -0.01]
        b = [-x for x in a]
        assert pairwise_correlation(a, b) < -0.999

    def test_uncorrelated_noise_is_small(self):
        rng = random.Random(3)
        a = [rng.gauss(0, 0.01) for _ in range(500)]
        b = [rng.gauss(0, 0.01) for _ in range(500)]
        assert abs(pairwise_correlation(a, b)) < 0.2

    def test_degenerate_series_is_zero(self):
        assert pairwise_correlation([0.01], [0.02]) == 0.0
        assert pairwise_correlation([0.0, 0.0], [0.01, -0.01]) == 0.0


class TestApplyPortfolioConstraints:
    def test_unconstrained_order_passes_through(self):
        out = apply_portfolio_constraints(_order(0.20), "NVDA", "Technology",
                                          holdings=[], limits=LIMITS)
        assert out.size == 0.20
        assert out.binding_constraint == "kelly"  # unchanged

    def test_gross_exposure_cap_scales_down(self):
        holdings = [Holding("AAPL", 0.50, "Technology"), Holding("JPM", 0.40, "Banking")]
        out = apply_portfolio_constraints(_order(0.25), "NVDA", "Semis",
                                          holdings=holdings, limits=LIMITS)
        assert math.isclose(out.size, 0.10, abs_tol=1e-9)  # 1.0 - 0.9
        assert out.binding_constraint == "portfolio_gross"

    def test_gross_exposure_full_vetoes_to_hold(self):
        holdings = [Holding("AAPL", 0.60, "Technology"), Holding("JPM", 0.40, "Banking")]
        out = apply_portfolio_constraints(_order(0.25), "NVDA", "Semis",
                                          holdings=holdings, limits=LIMITS)
        assert out.action == "hold" and out.size == 0.0
        assert out.binding_constraint == "portfolio_gross"

    def test_sector_cap_scales_down(self):
        holdings = [Holding("AAPL", 0.30, "Technology")]
        out = apply_portfolio_constraints(_order(0.25), "MSFT", "Technology",
                                          holdings=holdings, limits=LIMITS)
        assert math.isclose(out.size, 0.10, abs_tol=1e-9)  # 0.40 - 0.30
        assert out.binding_constraint == "sector_cap"

    def test_correlation_cap_counts_correlated_holdings(self):
        rng = random.Random(5)
        base = [rng.gauss(0.001, 0.02) for _ in range(200)]
        noisy = [x + rng.gauss(0, 0.002) for x in base]     # corr ~ 1
        indep = [rng.gauss(0.001, 0.02) for _ in range(200)]
        holdings = [Holding("AAPL", 0.40, "Technology", returns=tuple(noisy)),
                    Holding("JPM", 0.20, "Banking", returns=tuple(indep))]
        out = apply_portfolio_constraints(
            _order(0.25), "NVDA", "Semis", holdings=holdings, limits=LIMITS,
            candidate_returns=tuple(base),
        )
        # correlated bucket = AAPL 0.40; cap 0.50 -> only 0.10 headroom
        assert math.isclose(out.size, 0.10, abs_tol=1e-9)
        assert out.binding_constraint == "correlation_cap"

    def test_hold_order_passes_through(self):
        out = apply_portfolio_constraints(_order(0.0, action="hold"), "NVDA",
                                          "Semis", holdings=[], limits=LIMITS)
        assert out.action == "hold" and out.size == 0.0

    def test_binding_constraint_is_tightest(self):
        # sector headroom (0.40-0.30=0.10) < gross headroom (1.0-0.70=0.30)
        holdings = [Holding("AAPL", 0.30, "Technology"), Holding("JPM", 0.40, "Banking")]
        out = apply_portfolio_constraints(_order(0.25), "MSFT", "Technology",
                                          holdings=holdings, limits=LIMITS)
        assert out.binding_constraint == "sector_cap"
        assert math.isclose(out.size, 0.10, abs_tol=1e-9)
