"""Unit tests for the VaR / CVaR module of the quantitative risk engine.

VaR and CVaR are expressed as POSITIVE loss fractions (e.g. 0.04 == a 4% loss).
A 95% VaR is the loss not exceeded with 95% probability; CVaR (Expected
Shortfall) is the mean loss in the worst (1 - c) tail.
"""
import math

import pytest

from agentic_finance.risk_engine.var_cvar import (
    historical_var,
    historical_cvar,
    parametric_var,
    parametric_cvar,
)

pytestmark = pytest.mark.unit


# --- historical VaR -------------------------------------------------------

def test_historical_var_basic_percentile():
    # 100 returns from -0.50 ... +0.49 (step 0.01). The 5th percentile loss
    # sits around -0.45..-0.46, so 95% VaR is ~0.45 as a positive loss.
    returns = [(-50 + i) / 100.0 for i in range(100)]  # -0.50 .. 0.49
    var95 = historical_var(returns, confidence=0.95)
    assert 0.44 <= var95 <= 0.46


def test_historical_var_is_positive_loss_for_losing_tail():
    returns = [-0.10, -0.05, -0.02, 0.01, 0.03, 0.04, 0.06, 0.02, -0.01, 0.00]
    var = historical_var(returns, confidence=0.90)
    assert var > 0  # a loss magnitude


def test_historical_var_all_gains_is_non_positive():
    # No losing observations -> VaR loss is zero or negative (a "gain at risk").
    returns = [0.01, 0.02, 0.03, 0.04, 0.05]
    assert historical_var(returns, confidence=0.95) <= 0


def test_historical_var_higher_confidence_is_more_conservative():
    returns = [(-50 + i) / 100.0 for i in range(100)]
    assert historical_var(returns, 0.99) >= historical_var(returns, 0.95)


def test_historical_var_rejects_empty():
    with pytest.raises(ValueError):
        historical_var([], confidence=0.95)


def test_historical_var_rejects_bad_confidence():
    with pytest.raises(ValueError):
        historical_var([0.01, -0.01], confidence=1.5)


# --- historical CVaR ------------------------------------------------------

def test_historical_cvar_at_least_var():
    # Expected shortfall (mean tail loss) >= VaR (tail threshold).
    returns = [(-50 + i) / 100.0 for i in range(100)]
    var = historical_var(returns, 0.95)
    cvar = historical_cvar(returns, 0.95)
    assert cvar >= var


def test_historical_cvar_known_tail_mean():
    # Worst 10% of these 10 obs is just {-0.10}; CVaR90 == 0.10.
    returns = [-0.10, -0.05, -0.02, 0.01, 0.03, 0.04, 0.06, 0.02, -0.01, 0.00]
    assert historical_cvar(returns, confidence=0.90) == pytest.approx(0.10)


# --- parametric (Gaussian) VaR / CVaR ------------------------------------

def test_parametric_var_zero_mean_matches_z_sigma():
    # For mu=0, 95% VaR ~= 1.645 * sigma.
    sigma = 0.02
    returns = _normal_like(mu=0.0, sigma=sigma, n=2000)
    var = parametric_var(returns, confidence=0.95)
    assert var == pytest.approx(1.645 * sigma, rel=0.10)


def test_parametric_cvar_greater_than_parametric_var():
    returns = _normal_like(mu=0.0, sigma=0.02, n=2000)
    assert parametric_cvar(returns, 0.95) > parametric_var(returns, 0.95)


def test_parametric_var_rejects_single_point():
    with pytest.raises(ValueError):
        parametric_var([0.01], confidence=0.95)


# --- helper ---------------------------------------------------------------

def _normal_like(mu: float, sigma: float, n: int) -> list[float]:
    """Deterministic, roughly-normal sample via inverse-CDF over a uniform grid.

    Avoids RNG so the test is fully reproducible.
    """
    from statistics import NormalDist

    nd = NormalDist(mu, sigma)
    # Midpoint quantiles (i+0.5)/n -> evenly spaced probabilities in (0, 1).
    return [nd.inv_cdf((i + 0.5) / n) for i in range(n)]
