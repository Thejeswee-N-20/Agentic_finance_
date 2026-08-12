"""Value-at-Risk (VaR) and Conditional VaR / Expected Shortfall (CVaR).

Conventions
-----------
- Inputs are period **simple returns** (e.g. daily), as an iterable of floats.
- VaR and CVaR are returned as **positive loss fractions**: ``0.04`` means a
  4% loss. A non-positive value means the (1 - confidence) tail contains no
  loss (i.e. even the tail observation is a gain).
- ``confidence`` is the coverage level in ``(0, 1)`` — e.g. ``0.95`` for 95%.

Two estimators are provided:
- **historical**: non-parametric, from the empirical return distribution.
- **parametric**: Gaussian closed form using the sample mean and stdev.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable, List

__all__ = [
    "historical_var",
    "historical_cvar",
    "parametric_var",
    "parametric_cvar",
]


def _validate(returns: Iterable[float], confidence: float, min_points: int) -> List[float]:
    data = [float(r) for r in returns]
    if len(data) < min_points:
        raise ValueError(
            f"need at least {min_points} return observation(s), got {len(data)}"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return data


def _quantile(sorted_data: List[float], q: float) -> float:
    """Linear-interpolation quantile (matches numpy's default 'linear')."""
    if q <= 0:
        return sorted_data[0]
    if q >= 1:
        return sorted_data[-1]
    pos = q * (len(sorted_data) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_data[lo]
    frac = pos - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def historical_var(returns: Iterable[float], confidence: float = 0.95) -> float:
    """Non-parametric VaR as a positive loss fraction.

    The loss threshold is the negative of the ``(1 - confidence)`` quantile of
    the return distribution.
    """
    data = _validate(returns, confidence, min_points=1)
    data.sort()
    tail_q = 1.0 - confidence
    return -_quantile(data, tail_q)


def historical_cvar(returns: Iterable[float], confidence: float = 0.95) -> float:
    """Non-parametric Expected Shortfall: mean loss in the worst tail.

    Averages all returns at or below the ``(1 - confidence)`` quantile. Always
    ``>= historical_var`` for the same inputs.
    """
    data = _validate(returns, confidence, min_points=1)
    data.sort()
    tail_q = 1.0 - confidence
    threshold = _quantile(data, tail_q)
    tail = [r for r in data if r <= threshold]
    if not tail:  # degenerate (e.g. all equal above threshold); fall back to VaR
        return -threshold
    return -(sum(tail) / len(tail))


def parametric_var(returns: Iterable[float], confidence: float = 0.95) -> float:
    """Gaussian VaR using sample mean and (sample) standard deviation.

    ``VaR = -(mu + z_{1-c} * sigma)`` where ``z_{1-c}`` is the standard-normal
    quantile of the lower tail (negative), so the result is a positive loss
    when the tail return is negative.
    """
    data = _validate(returns, confidence, min_points=2)
    mu = sum(data) / len(data)
    var = sum((r - mu) ** 2 for r in data) / (len(data) - 1)
    sigma = math.sqrt(var)
    z = NormalDist().inv_cdf(1.0 - confidence)  # negative
    return -(mu + z * sigma)


def parametric_cvar(returns: Iterable[float], confidence: float = 0.95) -> float:
    """Gaussian Expected Shortfall.

    ``CVaR = -(mu - sigma * phi(z) / (1 - c))`` where ``z`` is the lower-tail
    standard-normal quantile and ``phi`` the standard-normal pdf. Strictly
    greater than the Gaussian VaR for sigma > 0.
    """
    data = _validate(returns, confidence, min_points=2)
    mu = sum(data) / len(data)
    var = sum((r - mu) ** 2 for r in data) / (len(data) - 1)
    sigma = math.sqrt(var)
    alpha = 1.0 - confidence
    z = NormalDist().inv_cdf(alpha)
    pdf_z = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return -(mu - sigma * pdf_z / alpha)
