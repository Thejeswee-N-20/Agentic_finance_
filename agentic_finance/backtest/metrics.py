"""Performance metrics for strategy evaluation.

Inputs are period **simple returns** (e.g. daily). Risk-adjusted ratios are
annualized by ``periods_per_year`` (default 252 trading days). These are the
core quantities behind the dissertation's headline results (and the
risk-adjusted win condition vs the baseline).
"""
from __future__ import annotations

import math
from typing import Iterable, List

__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "cumulative_return",
    "equity_curve",
]

# Volatilities below this are treated as zero: real per-period return stdevs are
# ~1e-2, so this only catches (near-)constant series where floating-point noise
# would otherwise blow the ratio up to ~1e16.
_EPS = 1e-12


def _as_list(returns: Iterable[float], min_points: int = 2) -> List[float]:
    data = [float(r) for r in returns]
    if len(data) < min_points:
        raise ValueError(f"need at least {min_points} returns, got {len(data)}")
    return data


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def _sample_std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def sharpe_ratio(
    returns: Iterable[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio.

    ``risk_free_rate`` is a per-period rate subtracted from each return. Returns
    0.0 when the return volatility is zero (no meaningful ratio).
    """
    data = _as_list(returns)
    excess = [r - risk_free_rate for r in data]
    sigma = _sample_std(excess)
    if sigma <= _EPS:
        return 0.0
    return (_mean(excess) / sigma) * math.sqrt(periods_per_year)


def sortino_ratio(
    returns: Iterable[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio (downside-deviation denominator).

    Returns ``+inf`` when there is no downside and a positive mean, ``-inf``
    when no downside and a negative mean, and 0.0 when the mean excess is zero.
    """
    data = _as_list(returns)
    excess = [r - risk_free_rate for r in data]
    mean_excess = _mean(excess)
    downside = [min(0.0, e) for e in excess]
    downside_dev = math.sqrt(sum(d * d for d in downside) / len(downside))
    if downside_dev <= _EPS:
        if mean_excess > 0:
            return math.inf
        if mean_excess < 0:
            return -math.inf
        return 0.0
    return (mean_excess / downside_dev) * math.sqrt(periods_per_year)


def equity_curve(returns: Iterable[float], initial: float = 1.0) -> List[float]:
    """Compounded equity curve. Length is ``len(returns) + 1`` (includes start)."""
    data = [float(r) for r in returns]
    curve = [initial]
    for r in data:
        curve.append(curve[-1] * (1.0 + r))
    return curve


def cumulative_return(returns: Iterable[float]) -> float:
    """Total compounded return over the series (e.g. 0.21 == +21%)."""
    total = 1.0
    for r in returns:
        total *= (1.0 + float(r))
    return total - 1.0


def max_drawdown(returns: Iterable[float]) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction (0.20 == 20%)."""
    curve = equity_curve(returns)
    peak = curve[0]
    mdd = 0.0
    for value in curve:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak
            if dd > mdd:
                mdd = dd
    return mdd


def calmar_ratio(returns: Iterable[float], periods_per_year: int = 252) -> float:
    """Annualized return divided by max drawdown.

    Returns ``+inf`` (positive annual return) / ``-inf`` (negative) when there
    is no drawdown.
    """
    data = _as_list(returns, min_points=1)
    total = 1.0
    for r in data:
        total *= (1.0 + r)
    annual_return = total ** (periods_per_year / len(data)) - 1.0
    mdd = max_drawdown(data)
    if mdd == 0:
        if annual_return > 0:
            return math.inf
        if annual_return < 0:
            return -math.inf
        return 0.0
    return annual_return / mdd
