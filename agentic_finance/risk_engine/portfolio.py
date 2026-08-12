"""Portfolio-level risk constraints: gross exposure, sector, and correlation caps.

The per-asset engine (:func:`assess_and_size`) bounds a single position; this
module bounds the *book*. ``apply_portfolio_constraints`` takes the per-asset
order plus the current holdings and enforces three limits, scaling the size to
the tightest remaining headroom (or vetoing to hold when none remains):

- **gross exposure**  : total absolute weight across the book
- **sector cap**      : combined weight per sector
- **correlation cap** : combined weight of holdings whose return series
  correlate with the candidate above a threshold — diversification that a
  per-name limit cannot see

Like the per-asset engine it is pure, deterministic, and records the binding
constraint (``portfolio_gross`` / ``sector_cap`` / ``correlation_cap``) for the
explainability layer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from agentic_finance.risk_engine.schemas import RiskedOrder

__all__ = ["PortfolioLimits", "Holding", "apply_portfolio_constraints",
           "pairwise_correlation"]


@dataclass(frozen=True)
class PortfolioLimits:
    """Book-level limits (fractions of equity)."""

    max_gross_exposure: float = 1.0
    max_per_sector: float = 0.40
    max_correlated_exposure: float = 0.50
    correlation_threshold: float = 0.70


@dataclass(frozen=True)
class Holding:
    """An existing position: weight, sector, and (optionally) its returns."""

    ticker: str
    weight: float
    sector: str = ""
    returns: Tuple[float, ...] = ()


def pairwise_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation of two return series (0.0 when degenerate)."""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    xs, ys = list(a[-n:]), list(b[-n:])
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (sx * sy)


def apply_portfolio_constraints(
    order: RiskedOrder,
    ticker: str,
    sector: str,
    holdings: Sequence[Holding],
    limits: PortfolioLimits,
    candidate_returns: Optional[Sequence[float]] = None,
) -> RiskedOrder:
    """Scale (or veto) a per-asset order to respect book-level limits.

    Returns a new :class:`RiskedOrder`; the input is never mutated. When a
    limit binds, ``binding_constraint`` names the tightest one.
    """
    if order.action == "hold" or order.size <= 0.0:
        return order

    headrooms: List[Tuple[str, float]] = []

    gross = sum(abs(h.weight) for h in holdings)
    headrooms.append(("portfolio_gross", limits.max_gross_exposure - gross))

    if sector:
        sector_weight = sum(abs(h.weight) for h in holdings if h.sector == sector)
        headrooms.append(("sector_cap", limits.max_per_sector - sector_weight))

    if candidate_returns:
        correlated = sum(
            abs(h.weight) for h in holdings
            if h.returns and pairwise_correlation(candidate_returns, h.returns)
            >= limits.correlation_threshold
        )
        headrooms.append(("correlation_cap",
                          limits.max_correlated_exposure - correlated))

    name, headroom = min(headrooms, key=lambda kv: kv[1])
    if headroom >= order.size:
        return order  # no book-level limit binds

    if headroom <= 0.0:
        return replace(order, action="hold", size=0.0, stop_price=0.0,
                       vetoed=True, binding_constraint=name,
                       reason=f"vetoed: no {name.replace('_', ' ')} headroom "
                              f"({headroom:+.2%}) for {ticker}")
    return replace(order, size=headroom, binding_constraint=name,
                   reason=f"scaled to {headroom:.2%} by {name.replace('_', ' ')} "
                          f"for {ticker}")
