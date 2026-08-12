"""Fundamental agent: company financial-health ratios -> typed Signal.

Assesses valuation, growth, and profitability from a small set of fundamental
ratios (as retrievable from Yahoo Finance) and emits a directional view. Three
deterministic votes decide direction:

- **growth**: revenue growth strong (+1) / shrinking (-1)
- **profitability**: healthy margins (+1) / loss-making (-1)
- **valuation**: forward P/E below trailing (earnings expected to grow, +1) /
  richening multiple (-1)

Direction follows the net vote; ``strength`` scales with vote unanimity;
``confidence`` rises with vote agreement and data coverage, and is penalised for
heavy leverage (high debt/equity). Missing data degrades gracefully: absent
metrics simply cast no vote, and with no data at all the agent emits a flat,
zero-confidence signal — so a fundamentals outage can never crash the pipeline.

Deterministic given its inputs; only :func:`fetch_fundamentals` touches the
network (defensively — any failure yields empty metrics).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import List, Optional

import yfinance as yf

from agentic_finance.agents_v2.schemas import Signal

__all__ = ["FundamentalMetrics", "fundamental_signal", "fetch_fundamentals"]

# Vote thresholds.
_GROWTH_STRONG = 0.10       # revenue growth >= 10% -> growth vote +1
_MARGIN_HEALTHY = 0.10      # net margin >= 10% -> profitability vote +1
_PE_IMPROVE = 0.95          # forward P/E <= 95% of trailing -> valuation vote +1
_PE_RICHEN = 1.05           # forward P/E >= 105% of trailing -> valuation vote -1
_HIGH_LEVERAGE = 150.0      # debt/equity above this (in %) starts the penalty
_MAX_LEV_PENALTY = 0.20


@dataclass(frozen=True)
class FundamentalMetrics:
    """Point-in-time fundamental ratios (all optional; missing = unknown).

    ``debt_to_equity`` follows the Yahoo Finance convention of a percentage
    (e.g. ``45.0`` means 0.45x equity).
    """

    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    return_on_equity: Optional[float] = None
    debt_to_equity: Optional[float] = None

    @property
    def coverage(self) -> float:
        """Fraction of metrics available, in [0, 1]."""
        values = [getattr(self, f.name) for f in fields(self)]
        return sum(v is not None for v in values) / len(values)


def _votes(m: FundamentalMetrics) -> List[int]:
    """Cast the growth / profitability / valuation votes (absent data: no vote)."""
    votes: List[int] = []
    if m.revenue_growth is not None:
        if m.revenue_growth >= _GROWTH_STRONG:
            votes.append(1)
        elif m.revenue_growth <= 0.0:
            votes.append(-1)
        else:
            votes.append(0)
    if m.profit_margin is not None:
        if m.profit_margin >= _MARGIN_HEALTHY:
            votes.append(1)
        elif m.profit_margin < 0.0:
            votes.append(-1)
        else:
            votes.append(0)
    if m.trailing_pe is not None and m.forward_pe is not None and m.trailing_pe > 0:
        ratio = m.forward_pe / m.trailing_pe
        if ratio <= _PE_IMPROVE:
            votes.append(1)
        elif ratio >= _PE_RICHEN:
            votes.append(-1)
        else:
            votes.append(0)
    return votes


def fundamental_signal(metrics: FundamentalMetrics) -> Signal:
    """Produce a fundamental-health Signal from the available ratios."""
    votes = _votes(metrics)
    if not votes:
        return Signal(source="fundamental", direction="flat", strength=0.0,
                      confidence=0.0, rationale="no fundamental data available")

    net = sum(votes)
    if net > 0:
        direction = "long"
    elif net < 0:
        direction = "short"
    else:
        direction = "flat"

    # Strength scales with how unanimously the cast votes point one way.
    strength = min(1.0, abs(net) / len(votes)) if direction != "flat" else 0.0

    # Confidence: agreement among cast votes + data coverage, minus a leverage
    # penalty that grows with debt/equity beyond the healthy range.
    aligned = sum(1 for v in votes if v == (1 if net > 0 else -1 if net < 0 else 0))
    agreement = aligned / len(votes)
    lev_penalty = 0.0
    if metrics.debt_to_equity is not None and metrics.debt_to_equity > _HIGH_LEVERAGE:
        overshoot = (metrics.debt_to_equity - _HIGH_LEVERAGE) / _HIGH_LEVERAGE
        lev_penalty = min(_MAX_LEV_PENALTY, 0.10 * (1.0 + overshoot))
    confidence = max(0.05, min(0.90, 0.25 + 0.35 * agreement
                               + 0.20 * metrics.coverage - lev_penalty))

    evidence: List[str] = []
    for name in ("trailing_pe", "forward_pe", "revenue_growth", "profit_margin",
                 "return_on_equity", "debt_to_equity"):
        value = getattr(metrics, name)
        if value is not None:
            evidence.append(f"{name}={value:+.3f}")
    evidence.append(f"votes(net)={net:+d}/{len(votes)}")

    rationale = (
        f"{direction} on fundamentals: {net:+d} net vote(s) across "
        f"growth/profitability/valuation; coverage {metrics.coverage:.0%}."
    )
    return Signal(source="fundamental", direction=direction, strength=strength,
                  confidence=confidence, rationale=rationale,
                  evidence=tuple(evidence))


def fetch_fundamentals(ticker: str) -> FundamentalMetrics:
    """Fetch fundamental ratios from Yahoo Finance (defensive; never raises)."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:  # noqa: BLE001 - a fundamentals outage must not break callers
        return FundamentalMetrics()

    def _num(key: str) -> Optional[float]:
        value = info.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    return FundamentalMetrics(
        trailing_pe=_num("trailingPE"),
        forward_pe=_num("forwardPE"),
        revenue_growth=_num("revenueGrowth"),
        profit_margin=_num("profitMargins"),
        return_on_equity=_num("returnOnEquity"),
        debt_to_equity=_num("debtToEquity"),
    )
