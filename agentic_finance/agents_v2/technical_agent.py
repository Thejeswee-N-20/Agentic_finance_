"""Technical agent: indicators -> typed Signal.

Design: time-series momentum is the *primary* directional driver — it is the
robust, hard-to-beat technical factor — and the slower trend (SMA20 vs SMA50),
MACD histogram, and realised volatility act as **confirmation gates** on
*confidence* rather than as equal votes on *direction*. The intuition is that a
momentum signal should be sized up only when slower indicators agree, and sized
down (or held) when they disagree or when volatility is high; this suppresses the
whipsaws that a naive equal-weight vote of the three indicators suffers. RSI
extremes trim conviction against chasing an exhausted move.

Deterministic and point-in-time (reuses
:func:`agentic_finance.features.extract_features`), so it needs no model and is
leakage-safe by construction.
"""
from __future__ import annotations

from typing import List, Sequence

from agentic_finance.agents_v2.schemas import Signal
from agentic_finance.features import extract_features

__all__ = ["technical_signal"]

# Momentum deadband: ignore tiny 20-day moves as directionless noise.
_MOM_DEADBAND = 0.002


def _sign(x: float, eps: float = 1e-9) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _flat(reason: str) -> Signal:
    return Signal(source="technical", direction="flat", strength=0.0,
                  confidence=0.0, rationale=reason)


def technical_signal(prices: Sequence[float]) -> Signal:
    """Produce a momentum-anchored, confirmation-gated technical Signal."""
    fv = extract_features(prices)
    if fv is None:
        return _flat("insufficient history for technical indicators")

    # Primary driver: time-series momentum. Long-only — like the technical
    # baselines and because shorting single names is materially riskier; a
    # negative-momentum reading yields no position rather than a short.
    if fv.ret_20d <= _MOM_DEADBAND:
        return _flat("momentum non-positive (long-only agent)")
    decided = 1
    direction = "long"

    # Confirmation gates: how many slower signals agree with momentum.
    trend = _sign(fv.sma20_sma50_ratio)
    macd = _sign(fv.macd_hist)
    confirms = sum(1 for x in (trend, macd) if x == decided)  # 0, 1, or 2

    # Strength scales with the magnitude of momentum (bounded).
    strength = min(1.0, abs(fv.ret_20d) * 6.0)
    if fv.rsi_14 > 75:  # overbought: trim conviction against chasing
        strength *= 0.7

    # Confidence: base (trades like momentum) + confirmation bonus, penalised by
    # high volatility. 0 confirmations -> ~0.55; 1 -> ~0.73; 2 -> ~0.91, minus a
    # volatility penalty so positions shrink in turbulent regimes.
    vol_penalty = min(0.25, fv.vol_20d * 5.0)
    confidence = max(0.05, min(0.95, 0.55 + 0.18 * confirms - vol_penalty))

    evidence: List[str] = [
        f"momentum(ret20d)={fv.ret_20d:+.4f}",
        f"trend(SMA20/50)={fv.sma20_sma50_ratio:+.4f}",
        f"macd_hist={fv.macd_hist:+.4f}",
        f"rsi14={fv.rsi_14:.1f}",
        f"vol20d={fv.vol_20d:.4f}",
        f"confirmations={confirms}/2",
    ]
    rationale = (
        f"{direction} on 20-day momentum, confirmed by {confirms}/2 slower signals "
        f"(trend/MACD); volatility-adjusted confidence."
    )
    return Signal(
        source="technical", direction=direction, strength=strength,
        confidence=confidence, rationale=rationale, evidence=tuple(evidence),
    )
