"""Signal-fusion layer: typed Signals -> risk-engine PreDecision.

A confidence-weighted vote across specialist agents, replacing TradingAgents'
bull/bear prose debate. Each signal contributes its ``signed_strength`` weighted
by its ``confidence`` (and an optional per-source weight). The net is a
conviction score in [-1, 1]; conflicting agents cancel, so genuine disagreement
collapses toward ``flat`` (which the risk engine then holds on). The output
plugs directly into :func:`agentic_finance.risk_engine.assess_and_size`.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from agentic_finance.agents_v2.schemas import Signal
from agentic_finance.risk_engine.schemas import PreDecision

__all__ = ["fuse_signals"]


def fuse_signals(
    signals: Sequence[Signal],
    weights: Optional[Mapping[str, float]] = None,
    deadband: float = 0.05,
) -> PreDecision:
    """Fuse agent signals into a directional PreDecision.

    ``weights`` optionally scales each source's influence (default 1.0 each).
    ``deadband`` is the minimum |net conviction| required to take a side.
    """
    if not signals:
        return PreDecision(direction="flat", confidence=0.0)

    def w(sig: Signal) -> float:
        return (weights or {}).get(sig.source, 1.0)

    total_weighted_conf = sum(w(s) * s.confidence for s in signals)
    if total_weighted_conf <= 0:
        return PreDecision(direction="flat", confidence=0.0)

    # Direction: confidence-weighted net signed strength (conflicts cancel).
    net = sum(w(s) * s.confidence * s.signed_strength for s in signals) / total_weighted_conf
    if net > deadband:
        direction = "long"
    elif net < -deadband:
        direction = "short"
    else:
        return PreDecision(direction="flat", confidence=0.0)

    # Confidence reflects RELIABILITY x CONSENSUS, independent of raw move
    # magnitude (the risk engine sizes magnitude via vol-targeting). Using |net|
    # here would discard agent confidence and suppress confident low-magnitude
    # signals (e.g. a steady low-vol uptrend), so confidence is the
    # confidence-weighted mean confidence of the agreeing agents times their
    # share of total directional confidence (consensus).
    dir_long = direction == "long"
    directional = [s for s in signals if s.signed_strength != 0.0]
    agreeing = [s for s in directional if (s.signed_strength > 0) == dir_long]
    dir_conf = sum(w(s) * s.confidence for s in directional)
    agree_conf = sum(w(s) * s.confidence for s in agreeing)
    consensus = agree_conf / dir_conf if dir_conf > 0 else 0.0
    mean_conf = (
        sum(w(s) * s.confidence * s.confidence for s in agreeing) / agree_conf
        if agree_conf > 0 else 0.0
    )
    confidence = max(0.0, min(0.99, mean_conf * consensus))
    return PreDecision(direction=direction, confidence=confidence, win_loss_ratio=1.0)
