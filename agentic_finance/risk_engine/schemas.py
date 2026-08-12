"""Immutable data contracts for the risk engine.

All frozen dataclasses: inputs are never mutated and the produced order is a new
value. These types are the boundary between the (LLM) agent layer and the
quantitative risk policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

__all__ = [
    "PreDecision",
    "MarketRiskInputs",
    "PortfolioState",
    "RiskBudget",
    "RiskedOrder",
]


@dataclass(frozen=True)
class PreDecision:
    """Directional view emitted by the signal-fusion layer.

    ``direction`` is one of ``"long"``, ``"short"``, ``"flat"``. ``confidence``
    is the fused confidence in ``[0, 1]``. ``win_loss_ratio`` lets the fusion
    layer pass an expected payoff asymmetry for Kelly sizing (default symmetric).
    """

    direction: str
    confidence: float
    win_loss_ratio: float = 1.0


@dataclass(frozen=True)
class MarketRiskInputs:
    """Point-in-time market features for the asset under consideration."""

    price: float
    atr: float
    returns: Tuple[float, ...]  # recent period returns (oldest -> newest)


@dataclass(frozen=True)
class PortfolioState:
    """Current portfolio state used for drawdown and budget checks."""

    equity: float
    peak_equity: float
    remaining_budget: float = 1.0  # unused fraction of equity available to allocate


@dataclass(frozen=True)
class RiskBudget:
    """Risk policy configuration (the firm's risk limits)."""

    target_vol: float = 0.15          # annualized volatility target
    max_position: float = 0.25        # per-position cap (fraction of equity)
    cvar_limit: float = 0.10          # max acceptable CVaR (loss fraction) per position
    atr_multiplier: float = 2.0       # ATR multiples for the stop
    max_drawdown: float = 0.20        # circuit-breaker drawdown limit
    kelly_fraction: float = 0.5       # fractional-Kelly scaler
    min_confidence: float = 0.55      # below this -> hold
    confidence_level: float = 0.95    # VaR/CVaR confidence
    periods_per_year: int = 252       # annualization factor for vol


@dataclass(frozen=True)
class RiskedOrder:
    """The enforced, risk-budgeted order (or veto) produced by the policy."""

    action: str               # "buy" | "sell" | "hold"
    size: float               # fraction of equity in [0, max_position]
    stop_price: float         # 0.0 when no position
    var: float                # historical VaR (positive loss fraction)
    cvar: float               # historical CVaR (positive loss fraction)
    binding_constraint: str   # which rule determined the outcome
    reason: str               # human-readable rationale (feeds the XAI layer)
    vetoed: bool = False
