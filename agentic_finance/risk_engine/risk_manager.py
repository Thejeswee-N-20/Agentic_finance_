"""The risk-manager policy node.

``assess_and_size`` is the single entry point: given a directional pre-decision
and point-in-time market + portfolio features, it returns an enforced
``RiskedOrder``. It composes the VaR/CVaR, position-sizing, and stop modules
and records *which* constraint was binding so the XAI layer can explain the
sized outcome.

Sizing logic (most conservative wins):
  1. Circuit breaker: if drawdown >= limit -> veto to hold.
  2. Gate: flat direction or confidence < min_confidence -> hold.
  3. Candidate sizes: volatility-target and fractional-Kelly; take the min.
  4. CVaR scaling: if position CVaR exceeds the limit, scale size by
     ``cvar_limit / cvar``.
  5. Caps: clamp to per-position cap and remaining portfolio budget.
The binding constraint is whichever step produced the smallest size.
"""
from __future__ import annotations

import math
from typing import List, Tuple

from agentic_finance.risk_engine.position_sizing import (
    apply_caps,
    fractional_kelly_size,
    volatility_target_size,
)
from agentic_finance.risk_engine.schemas import (
    MarketRiskInputs,
    PortfolioState,
    PreDecision,
    RiskBudget,
    RiskedOrder,
)
from agentic_finance.risk_engine.stops import (
    atr_stop_price,
    drawdown_breached,
)
from agentic_finance.risk_engine.var_cvar import historical_cvar, historical_var

_ACTION = {"long": "buy", "short": "sell"}


def _annualized_vol(returns: Tuple[float, ...], periods_per_year: int) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    mu = sum(returns) / n
    var = sum((r - mu) ** 2 for r in returns) / (n - 1)
    return math.sqrt(var) * math.sqrt(periods_per_year)


def _hold(binding: str, reason: str, var: float, cvar: float, vetoed: bool = False) -> RiskedOrder:
    return RiskedOrder(
        action="hold", size=0.0, stop_price=0.0, var=var, cvar=cvar,
        binding_constraint=binding, reason=reason, vetoed=vetoed,
    )


def assess_and_size(
    pre: PreDecision,
    market: MarketRiskInputs,
    portfolio: PortfolioState,
    budget: RiskBudget,
) -> RiskedOrder:
    """Convert a directional view into an enforced, risk-budgeted order."""
    var = historical_var(market.returns, budget.confidence_level)
    cvar = historical_cvar(market.returns, budget.confidence_level)

    # 1. Circuit breaker.
    if drawdown_breached(portfolio.equity, portfolio.peak_equity, budget.max_drawdown):
        return _hold(
            "max_drawdown",
            f"Max-drawdown circuit breaker tripped "
            f"(equity {portfolio.equity:.2f} vs peak {portfolio.peak_equity:.2f}).",
            var, cvar, vetoed=True,
        )

    # 2. Gate on signal / confidence.
    if pre.direction not in _ACTION:
        return _hold("no_signal", f"No directional signal (direction={pre.direction!r}).", var, cvar)
    if pre.confidence < budget.min_confidence:
        return _hold(
            "low_confidence",
            f"Confidence {pre.confidence:.2f} below minimum {budget.min_confidence:.2f}.",
            var, cvar,
        )

    # 3. Candidate sizes (track which is binding).
    asset_vol = _annualized_vol(market.returns, budget.periods_per_year)
    vol_size = volatility_target_size(budget.target_vol, asset_vol, max_leverage=1.0)
    kelly_size = fractional_kelly_size(pre.confidence, pre.win_loss_ratio, budget.kelly_fraction)

    candidates: List[Tuple[float, str]] = [(vol_size, "vol_target"), (kelly_size, "kelly")]
    size, binding = min(candidates, key=lambda c: c[0])

    # 4. CVaR scaling.
    if cvar > budget.cvar_limit and cvar > 0:
        scaled = size * (budget.cvar_limit / cvar)
        if scaled < size:
            size, binding = scaled, "cvar_limit"

    # 5. Caps.
    capped = apply_caps(size, budget.max_position, portfolio.remaining_budget)
    if capped < size:
        binding = "position_cap" if budget.max_position <= portfolio.remaining_budget else "portfolio_budget"
    size = capped

    if size <= 0.0:
        return _hold("zero_size", "Risk limits collapsed the position to zero.", var, cvar)

    stop = atr_stop_price(market.price, market.atr, budget.atr_multiplier, pre.direction)
    action = _ACTION[pre.direction]
    reason = (
        f"{action.upper()} sized to {size:.4f} of equity "
        f"(binding={binding}; asset_vol={asset_vol:.3f}, VaR={var:.4f}, CVaR={cvar:.4f}); "
        f"ATR stop @ {stop:.2f}."
    )
    return RiskedOrder(
        action=action, size=size, stop_price=stop, var=var, cvar=cvar,
        binding_constraint=binding, reason=reason, vetoed=False,
    )
