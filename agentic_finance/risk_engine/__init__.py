"""Quantitative risk engine for the Agentic Finance framework.

Turns an agent's directional view into an *enforced*, risk-budgeted order
(action, size, stop) — the quantitative replacement for TradingAgents'
prose-only risk debate. Pure-Python (stdlib), deterministic, and independently
testable without any LLM or API.

Public entry point: :func:`assess_and_size`.
"""

from agentic_finance.risk_engine.position_sizing import (
    apply_caps,
    fractional_kelly_size,
    volatility_target_size,
)
from agentic_finance.risk_engine.compliance import (
    CompliancePolicy,
    ComplianceResult,
    compliance_gate,
)
from agentic_finance.risk_engine.portfolio import (
    Holding,
    PortfolioLimits,
    apply_portfolio_constraints,
    pairwise_correlation,
)
from agentic_finance.risk_engine.risk_manager import assess_and_size
from agentic_finance.risk_engine.schemas import (
    MarketRiskInputs,
    PortfolioState,
    PreDecision,
    RiskBudget,
    RiskedOrder,
)
from agentic_finance.risk_engine.stops import (
    atr_stop_price,
    current_drawdown,
    drawdown_breached,
    stop_distance_fraction,
)
from agentic_finance.risk_engine.var_cvar import (
    historical_cvar,
    historical_var,
    parametric_cvar,
    parametric_var,
)

__all__ = [
    # entry point
    "assess_and_size",
    # portfolio-level constraints + compliance gate
    "PortfolioLimits", "Holding", "apply_portfolio_constraints",
    "pairwise_correlation", "CompliancePolicy", "ComplianceResult",
    "compliance_gate",
    # schemas
    "PreDecision",
    "MarketRiskInputs",
    "PortfolioState",
    "RiskBudget",
    "RiskedOrder",
    # var/cvar
    "historical_var",
    "historical_cvar",
    "parametric_var",
    "parametric_cvar",
    # sizing
    "volatility_target_size",
    "fractional_kelly_size",
    "apply_caps",
    # stops
    "atr_stop_price",
    "stop_distance_fraction",
    "current_drawdown",
    "drawdown_breached",
]
