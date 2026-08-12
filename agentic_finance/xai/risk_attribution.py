"""Risk-constraint attribution.

Explains the *size* of each enforced position by the constraint that bound it
(``kelly``, ``vol_target``, ``cvar_limit``, ``position_cap``, ``portfolio_budget``,
``max_drawdown``, ``low_confidence``, ...). This is the quantitative half of the
XAI story and the data source for the position-size-attribution figure.

Pure-Python: operates on the ``RiskedOrder`` sequence the risk-budgeted backtest
already produces, so it needs no model and no extra dependencies.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence

from agentic_finance.risk_engine.schemas import RiskedOrder

__all__ = ["attribute_constraints", "binding_summary", "explain_order"]


def attribute_constraints(orders: Sequence[RiskedOrder]) -> Dict[str, int]:
    """Count how many orders each binding constraint produced."""
    return dict(Counter(o.binding_constraint for o in orders))


def binding_summary(orders: Sequence[RiskedOrder]) -> Dict[str, float]:
    """Fraction of orders attributed to each binding constraint (sums to 1)."""
    counts = attribute_constraints(orders)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def explain_order(order: RiskedOrder) -> str:
    """One-line, human-readable rationale for a single enforced order."""
    risk = f"VaR={order.var * 100:.2f}%, CVaR={order.cvar * 100:.2f}%"
    if order.vetoed or order.action == "hold":
        return (
            f"HOLD — no position taken (binding constraint: {order.binding_constraint}; "
            f"{'risk breaker/veto engaged; ' if order.vetoed else ''}{risk})."
        )
    return (
        f"{order.action.upper()} sized to {order.size * 100:.2f}% of equity, "
        f"bound by '{order.binding_constraint}'; stop @ {order.stop_price:.2f} ({risk})."
    )
