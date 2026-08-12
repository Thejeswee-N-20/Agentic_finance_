"""Compliance gate: the final, non-negotiable check before an order stands.

Models the institutional controls that sit *after* risk sizing: restricted
instruments (vetoed outright), a hard per-position limit (clamped), and a
mandatory decision rationale (an unexplained order is not allowed to trade —
the operational face of the framework's explainability requirement).

Pure and deterministic; the result carries the (possibly modified) order plus
the list of violations so the audit trail records exactly what the gate did.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

from agentic_finance.risk_engine.schemas import RiskedOrder

__all__ = ["CompliancePolicy", "ComplianceResult", "compliance_gate"]


@dataclass(frozen=True)
class CompliancePolicy:
    """Institutional trading policy applied to every order."""

    restricted: Tuple[str, ...] = ()
    max_position: float = 0.25
    require_rationale: bool = True


@dataclass(frozen=True)
class ComplianceResult:
    """Outcome of the gate: approval flag, final order, and any violations."""

    approved: bool
    order: RiskedOrder
    violations: Tuple[str, ...] = ()


def compliance_gate(order: RiskedOrder, ticker: str,
                    policy: CompliancePolicy) -> ComplianceResult:
    """Apply the compliance policy to a sized order (never raises).

    Holds pass through untouched — there is nothing to gate. Restricted
    tickers and missing rationales veto to hold; oversized positions are
    clamped to the policy limit but remain approved.
    """
    if order.action == "hold" or order.size <= 0.0:
        return ComplianceResult(approved=True, order=order)

    ticker_norm = ticker.upper()
    restricted = {t.upper() for t in policy.restricted}

    if ticker_norm in restricted:
        vetoed = replace(order, action="hold", size=0.0, stop_price=0.0,
                         vetoed=True, binding_constraint="compliance_restricted",
                         reason=f"vetoed: {ticker} is on the restricted list")
        return ComplianceResult(approved=False, order=vetoed,
                                violations=(f"{ticker} is restricted",))

    if policy.require_rationale and not order.reason.strip():
        vetoed = replace(order, action="hold", size=0.0, stop_price=0.0,
                         vetoed=True, binding_constraint="compliance_no_rationale",
                         reason="vetoed: order carries no decision rationale")
        return ComplianceResult(approved=False, order=vetoed,
                                violations=("missing decision rationale",))

    if order.size > policy.max_position:
        clamped = replace(order, size=policy.max_position,
                          binding_constraint="compliance_position_limit",
                          reason=f"clamped to policy position limit "
                                 f"{policy.max_position:.0%}")
        return ComplianceResult(
            approved=True, order=clamped,
            violations=(f"size {order.size:.2%} exceeded position limit "
                        f"{policy.max_position:.0%}",),
        )

    return ComplianceResult(approved=True, order=order)
