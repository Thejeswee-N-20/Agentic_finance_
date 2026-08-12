"""Human-readable decision audit report.

Combines the enforced order's risk rationale (which constraint bound the size)
with the SHAP feature attributions (which signals drove the call) into a single
auditable paragraph — the regulator-facing "right to explanation" artifact. Pure
Python; no model or heavy deps.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from agentic_finance.agents_v2.schemas import Signal
from agentic_finance.risk_engine.schemas import PreDecision, RiskedOrder
from agentic_finance.xai.risk_attribution import explain_order

__all__ = ["decision_report", "decision_trace"]


def decision_report(
    order: RiskedOrder,
    local_shap: Mapping[str, float],
    fidelity: Optional[float] = None,
    top_k: int = 5,
) -> str:
    """Render a decision into an auditable explanation.

    ``local_shap`` maps feature name -> SHAP value for this decision; the top
    ``top_k`` features by magnitude are listed with their direction of
    contribution (``+`` supports the predicted action, ``-`` opposes it).
    """
    lines = [f"DECISION: {order.action.upper()}", explain_order(order)]

    if local_shap:
        ranked = sorted(local_shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        lines.append("Top feature drivers (SHAP):")
        for name, value in ranked:
            sign = "+" if value >= 0 else "-"
            lines.append(f"  [{sign}] {name}: {value:+.4f}")

    if fidelity is not None:
        lines.append(f"(Surrogate fidelity: {fidelity * 100:.1f}% — explanation trust level.)")

    return "\n".join(lines)


def decision_trace(signals: Sequence[Signal], pre: PreDecision,
                   order: RiskedOrder) -> str:
    """Render the full provenance of one decision: agents -> fusion -> risk gate.

    The complement of ``decision_report``: instead of *which features* drove
    the call, it shows *which stage* produced each part of the outcome — every
    agent's typed view, the fused conviction, and the risk engine's enforcement
    (action, size, stop, binding constraint). Pure Python, no deps.
    """
    lines = ["DECISION TRACE", "1. Agent signals:"]
    for s in signals:
        lines.append(
            f"   [{s.source:<11}] {s.direction:<5} strength={s.strength:.2f} "
            f"confidence={s.confidence:.2f}"
            + (f"  — {s.rationale}" if s.rationale else "")
        )
    lines.append(
        f"2. Fusion       : {pre.direction.upper()} at {pre.confidence * 100:.0f}% "
        f"confidence (confidence-weighted reconciliation of "
        f"{len(signals)} signal{'s' if len(signals) != 1 else ''})"
    )
    stop = f", stop {order.stop_price:.2f}" if order.stop_price else ""
    lines.append(
        f"3. Risk engine  : {order.action.upper()} {order.size * 100:.2f}% of equity"
        f"{stop} — bound by '{order.binding_constraint}' "
        f"(VaR {order.var * 100:.2f}%, CVaR {order.cvar * 100:.2f}%)"
    )
    return "\n".join(lines)
