"""Tests for the compliance gate (restricted lists, limits, rationale)."""
from __future__ import annotations

from agentic_finance.risk_engine.compliance import (
    CompliancePolicy,
    compliance_gate,
)
from agentic_finance.risk_engine.schemas import RiskedOrder


def _order(size: float = 0.20, action: str = "buy", reason: str = "kelly-bound long") -> RiskedOrder:
    return RiskedOrder(action=action, size=size, stop_price=95.0, var=0.03,
                       cvar=0.05, binding_constraint="kelly", reason=reason)


POLICY = CompliancePolicy(restricted=("BADCO", "XYZ.NS"), max_position=0.25,
                          require_rationale=True)


class TestComplianceGate:
    def test_clean_order_approved_unchanged(self):
        result = compliance_gate(_order(0.20), "NVDA", POLICY)
        assert result.approved
        assert result.violations == ()
        assert result.order.size == 0.20

    def test_restricted_ticker_is_vetoed(self):
        result = compliance_gate(_order(0.20), "BADCO", POLICY)
        assert not result.approved
        assert result.order.action == "hold"
        assert result.order.size == 0.0
        assert result.order.vetoed
        assert result.order.binding_constraint == "compliance_restricted"
        assert any("restricted" in v for v in result.violations)

    def test_restricted_match_is_case_insensitive(self):
        result = compliance_gate(_order(0.20), "xyz.ns", POLICY)
        assert not result.approved

    def test_oversized_order_clamped(self):
        result = compliance_gate(_order(0.40), "NVDA", POLICY)
        assert result.approved  # clamped, not vetoed
        assert result.order.size == 0.25
        assert result.order.binding_constraint == "compliance_position_limit"
        assert any("position" in v for v in result.violations)

    def test_missing_rationale_vetoed_when_required(self):
        result = compliance_gate(_order(0.20, reason=""), "NVDA", POLICY)
        assert not result.approved
        assert result.order.action == "hold"
        assert result.order.binding_constraint == "compliance_no_rationale"

    def test_missing_rationale_ok_when_not_required(self):
        lax = CompliancePolicy(restricted=(), max_position=0.25,
                               require_rationale=False)
        result = compliance_gate(_order(0.20, reason=""), "NVDA", lax)
        assert result.approved

    def test_hold_orders_always_approved(self):
        result = compliance_gate(_order(0.0, action="hold", reason=""),
                                 "BADCO", POLICY)
        assert result.approved  # nothing to gate on a hold
