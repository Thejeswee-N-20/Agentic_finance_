"""Unit tests for risk-constraint attribution.

Explains *why* a position ended up the size it did by attributing each enforced
order to the risk constraint that bound it — the quantitative half of the XAI
story (and the data behind the "position-size attribution" figure). Pure Python,
built from the ``RiskedOrder`` sequence the risk-budgeted backtest already emits.
"""
import pytest

from agentic_finance.risk_engine.schemas import RiskedOrder
from agentic_finance.xai.risk_attribution import (
    attribute_constraints,
    binding_summary,
    explain_order,
)

pytestmark = pytest.mark.unit


def _order(action="buy", size=0.2, binding="kelly", vetoed=False, var=0.03, cvar=0.04):
    return RiskedOrder(
        action=action, size=size, stop_price=95.0, var=var, cvar=cvar,
        binding_constraint=binding, reason="test", vetoed=vetoed,
    )


def test_attribute_constraints_counts_each():
    orders = [_order(binding="kelly"), _order(binding="kelly"),
              _order(binding="vol_target"), _order(action="hold", size=0.0,
                                                    binding="max_drawdown", vetoed=True)]
    counts = attribute_constraints(orders)
    assert counts["kelly"] == 2
    assert counts["vol_target"] == 1
    assert counts["max_drawdown"] == 1


def test_attribute_constraints_empty():
    assert attribute_constraints([]) == {}


def test_binding_summary_fractions_sum_to_one():
    orders = [_order(binding="kelly"), _order(binding="kelly"),
              _order(binding="cvar_limit"), _order(binding="position_cap")]
    summary = binding_summary(orders)
    assert summary["kelly"] == pytest.approx(0.5)
    assert sum(summary.values()) == pytest.approx(1.0)


def test_binding_summary_empty():
    assert binding_summary([]) == {}


def test_explain_order_buy_is_human_readable():
    text = explain_order(_order(action="buy", size=0.18, binding="vol_target",
                                var=0.025, cvar=0.031))
    assert "BUY" in text.upper()
    assert "vol_target" in text
    assert "%" in text  # size rendered as a percentage of equity


def test_explain_order_veto_mentions_breaker():
    text = explain_order(_order(action="hold", size=0.0, binding="max_drawdown", vetoed=True))
    assert "veto" in text.lower() or "halt" in text.lower() or "breaker" in text.lower()


def test_explain_order_includes_risk_metrics():
    text = explain_order(_order(var=0.05, cvar=0.07))
    assert "VaR" in text and "CVaR" in text
