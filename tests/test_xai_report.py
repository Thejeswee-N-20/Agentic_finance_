"""Unit tests for the human-readable decision audit report."""
import pytest

from agentic_finance.risk_engine.schemas import RiskedOrder
from agentic_finance.xai.report import decision_report

pytestmark = pytest.mark.unit


def _order(action="buy", size=0.2, binding="vol_target"):
    return RiskedOrder(
        action=action, size=size, stop_price=95.0, var=0.03, cvar=0.04,
        binding_constraint=binding, reason="r", vetoed=(action == "hold"),
    )


def test_report_includes_action_and_constraint():
    shap = {"ret_20d": 0.4, "rsi_14": -0.1, "vol_20d": 0.05}
    text = decision_report(_order(action="buy", binding="kelly"), shap)
    assert "BUY" in text.upper()
    assert "kelly" in text


def test_report_lists_top_features_by_magnitude():
    # distinctive names so substring checks don't collide with report prose.
    shap = {"feat_a": 0.5, "feat_b": -0.4, "feat_c": 0.01, "feat_d": 0.001}
    text = decision_report(_order(), shap, top_k=2)
    assert "feat_a" in text and "feat_b" in text
    assert "feat_d" not in text  # beyond top_k
    assert "feat_c" not in text


def test_report_marks_direction_of_contribution():
    shap = {"ret_20d": 0.5, "rsi_14": -0.3}
    text = decision_report(_order(), shap)
    # supporting (+) and opposing (-) contributions are visually distinguished
    assert "+" in text and "-" in text


def test_report_includes_fidelity_when_given():
    text = decision_report(_order(), {"x": 0.2}, fidelity=0.93)
    assert "93" in text or "0.93" in text


def test_report_handles_empty_shap():
    text = decision_report(_order(action="hold", binding="max_drawdown"), {})
    assert "HOLD" in text.upper()
