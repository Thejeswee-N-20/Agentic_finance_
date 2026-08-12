"""Tests for counterfactual explanations over the decision surrogate (offline)."""
from __future__ import annotations

import pytest

from agentic_finance.xai.counterfactual import (
    Counterfactual,
    counterfactual_report,
    find_counterfactuals,
)
from agentic_finance.xai.surrogate import train_surrogate

FEATURES = ["ret_20d", "vol_20d", "rsi_14"]


def _policy_surrogate():
    """Surrogate for a simple known policy: buy iff ret_20d > 0.02."""
    rows, labels = [], []
    for i in range(-20, 21):
        ret = i / 400.0                      # -0.05 .. 0.05
        for vol in (0.10, 0.30):
            rows.append([ret, vol, 50.0])
            labels.append("buy" if ret > 0.02 else "hold")
    return train_surrogate(rows, labels, feature_names=FEATURES)


RANGES = {"ret_20d": (-0.05, 0.05), "vol_20d": (0.05, 0.60), "rsi_14": (10.0, 90.0)}


class TestFindCounterfactuals:
    def test_finds_flip_on_true_driver(self):
        sur = _policy_surrogate()
        row = [0.04, 0.20, 55.0]              # predicted buy
        cfs = find_counterfactuals(sur, row, feature_ranges=RANGES)
        assert cfs, "expected at least one counterfactual"
        best = cfs[0]
        assert best.feature == "ret_20d"      # the true decision driver
        assert best.original_action == "buy"
        assert best.counterfactual_action == "hold"
        assert best.counterfactual_value < 0.03  # crossed the 0.02 boundary

    def test_counterfactual_is_minimal_ish(self):
        sur = _policy_surrogate()
        cfs = find_counterfactuals(sur, [0.04, 0.20, 55.0], feature_ranges=RANGES,
                                   steps=50)
        best = next(c for c in cfs if c.feature == "ret_20d")
        # the flip point is ~0.02, so the proposed change should not overshoot far
        assert abs(best.delta) < 0.04

    def test_target_action_filter(self):
        sur = _policy_surrogate()
        row = [-0.02, 0.20, 55.0]             # predicted hold
        cfs = find_counterfactuals(sur, row, target_action="buy",
                                   feature_ranges=RANGES)
        assert cfs and all(c.counterfactual_action == "buy" for c in cfs)

    def test_row_already_at_target_returns_empty(self):
        sur = _policy_surrogate()
        cfs = find_counterfactuals(sur, [0.04, 0.20, 55.0], target_action="buy",
                                   feature_ranges=RANGES)
        assert cfs == []

    def test_single_class_surrogate_returns_empty(self):
        rows = [[0.01, 0.2, 50.0], [0.02, 0.3, 60.0], [0.0, 0.1, 40.0]]
        sur = train_surrogate(rows, ["hold", "hold", "hold"], feature_names=FEATURES)
        assert find_counterfactuals(sur, [0.01, 0.2, 50.0], feature_ranges=RANGES) == []

    def test_results_sorted_by_relative_change(self):
        sur = _policy_surrogate()
        cfs = find_counterfactuals(sur, [0.04, 0.20, 55.0], feature_ranges=RANGES)
        rel = [abs(c.delta) / (RANGES[c.feature][1] - RANGES[c.feature][0]) for c in cfs]
        assert rel == sorted(rel)

    def test_ranges_derived_from_matrix_when_not_given(self):
        sur = _policy_surrogate()
        matrix = [[-0.05, 0.05, 10.0], [0.05, 0.60, 90.0]]
        cfs = find_counterfactuals(sur, [0.04, 0.20, 55.0], training_matrix=matrix)
        assert cfs and cfs[0].feature == "ret_20d"

    def test_no_ranges_and_no_matrix_raises(self):
        sur = _policy_surrogate()
        with pytest.raises(ValueError):
            find_counterfactuals(sur, [0.04, 0.20, 55.0])


class TestDecisionTrace:
    def test_trace_covers_all_stages(self):
        from agentic_finance.agents_v2.schemas import Signal
        from agentic_finance.risk_engine import PreDecision, RiskedOrder
        from agentic_finance.xai import decision_trace

        signals = [
            Signal(source="technical", direction="long", strength=0.7,
                   confidence=0.8, rationale="momentum"),
            Signal(source="fundamental", direction="long", strength=0.5,
                   confidence=0.6, rationale="healthy ratios"),
        ]
        pre = PreDecision(direction="long", confidence=0.75)
        order = RiskedOrder(action="buy", size=0.15, stop_price=98.5,
                            var=0.03, cvar=0.05, binding_constraint="kelly",
                            reason="kelly-bound long")
        text = decision_trace(signals, pre, order)
        assert "technical" in text and "fundamental" in text
        assert "LONG at 75%" in text
        assert "BUY 15.00%" in text
        assert "kelly" in text


class TestCounterfactualReport:
    def test_report_is_readable(self):
        cf = Counterfactual(feature="ret_20d", original_value=0.04,
                            counterfactual_value=0.015, delta=-0.025,
                            original_action="buy", counterfactual_action="hold")
        text = counterfactual_report([cf])
        assert "ret_20d" in text
        assert "buy" in text and "hold" in text

    def test_empty_report(self):
        assert "No counterfactual" in counterfactual_report([])
