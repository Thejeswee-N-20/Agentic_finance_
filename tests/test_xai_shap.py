"""Unit tests for the TreeSHAP explainer over the decision surrogate."""
import math

import pytest

from agentic_finance.xai.surrogate import train_surrogate
from agentic_finance.xai.shap_explainer import local_attributions, global_importance

pytestmark = pytest.mark.unit


def _f0_driven_dataset(n=200):
    """label depends ONLY on feature 0 -> SHAP should rank f0 highest."""
    names = ["f0", "f1", "f2"]
    matrix, labels = [], []
    for i in range(n):
        f0 = (i % 100) / 100.0
        # f1, f2 are noise uncorrelated with the label
        row = [f0, (i * 7 % 50) / 50.0, (i * 13 % 25) / 25.0]
        matrix.append(row)
        labels.append("buy" if f0 > 0.5 else "sell")
    return names, matrix, labels


def test_local_attributions_cover_all_features():
    names, matrix, labels = _f0_driven_dataset()
    model = train_surrogate(matrix, labels, feature_names=names)
    attr = local_attributions(model, matrix[0])
    assert set(attr.keys()) == set(names)
    assert all(math.isfinite(v) for v in attr.values())


def test_global_importance_non_negative_and_complete():
    names, matrix, labels = _f0_driven_dataset()
    model = train_surrogate(matrix, labels, feature_names=names)
    imp = global_importance(model, matrix)
    assert set(imp.keys()) == set(names)
    assert all(v >= 0 for v in imp.values())
    assert sum(imp.values()) > 0


def test_global_importance_ranks_driving_feature_first():
    names, matrix, labels = _f0_driven_dataset()
    model = train_surrogate(matrix, labels, feature_names=names)
    imp = global_importance(model, matrix)
    top = max(imp, key=imp.get)
    assert top == "f0"


def test_single_class_returns_zero_attributions_without_error():
    names = ["f0", "f1", "f2"]
    matrix = [[i / 10.0, 1.0, 2.0] for i in range(20)]
    labels = ["hold"] * 20
    model = train_surrogate(matrix, labels, feature_names=names)
    attr = local_attributions(model, matrix[0])
    assert set(attr.keys()) == set(names)
    assert all(v == 0.0 for v in attr.values())
