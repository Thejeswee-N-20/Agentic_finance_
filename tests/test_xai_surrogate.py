"""Unit tests for the decision-policy surrogate.

A gradient-boosted / forest surrogate is fit to mimic the agent's action labels
from the tabular feature matrix. SHAP then explains the surrogate, so we also
report **fidelity** (how faithfully the surrogate reproduces the policy) — an
explanation is only trustworthy to the extent the surrogate agrees with the
real decisions.
"""
import pytest

from agentic_finance.xai.surrogate import train_surrogate, surrogate_fidelity, SurrogateModel

pytestmark = pytest.mark.unit


def _separable_dataset(n=200):
    """label depends on feature 0: > 0.5 -> 'buy', else 'sell'. 3 features."""
    names = ["f0", "f1", "f2"]
    matrix, labels = [], []
    for i in range(n):
        f0 = (i % 100) / 100.0
        row = [f0, (i * 7 % 50) / 50.0, (i * 3 % 25) / 25.0]
        matrix.append(row)
        labels.append("buy" if f0 > 0.5 else "sell")
    return names, matrix, labels


def test_train_returns_surrogate_model():
    names, matrix, labels = _separable_dataset()
    model = train_surrogate(matrix, labels, feature_names=names)
    assert isinstance(model, SurrogateModel)
    assert model.feature_names == names


def test_fidelity_high_on_separable_data():
    names, matrix, labels = _separable_dataset()
    model = train_surrogate(matrix, labels, feature_names=names)
    fidelity = surrogate_fidelity(model, matrix, labels)
    assert fidelity > 0.9  # surrogate faithfully reproduces a clean rule


def test_predict_returns_known_classes():
    names, matrix, labels = _separable_dataset()
    model = train_surrogate(matrix, labels, feature_names=names)
    preds = model.predict(matrix[:10])
    assert all(p in {"buy", "sell"} for p in preds)
    assert len(preds) == 10


def test_handles_single_class_labels():
    names = ["f0", "f1", "f2"]
    matrix = [[i / 10.0, 0.0, 1.0] for i in range(20)]
    labels = ["hold"] * 20
    model = train_surrogate(matrix, labels, feature_names=names)
    assert surrogate_fidelity(model, matrix, labels) == pytest.approx(1.0)
    assert set(model.predict(matrix)) == {"hold"}


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        train_surrogate([[1.0, 2.0]], ["buy", "sell"])


def test_rejects_empty():
    with pytest.raises(ValueError):
        train_surrogate([], [])


def test_is_deterministic():
    names, matrix, labels = _separable_dataset()
    m1 = train_surrogate(matrix, labels, feature_names=names)
    m2 = train_surrogate(matrix, labels, feature_names=names)
    assert m1.predict(matrix) == m2.predict(matrix)
