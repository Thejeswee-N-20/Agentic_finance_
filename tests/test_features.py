"""Unit tests for the point-in-time feature bridge.

``extract_features`` turns a close-price history (as of a decision bar) into a
fixed, named, tabular feature vector used by both the risk engine and the SHAP
surrogate. ``build_feature_matrix`` walks the series point-in-time to produce
the surrogate's training matrix — and must never let future bars change a past
row (leakage).
"""
import pytest

from agentic_finance.features import (
    FeatureVector,
    extract_features,
    build_feature_matrix,
    MIN_HISTORY,
)

pytestmark = pytest.mark.unit


def _uptrend(n=120, start=100.0, step=0.005):
    return [start * (1 + step) ** i for i in range(n)]


def _mixed(n=120):
    # oscillating-but-drifting series
    return [100.0 + 10.0 * ((-1) ** i) * (i % 5) / 5.0 + 0.05 * i for i in range(n)]


# --- extract_features -----------------------------------------------------

def test_returns_none_when_insufficient_history():
    assert extract_features([100.0, 101.0, 102.0]) is None
    assert extract_features(_uptrend(MIN_HISTORY - 1)) is None


def test_returns_vector_when_enough_history():
    fv = extract_features(_uptrend(MIN_HISTORY))
    assert isinstance(fv, FeatureVector)


def test_names_and_values_aligned():
    fv = extract_features(_uptrend())
    assert len(fv.names()) == len(fv.values())
    assert list(fv.to_ordered_dict().keys()) == fv.names()
    assert list(fv.to_ordered_dict().values()) == pytest.approx(fv.values())


def test_ret_1d_matches_last_return():
    prices = _uptrend()
    fv = extract_features(prices)
    expected = prices[-1] / prices[-2] - 1.0
    assert fv.ret_1d == pytest.approx(expected)


def test_uptrend_has_high_rsi_and_positive_momentum():
    fv = extract_features(_uptrend())
    assert fv.rsi_14 == pytest.approx(100.0)
    assert fv.ret_20d > 0
    assert fv.price_sma20_ratio > 0  # price above its SMA in an uptrend


def test_risk_features_present_and_ordered():
    fv = extract_features(_mixed())
    # CVaR (mean tail loss) >= VaR (tail threshold) on a series with losses.
    assert fv.cvar_95 >= fv.var_95


def test_extra_features_appended_in_order():
    fv = extract_features(_uptrend(), extra={"sentiment": 0.8, "fund_score": -0.2})
    names = fv.names()
    # extras come after the core features, sorted by key for determinism
    assert names[-2:] == ["fund_score", "sentiment"]
    d = fv.to_ordered_dict()
    assert d["sentiment"] == pytest.approx(0.8)
    assert d["fund_score"] == pytest.approx(-0.2)


def test_feature_vector_is_frozen():
    import dataclasses
    fv = extract_features(_uptrend())
    with pytest.raises(dataclasses.FrozenInstanceError):
        fv.rsi_14 = 0.0  # type: ignore[misc]


def test_deterministic():
    prices = _mixed()
    assert extract_features(prices).values() == extract_features(prices).values()


# --- build_feature_matrix -------------------------------------------------

def test_matrix_shape_and_alignment():
    prices = _uptrend(120)
    names, matrix, indices = build_feature_matrix(prices, warmup=MIN_HISTORY)
    assert len(matrix) == len(indices)
    assert all(len(row) == len(names) for row in matrix)
    assert len(matrix) > 0


def test_matrix_names_match_feature_vector():
    prices = _uptrend(120)
    names, _, _ = build_feature_matrix(prices, warmup=MIN_HISTORY)
    assert names == extract_features(prices[:MIN_HISTORY]).names()


def test_matrix_is_leakage_safe_past_rows_unchanged_by_future():
    # Rows computed on a truncated series must equal the matching rows computed
    # on the full series — adding future bars cannot change a past feature row.
    prices = _mixed(140)
    names_a, mat_a, idx_a = build_feature_matrix(prices[:90], warmup=MIN_HISTORY)
    names_b, mat_b, idx_b = build_feature_matrix(prices, warmup=MIN_HISTORY)
    assert names_a == names_b
    overlap = [i for i in idx_a if i in idx_b]
    assert overlap  # there should be shared decision bars
    for i in overlap:
        row_a = mat_a[idx_a.index(i)]
        row_b = mat_b[idx_b.index(i)]
        assert row_a == pytest.approx(row_b)
