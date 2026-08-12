"""Unit tests for position sizing.

Sizes are expressed as a **fraction of portfolio equity** in [0, 1] for a long
(or the absolute exposure for a short). All sizers are pure functions and never
exceed the supplied caps.
"""
import pytest

from agentic_finance.risk_engine.position_sizing import (
    volatility_target_size,
    fractional_kelly_size,
    apply_caps,
)

pytestmark = pytest.mark.unit


# --- volatility targeting -------------------------------------------------

def test_vol_target_scales_inversely_with_realized_vol():
    # target 10% annual vol; double the asset vol -> half the size.
    low = volatility_target_size(target_vol=0.10, asset_vol=0.20)
    high = volatility_target_size(target_vol=0.10, asset_vol=0.40)
    assert high == pytest.approx(low / 2)


def test_vol_target_exact_value():
    # size = target / asset_vol = 0.10 / 0.20 = 0.5
    assert volatility_target_size(0.10, 0.20) == pytest.approx(0.5)


def test_vol_target_caps_at_max_leverage():
    # Very low asset vol would imply >1 size; clamp to max_leverage.
    assert volatility_target_size(0.10, 0.01, max_leverage=1.0) == 1.0


def test_vol_target_zero_vol_returns_max_leverage():
    assert volatility_target_size(0.10, 0.0, max_leverage=1.0) == 1.0


def test_vol_target_rejects_negative_target():
    with pytest.raises(ValueError):
        volatility_target_size(-0.1, 0.2)


# --- fractional Kelly -----------------------------------------------------

def test_kelly_positive_edge_positive_size():
    # win prob 0.6, win/loss payoff 1:1 -> full Kelly f* = 2p-1 = 0.2
    f = fractional_kelly_size(win_prob=0.6, win_loss_ratio=1.0, fraction=1.0)
    assert f == pytest.approx(0.2)


def test_kelly_half_fraction_halves_size():
    full = fractional_kelly_size(0.6, 1.0, fraction=1.0)
    half = fractional_kelly_size(0.6, 1.0, fraction=0.5)
    assert half == pytest.approx(full / 2)


def test_kelly_no_edge_is_zero_or_negative_clamped():
    # win prob 0.5, 1:1 -> edge 0 -> size 0 (never negative).
    assert fractional_kelly_size(0.5, 1.0) == pytest.approx(0.0)


def test_kelly_negative_edge_clamped_to_zero():
    assert fractional_kelly_size(0.4, 1.0) == 0.0


def test_kelly_rejects_bad_probability():
    with pytest.raises(ValueError):
        fractional_kelly_size(1.4, 1.0)


# --- caps -----------------------------------------------------------------

def test_apply_caps_respects_per_position_cap():
    assert apply_caps(0.8, per_position_cap=0.25) == 0.25


def test_apply_caps_respects_remaining_portfolio_budget():
    # 0.30 desired but only 0.10 portfolio budget left -> 0.10
    assert apply_caps(0.30, per_position_cap=0.50, remaining_budget=0.10) == pytest.approx(0.10)


def test_apply_caps_never_negative():
    assert apply_caps(-0.2, per_position_cap=0.5) == 0.0


def test_apply_caps_passthrough_when_within_limits():
    assert apply_caps(0.15, per_position_cap=0.25, remaining_budget=0.50) == pytest.approx(0.15)
