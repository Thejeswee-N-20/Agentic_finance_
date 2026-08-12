"""Unit tests for stop-loss levels and the max-drawdown circuit breaker."""
import pytest

from agentic_finance.risk_engine.stops import (
    atr_stop_price,
    stop_distance_fraction,
    current_drawdown,
    drawdown_breached,
)

pytestmark = pytest.mark.unit


# --- ATR stop price -------------------------------------------------------

def test_atr_stop_long_is_below_entry():
    # long: stop = entry - mult * ATR = 100 - 2*3 = 94
    assert atr_stop_price(entry=100.0, atr=3.0, multiplier=2.0, direction="long") == pytest.approx(94.0)


def test_atr_stop_short_is_above_entry():
    # short: stop = entry + mult * ATR = 100 + 2*3 = 106
    assert atr_stop_price(entry=100.0, atr=3.0, multiplier=2.0, direction="short") == pytest.approx(106.0)


def test_atr_stop_long_never_below_zero():
    assert atr_stop_price(entry=5.0, atr=10.0, multiplier=2.0, direction="long") == 0.0


def test_atr_stop_rejects_unknown_direction():
    with pytest.raises(ValueError):
        atr_stop_price(100.0, 3.0, 2.0, direction="sideways")


def test_atr_stop_rejects_negative_atr():
    with pytest.raises(ValueError):
        atr_stop_price(100.0, -1.0, 2.0, direction="long")


# --- stop distance fraction ----------------------------------------------

def test_stop_distance_fraction_long():
    # (100 - 94) / 100 = 0.06
    assert stop_distance_fraction(entry=100.0, stop=94.0) == pytest.approx(0.06)


def test_stop_distance_fraction_is_absolute():
    # short stop above entry still yields a positive distance.
    assert stop_distance_fraction(entry=100.0, stop=106.0) == pytest.approx(0.06)


# --- drawdown -------------------------------------------------------------

def test_current_drawdown_basic():
    # peak 120, now 90 -> (120-90)/120 = 0.25
    assert current_drawdown(equity=90.0, peak_equity=120.0) == pytest.approx(0.25)


def test_current_drawdown_at_peak_is_zero():
    assert current_drawdown(equity=120.0, peak_equity=120.0) == 0.0


def test_current_drawdown_above_peak_is_zero():
    assert current_drawdown(equity=130.0, peak_equity=120.0) == 0.0


def test_drawdown_breached_true_when_over_limit():
    assert drawdown_breached(equity=80.0, peak_equity=100.0, max_drawdown=0.15) is True


def test_drawdown_breached_false_within_limit():
    assert drawdown_breached(equity=90.0, peak_equity=100.0, max_drawdown=0.15) is False


def test_drawdown_breached_at_exact_limit_is_true():
    assert drawdown_breached(equity=85.0, peak_equity=100.0, max_drawdown=0.15) is True
