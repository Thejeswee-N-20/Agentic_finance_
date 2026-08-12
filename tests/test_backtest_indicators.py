"""Unit tests for technical indicators used by baseline strategies.

Pure functions over a close-price sequence. Each returns a value computed from
the trailing window so the backtest engine can call them point-in-time without
look-ahead.
"""
import pytest

from agentic_finance.backtest.indicators import sma, ema, macd, rsi

pytestmark = pytest.mark.unit


# --- SMA ------------------------------------------------------------------

def test_sma_simple_average_of_last_n():
    assert sma([1, 2, 3, 4, 5], period=3) == pytest.approx(4.0)  # (3+4+5)/3


def test_sma_uses_only_trailing_window():
    assert sma([10, 20, 30], period=2) == pytest.approx(25.0)  # (20+30)/2


def test_sma_none_when_insufficient_data():
    assert sma([1, 2], period=3) is None


# --- EMA ------------------------------------------------------------------

def test_ema_equals_sma_seed_for_flat_series():
    # constant series -> EMA equals the constant.
    assert ema([5, 5, 5, 5, 5], period=3) == pytest.approx(5.0)


def test_ema_reacts_more_than_sma_to_recent_jump():
    series = [10, 10, 10, 10, 20]
    assert ema(series, period=3) > sma(series, period=3)


def test_ema_none_when_insufficient_data():
    assert ema([1, 2], period=5) is None


# --- MACD -----------------------------------------------------------------

def test_macd_returns_line_and_signal():
    series = [float(i) for i in range(1, 60)]  # steady uptrend
    line, signal = macd(series, fast=12, slow=26, signal_period=9)
    assert line is not None and signal is not None
    # In a steady uptrend the MACD line is positive.
    assert line > 0


def test_macd_none_when_insufficient_data():
    line, signal = macd([1.0, 2.0, 3.0], fast=12, slow=26, signal_period=9)
    assert line is None and signal is None


# --- RSI ------------------------------------------------------------------

def test_rsi_is_100_for_only_gains():
    series = [float(i) for i in range(1, 20)]  # strictly increasing
    assert rsi(series, period=14) == pytest.approx(100.0)


def test_rsi_is_low_for_only_losses():
    series = [float(i) for i in range(20, 1, -1)]  # strictly decreasing
    assert rsi(series, period=14) == pytest.approx(0.0)


def test_rsi_mid_range_for_mixed():
    series = [10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10]
    val = rsi(series, period=14)
    assert 30 < val < 70


def test_rsi_none_when_insufficient_data():
    assert rsi([1, 2, 3], period=14) is None
