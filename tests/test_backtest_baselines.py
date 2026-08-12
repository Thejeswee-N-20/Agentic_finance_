"""Unit tests for baseline strategies.

A strategy is a callable ``f(prices_up_to_now) -> weight`` returning the target
position weight for the NEXT period, using only the supplied (past) prices.
Weights are long-only in [0, 1] for these baselines.
"""
import pytest

from agentic_finance.backtest.baselines import (
    buy_and_hold,
    sma_crossover,
    macd_strategy,
    rsi_strategy,
)

pytestmark = pytest.mark.unit


# --- buy & hold -----------------------------------------------------------

def test_buy_and_hold_always_full_weight():
    assert buy_and_hold([100.0]) == 1.0
    assert buy_and_hold([100.0, 90.0, 80.0]) == 1.0


# --- SMA crossover --------------------------------------------------------

def test_sma_crossover_long_when_short_above_long():
    strat = sma_crossover(short=2, long=4)
    prices = [1.0, 1.0, 1.0, 1.0, 5.0]  # recent jump -> short MA > long MA
    assert strat(prices) == 1.0


def test_sma_crossover_flat_when_short_below_long():
    strat = sma_crossover(short=2, long=4)
    prices = [5.0, 5.0, 5.0, 5.0, 1.0]  # recent drop -> short MA < long MA
    assert strat(prices) == 0.0


def test_sma_crossover_flat_when_insufficient_data():
    strat = sma_crossover(short=2, long=4)
    assert strat([1.0, 2.0]) == 0.0


# --- MACD -----------------------------------------------------------------

def test_macd_strategy_long_in_uptrend():
    strat = macd_strategy()
    # Accelerating uptrend keeps the MACD line rising above its (lagging) signal
    # line. (A purely linear ramp plateaus, so line == signal -> not long.)
    prices = [1.02 ** i for i in range(80)]
    assert strat(prices) == 1.0


def test_macd_strategy_flat_when_insufficient_data():
    strat = macd_strategy()
    assert strat([1.0, 2.0, 3.0]) == 0.0


# --- RSI ------------------------------------------------------------------

def test_rsi_strategy_flat_when_overbought():
    # strictly increasing -> RSI 100 -> overbought -> flat (mean reversion).
    strat = rsi_strategy(period=14, oversold=30, overbought=70)
    prices = [float(i) for i in range(1, 30)]
    assert strat(prices) == 0.0


def test_rsi_strategy_long_when_oversold():
    # strictly decreasing -> RSI 0 -> oversold -> long.
    strat = rsi_strategy(period=14, oversold=30, overbought=70)
    prices = [float(i) for i in range(40, 1, -1)]
    assert strat(prices) == 1.0


def test_rsi_strategy_flat_when_insufficient_data():
    strat = rsi_strategy()
    assert strat([1.0, 2.0, 3.0]) == 0.0
