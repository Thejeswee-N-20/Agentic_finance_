"""Baseline trading strategies for head-to-head comparison.

A *strategy* is a callable ``f(prices) -> weight`` where ``prices`` is the close
history available at decision time and ``weight`` is the target position for the
next period. These baselines are long-only (weight in {0.0, 1.0}); the engine
multiplies the weight by the next-period return. Keeping them stateless and
point-in-time makes them trivially leakage-safe.

These mirror the simple technical baselines the TradingAgents paper compares
against (Buy & Hold, SMA, MACD, RSI), so the dissertation's risk-adjusted win is
measured against the same yardsticks.
"""
from __future__ import annotations

from typing import Callable, Sequence

from agentic_finance.backtest.indicators import macd, rsi, sma

Strategy = Callable[[Sequence[float]], float]

__all__ = ["buy_and_hold", "sma_crossover", "macd_strategy", "rsi_strategy", "Strategy"]


def buy_and_hold(prices: Sequence[float]) -> float:
    """Always fully invested."""
    return 1.0


def sma_crossover(short: int = 20, long: int = 50) -> Strategy:
    """Long when the short SMA is above the long SMA, else flat."""
    if short >= long:
        raise ValueError(f"short ({short}) must be < long ({long})")

    def strategy(prices: Sequence[float]) -> float:
        short_ma = sma(prices, short)
        long_ma = sma(prices, long)
        if short_ma is None or long_ma is None:
            return 0.0
        return 1.0 if short_ma > long_ma else 0.0

    return strategy


def macd_strategy(fast: int = 12, slow: int = 26, signal_period: int = 9) -> Strategy:
    """Long when the MACD line is above its signal line, else flat."""

    def strategy(prices: Sequence[float]) -> float:
        line, signal = macd(prices, fast, slow, signal_period)
        if line is None or signal is None:
            return 0.0
        return 1.0 if line > signal else 0.0

    return strategy


def rsi_strategy(period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> Strategy:
    """Mean-reversion RSI: long when oversold, flat when overbought.

    Between the thresholds the position is held flat (conservative default) so
    the rule only takes a position on a clear oversold signal.
    """
    if not 0 < oversold < overbought < 100:
        raise ValueError("require 0 < oversold < overbought < 100")

    def strategy(prices: Sequence[float]) -> float:
        value = rsi(prices, period)
        if value is None:
            return 0.0
        if value <= oversold:
            return 1.0
        return 0.0

    return strategy
