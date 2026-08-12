"""Technical indicators for baseline strategies.

Pure functions over a sequence of close prices. Each returns the indicator value
as of the **last** element of the supplied window, so the backtest engine can
call them with ``prices[:i+1]`` and never see future bars. They return ``None``
when there is not enough data, which the strategies treat as "no position".

These are deliberately small, dependency-free reimplementations rather than
reuse of ``dataflows/stockstats_utils.py``: that module is coupled to the data
fetch/cache/LLM-tool layer (returns formatted strings, does date-range I/O),
whereas baselines need cheap, deterministic, unit-testable numeric functions.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

__all__ = ["sma", "ema", "macd", "rsi"]


def sma(prices: Sequence[float], period: int) -> Optional[float]:
    """Simple moving average of the last ``period`` closes."""
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    if len(prices) < period:
        return None
    window = prices[-period:]
    return sum(window) / period


def _ema_series(prices: Sequence[float], period: int) -> Optional[List[float]]:
    """Full EMA series, seeded with the SMA of the first ``period`` values."""
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    if len(prices) < period:
        return None
    k = 2.0 / (period + 1.0)
    seed = sum(prices[:period]) / period
    out = [seed]
    for price in prices[period:]:
        out.append(price * k + out[-1] * (1.0 - k))
    return out


def ema(prices: Sequence[float], period: int) -> Optional[float]:
    """Exponential moving average value as of the last close."""
    series = _ema_series(prices, period)
    return None if series is None else series[-1]


def macd(
    prices: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Tuple[Optional[float], Optional[float]]:
    """MACD line and signal line as of the last close.

    MACD line = EMA(fast) - EMA(slow); signal = EMA(signal_period) of the MACD
    line. Returns ``(None, None)`` when there is insufficient data.
    """
    fast_series = _ema_series(prices, fast)
    slow_series = _ema_series(prices, slow)
    if fast_series is None or slow_series is None:
        return None, None
    # Align the two EMA series on their shared (most recent) tail.
    n = min(len(fast_series), len(slow_series))
    macd_line_series = [f - s for f, s in zip(fast_series[-n:], slow_series[-n:])]
    if len(macd_line_series) < signal_period:
        return macd_line_series[-1], None
    signal_series = _ema_series(macd_line_series, signal_period)
    signal_val = None if signal_series is None else signal_series[-1]
    return macd_line_series[-1], signal_val


def rsi(prices: Sequence[float], period: int = 14) -> Optional[float]:
    """Wilder-style RSI over the last ``period`` changes (0..100).

    100 when every change is a gain, 0 when every change is a loss.
    """
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    if len(prices) < period + 1:
        return None
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    window = changes[-period:]
    gains = [c for c in window if c > 0]
    losses = [-c for c in window if c < 0]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    if avg_gain == 0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
