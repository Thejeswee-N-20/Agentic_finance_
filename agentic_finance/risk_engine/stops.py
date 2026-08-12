"""Stop-loss levels and the portfolio max-drawdown circuit breaker.

ATR-based stops use the Average True Range (already surfaced by the market
analyst / ``stockstats``) so the stop adapts to current volatility instead of a
fixed percentage. The drawdown breaker halts new risk once equity falls a set
fraction below its running peak.
"""
from __future__ import annotations

__all__ = [
    "atr_stop_price",
    "stop_distance_fraction",
    "current_drawdown",
    "drawdown_breached",
]

_LONG = "long"
_SHORT = "short"


def atr_stop_price(
    entry: float,
    atr: float,
    multiplier: float,
    direction: str,
) -> float:
    """ATR stop price for a long or short.

    Long stop sits ``multiplier * atr`` below entry; short stop the same
    distance above. A long stop is floored at 0 (price cannot go negative).
    """
    if atr < 0:
        raise ValueError(f"atr must be >= 0, got {atr}")
    if multiplier < 0:
        raise ValueError(f"multiplier must be >= 0, got {multiplier}")
    offset = multiplier * atr
    if direction == _LONG:
        return max(0.0, entry - offset)
    if direction == _SHORT:
        return entry + offset
    raise ValueError(f"direction must be '{_LONG}' or '{_SHORT}', got {direction!r}")


def stop_distance_fraction(entry: float, stop: float) -> float:
    """Absolute distance from entry to stop as a fraction of entry price.

    Used to translate a stop into a per-trade risk budget (risk-per-unit).
    """
    if entry <= 0:
        raise ValueError(f"entry must be > 0, got {entry}")
    return abs(entry - stop) / entry


def current_drawdown(equity: float, peak_equity: float) -> float:
    """Fractional drawdown from the running peak, floored at 0.

    Returns 0 when at or above the peak.
    """
    if peak_equity <= 0:
        raise ValueError(f"peak_equity must be > 0, got {peak_equity}")
    if equity >= peak_equity:
        return 0.0
    return (peak_equity - equity) / peak_equity


def drawdown_breached(equity: float, peak_equity: float, max_drawdown: float) -> bool:
    """True once current drawdown reaches/exceeds ``max_drawdown`` (e.g. 0.15)."""
    if not 0.0 < max_drawdown < 1.0:
        raise ValueError(f"max_drawdown must be in (0, 1), got {max_drawdown}")
    return current_drawdown(equity, peak_equity) >= max_drawdown
