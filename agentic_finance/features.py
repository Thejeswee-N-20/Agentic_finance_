"""Point-in-time feature bridge.

Turns a close-price history (as of a decision bar) into a fixed, named, tabular
feature vector. This is the shared substrate for two consumers:

- the **risk engine** (volatility, VaR/CVaR, momentum context), and
- the **XAI SHAP surrogate**, which needs a stable feature matrix + names.

Everything here is computed strictly from ``prices[:i+1]`` (closes up to the
decision bar), reusing :mod:`agentic_finance.backtest.indicators` and
:mod:`agentic_finance.risk_engine.var_cvar` rather than reimplementing them. The
optional ``extra`` mapping lets the agent layer inject sentiment / fundamental
scores later without changing the core schema.
"""
from __future__ import annotations

import dataclasses
import statistics
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence, Tuple

from agentic_finance.backtest.indicators import macd, rsi, sma
from agentic_finance.risk_engine.var_cvar import historical_cvar, historical_var

__all__ = ["FeatureVector", "extract_features", "build_feature_matrix", "MIN_HISTORY"]

# Need enough history for the longest core window (SMA-50) plus a prior bar.
MIN_HISTORY = 51

_CONFIDENCE = 0.95
_RISK_WINDOW = 30


@dataclass(frozen=True)
class FeatureVector:
    """A fixed, named tabular feature vector for one decision bar (immutable).

    Core fields are explicit and typed; ``extra`` carries agent-injected
    features (e.g. sentiment) appended deterministically (sorted by key).
    """

    ret_1d: float
    ret_5d: float
    ret_20d: float
    vol_20d: float
    rsi_14: float
    macd_line: float
    macd_signal: float
    macd_hist: float
    price_sma20_ratio: float
    sma20_sma50_ratio: float
    atr_pct: float
    bb_position: float
    dist_from_high_20: float
    dist_from_low_20: float
    var_95: float
    cvar_95: float
    extra: Tuple[Tuple[str, float], ...] = ()

    @staticmethod
    def _core_names() -> List[str]:
        return [f.name for f in dataclasses.fields(FeatureVector) if f.name != "extra"]

    def names(self) -> List[str]:
        return self._core_names() + [k for k, _ in self.extra]

    def values(self) -> List[float]:
        core = [getattr(self, n) for n in self._core_names()]
        return core + [v for _, v in self.extra]

    def to_ordered_dict(self) -> "OrderedDict[str, float]":
        return OrderedDict(zip(self.names(), self.values()))


def _cumulative(prices: Sequence[float], lookback: int) -> float:
    """Return over the last ``lookback`` bars: prices[-1]/prices[-1-lookback] - 1."""
    if len(prices) <= lookback:
        return 0.0
    return prices[-1] / prices[-1 - lookback] - 1.0


def _realized_vol(returns: Sequence[float]) -> float:
    if len(returns) < 2:
        return 0.0
    return statistics.pstdev(returns)


def _atr_proxy_pct(prices: Sequence[float], period: int = 14) -> float:
    if len(prices) < 2:
        return 0.0
    changes = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    window = changes[-period:]
    atr = sum(window) / len(window)
    return atr / prices[-1] if prices[-1] else 0.0


def extract_features(
    prices: Sequence[float],
    extra: Optional[Mapping[str, float]] = None,
) -> Optional[FeatureVector]:
    """Compute the feature vector as of the last close, or ``None`` if too short."""
    closes = [float(p) for p in prices]
    if len(closes) < MIN_HISTORY:
        return None

    price = closes[-1]
    daily_returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]

    sma20 = sma(closes, 20)
    sma50 = sma(closes, 50)
    macd_line, macd_signal = macd(closes)
    macd_line = macd_line if macd_line is not None else 0.0
    macd_signal = macd_signal if macd_signal is not None else 0.0

    last20 = closes[-20:]
    std20 = statistics.pstdev(last20) if len(last20) >= 2 else 0.0
    bb_position = (price - sma20) / (2.0 * std20) if (sma20 is not None and std20 > 0) else 0.0

    risk_window = tuple(daily_returns[-_RISK_WINDOW:])

    fv = FeatureVector(
        ret_1d=daily_returns[-1],
        ret_5d=_cumulative(closes, 5),
        ret_20d=_cumulative(closes, 20),
        vol_20d=_realized_vol(daily_returns[-20:]),
        rsi_14=rsi(closes, 14) or 0.0,
        macd_line=macd_line,
        macd_signal=macd_signal,
        macd_hist=macd_line - macd_signal,
        price_sma20_ratio=(price / sma20 - 1.0) if sma20 else 0.0,
        sma20_sma50_ratio=(sma20 / sma50 - 1.0) if (sma20 and sma50) else 0.0,
        atr_pct=_atr_proxy_pct(closes),
        bb_position=bb_position,
        dist_from_high_20=price / max(last20) - 1.0,
        dist_from_low_20=price / min(last20) - 1.0,
        var_95=historical_var(risk_window, _CONFIDENCE),
        cvar_95=historical_cvar(risk_window, _CONFIDENCE),
        extra=tuple(sorted((str(k), float(v)) for k, v in (extra or {}).items())),
    )
    return fv


def build_feature_matrix(
    prices: Sequence[float],
    warmup: int = MIN_HISTORY,
) -> Tuple[List[str], List[List[float]], List[int]]:
    """Walk the series point-in-time and build the surrogate training matrix.

    Mirrors the backtest engine's decision bars: for each bar ``i`` in
    ``[warmup, len-2]`` it computes features from ``prices[:i+1]`` only. Bars
    with insufficient history are skipped. Returns ``(names, matrix, indices)``
    where ``indices[k]`` is the decision bar that produced ``matrix[k]``.
    """
    closes = [float(p) for p in prices]
    start = max(warmup, MIN_HISTORY)
    names: List[str] = []
    matrix: List[List[float]] = []
    indices: List[int] = []
    for i in range(start, len(closes) - 1):
        fv = extract_features(closes[: i + 1])
        if fv is None:
            continue
        if not names:
            names = fv.names()
        matrix.append(fv.values())
        indices.append(i)
    return names, matrix, indices
