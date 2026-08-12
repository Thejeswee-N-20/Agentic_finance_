"""Point-in-time backtest engine.

Walks a close-price series one bar at a time. For each decision bar ``i`` the
strategy is shown only ``prices[:i+1]`` and returns a target weight for the next
period; that weight is multiplied by the realized ``i -> i+1`` return. Because
the strategy never sees ``prices[i+1]`` when deciding period ``i``, the loop is
structurally free of look-ahead — the property is asserted directly in the tests.

The result bundles the realized return series, equity curve, and the headline
risk-adjusted metrics (Sharpe / Sortino / Max Drawdown / Calmar).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

from agentic_finance.backtest.metrics import (
    calmar_ratio,
    cumulative_return,
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

Strategy = Callable[[Sequence[float]], float]

__all__ = ["run_backtest", "BacktestResult"]


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of a backtest run (immutable)."""

    returns: Tuple[float, ...]
    weights: Tuple[float, ...]
    equity: Tuple[float, ...]
    cumulative_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float


def run_backtest(
    prices: Sequence[float],
    strategy: Strategy,
    warmup: int = 0,
    initial_equity: float = 1.0,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Run ``strategy`` over ``prices`` point-in-time and summarize performance.

    ``warmup`` skips the first N decision bars (e.g. to let indicators form).
    """
    closes = [float(p) for p in prices]
    if len(closes) < 2:
        raise ValueError(f"need at least 2 prices, got {len(closes)}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    returns: List[float] = []
    weights: List[float] = []
    # Decision bar i in [warmup, len-2]; pay return prices[i+1]/prices[i].
    for i in range(warmup, len(closes) - 1):
        history = closes[: i + 1]            # info available at close i (no future)
        weight = float(strategy(history))
        period_return = closes[i + 1] / closes[i] - 1.0
        returns.append(weight * period_return)
        weights.append(weight)

    if not returns:
        raise ValueError("no periods to backtest after applying warmup")

    curve = equity_curve(returns, initial=initial_equity)
    # Metrics that need >= 2 points degrade gracefully on a single-period run.
    sharpe = sharpe_ratio(returns, risk_free_rate, periods_per_year) if len(returns) >= 2 else 0.0
    sortino = sortino_ratio(returns, risk_free_rate, periods_per_year) if len(returns) >= 2 else 0.0

    return BacktestResult(
        returns=tuple(returns),
        weights=tuple(weights),
        equity=tuple(curve),
        cumulative_return=cumulative_return(returns),
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown(returns),
        calmar=calmar_ratio(returns, periods_per_year),
    )
