"""Multi-regime, volatility-matched evaluation harness.

Motivation (see AGENTIC_FINANCE_IMPLEMENTATION.md > Findings & Analysis): a
risk-managed strategy deliberately runs at lower exposure than an unconstrained
benchmark, so a face-value raw-return comparison is unfair — it compares returns
at *different risk levels*. This harness:

- runs any strategy as a uniform "return-series producer" over a price window,
- reports risk-adjusted metrics (Sharpe / Sortino / Max DD / Calmar), and
- reports a **volatility-matched return**: the return the strategy would have
  produced scaled to a common target volatility, i.e. an equal-risk comparison.

``volatility_scale`` is an *ex-post* normalization for reporting/comparison (it
uses the full-sample realized vol), not a tradeable signal. The proper ex-ante
counterpart is the risk engine's point-in-time ``target_vol`` sizing; this is the
analyst's lens for "who wins per unit of risk".
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence

from agentic_finance.backtest.engine import run_backtest
from agentic_finance.backtest.metrics import (
    calmar_ratio,
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from agentic_finance.backtest.risk_budgeted import (
    DirectionSignal,
    run_risk_budgeted_backtest,
)
from agentic_finance.backtest.baselines import Strategy
from agentic_finance.risk_engine.schemas import RiskBudget

ReturnSeriesProducer = Callable[[Sequence[float]], Sequence[float]]

__all__ = [
    "realized_vol",
    "volatility_scale",
    "StrategyResult",
    "evaluate_window",
    "aggregate",
    "baseline_producer",
    "risk_budgeted_producer",
]

_EPS = 1e-12


def realized_vol(returns: Sequence[float], periods_per_year: int = 252) -> float:
    """Annualized realized volatility (sample std) of a return series."""
    data = [float(r) for r in returns]
    if len(data) < 2:
        return 0.0
    mu = sum(data) / len(data)
    var = sum((r - mu) ** 2 for r in data) / (len(data) - 1)
    return math.sqrt(var) * math.sqrt(periods_per_year)


def volatility_scale(
    returns: Sequence[float],
    target_vol: float,
    periods_per_year: int = 252,
) -> List[float]:
    """Scale a return series to ``target_vol`` annualized (ex-post, for comparison).

    Returns the original series unchanged-to-zero when realized vol is ~0.
    """
    data = [float(r) for r in returns]
    rv = realized_vol(data, periods_per_year)
    if rv <= _EPS:
        return [0.0 for _ in data]
    factor = target_vol / rv
    return [r * factor for r in data]


@dataclass(frozen=True)
class StrategyResult:
    """Per-strategy, per-window evaluation outcome (immutable)."""

    name: str
    regime: str
    cum_return: float
    vol_matched_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float


def baseline_producer(strategy: Strategy, warmup: int = 50) -> ReturnSeriesProducer:
    """Wrap a weight-based baseline strategy as a return-series producer."""
    def produce(prices: Sequence[float]) -> Sequence[float]:
        return run_backtest(prices, strategy, warmup=warmup).returns
    return produce


def risk_budgeted_producer(
    direction_signal: DirectionSignal,
    budget: RiskBudget,
    warmup: int = 50,
) -> ReturnSeriesProducer:
    """Wrap the risk-engine-driven backtest as a return-series producer."""
    def produce(prices: Sequence[float]) -> Sequence[float]:
        return run_risk_budgeted_backtest(prices, direction_signal, budget, warmup=warmup).performance.returns
    return produce


def evaluate_window(
    prices: Sequence[float],
    producers: Mapping[str, ReturnSeriesProducer],
    regime: str = "",
    target_vol: float = 0.15,
    periods_per_year: int = 252,
) -> List[StrategyResult]:
    """Run every producer over ``prices`` and summarize at equal risk."""
    results: List[StrategyResult] = []
    for name, produce in producers.items():
        returns = list(produce(prices))
        matched = volatility_scale(returns, target_vol, periods_per_year)
        results.append(
            StrategyResult(
                name=name,
                regime=regime,
                cum_return=cumulative_return(returns),
                vol_matched_return=cumulative_return(matched),
                sharpe=sharpe_ratio(returns, 0.0, periods_per_year) if len(returns) >= 2 else 0.0,
                sortino=sortino_ratio(returns, 0.0, periods_per_year) if len(returns) >= 2 else 0.0,
                max_drawdown=max_drawdown(returns),
                calmar=calmar_ratio(returns, periods_per_year),
            )
        )
    return results


def aggregate(results: Sequence[StrategyResult], metric: str = "sharpe") -> Dict[str, float]:
    """Average ``metric`` across results, grouped by strategy name.

    Useful for "average Sharpe across regimes/tickers". Non-finite values
    (inf from zero-drawdown Calmar/Sortino) are skipped in the mean.
    """
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for r in results:
        value = getattr(r, metric)
        if not math.isfinite(value):
            continue
        sums[r.name] = sums.get(r.name, 0.0) + value
        counts[r.name] = counts.get(r.name, 0) + 1
    return {name: sums[name] / counts[name] for name in sums if counts[name] > 0}
