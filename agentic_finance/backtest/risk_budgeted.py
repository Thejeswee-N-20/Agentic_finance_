"""Risk-budgeted backtest — the quantitative risk engine driving a backtest.

A *direction signal* maps point-in-time price history to a ``PreDecision``
(direction + confidence). At each bar the engine builds the market/portfolio
risk inputs, calls :func:`assess_and_size`, and applies the resulting enforced
position. Portfolio equity and its running peak are threaded through the loop so
the max-drawdown circuit breaker engages exactly as it would live.

This is what separates the framework from TradingAgents: positions are *sized
and risk-limited*, not a prose rating. The per-bar ``RiskedOrder`` sequence is
returned alongside the performance summary so the XAI layer can later attribute
each position to its binding risk constraint.

Notes / scope:
- ATR is approximated from close-to-close moves (true ATR needs OHLC); the live
  agent path uses the real ATR surfaced by the market analyst.
- Stops are computed and recorded but not executed intra-bar in this daily loop;
  the active protection here is the drawdown breaker. Intra-bar stop execution
  is a later refinement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

from agentic_finance.backtest.engine import BacktestResult
from agentic_finance.backtest.metrics import (
    calmar_ratio,
    cumulative_return,
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from agentic_finance.risk_engine.risk_manager import assess_and_size
from agentic_finance.risk_engine.schemas import (
    MarketRiskInputs,
    PortfolioState,
    PreDecision,
    RiskBudget,
    RiskedOrder,
)

DirectionSignal = Callable[[Sequence[float]], PreDecision]

__all__ = [
    "run_risk_budgeted_backtest",
    "RiskBudgetedResult",
    "constant_long",
    "momentum_direction",
]


@dataclass(frozen=True)
class RiskBudgetedResult:
    """Performance summary plus the per-bar enforced orders."""

    performance: BacktestResult
    orders: Tuple[RiskedOrder, ...]


# --- direction signals ----------------------------------------------------

def constant_long(confidence: float) -> DirectionSignal:
    """Always-long signal at a fixed confidence (useful for tests/demos)."""
    def signal(prices: Sequence[float]) -> PreDecision:
        return PreDecision(direction="long", confidence=confidence)
    return signal


def momentum_direction(lookback: int = 20, confidence: float = 0.7) -> DirectionSignal:
    """Long when price is above its value ``lookback`` bars ago, else flat."""
    def signal(prices: Sequence[float]) -> PreDecision:
        if len(prices) <= lookback:
            return PreDecision(direction="flat", confidence=0.0)
        direction = "long" if prices[-1] > prices[-1 - lookback] else "flat"
        return PreDecision(direction=direction, confidence=confidence if direction == "long" else 0.0)
    return signal


# --- helpers --------------------------------------------------------------

def _atr_proxy(prices: Sequence[float], period: int) -> float:
    """Close-to-close ATR proxy: mean absolute change over the last ``period``."""
    if len(prices) < 2:
        return 0.0
    changes = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    window = changes[-period:]
    return sum(window) / len(window)


def _returns_window(prices: Sequence[float], window: int) -> Tuple[float, ...]:
    rets = [prices[i] / prices[i - 1] - 1.0 for i in range(1, len(prices))]
    return tuple(rets[-window:])


# --- main loop ------------------------------------------------------------

def run_risk_budgeted_backtest(
    prices: Sequence[float],
    direction_signal: DirectionSignal,
    budget: RiskBudget,
    warmup: int = 20,
    initial_equity: float = 1.0,
    atr_period: int = 14,
    risk_window: int = 30,
    periods_per_year: int = 252,
) -> RiskBudgetedResult:
    """Run the risk-engine-driven backtest over ``prices`` point-in-time."""
    closes = [float(p) for p in prices]
    if len(closes) < 2:
        raise ValueError(f"need at least 2 prices, got {len(closes)}")
    if warmup < 1:
        raise ValueError("warmup must be >= 1 so a returns window exists")

    equity = initial_equity
    peak = initial_equity
    returns: List[float] = []
    weights: List[float] = []
    orders: List[RiskedOrder] = []

    for i in range(warmup, len(closes) - 1):
        history = closes[: i + 1]  # info available at close i (no future)
        pre = direction_signal(history)
        market = MarketRiskInputs(
            price=closes[i],
            atr=_atr_proxy(history, atr_period),
            returns=_returns_window(history, risk_window),
        )
        state = PortfolioState(equity=equity, peak_equity=peak, remaining_budget=1.0)
        order = assess_and_size(pre, market, state, budget)

        if order.action == "buy":
            weight = order.size
        elif order.action == "sell":
            weight = -order.size
        else:
            weight = 0.0

        period_return = closes[i + 1] / closes[i] - 1.0
        realized = weight * period_return
        equity *= (1.0 + realized)
        peak = max(peak, equity)

        returns.append(realized)
        weights.append(weight)
        orders.append(order)

    if not returns:
        raise ValueError("no periods to backtest after applying warmup")

    curve = equity_curve(returns, initial=initial_equity)
    sharpe = sharpe_ratio(returns, 0.0, periods_per_year) if len(returns) >= 2 else 0.0
    sortino = sortino_ratio(returns, 0.0, periods_per_year) if len(returns) >= 2 else 0.0
    performance = BacktestResult(
        returns=tuple(returns),
        weights=tuple(weights),
        equity=tuple(curve),
        cumulative_return=cumulative_return(returns),
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown(returns),
        calmar=calmar_ratio(returns, periods_per_year),
    )
    return RiskBudgetedResult(performance=performance, orders=tuple(orders))
