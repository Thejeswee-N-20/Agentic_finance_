"""Leakage-aware backtesting for the Agentic Finance framework.

- ``metrics``     : Sharpe / Sortino / Max Drawdown / Calmar / equity curve.
- ``indicators``  : SMA / EMA / MACD / RSI (pure, point-in-time).
- ``baselines``   : Buy & Hold / SMA crossover / MACD / RSI strategies.
- ``engine``      : point-in-time backtest loop (no look-ahead).
- ``risk_budgeted``: the risk-engine-driven backtest (the headline integration).
"""

from agentic_finance.backtest.baselines import (
    buy_and_hold,
    macd_strategy,
    rsi_strategy,
    sma_crossover,
)
from agentic_finance.backtest.engine import BacktestResult, run_backtest
from agentic_finance.backtest.indicators import ema, macd, rsi, sma
from agentic_finance.backtest.metrics import (
    calmar_ratio,
    cumulative_return,
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from agentic_finance.backtest.risk_budgeted import (
    RiskBudgetedResult,
    constant_long,
    momentum_direction,
    run_risk_budgeted_backtest,
)
from agentic_finance.backtest.evaluation import (
    StrategyResult,
    aggregate,
    baseline_producer,
    evaluate_window,
    realized_vol,
    risk_budgeted_producer,
    volatility_scale,
)
from agentic_finance.backtest.statistics import (
    bootstrap_confidence_interval,
    deflated_sharpe_ratio,
)

__all__ = [
    # metrics
    "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
    "cumulative_return", "equity_curve",
    # statistical rigor
    "bootstrap_confidence_interval", "deflated_sharpe_ratio",
    # indicators
    "sma", "ema", "macd", "rsi",
    # baselines
    "buy_and_hold", "sma_crossover", "macd_strategy", "rsi_strategy",
    # engine
    "run_backtest", "BacktestResult",
    # risk-budgeted
    "run_risk_budgeted_backtest", "RiskBudgetedResult",
    "constant_long", "momentum_direction",
    # evaluation harness
    "evaluate_window", "aggregate", "StrategyResult",
    "baseline_producer", "risk_budgeted_producer",
    "volatility_scale", "realized_vol",
]
