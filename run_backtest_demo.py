"""End-to-end backtest demo on real market data.

Pulls daily closes from Yahoo Finance and compares the baseline strategies
against the **risk-budgeted** strategy (momentum direction signal sized by the
quantitative risk engine). Prints a metrics table — the first concrete
head-to-head the dissertation needs. No LLM/API key required.

Usage:
    python run_backtest_demo.py                  # NVDA, ~2 years
    REPRO_TICKER=AAPL REPRO_START=2022-01-01 REPRO_END=2024-01-01 python run_backtest_demo.py
"""
import os
from typing import List

import yfinance as yf

from agentic_finance.backtest import (
    buy_and_hold,
    macd_strategy,
    momentum_direction,
    run_backtest,
    run_risk_budgeted_backtest,
    sma_crossover,
)
from agentic_finance.risk_engine.schemas import RiskBudget

TICKER = os.environ.get("REPRO_TICKER", "NVDA")
START = os.environ.get("REPRO_START", "2022-06-01")
END = os.environ.get("REPRO_END", "2024-06-01")


def _closes(ticker: str, start: str, end: str) -> List[float]:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        raise SystemExit(f"No data for {ticker} {start}..{end}")
    return [float(x) for x in df["Close"].iloc[:, 0].tolist()]


def _row(name: str, cum: float, sharpe: float, sortino: float, mdd: float, calmar: float) -> str:
    return (f"{name:<22} {cum*100:>8.2f}%  {sharpe:>7.2f}  {sortino:>8.2f}  "
            f"{mdd*100:>7.2f}%  {calmar:>7.2f}")


def main() -> None:
    closes = _closes(TICKER, START, END)
    print(f"\n{TICKER}  {START} -> {END}   ({len(closes)} trading days)\n")
    print(f"{'Strategy':<22} {'CumRet':>9}  {'Sharpe':>7}  {'Sortino':>8}  {'MaxDD':>8}  {'Calmar':>7}")
    print("-" * 72)

    warmup = 60  # let the 50-day SMA / momentum lookback form
    for name, strat in [
        ("Buy & Hold", buy_and_hold),
        ("SMA crossover 20/50", sma_crossover(20, 50)),
        ("MACD", macd_strategy()),
    ]:
        r = run_backtest(closes, strat, warmup=warmup)
        print(_row(name, r.cumulative_return, r.sharpe, r.sortino, r.max_drawdown, r.calmar))

    budget = RiskBudget(target_vol=0.15, max_position=1.0, max_drawdown=0.20,
                        cvar_limit=0.08, min_confidence=0.55)
    rb = run_risk_budgeted_backtest(closes, momentum_direction(lookback=20, confidence=0.75),
                                    budget, warmup=warmup)
    p = rb.performance
    print(_row("Risk-budgeted (ours)", p.cumulative_return, p.sharpe, p.sortino,
               p.max_drawdown, p.calmar))

    vetoes = sum(1 for o in rb.orders if o.vetoed)
    constraints = {}
    for o in rb.orders:
        constraints[o.binding_constraint] = constraints.get(o.binding_constraint, 0) + 1
    print(f"\nRisk-budgeted binding constraints: {constraints}")
    print(f"Drawdown-breaker vetoes: {vetoes} / {len(rb.orders)} bars")


if __name__ == "__main__":
    main()
