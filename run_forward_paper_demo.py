"""Forward paper-trading test on a held-out recent period, with statistical rigor.

The only fully leakage-proof test of a strategy is a period the developers
never saw: every configuration in this repository was frozen on data up to
mid-2025, so the window below (2026 onwards by default) is genuinely
out-of-sample "forward" data. The deterministic price-only strategy is used so
the test is exactly reproducible and free (no API calls).

For each ticker it reports Sharpe with a moving-block bootstrap 95% CI and the
deflated Sharpe ratio (DSR): the probability that the observed Sharpe beats
the expected maximum of ``N_TRIALS`` unskilled strategies — the
Bailey & López de Prado correction for selection bias.

    python run_forward_paper_demo.py
    REPRO_START=2026-01-01 python run_forward_paper_demo.py
"""
import os
from typing import List

import yfinance as yf

from agentic_finance.backtest import (
    bootstrap_confidence_interval,
    cumulative_return,
    deflated_sharpe_ratio,
    max_drawdown,
    momentum_direction,
    run_risk_budgeted_backtest,
    sharpe_ratio,
)
from agentic_finance.risk_engine.schemas import RiskBudget

TICKERS = os.environ.get("REPRO_TICKERS", "NVDA,AAPL,MSFT,GOOGL,^NSEI,BTC-USD").split(",")
START = os.environ.get("REPRO_START", "2026-01-01")  # held-out forward period
WARMUP = 60
# Strategy variants ever evaluated in this project (baselines + ours) — the
# honest trial count for the deflated-Sharpe selection-bias correction.
N_TRIALS = 8


def _closes(ticker: str, start: str) -> List[float]:
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        return []
    return [float(x) for x in df["Close"].iloc[:, 0].tolist()]


def main() -> None:
    budget = RiskBudget(target_vol=0.15, max_position=0.5, cvar_limit=0.10,
                        max_drawdown=0.20, min_confidence=0.55)
    print(f"=== Forward paper-trading (held-out from {START}) ===")
    print(f"Deterministic price-only risk-budgeted strategy; DSR vs N={N_TRIALS} trials\n")
    print(f"{'Ticker':<10}{'Bars':>6}{'CumRet':>9}{'Sharpe':>8}"
          f"{'95% CI':>18}{'DSR':>6}{'MaxDD':>8}{'B&H CumRet':>12}")
    print("-" * 78)
    for ticker in TICKERS:
        closes = _closes(ticker.strip(), START)
        if len(closes) < WARMUP + 20:
            print(f"{ticker:<10}  insufficient data ({len(closes)} bars) — skipped")
            continue
        rb = run_risk_budgeted_backtest(closes, momentum_direction(20, 0.75),
                                        budget, warmup=WARMUP)
        rets = list(rb.performance.returns)
        bh = [closes[i + 1] / closes[i] - 1.0 for i in range(WARMUP, len(closes) - 1)]
        lo, hi = bootstrap_confidence_interval(rets, sharpe_ratio, n_boot=1000, seed=0)
        dsr = deflated_sharpe_ratio(rets, n_trials=N_TRIALS)
        print(f"{ticker:<10}{len(rets):>6}{cumulative_return(rets)*100:>8.2f}%"
              f"{sharpe_ratio(rets):>8.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>18}"
              f"{dsr:>6.2f}{max_drawdown(rets)*100:>7.2f}%"
              f"{cumulative_return(bh)*100:>11.2f}%")
    print("\nDSR reading: >0.95 = skill unlikely to be a multiple-testing artefact;"
          "\n~0.5 or below = consistent with selection from unskilled noise.")


if __name__ == "__main__":
    main()
