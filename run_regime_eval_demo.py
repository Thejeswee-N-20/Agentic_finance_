"""Multi-regime, volatility-matched evaluation demo.

The fair, honest version of the head-to-head: every strategy is compared across
distinct market regimes (bull / bear / chop) and at *equal risk* (vol-matched
return) — the framing argued for in AGENTIC_FINANCE_IMPLEMENTATION.md. No
LLM/API key required.

Adds a per-regime table plus an aggregate (averaged across windows). Reports
raw return AND volatility-matched return side by side so the exposure mismatch
that flatters Buy & Hold in a bull run is removed.
"""
from typing import Dict, List

import yfinance as yf

from agentic_finance.backtest import (
    aggregate,
    baseline_producer,
    buy_and_hold,
    evaluate_window,
    macd_strategy,
    momentum_direction,
    risk_budgeted_producer,
    sma_crossover,
)
from agentic_finance.risk_engine.schemas import RiskBudget

WARMUP = 60
TARGET_VOL = 0.20  # common risk level for the vol-matched column

# (label, ticker, start, end) — chosen to isolate regimes.
# Spans three asset classes: US equities/ETF, the Indian NIFTY-50 index, and
# crypto (BTC trades 7d/week; annualisation uses the same 252-day convention
# for comparability, noted in the report).
WINDOWS = [
    ("bull  SPY 2023-2024", "SPY", "2023-01-01", "2024-12-31"),
    ("bear  SPY 2022", "SPY", "2022-01-01", "2022-12-31"),
    ("bull  NVDA 2023", "NVDA", "2023-01-01", "2023-12-31"),
    ("bear  NVDA 2021H2-2022", "NVDA", "2021-08-01", "2023-01-01"),
    ("chop  AAPL 2015-2016", "AAPL", "2015-01-01", "2016-12-31"),
    ("bull  NIFTY 2023-2024", "^NSEI", "2023-01-01", "2024-12-31"),
    ("chop  NIFTY 2022", "^NSEI", "2022-01-01", "2022-12-31"),
    ("bear  BTC 2022", "BTC-USD", "2022-01-01", "2022-12-31"),
    ("bull  BTC 2023", "BTC-USD", "2023-01-01", "2023-12-31"),
]


def _closes(ticker: str, start: str, end: str) -> List[float]:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        return []
    return [float(x) for x in df["Close"].iloc[:, 0].tolist()]


def _producers():
    budget = RiskBudget(target_vol=0.15, max_position=1.0, max_drawdown=0.20,
                        cvar_limit=0.08, min_confidence=0.55)
    return {
        "Buy & Hold": baseline_producer(buy_and_hold, warmup=WARMUP),
        "SMA 20/50": baseline_producer(sma_crossover(20, 50), warmup=WARMUP),
        "MACD": baseline_producer(macd_strategy(), warmup=WARMUP),
        "Risk-budgeted (ours)": risk_budgeted_producer(
            momentum_direction(20, 0.75), budget, warmup=WARMUP),
    }


def _print_table(title: str, results) -> None:
    print(f"\n{title}")
    print(f"{'Strategy':<24} {'CumRet':>9} {'VolMatch':>9} {'Sharpe':>7} "
          f"{'Sortino':>8} {'MaxDD':>8} {'Calmar':>7}")
    print("-" * 80)
    for r in results:
        print(f"{r.name:<24} {r.cum_return*100:>8.2f}% {r.vol_matched_return*100:>8.2f}% "
              f"{r.sharpe:>7.2f} {r.sortino:>8.2f} {r.max_drawdown*100:>7.2f}% {r.calmar:>7.2f}")


def main() -> None:
    all_results = []
    for label, ticker, start, end in WINDOWS:
        closes = _closes(ticker, start, end)
        if len(closes) < WARMUP + 30:
            print(f"\n[skip] {label}: insufficient data ({len(closes)} bars)")
            continue
        regime = label.split()[0]
        results = evaluate_window(closes, _producers(), regime=regime, target_vol=TARGET_VOL)
        all_results.extend(results)
        _print_table(f"{label}  ({len(closes)} days)", results)

    print("\n\n================ AGGREGATE ACROSS ALL WINDOWS ================")
    for metric in ("vol_matched_return", "sharpe", "sortino", "max_drawdown", "calmar"):
        agg = aggregate(all_results, metric=metric)
        pretty = {k: round(v, 3) for k, v in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)}
        print(f"\nMean {metric} by strategy:")
        for name, val in pretty.items():
            suffix = "%" if "return" in metric or "drawdown" in metric else ""
            shown = val * 100 if suffix else val
            print(f"  {name:<24} {shown:>8.2f}{suffix}")


if __name__ == "__main__":
    main()
