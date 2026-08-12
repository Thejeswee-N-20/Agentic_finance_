"""Ablation: placeholder momentum signal vs the technical-agent signal.

Same quantitative risk engine, same windows, same equal-risk (vol-matched)
evaluation — only the *direction signal* changes. This is the dissertation's key
ablation isolating the contribution of the (real) agent signal from the risk
engine. No LLM/API key required (the technical agent is deterministic).
"""
from typing import List

import yfinance as yf

from agentic_finance.agents_v2 import agent_direction_signal, technical_provider
from agentic_finance.backtest import (
    aggregate,
    evaluate_window,
    momentum_direction,
    risk_budgeted_producer,
)
from agentic_finance.risk_engine.schemas import RiskBudget

WARMUP = 60
TARGET_VOL = 0.20

WINDOWS = [
    ("bull SPY 2023-2024", "SPY", "2023-01-01", "2024-12-31"),
    ("bear SPY 2022", "SPY", "2022-01-01", "2022-12-31"),
    ("bull NVDA 2023", "NVDA", "2023-01-01", "2023-12-31"),
    ("bear NVDA 2021H2-2022", "NVDA", "2021-08-01", "2023-01-01"),
    ("chop AAPL 2015-2016", "AAPL", "2015-01-01", "2016-12-31"),
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
        "Placeholder momentum": risk_budgeted_producer(
            momentum_direction(20, 0.75), budget, warmup=WARMUP),
        "Technical agent": risk_budgeted_producer(
            agent_direction_signal([technical_provider()]), budget, warmup=WARMUP),
    }


def main() -> None:
    all_results = []
    for label, ticker, start, end in WINDOWS:
        closes = _closes(ticker, start, end)
        if len(closes) < WARMUP + 30:
            print(f"[skip] {label}")
            continue
        results = evaluate_window(closes, _producers(),
                                  regime=label.split()[0], target_vol=TARGET_VOL)
        all_results.extend(results)
        print(f"\n{label}  ({len(closes)} days)")
        for r in results:
            print(f"  {r.name:<22} cum={r.cum_return*100:7.2f}%  "
                  f"volmatch={r.vol_matched_return*100:7.2f}%  "
                  f"Sharpe={r.sharpe:5.2f}  Sortino={r.sortino:5.2f}  "
                  f"MaxDD={r.max_drawdown*100:6.2f}%")

    print("\n\n=========== ABLATION AGGREGATE (mean across windows) ===========")
    for metric in ("vol_matched_return", "sharpe", "sortino", "max_drawdown", "calmar"):
        agg = aggregate(all_results, metric=metric)
        print(f"\n{metric}:")
        for name, val in sorted(agg.items(), key=lambda kv: kv[1], reverse=True):
            suffix = "%" if ("return" in metric or "drawdown" in metric) else ""
            shown = val * 100 if suffix else val
            print(f"  {name:<22} {shown:8.2f}{suffix}")


if __name__ == "__main__":
    main()
