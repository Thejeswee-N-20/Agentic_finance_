"""End-to-end XAI demo on real market data.

Pipeline: pull closes -> run the risk-budgeted backtest -> align the
point-in-time feature matrix with the per-bar actions -> train a faithful
surrogate of the decision policy -> report SHAP global importance, a local
attribution for one decision, the risk-constraint attribution, and an auditable
decision report. No LLM/API key required.

Usage:
    python run_xai_demo.py
    REPRO_TICKER=AAPL python run_xai_demo.py
"""
import os
from typing import List

import yfinance as yf

from agentic_finance.backtest import momentum_direction, run_risk_budgeted_backtest
from agentic_finance.features import build_feature_matrix
from agentic_finance.risk_engine.schemas import RiskBudget
from agentic_finance.xai import (
    binding_summary,
    counterfactual_report,
    decision_report,
    find_counterfactuals,
    global_importance,
    local_attributions,
    surrogate_fidelity,
    train_surrogate,
)

TICKER = os.environ.get("REPRO_TICKER", "NVDA")
START = os.environ.get("REPRO_START", "2021-01-01")
END = os.environ.get("REPRO_END", "2024-06-01")
WARMUP = 60


def _closes(ticker: str, start: str, end: str) -> List[float]:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        raise SystemExit(f"No data for {ticker}")
    return [float(x) for x in df["Close"].iloc[:, 0].tolist()]


def main() -> None:
    closes = _closes(TICKER, START, END)
    budget = RiskBudget(target_vol=0.15, max_position=1.0, max_drawdown=0.20,
                        cvar_limit=0.08, min_confidence=0.55)
    rb = run_risk_budgeted_backtest(closes, momentum_direction(20, 0.75), budget, warmup=WARMUP)

    # Orders correspond to decision bars WARMUP .. len-2 (one per realized period).
    order_by_bar = {WARMUP + k: o for k, o in enumerate(rb.orders)}
    names, matrix, feat_idx = build_feature_matrix(closes, warmup=WARMUP)

    rows, labels, kept_bars = [], [], []
    for j, bar in enumerate(feat_idx):
        if bar in order_by_bar:
            rows.append(matrix[j])
            labels.append(order_by_bar[bar].action)
            kept_bars.append(bar)

    print(f"\n=== XAI demo: {TICKER} {START} -> {END} ===")
    print(f"Decisions: {len(labels)} | action mix: "
          f"{ {a: labels.count(a) for a in sorted(set(labels))} }")

    if len(set(labels)) < 2:
        print("Only one action class in this window — surrogate/SHAP need >=2 classes.")
        print(f"Risk-constraint attribution: {binding_summary(rb.orders)}")
        return

    model = train_surrogate(rows, labels, feature_names=names)
    fidelity = surrogate_fidelity(model, rows, labels)
    importance = global_importance(model, rows)

    print(f"\nSurrogate fidelity (policy agreement): {fidelity * 100:.1f}%")
    print("\nGlobal SHAP feature importance (top 8):")
    for name, val in sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:8]:
        print(f"  {name:<20} {val:.4f}")

    print(f"\nRisk-constraint attribution (fraction of bars):")
    for k, v in sorted(binding_summary(rb.orders).items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k:<18} {v * 100:5.1f}%")

    # Local explanation for the first non-hold decision.
    for bar in kept_bars:
        order = order_by_bar[bar]
        if order.action != "hold":
            row = rows[kept_bars.index(bar)]
            shap_local = local_attributions(model, row)
            print("\n--- Audit report for one decision ---")
            print(decision_report(order, shap_local, fidelity=fidelity, top_k=5))

            # Counterfactual: the smallest input change that flips this call.
            cfs = find_counterfactuals(model, row, training_matrix=rows)
            print("\n--- Counterfactual explanation ---")
            print(counterfactual_report(cfs))
            break


if __name__ == "__main__":
    main()
