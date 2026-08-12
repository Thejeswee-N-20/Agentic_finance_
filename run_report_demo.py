"""Generate the dissertation figure set from real data.

Runs the multi-regime evaluation + an XAI pass and writes publication-style PNGs
to reports/figures/. No LLM/API key required. This is the one command that turns
the framework's outputs into thesis-ready figures.
"""
import os
from typing import List

import yfinance as yf

from agentic_finance.backtest import (
    baseline_producer,
    buy_and_hold,
    evaluate_window,
    macd_strategy,
    momentum_direction,
    risk_budgeted_producer,
    run_backtest,
    run_risk_budgeted_backtest,
    sma_crossover,
)
from agentic_finance.features import build_feature_matrix
from agentic_finance.reporting import (
    plot_drawdown,
    plot_equity_curves,
    plot_metric_bar,
    plot_regime_heatmap,
    plot_risk_attribution,
    plot_shap_importance,
)
from agentic_finance.risk_engine.schemas import RiskBudget
from agentic_finance.xai import binding_summary, global_importance, surrogate_fidelity, train_surrogate

WARMUP = 60
OUT = os.path.join("reports", "figures")
WINDOWS = [
    ("bull SPY 23-24", "SPY", "2023-01-01", "2024-12-31"),
    ("bear SPY 22", "SPY", "2022-01-01", "2022-12-31"),
    ("bull NVDA 23", "NVDA", "2023-01-01", "2023-12-31"),
    ("bear NVDA 21-22", "NVDA", "2021-08-01", "2023-01-01"),
    ("chop AAPL 15-16", "AAPL", "2015-01-01", "2016-12-31"),
]
SHOWCASE = ("NVDA", "2021-08-01", "2023-01-01")  # bear window: drawdown protection


def _closes(ticker: str, start: str, end: str) -> List[float]:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    return [] if df is None or len(df) == 0 else [float(x) for x in df["Close"].iloc[:, 0].tolist()]


def _budget():
    return RiskBudget(target_vol=0.15, max_position=1.0, max_drawdown=0.20,
                      cvar_limit=0.08, min_confidence=0.55)


def _producers():
    return {
        "Buy & Hold": baseline_producer(buy_and_hold, warmup=WARMUP),
        "SMA 20/50": baseline_producer(sma_crossover(20, 50), warmup=WARMUP),
        "MACD": baseline_producer(macd_strategy(), warmup=WARMUP),
        "Risk-budgeted": risk_budgeted_producer(momentum_direction(20, 0.75), _budget(), warmup=WARMUP),
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    written = []

    # 1-2. Multi-regime metrics: heatmaps + bars.
    all_results = []
    for label, ticker, start, end in WINDOWS:
        closes = _closes(ticker, start, end)
        if len(closes) < WARMUP + 30:
            continue
        all_results.extend(evaluate_window(closes, _producers(), regime=label, target_vol=0.20))
    if all_results:
        written.append(plot_regime_heatmap(all_results, "sharpe", f"{OUT}/heatmap_sharpe.png"))
        written.append(plot_regime_heatmap(all_results, "max_drawdown", f"{OUT}/heatmap_maxdd.png"))
        written.append(plot_metric_bar(all_results, "max_drawdown", f"{OUT}/bar_maxdd.png"))
        written.append(plot_metric_bar(all_results, "vol_matched_return", f"{OUT}/bar_volmatched.png"))
        written.append(plot_metric_bar(all_results, "sharpe", f"{OUT}/bar_sharpe.png"))

    # 3-4. Equity + drawdown on the showcase (bear) window.
    closes = _closes(*SHOWCASE)
    if len(closes) >= WARMUP + 30:
        curves = {
            "Buy & Hold": run_backtest(closes, buy_and_hold, warmup=WARMUP).equity,
            "SMA 20/50": run_backtest(closes, sma_crossover(20, 50), warmup=WARMUP).equity,
            "MACD": run_backtest(closes, macd_strategy(), warmup=WARMUP).equity,
        }
        rb = run_risk_budgeted_backtest(closes, momentum_direction(20, 0.75), _budget(), warmup=WARMUP)
        curves["Risk-budgeted"] = rb.performance.equity
        written.append(plot_equity_curves(curves, f"{OUT}/equity_bear_NVDA.png",
                                           title="Equity curves — NVDA bear window (2021H2–2022)"))
        written.append(plot_drawdown(curves, f"{OUT}/drawdown_bear_NVDA.png"))
        written.append(plot_risk_attribution(binding_summary(rb.orders), f"{OUT}/risk_attribution.png"))

    # 5. XAI: surrogate + SHAP global importance on a trending window.
    closes = _closes("NVDA", "2021-01-01", "2024-06-01")
    if len(closes) >= WARMUP + 60:
        rb = run_risk_budgeted_backtest(closes, momentum_direction(20, 0.75), _budget(), warmup=WARMUP)
        order_by_bar = {WARMUP + k: o for k, o in enumerate(rb.orders)}
        names, matrix, feat_idx = build_feature_matrix(closes, warmup=WARMUP)
        rows, labels = [], []
        for j, bar in enumerate(feat_idx):
            if bar in order_by_bar:
                rows.append(matrix[j]); labels.append(order_by_bar[bar].action)
        if len(set(labels)) >= 2:
            model = train_surrogate(rows, labels, feature_names=names)
            fid = surrogate_fidelity(model, rows, labels)
            imp = global_importance(model, rows)
            written.append(plot_shap_importance(
                imp, f"{OUT}/shap_importance.png",
                title=f"Global SHAP importance (surrogate fidelity {fid*100:.0f}%)"))

    print(f"Wrote {len(written)} figures to {OUT}/:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
