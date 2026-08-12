"""Figure generators for the dissertation.

Headless (Agg) matplotlib. Each function returns the output path it wrote. Data
comes from the backtest / evaluation / XAI modules so figures are reproducible
from run artifacts. Kept dependency-light: pure matplotlib (no seaborn).
"""
from __future__ import annotations

from typing import List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless; safe for tests/CI
import matplotlib.pyplot as plt  # noqa: E402

from agentic_finance.backtest.evaluation import StrategyResult  # noqa: E402

__all__ = [
    "plot_equity_curves",
    "plot_drawdown",
    "plot_metric_bar",
    "plot_risk_attribution",
    "plot_shap_importance",
    "plot_regime_heatmap",
]

_METRICS = {"cum_return", "vol_matched_return", "sharpe", "sortino", "max_drawdown", "calmar"}


def _save(fig, outpath: str) -> str:
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def _running_drawdown(curve: Sequence[float]) -> List[float]:
    peak = curve[0]
    dd = []
    for v in curve:
        peak = max(peak, v)
        dd.append((v - peak) / peak if peak > 0 else 0.0)
    return dd


def plot_equity_curves(curves: Mapping[str, Sequence[float]], outpath: str,
                       title: str = "Equity curves") -> str:
    """Overlaid equity curves per strategy (the headline performance figure)."""
    if not curves:
        raise ValueError("no curves to plot")
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, curve in curves.items():
        ax.plot(range(len(curve)), curve, label=name, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Equity (growth of 1)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return _save(fig, outpath)


def plot_drawdown(curves: Mapping[str, Sequence[float]], outpath: str,
                  title: str = "Drawdown (underwater)") -> str:
    """Underwater drawdown plot per strategy."""
    if not curves:
        raise ValueError("no curves to plot")
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, curve in curves.items():
        dd = [d * 100 for d in _running_drawdown(list(curve))]
        ax.plot(range(len(dd)), dd, label=name, linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return _save(fig, outpath)


def plot_metric_bar(results: Sequence[StrategyResult], metric: str, outpath: str) -> str:
    """Bar chart of one metric per strategy, averaged across the supplied results."""
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(_METRICS)}")
    if not results:
        raise ValueError("no results to plot")
    agg: dict = {}
    counts: dict = {}
    for r in results:
        v = getattr(r, metric)
        agg[r.name] = agg.get(r.name, 0.0) + v
        counts[r.name] = counts.get(r.name, 0) + 1
    names = list(agg)
    values = [agg[n] / counts[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, values, color="#3b6ea5")
    ax.set_title(f"Mean {metric} by strategy")
    ax.set_ylabel(metric)
    ax.axhline(0, color="black", linewidth=0.8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, outpath)


def plot_risk_attribution(summary: Mapping[str, float], outpath: str,
                          title: str = "Risk-constraint attribution") -> str:
    """Bar chart of the fraction of decisions bound by each risk constraint."""
    if not summary:
        raise ValueError("empty attribution summary")
    items = sorted(summary.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v * 100 for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#a5683b")
    ax.set_title(title)
    ax.set_ylabel("Share of decisions (%)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, outpath)


def plot_shap_importance(importance: Mapping[str, float], outpath: str,
                         top_k: int = 12, title: str = "Global SHAP feature importance") -> str:
    """Horizontal bar of mean |SHAP| per feature (the summary-bar view)."""
    if not importance:
        raise ValueError("empty importance mapping")
    items = sorted(importance.items(), key=lambda kv: kv[1])[-top_k:]
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.4 * len(labels))))
    ax.barh(labels, values, color="#4a8a5a")
    ax.set_title(title)
    ax.set_xlabel("mean |SHAP value|")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, outpath)


def plot_regime_heatmap(results: Sequence[StrategyResult], metric: str, outpath: str) -> str:
    """Strategy x regime heatmap of a metric."""
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(_METRICS)}")
    if not results:
        raise ValueError("no results to plot")
    strategies = list(dict.fromkeys(r.name for r in results))
    regimes = list(dict.fromkeys(r.regime for r in results))
    grid = [[float("nan")] * len(regimes) for _ in strategies]
    for r in results:
        i = strategies.index(r.name)
        j = regimes.index(r.regime)
        grid[i][j] = getattr(r, metric)

    fig, ax = plt.subplots(figsize=(1.6 * len(regimes) + 3, 0.7 * len(strategies) + 2))
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(regimes)), labels=regimes)
    ax.set_yticks(range(len(strategies)), labels=strategies)
    for i in range(len(strategies)):
        for j in range(len(regimes)):
            val = grid[i][j]
            if val == val:  # not NaN
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"{metric} by strategy x regime")
    fig.colorbar(im, ax=ax, label=metric, fraction=0.046, pad=0.04)
    return _save(fig, outpath)
