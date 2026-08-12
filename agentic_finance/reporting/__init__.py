"""Reporting: dissertation figures from backtest + XAI artifacts.

Each function takes plain data (curves, ``StrategyResult`` lists, importance /
attribution dicts) and writes a publication-style PNG, so figures regenerate
deterministically from run outputs. Uses a headless Agg backend; matplotlib is
the only extra dependency (``reporting`` extra).
"""

from agentic_finance.reporting.figures import (
    plot_drawdown,
    plot_equity_curves,
    plot_metric_bar,
    plot_regime_heatmap,
    plot_risk_attribution,
    plot_shap_importance,
)

__all__ = [
    "plot_equity_curves",
    "plot_drawdown",
    "plot_metric_bar",
    "plot_risk_attribution",
    "plot_shap_importance",
    "plot_regime_heatmap",
]
