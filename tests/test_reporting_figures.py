"""Unit tests for the reporting/figures module.

Plotting code is verified by generating each figure to a temp PNG and asserting
a non-empty file is written (headless Agg backend). This catches API misuse and
data-shape errors without asserting pixels.
"""
import os

import pytest

from agentic_finance.backtest.evaluation import StrategyResult
from agentic_finance.reporting.figures import (
    plot_equity_curves,
    plot_drawdown,
    plot_metric_bar,
    plot_risk_attribution,
    plot_shap_importance,
    plot_regime_heatmap,
)

pytestmark = pytest.mark.unit


def _curves():
    return {
        "Buy & Hold": [1.0, 1.1, 1.05, 1.2, 1.15],
        "Risk-budgeted": [1.0, 1.02, 1.03, 1.04, 1.05],
    }


def _results():
    rows = []
    for regime in ("bull", "bear"):
        for name, sharpe, mdd in [("Buy & Hold", 1.2, 0.3), ("Risk-budgeted", 0.9, 0.08)]:
            rows.append(StrategyResult(
                name=name, regime=regime, cum_return=0.2, vol_matched_return=0.15,
                sharpe=sharpe, sortino=sharpe * 1.5, max_drawdown=mdd, calmar=0.5))
    return rows


def _exists_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def test_plot_equity_curves(tmp_path):
    out = str(tmp_path / "equity.png")
    assert plot_equity_curves(_curves(), out) == out
    assert _exists_nonempty(out)


def test_plot_drawdown(tmp_path):
    out = str(tmp_path / "dd.png")
    plot_drawdown(_curves(), out)
    assert _exists_nonempty(out)


def test_plot_metric_bar(tmp_path):
    out = str(tmp_path / "bar.png")
    plot_metric_bar(_results(), "sharpe", out)
    assert _exists_nonempty(out)


def test_plot_metric_bar_rejects_unknown_metric(tmp_path):
    with pytest.raises(ValueError):
        plot_metric_bar(_results(), "nonsense", str(tmp_path / "x.png"))


def test_plot_risk_attribution(tmp_path):
    out = str(tmp_path / "risk.png")
    plot_risk_attribution({"kelly": 0.5, "vol_target": 0.3, "cvar_limit": 0.2}, out)
    assert _exists_nonempty(out)


def test_plot_shap_importance(tmp_path):
    out = str(tmp_path / "shap.png")
    importance = {f"feat_{i}": (10 - i) / 10.0 for i in range(10)}
    plot_shap_importance(importance, out, top_k=5)
    assert _exists_nonempty(out)


def test_plot_regime_heatmap(tmp_path):
    out = str(tmp_path / "heat.png")
    plot_regime_heatmap(_results(), "sharpe", out)
    assert _exists_nonempty(out)


def test_empty_inputs_raise(tmp_path):
    with pytest.raises(ValueError):
        plot_equity_curves({}, str(tmp_path / "e.png"))
    with pytest.raises(ValueError):
        plot_risk_attribution({}, str(tmp_path / "r.png"))
