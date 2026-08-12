"""Unit tests for presets, market-regime, screening and suggestions (offline)."""
import math

from agentic_finance.risk_engine import RiskBudget
from agentic_finance.screening import (
    RISK_PRESETS,
    Regime,
    StockScore,
    budget_from_preset,
    market_regime,
    screen,
    suggest,
)


def _trend(n=200, start=100.0, step=0.004):
    return [start * (1.0 + step) ** i * (1.0 + 0.01 * math.sin(i / 3.0)) for i in range(n)]


def test_presets_build_valid_budgets():
    for name in ("Conservative", "Moderate", "Aggressive"):
        b = budget_from_preset(name)
        assert isinstance(b, RiskBudget)
    # conservative must be stricter than aggressive
    c, a = budget_from_preset("Conservative"), budget_from_preset("Aggressive")
    assert c.max_position < a.max_position
    assert c.max_drawdown < a.max_drawdown
    assert c.target_vol < a.target_vol


def test_market_regime_bull_bear():
    bull = _trend(step=0.006)
    assert market_regime(bull).label == "Bullish"
    bear = _trend(step=-0.006, start=300.0)
    assert market_regime(bear).label == "Bearish"
    # too-short history is treated as choppy/neutral
    assert market_regime([100, 101, 102]).label == "Choppy"


def test_screen_ranks_and_scores():
    # three synthetic peers with different momentum
    data = {
        "A": _trend(step=0.006),   # strong up
        "B": _trend(step=0.002),   # mild up
        "C": _trend(step=-0.004, start=300.0),  # down
    }
    load = lambda t: data.get(t)
    budget = budget_from_preset("Aggressive")
    reg = Regime("Bullish", 0.1, 0.2, 1.3)
    scores = screen(list(data), budget, "Aggressive", reg, load)
    assert scores and all(isinstance(s, StockScore) for s in scores)
    # ranked descending by fit; aggressive appetite should favor the strongest up-trend
    assert scores == sorted(scores, key=lambda s: s.fit_score, reverse=True)
    assert scores[0].ticker == "A"


def test_suggest_ranks_injected_peers_no_network():
    # peer_source + enrich_names=False keep this fully offline
    load = lambda t: _trend()          # every ticker returns a valid series
    peers = lambda t, n: ["TCS.NS", "HCLTECH.NS", "WIPRO.NS"]
    budget = budget_from_preset("Moderate")
    out = suggest("INFY.NS", budget, "Moderate", load, top_k=4, deep_dive=False,
                  peer_source=peers, enrich_names=False)
    assert 0 < len(out) <= 4
    assert all(s.ticker != "INFY.NS" for s in out)   # self excluded upstream
    assert all(s.agentic_note is None for s in out)   # deep-dive was off


def test_suggest_no_peers_returns_empty():
    load = lambda t: _trend()
    out = suggest("ZZZZ", budget_from_preset("Moderate"), "Moderate", load,
                  deep_dive=False, peer_source=lambda t, n: [], enrich_names=False)
    assert out == []
