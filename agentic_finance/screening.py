"""Risk-appetite presets, market-regime detection, and the suggestion engine.

Built entirely on existing primitives — no change to the risk engine or fusion.

- ``RISK_PRESETS`` / ``budget_from_preset`` map a Conservative/Moderate/Aggressive
  appetite onto a concrete ``RiskBudget`` (so the *decision* follows appetite).
- ``market_regime`` classifies current conditions from a benchmark index.
- ``screen`` scores a set of tickers deterministically (no LLM) and ranks them to
  the user's appetite and the prevailing regime.
- ``suggest`` finds sector peers of a chosen stock, screens them, and (optionally)
  runs the full agentic pipeline on the top few for a richer view — the hybrid
  design. Output is explicitly *ideas to review, not investment advice*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from agentic_finance import universe
from agentic_finance.agents_v2 import technical_signal
from agentic_finance.backtest import momentum_direction, run_risk_budgeted_backtest
from agentic_finance.decision import agentic_decision
from agentic_finance.features import extract_features
from agentic_finance.risk_engine import RiskBudget

__all__ = [
    "RISK_PRESETS", "APPETITES", "budget_from_preset",
    "Regime", "market_regime", "StockScore", "screen", "suggest",
]

# Loader contract: ticker -> list of daily closes (or None if unavailable).
PriceLoader = Callable[[str], Optional[Sequence[float]]]

APPETITES = ("Conservative", "Moderate", "Aggressive")

# Preset -> RiskBudget field values. Moderate == the framework's default budget.
RISK_PRESETS: Dict[str, Dict[str, float]] = {
    "Conservative": dict(target_vol=0.08, max_position=0.10, cvar_limit=0.06,
                         max_drawdown=0.10, min_confidence=0.65,
                         kelly_fraction=0.35, atr_multiplier=2.5),
    "Moderate": dict(target_vol=0.15, max_position=0.25, cvar_limit=0.10,
                     max_drawdown=0.20, min_confidence=0.55,
                     kelly_fraction=0.50, atr_multiplier=2.0),
    "Aggressive": dict(target_vol=0.25, max_position=0.50, cvar_limit=0.15,
                       max_drawdown=0.30, min_confidence=0.50,
                       kelly_fraction=0.70, atr_multiplier=1.5),
}


def budget_from_preset(name: str) -> RiskBudget:
    """Build a RiskBudget from an appetite preset (defaults to Moderate)."""
    return RiskBudget(**RISK_PRESETS.get(name, RISK_PRESETS["Moderate"]))


# --- market regime ---------------------------------------------------------
@dataclass(frozen=True)
class Regime:
    label: str        # "Bullish" | "Bearish" | "Choppy"
    ret_60d: float    # 60-day benchmark return
    vol: float        # annualized 20-day volatility
    tilt: float       # momentum weight multiplier for ranking

    def describe(self) -> str:
        return (f"{self.label} market — 60-day move {self.ret_60d * 100:+.1f}%, "
                f"volatility {self.vol * 100:.0f}% (annualized).")


def _ann_vol(closes: Sequence[float], window: int = 20) -> float:
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))][-window:]
    if len(rets) < 2:
        return 0.0
    mu = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252)


def market_regime(index_closes: Optional[Sequence[float]]) -> Regime:
    """Classify current market conditions from a benchmark index's closes."""
    if not index_closes or len(index_closes) < 61:
        return Regime("Choppy", 0.0, 0.0, 1.0)
    closes = [float(x) for x in index_closes]
    ret_60 = closes[-1] / closes[-61] - 1.0
    sma50 = sum(closes[-50:]) / 50.0
    vol = _ann_vol(closes)
    above = closes[-1] > sma50
    if ret_60 > 0.02 and above:
        return Regime("Bullish", ret_60, vol, 1.3)
    if ret_60 < -0.02 and not above:
        return Regime("Bearish", ret_60, vol, 0.6)
    return Regime("Choppy", ret_60, vol, 1.0)


# --- per-stock score -------------------------------------------------------
@dataclass
class StockScore:
    ticker: str
    name: str
    sector: str
    currency: str
    last_price: float
    ret_20d: float
    vol_20d_ann: float
    sharpe: float
    max_drawdown: float
    suggested_size: float
    direction: str
    fit_score: float = 0.0
    rationale: str = ""
    agentic_note: Optional[str] = None   # filled for the top-N deep-dive

    def summary(self) -> str:
        return (f"{self.direction}, 20d {self.ret_20d * 100:+.1f}%, "
                f"Sharpe {self.sharpe:.2f}, maxDD {self.max_drawdown * 100:.1f}%")


def _rank01(values: List[float], higher_better: bool = True) -> List[float]:
    """Min-max normalize to [0,1] across the set (0.5 if all equal)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5] * len(values)
    norm = [(v - lo) / (hi - lo) for v in values]
    return norm if higher_better else [1.0 - n for n in norm]


def _raw_metrics(ticker: str, closes: Sequence[float], budget: RiskBudget):
    """Deterministic per-stock metrics; returns None if history is insufficient."""
    fv = extract_features(closes)
    if fv is None:
        return None
    tech = technical_signal(closes)
    warmup = min(60, max(20, len(closes) // 4))
    if len(closes) <= warmup + 5:
        return None
    rb = run_risk_budgeted_backtest(closes, momentum_direction(20, 0.75), budget, warmup=warmup)
    # Fast deterministic screen: skip news AND fundamentals (no network per ticker).
    dec = agentic_decision(ticker, closes, budget, fetch_news=False, fundamentals=False)
    return {
        "fv": fv, "tech": tech,
        "sharpe": rb.performance.sharpe,
        "max_drawdown": rb.performance.max_drawdown,
        "suggested_size": dec.order.size,
        "direction": dec.order.action,
        "last_price": float(closes[-1]),
    }


def screen(
    tickers: Sequence[str],
    budget: RiskBudget,
    appetite: str,
    regime: Regime,
    load_closes: PriceLoader,
) -> List[StockScore]:
    """Score and rank ``tickers`` deterministically (no LLM/news calls)."""
    rows = []
    for t in tickers:
        try:
            closes = load_closes(t)
        except Exception:  # noqa: BLE001 - a bad fetch shouldn't kill the screen
            closes = None
        if not closes or len(closes) < 55:
            continue
        m = _raw_metrics(t, closes, budget)
        if m is None:
            continue
        comp = universe.get_company(t)
        rows.append((t, comp, m))

    if not rows:
        return []

    momentum = [m["fv"].ret_20d for _, _, m in rows]
    sharpe = [m["sharpe"] for _, _, m in rows]
    maxdd = [m["max_drawdown"] for _, _, m in rows]
    vol = [m["fv"].vol_20d * math.sqrt(252) for _, _, m in rows]
    strength = [m["tech"].strength for _, _, m in rows]

    n_mom = _rank01(momentum, higher_better=True)
    n_shp = _rank01(sharpe, higher_better=True)
    n_dd = _rank01(maxdd, higher_better=False)   # lower drawdown -> higher score
    n_vol = _rank01(vol, higher_better=False)    # lower vol -> higher score

    # Appetite weights: (momentum, sharpe, low-drawdown, low-vol, strength).
    weights = {
        "Conservative": (0.10, 0.25, 0.35, 0.25, 0.05),
        "Moderate": (0.25, 0.30, 0.25, 0.15, 0.05),
        "Aggressive": (0.45, 0.30, 0.05, 0.05, 0.15),
    }.get(appetite, (0.25, 0.30, 0.25, 0.15, 0.05))
    w_mom, w_shp, w_dd, w_vol, w_str = weights

    scores: List[StockScore] = []
    for i, (t, comp, m) in enumerate(rows):
        fit = (w_mom * n_mom[i] * regime.tilt + w_shp * n_shp[i]
               + w_dd * n_dd[i] + w_vol * n_vol[i] + w_str * strength[i])
        s = StockScore(
            ticker=t,
            name=comp.name if comp else t,
            sector=comp.sector if comp else (universe.sector_of(t) or "—"),
            currency=universe.currency_symbol(t),
            last_price=m["last_price"],
            ret_20d=m["fv"].ret_20d,
            vol_20d_ann=vol[i],
            sharpe=m["sharpe"],
            max_drawdown=m["max_drawdown"],
            suggested_size=m["suggested_size"],
            direction=m["direction"],
            fit_score=fit,
        )
        s.rationale = s.summary()
        scores.append(s)

    scores.sort(key=lambda x: x.fit_score, reverse=True)
    return scores


def _benchmark_for(ticker: str) -> str:
    return "^NSEI" if universe.region_of(ticker) == "IN" else "SPY"


def _default_peer_source(ticker: str, max_peers: int) -> List[str]:
    """Dynamic peers: Yahoo 'similar symbols' first, static sector map as fallback."""
    from agentic_finance import discovery
    peers = discovery.similar_symbols(ticker, limit=max_peers)
    if not peers:
        # offline / no result -> curated same-sector peers (prefer same market)
        comps = universe.peers(ticker)
        region = universe.region_of(ticker)
        same = [c for c in comps if c.region == region]
        comps = same if len(same) >= 3 else comps
        peers = [c.ticker for c in comps]
    return [p for p in peers if p.upper() != ticker.upper()][:max_peers]


def suggest(
    ticker: str,
    budget: RiskBudget,
    appetite: str,
    load_closes: PriceLoader,
    top_k: int = 6,
    deep: int = 3,
    deep_dive: bool = True,
    max_peers: int = 15,
    peer_source: Optional[Callable[[str, int], List[str]]] = None,
    enrich_names: bool = True,
) -> List[StockScore]:
    """Similar-stock suggestions ranked to appetite + regime (hybrid deep-dive).

    Candidates come from Yahoo's live "similar symbols" (any company worldwide),
    falling back to the curated sector map when offline. The top few are ranked by
    fit to the user's risk appetite and the current market regime; when
    ``deep_dive`` is true the top ``deep`` also get a full agentic pipeline run
    whose one-line view is attached as ``agentic_note``. Ideas to review, not advice.

    ``peer_source(ticker, max_peers) -> list[str]`` and ``enrich_names`` are
    injectable so tests can run fully offline.
    """
    source = peer_source or _default_peer_source
    try:
        peer_tickers = source(ticker, max_peers)
    except Exception:  # noqa: BLE001
        peer_tickers = []
    if not peer_tickers:
        return []

    try:
        bench = load_closes(_benchmark_for(ticker))
    except Exception:  # noqa: BLE001
        bench = None
    regime = market_regime(bench)

    ranked = screen(peer_tickers, budget, appetite, regime, load_closes)[:top_k]

    # Enrich human-readable names for anything not already in the static registry.
    if enrich_names:
        from agentic_finance import discovery
        for s in ranked:
            if s.name == s.ticker:
                s.name = discovery.company_name(s.ticker) or s.ticker

    if deep_dive and deep > 0:
        for s in ranked[:deep]:
            try:
                closes = load_closes(s.ticker)
                if closes and len(closes) >= 55:
                    dec = agentic_decision(s.ticker, closes, budget, fetch_news=True)
                    s.direction = dec.order.action
                    s.suggested_size = dec.order.size
                    s.agentic_note = (
                        f"Agentic view: {dec.pre.direction} "
                        f"({dec.pre.confidence * 100:.0f}% conf) → "
                        f"{dec.order.action.upper()} {dec.order.size * 100:.1f}%, "
                        f"bound by '{dec.order.binding_constraint}'."
                    )
            except Exception:  # noqa: BLE001 - deep-dive is best-effort
                s.agentic_note = None

    return ranked
