"""Reusable end-to-end agentic decision — the pipeline as one callable.

Extracted from the Streamlit UI so the *same* logic drives the Live-Decision tab
and the Suggestions engine's top-N deep-dive, and so it is unit-testable without
a UI. Given a ticker and its close-price history, it runs the full pipeline:

    technical + sentiment + news-reasoning agents -> fusion -> risk engine -> order

and returns every intermediate signal for display/audit. Price fetching is kept
out of here (the caller supplies ``closes``) so this stays deterministic and
network-free except for the optional news/LLM/fundamentals calls, which degrade
gracefully.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from agentic_finance.agents_v2 import (
    fetch_fundamentals,
    fundamental_signal,
    fuse_signals,
    llm_signal,
    sentiment_signal,
    technical_signal,
)
from agentic_finance.agents_v2.schemas import Signal
from agentic_finance.features import extract_features
from agentic_finance.news import get_headlines
from agentic_finance.rag.ingest import ensure_ingested, retrieve_filing_context
from agentic_finance.risk_engine import (
    CompliancePolicy,
    Holding,
    MarketRiskInputs,
    PortfolioLimits,
    PortfolioState,
    PreDecision,
    RiskBudget,
    RiskedOrder,
    apply_portfolio_constraints,
    assess_and_size,
    compliance_gate,
)
from agentic_finance.slm import get_chat_model, get_sentiment_model

__all__ = ["AgenticDecision", "agentic_decision", "atr_proxy"]


@dataclass(frozen=True)
class AgenticDecision:
    """All outputs of one agentic decision (signals + fused view + sized order)."""

    ticker: str
    technical: Signal
    sentiment: Optional[Signal]
    fundamental: Optional[Signal]
    news: Optional[Signal]
    pre: PreDecision
    order: RiskedOrder
    headlines: List[str]
    filings: Tuple[str, ...] = ()  # SEC filing excerpts fed to the news agent (RAG)
    compliance_violations: Tuple[str, ...] = ()  # what the compliance gate flagged


def atr_proxy(prices: Sequence[float], period: int = 14) -> float:
    """Close-to-close ATR proxy: mean absolute change over the last ``period``."""
    if len(prices) < 2:
        return 0.0
    changes = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    window = changes[-period:]
    return sum(window) / len(window) if window else 0.0


def _default_policy() -> CompliancePolicy:
    """Compliance policy from the environment.

    ``AGENTIC_RESTRICTED_TICKERS`` (comma-separated) vetoes named instruments;
    a rationale is always required (the operational face of explainability).
    Position size is left to the user's ``RiskBudget`` — the policy only clamps
    when an institution configures a stricter limit of its own.
    """
    restricted = tuple(
        t.strip().upper()
        for t in os.environ.get("AGENTIC_RESTRICTED_TICKERS", "").split(",")
        if t.strip()
    )
    return CompliancePolicy(restricted=restricted, max_position=1.0,
                            require_rationale=True)


def agentic_decision(
    ticker: str,
    closes: Sequence[float],
    budget: RiskBudget,
    fetch_news: bool = True,
    equity: float = 100_000.0,
    fundamentals: bool = True,
    holdings: Sequence[Holding] = (),
    limits: Optional[PortfolioLimits] = None,
    policy: Optional[CompliancePolicy] = None,
) -> AgenticDecision:
    """Run the full agentic pipeline for one instrument and return all outputs.

    ``fetch_news=False`` skips the news/LLM path entirely (fast, offline) — the
    decision then rests on the deterministic technical agent alone, which is what
    the suggestion screener uses before the top-N deep-dive. ``fundamentals=False``
    likewise skips the fundamental agent's Yahoo Finance lookup.

    After per-asset risk sizing the order passes through the book-level
    constraints (when ``holdings`` are supplied) and then the compliance gate —
    the complete institutional ladder: size -> portfolio -> compliance.
    """
    closes = [float(x) for x in closes]
    last_price = closes[-1]
    fv = extract_features(closes)

    # 1. Technical agent (deterministic, always available).
    tech = technical_signal(closes)
    signals: List[Signal] = [tech]

    # 2. Fundamental agent (financial-health ratios; degrades to None on outage).
    fund: Optional[Signal] = None
    if fundamentals:
        metrics = fetch_fundamentals(ticker)
        candidate = fundamental_signal(metrics)
        if candidate.confidence > 0.0:  # skip the no-data flat signal
            fund = candidate
            signals.append(fund)

    sent: Optional[Signal] = None
    news_sig: Optional[Signal] = None
    headlines: List[str] = []
    filings: List[str] = []

    if fetch_news:
        try:
            headlines = get_headlines(f"{ticker} stock", limit=8)
        except Exception:  # noqa: BLE001 - news failure must not break a decision
            headlines = []

        # RAG: point-in-time SEC-filing excerpts for the news agent's context.
        # Idempotent (first call per ticker ingests; later calls hit the local
        # index) and fully graceful — a ticker with no EDGAR filings (e.g. a
        # purely domestic Indian name) just yields no excerpts.
        if os.environ.get("AGENTIC_RAG", "on").lower() not in ("off", "0", "false"):
            try:
                ensure_ingested(ticker)
                filings = retrieve_filing_context(
                    ticker, f"{ticker} business outlook, risks, and revenue growth",
                    k=3,
                )
            except Exception:  # noqa: BLE001 - RAG failure must not break a decision
                filings = []

        if headlines:
            sent = sentiment_signal(headlines, get_sentiment_model())
            signals.append(sent)

        price_summary = (
            f"{ticker} last price {last_price:.2f}; 20-day return "
            f"{fv.ret_20d * 100:.1f}%; RSI14 {fv.rsi_14:.0f}; "
            f"price/SMA20 {fv.price_sma20_ratio * 100:+.1f}%; "
            f"realized vol(20d) {fv.vol_20d * 100:.1f}%."
            if fv else f"{ticker} last price {last_price:.2f}."
        )
        context = price_summary + (
            "\n\nHeadlines:\n" + "\n".join(headlines) if headlines else ""
        ) + (
            "\n\nSEC filing excerpts (point-in-time):\n" + "\n---\n".join(filings)
            if filings else ""
        )
        try:
            news_sig = llm_signal(context, get_chat_model(), source="news")
            signals.append(news_sig)
        except Exception:  # noqa: BLE001 - model failure -> skip news signal
            news_sig = None

    # 2. Fusion -> PreDecision.
    pre = fuse_signals(signals)

    # 3. Risk engine -> enforced sized order.
    returns = tuple(
        closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))
    )[-30:]
    market = MarketRiskInputs(price=last_price, atr=atr_proxy(closes), returns=returns)
    portfolio = PortfolioState(equity=equity, peak_equity=equity, remaining_budget=1.0)
    order = assess_and_size(pre, market, portfolio, budget)

    # 4. Book-level limits (only bind when an actual book is supplied) ...
    if holdings:
        from agentic_finance import universe
        sector = universe.sector_of(ticker) or ""
        order = apply_portfolio_constraints(
            order, ticker, sector, holdings, limits or PortfolioLimits(),
            candidate_returns=returns,
        )

    # 5. ... then the compliance gate — the final, non-negotiable check.
    comp = compliance_gate(order, ticker, policy or _default_policy())

    return AgenticDecision(
        ticker=ticker, technical=tech, sentiment=sent, fundamental=fund,
        news=news_sig, pre=pre, order=comp.order, headlines=headlines,
        filings=tuple(filings), compliance_violations=comp.violations,
    )
