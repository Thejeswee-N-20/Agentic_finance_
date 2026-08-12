"""Point-in-time, news-driven backtest at weekly cadence.

The experiment that tests whether the *agentic* signal (technical + sentiment +
LLM reasoning over as-of news) improves on a price-only signal under the SAME risk
engine. Three strategies over the same window/cadence:

  - Buy & Hold
  - Risk-budgeted (price-only)  : risk engine + technical(momentum) signal
  - Risk-budgeted (agentic)     : risk engine + fusion(technical, sentiment, LLM-news)

Only the SIGNAL differs between the two risk-budgeted strategies, isolating the
contribution of the news/LLM agents.

Cost: one Tavily search + one LLM call per rebalance (~weekly). Leakage controls:
prices are used point-in-time, news is restricted to a lookback window ending at
the rebalance date, and the LLM is instructed to reason only from supplied context.
A residual parametric-recall caveat remains (see the dissertation, Ch. 7).

    REPRO_TICKER=NVDA REPRO_START=2024-06-01 REPRO_END=2025-06-01 python run_news_backtest_demo.py
"""
import os
from datetime import timedelta
from typing import List

import pandas as pd
import yfinance as yf

import agentic_finance  # loads .env
from agentic_finance.agents_v2 import fuse_signals, llm_signal, sentiment_signal, technical_signal
from agentic_finance.backtest.metrics import (cumulative_return, max_drawdown,
                                            sharpe_ratio, sortino_ratio)
from agentic_finance.features import extract_features
from agentic_finance.news import get_news_provider, news_provider
from agentic_finance.risk_engine import (MarketRiskInputs, PortfolioState, RiskBudget,
                                       assess_and_size)
from agentic_finance.slm import get_chat_model, get_sentiment_model, sentiment_backend, slm_provider

# Fusion weights: the proven technical (momentum) factor leads; news + sentiment
# tilt rather than override. (These weights reproduce the reported headline results.)
FUSION_WEIGHTS = {"technical": 2.5, "sentiment": 1.0, "news": 1.0}

TICKER = os.environ.get("REPRO_TICKER", "NVDA")
START = os.environ.get("REPRO_START", "2024-06-01")
END = os.environ.get("REPRO_END", "2025-06-01")
STEP = int(os.environ.get("REPRO_STEP", "5"))      # rebalance cadence (trading days)
WARMUP = 60
NEWS_LOOKBACK = 10


def _weight(order):
    if order.action == "buy":
        return order.size
    if order.action == "sell":
        return -order.size
    return 0.0


def _atr(closes, i, period=14):
    lo = max(1, i - period + 1)
    diffs = [abs(closes[j] - closes[j - 1]) for j in range(lo, i + 1)]
    return sum(diffs) / len(diffs) if diffs else 0.0


def main() -> None:
    print(f"=== News-driven backtest: {TICKER} {START}->{END} (every {STEP}d) ===")
    print(f"SLM provider: {slm_provider()} | news provider: {news_provider()}")
    df = yf.download(TICKER, start=START, end=END, progress=False, auto_adjust=True)
    closes = [float(x) for x in df["Close"].iloc[:, 0].tolist()]
    dates = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df.index)]
    n = len(closes)
    if n < WARMUP + STEP * 3:
        raise SystemExit(f"insufficient data ({n} bars)")

    provider = get_news_provider()
    chat = get_chat_model()
    sent_model = get_sentiment_model()  # configured backend (gemini | heuristic)
    print(f"sentiment backend: {sentiment_backend()} | chat model: {chat.name}")
    budget = RiskBudget(target_vol=0.15, max_position=0.5, cvar_limit=0.10,
                        max_drawdown=0.20, min_confidence=0.55)

    rebalances = list(range(WARMUP, n - 1, STEP))
    # Per-day weights for each strategy (forward-filled from each rebalance).
    w_mom = [0.0] * n
    w_ag = [0.0] * n
    eq_mom = eq_ag = 100_000.0
    peak_mom = peak_ag = 100_000.0

    print(f"Rebalances: {len(rebalances)} (each = 1 Tavily + 1 {slm_provider()} call)")
    for r, i in enumerate(rebalances):
        history = closes[: i + 1]
        d = dates[i]
        fv = extract_features(history)
        price, atr = closes[i], _atr(closes, i)
        rets = tuple(closes[k] / closes[k - 1] - 1.0 for k in range(1, i + 1))[-30:]
        market = MarketRiskInputs(price=price, atr=atr, returns=rets)

        tech = technical_signal(history)

        # Agentic signal: technical + as-of sentiment + LLM news reasoning.
        start_news = (pd.Timestamp(d) - timedelta(days=NEWS_LOOKBACK)).strftime("%Y-%m-%d")
        try:
            articles = provider.fetch(f"{TICKER} stock", start=start_news, end=d, limit=6)
        except Exception:
            articles = []
        headlines = [a.title for a in articles]
        agent_signals = [tech]
        if headlines:
            agent_signals.append(sentiment_signal(headlines, sent_model))
            ctx = (f"As of {d}, {TICKER} at {price:.2f}; 20d return "
                   f"{(fv.ret_20d*100 if fv else 0):.1f}%; RSI {fv.rsi_14:.0f}.\n"
                   "Headlines:\n" + "\n".join(headlines))
            agent_signals.append(llm_signal(ctx, chat, source="news"))

        pre_mom = fuse_signals([tech])
        pre_ag = fuse_signals(agent_signals, weights=FUSION_WEIGHTS)

        order_mom = assess_and_size(pre_mom, market,
                                    PortfolioState(eq_mom, peak_mom), budget)
        order_ag = assess_and_size(pre_ag, market,
                                   PortfolioState(eq_ag, peak_ag), budget)

        end_i = min(i + STEP, n - 1)
        for day in range(i, end_i):
            w_mom[day] = _weight(order_mom)
            w_ag[day] = _weight(order_ag)
            dr = closes[day + 1] / closes[day] - 1.0
            eq_mom *= 1 + w_mom[day] * dr
            eq_ag *= 1 + w_ag[day] * dr
            peak_mom, peak_ag = max(peak_mom, eq_mom), max(peak_ag, eq_ag)
        if r % 10 == 0:
            print(f"  [{r+1}/{len(rebalances)}] {d}  mom={pre_mom.direction}"
                  f"({pre_mom.confidence:.2f}) agentic={pre_ag.direction}"
                  f"({pre_ag.confidence:.2f}) news={len(headlines)}")

    # Build daily return series over the traded span for metrics.
    span = range(WARMUP, min(rebalances[-1] + STEP, n - 1))
    def series(weights):
        return [weights[day] * (closes[day + 1] / closes[day] - 1.0) for day in span]
    bh = [closes[day + 1] / closes[day] - 1.0 for day in span]
    rb_mom, rb_ag = series(w_mom), series(w_ag)

    print(f"\n{'Strategy':<28}{'CumRet':>9}{'Sharpe':>8}{'Sortino':>9}{'MaxDD':>8}")
    print("-" * 62)
    for name, r in [("Buy & Hold", bh),
                    ("Risk-budgeted (price-only)", rb_mom),
                    ("Risk-budgeted (agentic+news)", rb_ag)]:
        print(f"{name:<28}{cumulative_return(r)*100:>8.2f}%{sharpe_ratio(r):>8.2f}"
              f"{sortino_ratio(r):>9.2f}{max_drawdown(r)*100:>7.2f}%")


if __name__ == "__main__":
    main()
