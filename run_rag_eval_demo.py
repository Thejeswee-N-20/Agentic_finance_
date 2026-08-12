"""RAG ablation: does point-in-time filing context improve the agentic signal?

Same protocol as run_news_backtest_demo.py (weekly cadence, leakage-controlled,
identical risk engine), with ONE extra strategy: the agentic signal whose LLM
context additionally contains SEC-filing excerpts retrieved with the as-of date
filter (filing_date <= decision date — no look-ahead). Comparing the two
agentic strategies isolates the marginal contribution of RAG:

  - Buy & Hold
  - Risk-budgeted (price-only)         : technical(momentum) only
  - Risk-budgeted (agentic+news)       : fusion(technical, sentiment, LLM-news)
  - Risk-budgeted (agentic+news+RAG)   : same, LLM context + as-of filing excerpts

Requires the ticker's filings in the local index first:
    python -c "from agentic_finance.rag.ingest import ingest_ticker; ingest_ticker('NVDA', max_filings=10)"
Cost: one Tavily search + TWO LLM calls per rebalance (news signal with and
without RAG context).

    REPRO_TICKER=NVDA REPRO_START=2024-06-01 REPRO_END=2025-06-01 python run_rag_eval_demo.py
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
from agentic_finance.rag.ingest import retrieve_filing_context
from agentic_finance.risk_engine import (MarketRiskInputs, PortfolioState, RiskBudget,
                                         assess_and_size)
from agentic_finance.slm import get_chat_model, get_sentiment_model, sentiment_backend, slm_provider

FUSION_WEIGHTS = {"technical": 2.5, "sentiment": 1.0, "news": 1.0}

TICKER = os.environ.get("REPRO_TICKER", "NVDA")
START = os.environ.get("REPRO_START", "2024-06-01")
END = os.environ.get("REPRO_END", "2025-06-01")
STEP = int(os.environ.get("REPRO_STEP", "5"))
WARMUP = 60
NEWS_LOOKBACK = 10
RAG_K = 2  # filing excerpts per decision


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
    print(f"=== RAG ablation backtest: {TICKER} {START}->{END} (every {STEP}d) ===")
    print(f"SLM provider: {slm_provider()} | news provider: {news_provider()}")
    df = yf.download(TICKER, start=START, end=END, progress=False, auto_adjust=True)
    closes = [float(x) for x in df["Close"].iloc[:, 0].tolist()]
    dates = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df.index)]
    n = len(closes)
    if n < WARMUP + STEP * 3:
        raise SystemExit(f"insufficient data ({n} bars)")

    provider = get_news_provider()
    chat = get_chat_model()
    sent_model = get_sentiment_model()
    print(f"sentiment backend: {sentiment_backend()} | chat model: {chat.name}")
    budget = RiskBudget(target_vol=0.15, max_position=0.5, cvar_limit=0.10,
                        max_drawdown=0.20, min_confidence=0.55)

    rebalances = list(range(WARMUP, n - 1, STEP))
    w_mom = [0.0] * n
    w_ag = [0.0] * n
    w_rag = [0.0] * n
    eq = {k: 100_000.0 for k in ("mom", "ag", "rag")}
    peak = dict(eq)
    rag_hits = 0

    print(f"Rebalances: {len(rebalances)} (each = 1 Tavily + 2 {slm_provider()} calls)")
    for r, i in enumerate(rebalances):
        history = closes[: i + 1]
        d = dates[i]
        as_of = int(d.replace("-", ""))
        fv = extract_features(history)
        price, atr = closes[i], _atr(closes, i)
        rets = tuple(closes[k] / closes[k - 1] - 1.0 for k in range(1, i + 1))[-30:]
        market = MarketRiskInputs(price=price, atr=atr, returns=rets)

        tech = technical_signal(history)

        start_news = (pd.Timestamp(d) - timedelta(days=NEWS_LOOKBACK)).strftime("%Y-%m-%d")
        try:
            articles = provider.fetch(f"{TICKER} stock", start=start_news, end=d, limit=6)
        except Exception:
            articles = []
        headlines = [a.title for a in articles]

        # Point-in-time filing excerpts (as-of filtered — no look-ahead).
        excerpts = retrieve_filing_context(
            TICKER, f"{TICKER} business outlook, risks, and revenue growth",
            as_of=as_of, k=RAG_K,
        )
        if excerpts:
            rag_hits += 1

        signals_ag: List = [tech]
        signals_rag: List = [tech]
        if headlines:
            sent = sentiment_signal(headlines, sent_model)
            signals_ag.append(sent)
            signals_rag.append(sent)
            base_ctx = (f"As of {d}, {TICKER} at {price:.2f}; 20d return "
                        f"{(fv.ret_20d*100 if fv else 0):.1f}%; RSI {fv.rsi_14:.0f}.\n"
                        "Headlines:\n" + "\n".join(headlines))
            signals_ag.append(llm_signal(base_ctx, chat, source="news"))
            rag_ctx = base_ctx + (
                "\n\nSEC filing excerpts (point-in-time, filed on or before "
                f"{d}):\n" + "\n---\n".join(excerpts) if excerpts else ""
            )
            signals_rag.append(llm_signal(rag_ctx, chat, source="news"))

        pre_mom = fuse_signals([tech])
        pre_ag = fuse_signals(signals_ag, weights=FUSION_WEIGHTS)
        pre_rag = fuse_signals(signals_rag, weights=FUSION_WEIGHTS)

        order_mom = assess_and_size(pre_mom, market, PortfolioState(eq["mom"], peak["mom"]), budget)
        order_ag = assess_and_size(pre_ag, market, PortfolioState(eq["ag"], peak["ag"]), budget)
        order_rag = assess_and_size(pre_rag, market, PortfolioState(eq["rag"], peak["rag"]), budget)

        end_i = min(i + STEP, n - 1)
        for day in range(i, end_i):
            w_mom[day] = _weight(order_mom)
            w_ag[day] = _weight(order_ag)
            w_rag[day] = _weight(order_rag)
            dr = closes[day + 1] / closes[day] - 1.0
            eq["mom"] *= 1 + w_mom[day] * dr
            eq["ag"] *= 1 + w_ag[day] * dr
            eq["rag"] *= 1 + w_rag[day] * dr
            for k in eq:
                peak[k] = max(peak[k], eq[k])
        if r % 10 == 0:
            print(f"  [{r+1}/{len(rebalances)}] {d}  agentic={pre_ag.direction}"
                  f"({pre_ag.confidence:.2f}) +RAG={pre_rag.direction}"
                  f"({pre_rag.confidence:.2f}) news={len(headlines)} filings={len(excerpts)}")

    print(f"\nDecisions with filing context available: {rag_hits}/{len(rebalances)}")
    span = range(WARMUP, min(rebalances[-1] + STEP, n - 1))

    def series(weights):
        return [weights[day] * (closes[day + 1] / closes[day] - 1.0) for day in span]

    bh = [closes[day + 1] / closes[day] - 1.0 for day in span]
    print(f"\n{'Strategy':<32}{'CumRet':>9}{'Sharpe':>8}{'Sortino':>9}{'MaxDD':>8}")
    print("-" * 66)
    for name, rr in [("Buy & Hold", bh),
                     ("Risk-budgeted (price-only)", series(w_mom)),
                     ("Risk-budgeted (agentic+news)", series(w_ag)),
                     ("Risk-budgeted (agentic+news+RAG)", series(w_rag))]:
        print(f"{name:<32}{cumulative_return(rr)*100:>8.2f}%{sharpe_ratio(rr):>8.2f}"
              f"{sortino_ratio(rr):>9.2f}{max_drawdown(rr)*100:>7.2f}%")


if __name__ == "__main__":
    main()
