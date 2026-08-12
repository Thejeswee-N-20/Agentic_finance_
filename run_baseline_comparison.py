"""Head-to-head: Agentic Finance vs the TradingAgents baseline.

Runs OUR full pipeline as-of a historical date (point-in-time prices, news,
fundamentals, and filing excerpts) and prints the decision artefacts plus
measured latency and LLM-call count. Run the TradingAgents baseline separately
on the same ticker/date (its repo, same Gemini model) and compare:

    # ours (this repo)
    REPRO_TICKER=NVDA REPRO_ASOF=2024-05-10 python run_baseline_comparison.py
    # baseline (sibling repo, measured with /usr/bin/time)
    cd ../trading_agents_repo && REPRO_TICKER=NVDA REPRO_DATE=2024-05-10 python run_repro.py

The comparison axes (reported in the dissertation):
decision expressiveness (sized order + stop + tail-risk vs categorical
rating), explainability artefacts, enforced risk, latency, and LLM calls.
"""
import os
import time
from datetime import datetime, timedelta
from typing import List

import yfinance as yf

import agentic_finance  # loads .env
from agentic_finance.agents_v2 import (
    fetch_fundamentals,
    fundamental_signal,
    fuse_signals,
    llm_signal,
    sentiment_signal,
    technical_signal,
)
from agentic_finance.features import extract_features
from agentic_finance.news import get_news_provider, news_provider
from agentic_finance.rag.ingest import retrieve_filing_context
from agentic_finance.risk_engine import (
    MarketRiskInputs,
    PortfolioState,
    RiskBudget,
    assess_and_size,
)
from agentic_finance.slm import get_chat_model, slm_provider
from agentic_finance.slm.gemini import GeminiSentimentModel
from agentic_finance.xai import decision_trace, explain_order

TICKER = os.environ.get("REPRO_TICKER", "NVDA")
ASOF = os.environ.get("REPRO_ASOF", "2024-05-10")
LOOKBACK_DAYS = 10


def _closes_until(ticker: str, asof: str) -> List[float]:
    end = (datetime.fromisoformat(asof) + timedelta(days=1)).strftime("%Y-%m-%d")
    start = (datetime.fromisoformat(asof) - timedelta(days=400)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        raise SystemExit(f"no price data for {ticker} up to {asof}")
    return [float(x) for x in df["Close"].iloc[:, 0].tolist()]


def main() -> None:
    print(f"=== Agentic Finance as-of decision: {TICKER} @ {ASOF} ===")
    print(f"SLM provider: {slm_provider()} | news provider: {news_provider()}")
    t0 = time.perf_counter()
    llm_calls = 0

    closes = _closes_until(TICKER, ASOF)
    fv = extract_features(closes)
    price = closes[-1]
    returns = tuple(closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)))[-30:]

    signals = [technical_signal(closes)]

    fund = fundamental_signal(fetch_fundamentals(TICKER))
    if fund.confidence > 0:
        signals.append(fund)  # note: ratios are current-day (yfinance limitation)

    news_start = (datetime.fromisoformat(ASOF) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    try:
        articles = get_news_provider().fetch(f"{TICKER} stock", start=news_start,
                                             end=ASOF, limit=8)
        headlines = [a.title for a in articles]
    except Exception:
        headlines = []
    if headlines:
        signals.append(sentiment_signal(headlines, GeminiSentimentModel()))
        llm_calls += 1

    excerpts = retrieve_filing_context(
        TICKER, f"{TICKER} business outlook, risks, and revenue growth",
        as_of=int(ASOF.replace("-", "")), k=2,
    )
    ctx = (f"As of {ASOF}: {TICKER} at {price:.2f}; 20d return "
           f"{(fv.ret_20d * 100 if fv else 0):.1f}%; RSI {fv.rsi_14:.0f}.\n")
    if headlines:
        ctx += "Headlines:\n" + "\n".join(headlines)
    if excerpts:
        ctx += "\n\nSEC filing excerpts (point-in-time):\n" + "\n---\n".join(excerpts)
    signals.append(llm_signal(ctx, get_chat_model(), source="news"))
    llm_calls += 1

    pre = fuse_signals(signals)
    budget = RiskBudget(target_vol=0.15, max_position=0.25, cvar_limit=0.10,
                        max_drawdown=0.20, min_confidence=0.55)
    atr = sum(abs(closes[i] - closes[i - 1]) for i in range(len(closes) - 14,
              len(closes))) / 14
    market = MarketRiskInputs(price=price, atr=atr, returns=returns)
    order = assess_and_size(pre, market, PortfolioState(100_000.0, 100_000.0), budget)
    elapsed = time.perf_counter() - t0

    print("\n" + decision_trace(signals, pre, order))
    print(f"\nXAI: {explain_order(order)}")
    print(f"Filing excerpts used: {len(excerpts)} | headlines: {len(headlines)}")
    print("\n--- Measured (ours) ---")
    print(f"latency: {elapsed:.1f}s | LLM calls: {llm_calls} "
          f"(+{len(signals) - llm_calls} deterministic agents) | "
          f"output: sized order (action/size/stop/VaR/CVaR + binding constraint)")
    print("\nBaseline for the same ticker/date (run in ../trading_agents_repo):")
    print(f"  REPRO_TICKER={TICKER} REPRO_DATE={ASOF} python run_repro.py")
    print("  -> categorical rating (Buy/Hold/Sell), ~13+ LLM calls "
          "(4 analysts, 2x researcher debate, trader, 3x risk debate, 2 managers),")
    print("     no enforced size/stop, no feature-level explanation.")


if __name__ == "__main__":
    main()
