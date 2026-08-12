"""Point-in-time agentic decision (as-of a historical date).

Unlike run_gemini_e2e_demo.py (which uses "now" news), this makes a decision as it
would have been made on a PAST date, using only information available up to that
date:
  - prices truncated at the decision date (no future bars),
  - news restricted to a lookback window ENDING at the decision date (Tavily date
    range — point-in-time).

This is the building block for a news-driven *backtest*: one honest as-of
decision. Needs Tavily (TAVILY_NEWS_API_KEY) + the SLM provider (Gemini/Ollama).

    REPRO_TICKER=NVDA REPRO_ASOF=2024-05-24 python run_pit_decision_demo.py
"""
import os
from datetime import datetime, timedelta
from typing import List

import yfinance as yf

import agentic_finance  # loads .env
from agentic_finance.agents_v2 import fuse_signals, llm_signal, sentiment_signal, technical_signal
from agentic_finance.features import extract_features
from agentic_finance.news import get_news_provider, news_provider
from agentic_finance.risk_engine import MarketRiskInputs, PortfolioState, RiskBudget, assess_and_size
from agentic_finance.slm import get_chat_model, slm_provider
from agentic_finance.slm.gemini import GeminiSentimentModel
from agentic_finance.xai import explain_order

TICKER = os.environ.get("REPRO_TICKER", "NVDA")
ASOF = os.environ.get("REPRO_ASOF", "2024-05-24")
LOOKBACK_DAYS = int(os.environ.get("REPRO_NEWS_LOOKBACK", "10"))


def _closes_until(ticker: str, asof: str) -> List[float]:
    # Pull a generous window ending at the as-of date (inclusive-ish).
    end = (datetime.fromisoformat(asof) + timedelta(days=1)).strftime("%Y-%m-%d")
    start = (datetime.fromisoformat(asof) - timedelta(days=400)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        raise SystemExit(f"no price data for {ticker} up to {asof}")
    return [float(x) for x in df["Close"].iloc[:, 0].tolist()]


def main() -> None:
    print(f"=== Point-in-time decision: {TICKER} as of {ASOF} ===")
    print(f"SLM provider: {slm_provider()} | news provider: {news_provider()}\n")

    closes = _closes_until(TICKER, ASOF)           # no future bars beyond ASOF
    fv = extract_features(closes)
    price = closes[-1]
    returns = tuple(closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)))[-30:]

    # News strictly within [ASOF - lookback, ASOF] — point-in-time.
    news_start = (datetime.fromisoformat(ASOF) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    provider = get_news_provider()
    try:
        articles = provider.fetch(f"{TICKER} stock", start=news_start, end=ASOF, limit=8)
    except Exception as exc:
        print(f"[news] fetch failed: {exc}")
        articles = []
    headlines = [a.title for a in articles]
    print(f"As-of news ({news_start}..{ASOF}): {len(headlines)} headlines")
    for a in articles[:5]:
        print(f"  - [{a.published[:16]}] {a.title[:80]}")

    signals = [technical_signal(closes)]
    if headlines:
        signals.append(sentiment_signal(headlines, GeminiSentimentModel()))
        context = (f"As of {ASOF}, {TICKER} at {price:.2f}; 20d return {fv.ret_20d*100:.1f}%; "
                   f"RSI14 {fv.rsi_14:.0f}.\nHeadlines:\n" + "\n".join(headlines))
        signals.append(llm_signal(context, get_chat_model(), source="news"))

    print("\n--- Agent signals ---")
    for s in signals:
        print(f"  [{s.source:<10}] {s.direction:<5} strength={s.strength:.2f} conf={s.confidence:.2f}")

    pre = fuse_signals(signals)
    budget = RiskBudget(target_vol=0.15, max_position=0.25, cvar_limit=0.10,
                        max_drawdown=0.20, min_confidence=0.55)
    market = MarketRiskInputs(
        price=price,
        atr=sum(abs(closes[i] - closes[i - 1]) for i in range(len(closes) - 14, len(closes))) / 14,
        returns=returns,
    )
    order = assess_and_size(pre, market, PortfolioState(100_000.0, 100_000.0), budget)

    print(f"\nFused: {pre.direction} (conf {pre.confidence:.2f})")
    print(f"Decision: {order.action} size={order.size*100:.2f}% stop={order.stop_price:.2f} "
          f"[binding={order.binding_constraint}]")
    print(f"XAI: {explain_order(order)}")


if __name__ == "__main__":
    main()
