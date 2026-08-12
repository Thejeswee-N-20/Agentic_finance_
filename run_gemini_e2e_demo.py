"""End-to-end live decision using the configured SLM provider (Gemini by default).

Full agentic pipeline on a single, live decision:
  real prices ─► technical agent
  live headlines ─► Gemini sentiment agent + Gemini LLM reasoning agent
        │
        └─► signal fusion ─► PreDecision ─► quantitative risk engine ─► sized order
                                                              └─► XAI explanation

Exercises the Gemini backend across sentiment + reasoning AND the deterministic
risk engine. Honest about scope: this is a *live* decision (news is "now"), not a
point-in-time backtest. Needs GOOGLE_API_KEY (in .env); flip AGENTIC_SLM_PROVIDER
=ollama to run the same flow locally.
"""
import os
from typing import List

import yfinance as yf

import agentic_finance  # loads .env (GOOGLE_API_KEY, AGENTIC_*)
from agentic_finance.agents_v2 import fuse_signals, llm_signal, sentiment_signal, technical_signal
from agentic_finance.features import extract_features
from agentic_finance.news import get_headlines, news_provider
from agentic_finance.risk_engine import (
    MarketRiskInputs,
    PortfolioState,
    RiskBudget,
    assess_and_size,
)
from agentic_finance.slm import get_chat_model, slm_provider
from agentic_finance.slm.gemini import GeminiSentimentModel
from agentic_finance.xai import explain_order

TICKER = os.environ.get("REPRO_TICKER", "NVDA")
# Optional point-in-time window for news (YYYY-MM-DD); Tavily honors it.
NEWS_START = os.environ.get("REPRO_START")
NEWS_END = os.environ.get("REPRO_END")


def _closes(ticker: str) -> List[float]:
    df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        raise SystemExit(f"no price data for {ticker}")
    return [float(x) for x in df["Close"].iloc[:, 0].tolist()]


def _headlines(ticker: str, limit: int = 8) -> List[str]:
    try:
        return get_headlines(f"{ticker} stock", start=NEWS_START, end=NEWS_END, limit=limit)
    except Exception as exc:
        print(f"[news] fetch failed ({exc}); continuing without headlines")
        return []


def _atr_proxy(prices: List[float], period: int = 14) -> float:
    changes = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    window = changes[-period:]
    return sum(window) / len(window) if window else 0.0


def main() -> None:
    print(f"=== Gemini end-to-end decision: {TICKER} ===")
    window = f" (news {NEWS_START}..{NEWS_END})" if NEWS_START or NEWS_END else ""
    print(f"SLM provider: {slm_provider()} | news provider: {news_provider()}{window}\n")

    closes = _closes(TICKER)
    fv = extract_features(closes)
    price = closes[-1]
    returns = tuple(closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)))[-30:]

    # 1. Technical agent (deterministic, from prices).
    tech = technical_signal(closes)

    # 2. Live headlines -> Gemini sentiment + Gemini LLM reasoning.
    headlines = _headlines(TICKER)
    print(f"Pulled {len(headlines)} headlines.")
    signals = [tech]

    if headlines:
        sent = sentiment_signal(headlines, GeminiSentimentModel())
        signals.append(sent)
    else:
        sent = None

    price_summary = (
        f"{TICKER} last price {price:.2f}; 20-day return {fv.ret_20d*100:.1f}%; "
        f"RSI14 {fv.rsi_14:.0f}; price/SMA20 {fv.price_sma20_ratio*100:+.1f}%; "
        f"realized vol(20d) {fv.vol_20d*100:.1f}%."
    ) if fv else f"{TICKER} last price {price:.2f}."
    context = price_summary + ("\n\nHeadlines:\n" + "\n".join(headlines) if headlines else "")
    news_sig = llm_signal(context, get_chat_model(), source="news")
    signals.append(news_sig)

    # 3. Show each agent's typed signal.
    print("\n--- Agent signals ---")
    for s in signals:
        print(f"  [{s.source:<10}] {s.direction:<5} strength={s.strength:.2f} "
              f"conf={s.confidence:.2f}  {s.rationale[:90]}")

    # 4. Fusion -> PreDecision.
    pre = fuse_signals(signals)
    print(f"\nFused PreDecision: {pre.direction} (confidence {pre.confidence:.2f})")

    # 5. Risk engine -> sized order.
    budget = RiskBudget(target_vol=0.15, max_position=0.25, cvar_limit=0.10,
                        max_drawdown=0.20, min_confidence=0.55)
    market = MarketRiskInputs(price=price, atr=_atr_proxy(closes), returns=returns)
    portfolio = PortfolioState(equity=100_000.0, peak_equity=100_000.0, remaining_budget=1.0)
    order = assess_and_size(pre, market, portfolio, budget)

    # 6. Decision + XAI explanation.
    print("\n--- Final risk-budgeted decision ---")
    print(f"  action={order.action}  size={order.size*100:.2f}% of equity  "
          f"stop={order.stop_price:.2f}")
    print(f"  binding constraint: {order.binding_constraint}")
    print(f"  XAI: {explain_order(order)}")


if __name__ == "__main__":
    main()
