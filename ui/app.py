"""Agentic Finance — interactive dashboard.

A user-friendly Streamlit front-end over the agentic_finance framework. It drives
the *real* package code (no reimplementation): typed-signal agents -> fusion ->
quantitative risk engine -> explainability, plus the leakage-aware backtest and
the SHAP surrogate.

Run from the repo root:
    streamlit run ui/app.py
or:
    python run_ui.py
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Importing the package loads .env (API keys / AGENTIC_*) and, on Windows behind a
# TLS-scanning AV/proxy, installs the combined CA bundle so yfinance/Tavily work.
import agentic_finance  # noqa: F401

from agentic_finance import discovery, universe
from agentic_finance.agents_v2.pipeline import agent_direction_signal, technical_provider
from agentic_finance.backtest import (
    buy_and_hold,
    equity_curve,
    macd_strategy,
    momentum_direction,
    rsi_strategy,
    run_backtest,
    run_risk_budgeted_backtest,
    sma_crossover,
)
from agentic_finance.decision import agentic_decision
from agentic_finance.features import build_feature_matrix, extract_features
from agentic_finance.news import news_provider
from agentic_finance.risk_engine import RiskBudget
from agentic_finance.screening import RISK_PRESETS, market_regime, suggest
from agentic_finance.slm import sentiment_backend, slm_provider
from agentic_finance.xai import attribute_constraints, explain_order

# --------------------------------------------------------------------------- #
# Page config + light styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Agentic Finance",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
      .badge {display:inline-block; padding:3px 12px; border-radius:999px;
              font-weight:700; font-size:0.85rem; letter-spacing:.3px;}
      .badge-long {background:#d6ebd9; color:#1a6b32;}
      .badge-short {background:#f5d6cf; color:#9b2616;}
      .badge-flat {background:#e8e8e8; color:#555;}
      .agent-card {border:1px solid #e6e6e6; border-radius:12px; padding:14px 16px;
                   background:#fafafa; height:100%;}
      .agent-name {font-weight:700; font-size:0.95rem; color:#333; margin-bottom:6px;}
      .muted {color:#777; font-size:0.85rem;}
      .big-action {font-size:2.2rem; font-weight:800; line-height:1;}
    </style>
    """,
    unsafe_allow_html=True,
)

DIR_COLORS = {"long": "#1a6b32", "short": "#9b2616", "flat": "#777"}
ACTION_COLORS = {"buy": "#1a6b32", "sell": "#9b2616", "hold": "#777"}


# --------------------------------------------------------------------------- #
# Data helpers (cached)
# --------------------------------------------------------------------------- #
def _close_series(df) -> Optional[pd.Series]:
    if df is None or len(df) == 0:
        return None
    close = df["Close"]
    if hasattr(close, "ndim") and close.ndim == 2:  # yfinance multi-index frame
        close = close.iloc[:, 0]
    return close.dropna()


@st.cache_data(ttl=900, show_spinner=False)
def load_history(ticker: str, period: str = "1y") -> Optional[pd.Series]:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    return _close_series(df)


@st.cache_data(ttl=900, show_spinner=False)
def load_range(ticker: str, start: str, end: str) -> Optional[pd.Series]:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    return _close_series(df)


def _atr_proxy(prices: List[float], period: int = 14) -> float:
    if len(prices) < 2:
        return 0.0
    changes = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    window = changes[-period:]
    return sum(window) / len(window) if window else 0.0


def _fmt_ratio(x: float) -> str:
    if x == math.inf:
        return "∞"
    if x == -math.inf:
        return "−∞"
    return f"{x:.2f}"


# --------------------------------------------------------------------------- #
# Plain-language helpers (make finance jargon approachable)
# --------------------------------------------------------------------------- #
# One-line, jargon-free tooltips reused across the app.
HELP = {
    "size": "How much of your money the system would put into this single trade "
            "(as a % of the total account).",
    "stop": "A pre-set protective exit price. If the price falls to here, you'd close "
            "the trade to cap the loss.",
    "conviction": "The agents' combined view after fusion, plus how confident they are. "
                  "Low agreement collapses to HOLD.",
    "binding": "The one risk rule that limited the trade — i.e. the reason the position "
               "isn't any bigger.",
    "var": "Value at Risk: a loss the position is unlikely to exceed on a normal day "
           "(here, 95% of days should be better than this).",
    "cvar": "If one of those rare bad days does happen, the average loss you'd expect. "
            "Always worse than VaR.",
    "cumret": "Total return over the whole period (e.g. +50% means the money grew 1.5×).",
    "sharpe": "Return earned per unit of overall risk. Higher is better; above 1 is good, "
              "above 2 is excellent.",
    "sortino": "Like Sharpe, but only counts downside (bad) volatility. Higher is better.",
    "maxdd": "Maximum drawdown — the worst peak-to-trough drop along the way. Smaller is "
             "safer. This is the framework's headline risk metric.",
    "calmar": "Annual return divided by the worst drawdown. Higher means you were paid "
              "more for the pain you endured.",
}

# Binding-constraint codes -> plain English.
CONSTRAINT_PLAIN = {
    "kelly": "bet-sizing math (fractional-Kelly) capped how much to risk",
    "vol_target": "the volatility target — the stock was too choppy to size up further",
    "cvar_limit": "the worst-case-loss (CVaR) limit",
    "position_cap": "the hard per-trade position cap",
    "portfolio_budget": "the remaining portfolio budget",
    "max_drawdown": "the drawdown circuit breaker (portfolio fell too far from its peak)",
    "low_confidence": "the agents weren't confident enough to act",
    "no_signal": "no clear directional signal",
    # book-level constraints
    "portfolio_gross": "the portfolio's total-exposure cap — the book is already full",
    "sector_cap": "the per-sector exposure cap — too much already in this sector",
    "correlation_cap": "the correlation cap — too much in holdings that move together",
    # compliance gate
    "compliance_restricted": "compliance — this instrument is on the restricted list",
    "compliance_no_rationale": "compliance — no decision rationale was recorded",
    "compliance_position_limit": "compliance — the institutional position limit",
}


def plain_constraint(code: str) -> str:
    return CONSTRAINT_PLAIN.get(code, code)


def plain_verdict(order, pre, ticker: str) -> str:
    """A single friendly sentence summarizing the decision."""
    if order.action == "hold" or order.vetoed:
        return (
            f"🟡 **HOLD — no trade in {ticker} right now.** The agents didn't reach a "
            f"confident, aligned view, so the safe choice is to stay out "
            f"({plain_constraint(order.binding_constraint)})."
        )
    verb = "BUY" if order.action == "buy" else "SELL"
    emoji = "🟢" if order.action == "buy" else "🔴"
    return (
        f"{emoji} **{verb} a {order.size * 100:.0f}% position in {ticker}.** The agents lean "
        f"**{pre.direction}** with {pre.confidence * 100:.0f}% confidence. The size was held "
        f"back by {plain_constraint(order.binding_constraint)}; a protective stop sits at "
        f"{order.stop_price:.2f}."
    )


def plain_backtest_summary(rows: list) -> str:
    """Interpret the metrics table in one friendly paragraph."""
    by_name = {r[0]: r for r in rows}
    ours = by_name.get("Risk-budgeted (ours)")
    if not ours:
        return ""
    _, cum, sharpe, _, mdd, _ = ours
    parts = [
        f"In plain terms: the **risk-budgeted** strategy returned **{cum * 100:.0f}%** while "
        f"keeping its worst drop (max drawdown) to **{mdd * 100:.0f}%**."
    ]
    bh = by_name.get("Buy & Hold")
    if bh:
        bcum, bmdd = bh[1], bh[4]
        parts.append(
            f"Buy & Hold made more (**{bcum * 100:.0f}%**) but only by enduring a far deeper "
            f"**{bmdd * 100:.0f}%** fall. That's the deliberate trade-off here: **give up some "
            f"upside for much smaller, smoother losses**."
        )
    parts.append(
        f"A Sharpe of **{_fmt_ratio(sharpe)}** means solid return for the risk taken "
        f"(higher = better)."
    )
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Sidebar — backend status + ticker + risk budget
# --------------------------------------------------------------------------- #
def backend_status() -> dict:
    import os

    return {
        "slm_provider": slm_provider(),
        "gemini_model": os.environ.get("AGENTIC_GEMINI_MODEL", "gemini-2.5-flash"),
        "ollama_model": os.environ.get("AGENTIC_OLLAMA_MODEL", "qwen2.5:3b"),
        "news_provider": news_provider(),
        "sentiment_backend": sentiment_backend(),
        "google_key": bool(os.environ.get("GOOGLE_API_KEY")),
        "tavily_key": bool(os.environ.get("TAVILY_NEWS_API_KEY")),
    }


with st.sidebar:
    st.markdown("## 📈 Agentic Finance")
    st.caption("Understand any stock — and what to do about it — in one click.")
    st.markdown("**1.** Pick a company&nbsp;&nbsp;**2.** Set your risk comfort&nbsp;&nbsp;"
                "**3.** Get your recommendation", help=None)

    # --- Company search: any company worldwide (live Yahoo search) ---
    st.markdown("### 🔎 Company")
    query = st.text_input(
        "Search a company or symbol",
        value=st.session_state.get("company_query", "Apple"),
        help="Type any company or ticker — 'Reliance', 'Apple', 'TSLA', 'BTC-USD'. "
             "Searches all Yahoo Finance markets (US, India, and global).")
    st.session_state["company_query"] = query

    @st.cache_data(ttl=600, show_spinner=False)
    def _search(q):
        return discovery.search_symbols(q, limit=15)

    hits = _search(query) if query.strip() else []
    if hits:
        labels = [h.label for h in hits]
        # Key depends on the query so a new search resets to its best (top) match.
        sel = st.selectbox("Select a match", range(len(labels)),
                           format_func=lambda i: labels[i], key=f"match_{query.strip().lower()}",
                           help="Multiple listings/exchanges may match — pick the right one.")
        ticker = hits[sel].symbol
        ticker_name = hits[sel].name
    else:
        # Offline / no live match -> static registry, else use the text as a symbol.
        ticker = universe.resolve(query) if query.strip() else "AAPL"
        _c = universe.get_company(ticker)
        ticker_name = _c.name if _c else ticker
        if query.strip():
            st.caption("No live matches (offline?) — using the text as a raw symbol.")

    st.divider()
    st.markdown("### 🎯 Risk appetite")
    st.caption("New to investing? Just pick one — it sets everything sensibly for you.")
    appetite = st.radio(
        "How much risk are you comfortable with?",
        ["Conservative", "Moderate", "Aggressive"],
        index=1, horizontal=True, key="appetite")
    st.caption({
        "Conservative": "🛡️ Plays it safe — small positions, tight safety stops.",
        "Moderate": "⚖️ Balanced — a sensible middle ground (good place to start).",
        "Aggressive": "🚀 Bolder — larger positions, chasing more upside.",
    }[appetite])

    # Seed the sliders whenever appetite changes (first run seeds Moderate defaults).
    if st.session_state.get("_last_appetite") != appetite:
        for _k, _v in RISK_PRESETS[appetite].items():
            st.session_state[f"rb_{_k}"] = _v
        st.session_state["_last_appetite"] = appetite

    with st.expander("⚙️ Advanced: fine-tune risk limits (optional)", expanded=False):
        st.caption("You can skip this — your risk appetite already sets sensible limits. "
                   "Open only if you're comfortable with these terms.")
        st.slider("Target volatility (annual)", 0.05, 0.40, step=0.01, key="rb_target_vol",
                  help="Yearly price-swing the engine sizes toward. Lower = more cautious.")
        st.slider("Max position (frac. of equity)", 0.05, 1.0, step=0.05, key="rb_max_position",
                  help="Biggest share of your money allowed in a single trade.")
        st.slider("CVaR limit (per position)", 0.02, 0.20, step=0.01, key="rb_cvar_limit",
                  help="Caps the average loss on the worst ~5% of days. Lower = stricter.")
        st.slider("Max drawdown (circuit breaker)", 0.05, 0.50, step=0.01, key="rb_max_drawdown",
                  help="Trading halts if the account falls this far below its peak.")
        st.slider("Min confidence to act", 0.30, 0.90, step=0.01, key="rb_min_confidence",
                  help="Agents must be this sure before any trade — otherwise HOLD.")
        st.slider("Fractional-Kelly scaler", 0.1, 1.0, step=0.05, key="rb_kelly_fraction",
                  help="How aggressively to size bets. 0.5 is a common safe choice.")
        st.slider("ATR stop multiplier", 1.0, 4.0, step=0.5, key="rb_atr_multiplier",
                  help="How far below price the protective stop sits.")
        st.caption("Lower limits = safer, smaller trades. Changing the appetite above "
                   "resets these to its defaults.")

    budget = RiskBudget(
        target_vol=st.session_state["rb_target_vol"],
        max_position=st.session_state["rb_max_position"],
        cvar_limit=st.session_state["rb_cvar_limit"],
        max_drawdown=st.session_state["rb_max_drawdown"],
        min_confidence=st.session_state["rb_min_confidence"],
        kelly_fraction=st.session_state["rb_kelly_fraction"],
        atr_multiplier=st.session_state["rb_atr_multiplier"],
    )

    st.divider()
    with st.expander("⚙️ System status", expanded=False):
        s = backend_status()
        st.caption(
            f"AI model: `{s['slm_provider']}` · sentiment: `{s['sentiment_backend']}` · "
            f"news: `{s['news_provider']}`")
        st.caption(("✅" if s["google_key"] else "⬜") + " AI key   "
                   + ("✅" if s["tavily_key"] else "⬜") + " News key")


# --------------------------------------------------------------------------- #
# Header + live price
# --------------------------------------------------------------------------- #
sym = universe.currency_symbol(ticker)
_disp = f"{ticker_name} ({ticker})" if ticker_name and ticker_name != ticker else ticker
st.markdown(f"# {_disp}")

price_series = load_history(ticker, "1y")
hcol1, hcol2, hcol3, hcol4 = st.columns([1.2, 1, 1, 1.4])
if price_series is None or len(price_series) < 2:
    st.error(
        f"No price data for **{ticker}**. Check the symbol, or your network "
        f"(live HTTPS may be blocked by an AV/proxy)."
    )
    st.stop()

closes = [float(x) for x in price_series.tolist()]
last_price = closes[-1]
prev_price = closes[-2]
day_change = last_price / prev_price - 1.0
yr_change = last_price / closes[0] - 1.0

hcol1.metric("Price", f"{sym}{last_price:,.2f}", f"{day_change * 100:+.2f}% today",
             help="Latest closing price, and how it moved versus the previous day.")
hcol2.metric("Past year", f"{yr_change * 100:+.1f}%",
             help="How much the price has changed over the last 12 months.")
fv_now = extract_features(closes)
if fv_now:
    _rsi = fv_now.rsi_14
    _mom = "Overbought" if _rsi > 70 else "Oversold" if _rsi < 30 else "Neutral"
    hcol3.metric("Momentum", _mom,
                 help="Is the stock looking over-bought (may be pricey), over-sold (may be "
                      "cheap), or neutral right now? (Based on RSI.)")
    _volp = fv_now.vol_20d * math.sqrt(252) * 100
    _risk = "Low" if _volp < 20 else "High" if _volp > 40 else "Medium"
    hcol4.metric("Volatility", _risk, f"{_volp:.0f}% / yr", delta_color="off",
                 help="How much the price swings around. Higher = riskier, choppier.")
st.caption(f"Data as of {price_series.index[-1].date()}")

with st.expander("❓ How do I use this? (30-second guide)"):
    st.markdown(
        """
**In 3 steps:**
1. **Pick a company** in the left sidebar (search any stock, US or global).
2. **Choose your risk comfort** — Conservative, Moderate, or Aggressive.
3. Open **🧠 Recommendation** and press the button — you'll get a clear
   **Buy / Sell / Hold**, how much to invest, and a plain-English reason.

**The tabs, in plain words**
- **🧠 Recommendation** — what to do with this stock right now, and why.
- **📊 Track Record** — how this approach would have done in the past.
- **🔍 Why** — the signals behind the decisions.
- **🎯 Ideas for You** — similar companies that fit your risk comfort.

> 💡 Hover any small **ⓘ** icon for a plain-English explanation of a term.
        """
    )

if st.button("🔄 Refresh prices", help="Re-download the latest prices for this company."):
    load_history.clear()
    st.rerun()


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_live, tab_bt, tab_xai, tab_suggest, tab_about = st.tabs(
    ["🧠 Recommendation", "📊 Track Record", "🔍 Why", "🎯 Ideas for You", "ℹ️ Help"]
)


def signal_badge(direction: str) -> str:
    cls = {"long": "badge-long", "short": "badge-short", "flat": "badge-flat"}[direction]
    return f'<span class="badge {cls}">{direction.upper()}</span>'


def render_agent_card(col, sig, title: str):
    with col:
        st.markdown(
            f'<div class="agent-card">'
            f'<div class="agent-name">{title}</div>'
            f'{signal_badge(sig.direction)}'
            f'<div style="margin-top:10px;" class="muted">strength</div>',
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, sig.strength), text=f"{sig.strength:.2f}")
        st.markdown('<div class="muted">confidence</div>', unsafe_allow_html=True)
        st.progress(min(1.0, sig.confidence), text=f"{sig.confidence:.2f}")
        st.caption(sig.rationale or "—")


# ------------------------------ LIVE DECISION ------------------------------ #
with tab_live:
    # Price line chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_series.index, y=price_series.values, mode="lines",
            line=dict(color="#3a6ea5", width=1.6), name=ticker, fill="tozeroy",
            fillcolor="rgba(58,110,165,0.07)",
        )
    )
    fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Price", xaxis_rangeslider_visible=False,
        showlegend=False, template="plotly_white",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        "We'll read the **price trend**, the **news mood**, and the **risks** — then "
        "suggest **what to do**, how much to invest, and **why**."
    )
    run = st.button("🔍 Analyze this stock & recommend", type="primary", width="stretch")

    if run:
        with st.spinner("Querying agents (prices · fundamentals · news · reasoning)…"):
            # Full agentic pipeline (shared with the Suggestions deep-dive).
            dec = agentic_decision(ticker, closes, budget, fetch_news=True)
        st.session_state["live"] = {
            "ticker": ticker, "tech": dec.technical, "sent": dec.sentiment,
            "fund": dec.fundamental, "news": dec.news, "pre": dec.pre,
            "order": dec.order, "headlines": dec.headlines,
            "filings": list(dec.filings), "ts": datetime.now(),
            "violations": list(dec.compliance_violations),
        }

    live = st.session_state.get("live")
    if live and live["ticker"] == ticker:
        order = live["order"]
        pre = live["pre"]

        st.divider()
        # Plain-language verdict first — the one thing a newcomer should read.
        st.markdown(f"### {plain_verdict(order, pre, ticker)}")

        # Headline decision
        d1, d2, d3, d4 = st.columns([1.3, 1, 1, 1])
        action_color = ACTION_COLORS.get(order.action, "#555")
        d1.markdown(
            f'<div class="muted">Decision</div>'
            f'<div class="big-action" style="color:{action_color}">{order.action.upper()}</div>',
            unsafe_allow_html=True,
        )
        d2.metric("Position size", f"{order.size * 100:.2f}%", help=HELP["size"])
        d3.metric("Stop price", f"{sym}{order.stop_price:.2f}" if order.stop_price else "—",
                  help=HELP["stop"])
        d4.metric("Fused conviction", f"{pre.direction.upper()}", f"{pre.confidence * 100:.0f}% conf",
                  help=HELP["conviction"])

        r1, r2, r3 = st.columns(3)
        r1.metric("Binding constraint", order.binding_constraint, help=HELP["binding"])
        r2.metric("VaR (95%)", f"{order.var * 100:.2f}%", help=HELP["var"])
        r3.metric("CVaR (95%)", f"{order.cvar * 100:.2f}%", help=HELP["cvar"])
        st.caption(f"ⓘ Why this size? {plain_constraint(order.binding_constraint).capitalize()}.")

        for v in live.get("violations", []):
            st.warning(f"🛡️ Compliance gate: {v}", icon="🛡️")

        st.markdown("#### Agent signals")
        st.caption("Each specialist gives a direction (long = expect up, short = expect down, "
                   "flat = no view), how strong it is, and how confident it is.")
        cols = st.columns(4)
        render_agent_card(cols[0], live["tech"], "🔧 Technical")
        if live.get("fund") is not None:
            render_agent_card(cols[1], live["fund"], "🏛️ Fundamental")
        else:
            cols[1].info("Fundamentals skipped — no ratio data.")
        if live["sent"] is not None:
            render_agent_card(cols[2], live["sent"], "💬 Sentiment")
        else:
            cols[2].info("Sentiment skipped — no headlines.")
        if live["news"] is not None:
            render_agent_card(cols[3], live["news"], "📰 News reasoning")
        else:
            cols[3].info("News reasoning skipped — no headlines / model unavailable.")

        st.markdown("#### Explanation")
        st.success(explain_order(order))

        with st.expander("🔍 Decision trace (agents → fusion → risk gate)"):
            from agentic_finance.xai import decision_trace as _trace

            trace_signals = [s for s in (live["tech"], live.get("fund"),
                                         live["sent"], live["news"]) if s]
            st.code(_trace(trace_signals, pre, order), language=None)

        with st.expander(f"📰 Headlines used ({len(live['headlines'])})"):
            if live["headlines"]:
                for h in live["headlines"]:
                    st.markdown(f"- {h}")
            else:
                st.caption("No headlines retrieved.")

        filings = live.get("filings", [])
        with st.expander(f"📄 SEC filing excerpts used ({len(filings)})"):
            if filings:
                st.caption("Point-in-time excerpts retrieved from the local "
                           "filing index (SEC EDGAR) and given to the news agent.")
                for f in filings:
                    st.markdown(f"> {f}")
            else:
                st.caption("No filings indexed for this ticker (non-SEC-registered "
                           "tickers are skipped automatically).")

        if fv_now:
            with st.expander("🔢 Point-in-time feature vector"):
                fd = fv_now.to_ordered_dict()
                st.dataframe(
                    pd.DataFrame({"feature": list(fd.keys()), "value": list(fd.values())}),
                    width="stretch", hide_index=True,
                )
        st.caption(f"Decision computed {live['ts']:%Y-%m-%d %H:%M:%S}")
    else:
        st.info("👆 Press **Analyze this stock & recommend** to get your recommendation.")


# --------------------------------- BACKTEST -------------------------------- #
with tab_bt:
    st.markdown(
        "**Replay history without cheating.** This walks through the chosen dates day by "
        "day — each decision sees only the past — and compares our **risk-budgeted** "
        "strategy against simple textbook strategies. No API keys needed.")
    c1, c2, c3 = st.columns(3)
    start = c1.date_input("Start", value=date(2022, 6, 1),
                          help="First day of the period to replay.")
    end = c2.date_input("End", value=date(2024, 6, 1),
                        help="Last day of the period to replay.")
    sig_choice = c3.selectbox(
        "Risk-engine direction signal",
        ["Momentum (20d)", "Agentic technical agent"],
        help="What tells the risk engine which way to lean: a simple momentum rule, or "
             "the framework's own technical agent.")
    baselines_chosen = st.multiselect(
        "Baselines to compare",
        ["Buy & Hold", "SMA 20/50", "MACD", "RSI(14)"],
        default=["Buy & Hold", "SMA 20/50", "MACD"],
        help="Simple reference strategies to benchmark against. Buy & Hold = just hold the "
             "stock the whole time.")
    warmup = st.slider("Warm-up bars (let indicators form)", 20, 120, 60, 5,
                       help="Skip the first N days so moving averages have enough history "
                            "before any trading starts.")

    if st.button("▶️ Run backtest", type="primary", width="stretch"):
        bt_series = load_range(ticker, str(start), str(end))
        if bt_series is None or len(bt_series) < warmup + 5:
            st.error("Not enough data for that ticker / range / warm-up.")
            st.stop()
        bt_closes = [float(x) for x in bt_series.tolist()]
        dates = bt_series.index

        baseline_fns = {
            "Buy & Hold": buy_and_hold,
            "SMA 20/50": sma_crossover(20, 50),
            "MACD": macd_strategy(),
            "RSI(14)": rsi_strategy(),
        }

        rows = []
        fig_eq = go.Figure()

        def add_curve(name, returns, color, width=1.6, dash=None):
            curve = equity_curve(returns)
            x = dates[warmup: warmup + len(curve)]
            fig_eq.add_trace(go.Scatter(
                x=x, y=curve, mode="lines", name=name,
                line=dict(color=color, width=width, dash=dash),
            ))

        palette = ["#9aa0a6", "#6c8ebf", "#b07aa1", "#e0a458"]
        for i, name in enumerate(baselines_chosen):
            r = run_backtest(bt_closes, baseline_fns[name], warmup=warmup)
            add_curve(name, r.returns, palette[i % len(palette)])
            rows.append([name, r.cumulative_return, r.sharpe, r.sortino, r.max_drawdown, r.calmar])

        # Risk-budgeted (ours)
        if sig_choice.startswith("Momentum"):
            direction_signal = momentum_direction(lookback=20, confidence=0.75)
        else:
            direction_signal = agent_direction_signal([technical_provider()])
        rb = run_risk_budgeted_backtest(bt_closes, direction_signal, budget, warmup=warmup)
        p = rb.performance
        add_curve("Risk-budgeted (ours)", p.returns, "#1a6b32", width=2.6)
        rows.append(["Risk-budgeted (ours)", p.cumulative_return, p.sharpe,
                     p.sortino, p.max_drawdown, p.calmar])

        fig_eq.update_layout(
            height=420, template="plotly_white", title="Growth of $1 (point-in-time)",
            margin=dict(l=0, r=0, t=40, b=0), yaxis_title="Equity (×)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_eq, width="stretch")

        # Metrics table
        df_m = pd.DataFrame(rows, columns=["Strategy", "CumRet", "Sharpe", "Sortino", "MaxDD", "Calmar"])
        styled = df_m.copy()
        styled["CumRet"] = styled["CumRet"].map(lambda v: f"{v * 100:.2f}%")
        styled["MaxDD"] = styled["MaxDD"].map(lambda v: f"{v * 100:.2f}%")
        styled["Sharpe"] = styled["Sharpe"].map(_fmt_ratio)
        styled["Sortino"] = styled["Sortino"].map(_fmt_ratio)
        styled["Calmar"] = styled["Calmar"].map(_fmt_ratio)
        st.markdown("#### Performance")
        st.dataframe(
            styled, width="stretch", hide_index=True,
            column_config={
                "CumRet": st.column_config.TextColumn("CumRet", help=HELP["cumret"]),
                "Sharpe": st.column_config.TextColumn("Sharpe", help=HELP["sharpe"]),
                "Sortino": st.column_config.TextColumn("Sortino", help=HELP["sortino"]),
                "MaxDD": st.column_config.TextColumn("MaxDD", help=HELP["maxdd"]),
                "Calmar": st.column_config.TextColumn("Calmar", help=HELP["calmar"]),
            },
        )
        st.caption("**CumRet** = total return · **Sharpe/Sortino** = return per unit of risk "
                   "(higher better) · **MaxDD** = worst drop (smaller better) · "
                   "**Calmar** = return ÷ worst drop. Hover any column header for more.")
        summary = plain_backtest_summary(rows)
        if summary:
            st.info(summary)

        # Risk-constraint attribution + drawdown breaker
        st.markdown("#### Why each trade was the size it was")
        st.caption("Every position is limited by exactly one risk rule. This shows how "
                   "often each rule was the deciding factor — the system explaining itself.")
        ac1, ac2 = st.columns([1.4, 1])
        constraints = attribute_constraints(rb.orders)
        if constraints:
            labels = [plain_constraint(k) for k in constraints]
            values = list(constraints.values())
            donut = go.Figure(go.Pie(labels=labels, values=values, hole=0.5,
                                     textinfo="percent"))
            donut.update_layout(height=340, template="plotly_white",
                                margin=dict(l=0, r=0, t=10, b=0),
                                legend=dict(font=dict(size=11)))
            ac1.plotly_chart(donut, width="stretch")
        vetoes = sum(1 for o in rb.orders if o.vetoed)
        ac2.metric("Decisions made", len(rb.orders),
                   help="Total number of daily trade/hold decisions in this backtest.")
        ac2.metric("Circuit-breaker stops", f"{vetoes} / {len(rb.orders)}",
                   help="Days the drawdown breaker forced no trade because losses got too deep.")


# ------------------------------ EXPLAINABILITY ----------------------------- #
with tab_xai:
    st.markdown(
        "**Which signals actually drive the decisions?** We train a simple stand-in model "
        "that mimics the system, then use **SHAP** to rank what it pays attention to. We "
        "also report **fidelity** — how well the stand-in matches the real system (an "
        "explanation you can only trust as much as its fidelity)."
    )
    cx1, cx2 = st.columns(2)
    xs = cx1.date_input("Start ", value=date(2021, 1, 1), key="xai_start",
                        help="Start of the history used to learn what drives decisions.")
    xe = cx2.date_input("End ", value=date(2024, 6, 1), key="xai_end",
                        help="End of that history.")
    xwarm = st.slider("Warm-up bars ", 51, 120, 60, 1, key="xai_warm",
                      help="Days skipped at the start so indicators have enough history.")

    if st.button("🔬 Train surrogate + SHAP", type="primary", width="stretch"):
        with st.spinner("Backtesting, training surrogate, computing SHAP…"):
            from agentic_finance.xai import (  # lazy: pulls sklearn / shap
                global_importance,
                surrogate_fidelity,
                train_surrogate,
            )

            xseries = load_range(ticker, str(xs), str(xe))
            if xseries is None or len(xseries) < xwarm + 30:
                st.error("Not enough data for that range / warm-up.")
                st.stop()
            xcloses = [float(v) for v in xseries.tolist()]

            rb = run_risk_budgeted_backtest(
                xcloses, momentum_direction(20, 0.75), budget, warmup=xwarm
            )
            order_by_bar = {xwarm + k: o for k, o in enumerate(rb.orders)}
            names, matrix, feat_idx = build_feature_matrix(xcloses, warmup=xwarm)
            rows_x, labels = [], []
            for j, bar in enumerate(feat_idx):
                if bar in order_by_bar:
                    rows_x.append(matrix[j])
                    labels.append(order_by_bar[bar].action)

        if len(set(labels)) < 2:
            st.warning(
                f"Surrogate needs ≥2 distinct actions to explain; this run produced "
                f"only `{set(labels)}`. Try a longer window or different risk budget."
            )
        else:
            model = train_surrogate(rows_x, labels, feature_names=names)
            fidelity = surrogate_fidelity(model, rows_x, labels)
            importance = global_importance(model, rows_x)

            st.metric("Surrogate fidelity", f"{fidelity * 100:.1f}%",
                      help="How often the stand-in model matches the real system. "
                           "Higher = the explanation below is more trustworthy.")
            top = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:12][::-1]
            fig_imp = go.Figure(go.Bar(
                x=[v for _, v in top], y=[k for k, _ in top],
                orientation="h", marker_color="#6c8ebf",
            ))
            fig_imp.update_layout(
                height=460, template="plotly_white",
                title="What drives the decisions (longer bar = more influence)",
                margin=dict(l=0, r=0, t=40, b=0), xaxis_title="influence (mean |SHAP value|)",
            )
            st.plotly_chart(fig_imp, width="stretch")
            st.caption("Each bar is a feature (e.g. recent return, RSI, volatility). The "
                       "longer it is, the more that feature sways the system's choices.")

            # Counterfactual: the smallest change that would flip the latest call.
            st.markdown("#### 🔀 What would change the decision?")
            st.caption("A *counterfactual* asks: what's the smallest change in the "
                       "inputs that would have flipped the most recent decision?")
            try:
                from agentic_finance.xai import counterfactual_report, find_counterfactuals

                cfs = find_counterfactuals(model, rows_x[-1], training_matrix=rows_x)
                if cfs:
                    st.code(counterfactual_report(cfs), language=None)
                else:
                    st.caption("No nearby flip found — the latest decision is robust "
                               "to small input changes.")
            except Exception as exc:  # noqa: BLE001 - explanation is best-effort
                st.caption(f"Counterfactual search unavailable: {exc}")

            dist = pd.Series(labels).value_counts()
            st.caption("Decision label distribution: " +
                       " · ".join(f"**{k}** {v}" for k, v in dist.items()))


# ------------------------------- SUGGESTIONS ------------------------------- #
with tab_suggest:
    st.markdown(
        "**You might also be interested in…** — sector peers of the selected company, "
        "ranked to your **risk appetite** and the **current market condition**."
    )
    c1, c2 = st.columns([1, 1])
    c1.write(f"Selected: **{ticker_name}** ({ticker})")
    c2.write(f"Risk appetite: **{appetite}**")
    deep_dive = st.checkbox(
        "Deep-dive the top 3 with the full agentic pipeline",
        value=False,
        help="Off = fast, deterministic screen only. On = also runs the full "
             "agents→fusion→risk pipeline on the top 3.")

    if st.button("🔎 Find similar companies", type="primary", width="stretch"):
        def _loader(t):
            srs = load_history(t, "1y")
            return [float(x) for x in srs.tolist()] if srs is not None and len(srs) else None

        with st.spinner("Finding and screening similar companies…"):
            bench = "^NSEI" if universe.region_of(ticker) == "IN" else "SPY"
            regime = market_regime(_loader(bench))
            ranked = suggest(ticker, budget, appetite, _loader,
                             top_k=6, deep=3, deep_dive=deep_dive)
            sector = universe.sector_of(ticker) or discovery.get_sector(ticker)
        st.session_state["suggest"] = {
            "ticker": ticker, "appetite": appetite, "regime": regime,
            "ranked": ranked, "deep": deep_dive, "sector": sector,
        }

    sug = st.session_state.get("suggest")
    if sug and sug["ticker"] == ticker:
        ranked = sug["ranked"]
        if sug.get("sector"):
            st.write(f"Sector: **{sug['sector']}**")
        st.info("📈 Current market condition: " + sug["regime"].describe())
        if not ranked:
            st.warning(
                "No similar companies could be scored (an index/crypto with no peers, or "
                "insufficient price history). Try a listed company like Apple or Reliance.")
        else:
            sym_s = universe.currency_symbol(ticker)
            rows = []
            for i, s in enumerate(ranked, 1):
                rows.append({
                    "#": i, "Company": s.name, "Ticker": s.ticker,
                    "Signal": s.direction.upper(),
                    "Suggested size": f"{s.suggested_size * 100:.1f}%",
                    "20d return": f"{s.ret_20d * 100:+.1f}%",
                    "Sharpe": f"{s.sharpe:.2f}",
                    "Max DD": f"{s.max_drawdown * 100:.1f}%",
                    "Vol (ann.)": f"{s.vol_20d_ann * 100:.0f}%",
                    "Fit": f"{s.fit_score:.2f}",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                         column_config={
                             "Fit": st.column_config.TextColumn(
                                 "Fit", help="How well the stock matches your risk "
                                 "appetite + current market (higher = better fit)."),
                             "Suggested size": st.column_config.TextColumn(
                                 "Suggested size", help="Position size the risk engine "
                                 "would allow under your current risk budget."),
                         })
            st.caption("Ranked by **fit** to your appetite and the market regime. "
                       "'Suggested size' is what the risk engine would allow under your budget.")

            deep_notes = [s for s in ranked if s.agentic_note]
            if deep_notes:
                st.markdown("#### 🔬 Agentic deep-dive (top matches)")
                for s in deep_notes:
                    with st.container(border=True):
                        st.markdown(f"**{s.name}** ({s.ticker}) — {sym_s}{s.last_price:,.2f}")
                        st.write(s.agentic_note)
            elif sug["deep"]:
                st.caption("Deep-dive produced no additional notes (news/model unavailable).")
    else:
        st.caption("Press **Find similar companies** to get ideas for your risk appetite.")


# --------------------------------- ABOUT ----------------------------------- #
with tab_about:
    st.markdown(
        """
### What is this? (in plain words)
It looks at a stock's **price trend**, the **mood of recent news**, and the **risk**, then
gives you a clear **Buy / Sell / Hold** — sized to how much risk you're comfortable with —
and explains **why** in simple terms. Think of it as a careful, transparent second opinion.

> 🛡️ It always puts **safety first**: it caps how much to risk, sets an automatic exit, and
> holds (does nothing) when it isn't confident.

---

### For the technically curious
A multi-agent, **risk-mitigated**, **explainable** trading-decision framework that runs
on configurable small or large language models.

**Pipeline**
1. **Specialist agents** emit typed `Signal`s — Technical (deterministic), Sentiment
   (lexicon/Gemini), News-reasoning (Gemini/Ollama LLM).
2. **Fusion** reconciles them by confidence into a `PreDecision`.
3. The **quantitative risk engine** turns conviction into an *enforced* sized order
   (VaR/CVaR, volatility-target & fractional-Kelly sizing, ATR stop, drawdown breaker)
   and records the **binding constraint**.
4. The **explainability layer** (SHAP surrogate + risk-constraint attribution) produces
   an auditable account of the decision.

**Honesty notes.** The *Recommendation* tab is a live snapshot (news is "now"), not a
leakage-controlled backtest. LLM agents are stochastic; the deterministic core (risk,
metrics, features, fusion, surrogate) is exactly reproducible. Validation here is
paper-trading / simulation only — no real capital.
        """
    )
    st.json(backend_status())
