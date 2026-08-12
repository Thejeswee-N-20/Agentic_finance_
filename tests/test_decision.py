"""Unit tests for the reusable agentic_decision pipeline (offline path)."""
import math

from agentic_finance.decision import AgenticDecision, agentic_decision, atr_proxy
from agentic_finance.risk_engine import RiskBudget


def _uptrend(n=200, start=100.0, step=0.004):
    # deterministic gentle uptrend with mild oscillation (non-degenerate vol/RSI)
    return [start * (1.0 + step) ** i * (1.0 + 0.01 * math.sin(i / 3.0)) for i in range(n)]


def _budget():
    return RiskBudget(target_vol=0.15, max_position=0.25, cvar_limit=0.10,
                      max_drawdown=0.20, min_confidence=0.55)


def test_agentic_decision_offline_returns_order():
    closes = _uptrend()
    dec = agentic_decision("TEST", closes, _budget(), fetch_news=False,
                           fundamentals=False)
    assert isinstance(dec, AgenticDecision)
    # no news/fundamentals path was taken
    assert dec.sentiment is None and dec.news is None and dec.headlines == []
    assert dec.fundamental is None
    # technical agent produced a signal; risk engine produced an enforced order
    assert dec.technical.source == "technical"
    assert dec.order.action in {"buy", "sell", "hold"}
    assert 0.0 <= dec.order.size <= 0.25


def test_uptrend_is_long_and_sized():
    closes = _uptrend()
    dec = agentic_decision("TEST", closes, _budget(), fetch_news=False,
                           fundamentals=False)
    # a clean uptrend should be long/buy with a positive size
    assert dec.pre.direction == "long"
    assert dec.order.action == "buy"
    assert dec.order.size > 0.0


def test_fundamental_agent_joins_fusion_when_data_available(monkeypatch):
    from agentic_finance.agents_v2.fundamental_agent import FundamentalMetrics
    import agentic_finance.decision as decision_mod

    monkeypatch.setattr(
        decision_mod, "fetch_fundamentals",
        lambda ticker: FundamentalMetrics(
            trailing_pe=30.0, forward_pe=24.0, revenue_growth=0.25,
            profit_margin=0.30, return_on_equity=0.40, debt_to_equity=40.0,
        ),
    )
    dec = agentic_decision("TEST", _uptrend(), _budget(), fetch_news=False)
    assert dec.fundamental is not None
    assert dec.fundamental.source == "fundamental"
    assert dec.fundamental.direction == "long"


def test_fundamentals_outage_degrades_to_none(monkeypatch):
    from agentic_finance.agents_v2.fundamental_agent import FundamentalMetrics
    import agentic_finance.decision as decision_mod

    # outage -> empty metrics -> flat/zero-confidence signal is skipped
    monkeypatch.setattr(decision_mod, "fetch_fundamentals",
                        lambda ticker: FundamentalMetrics())
    dec = agentic_decision("TEST", _uptrend(), _budget(), fetch_news=False)
    assert dec.fundamental is None
    assert dec.order.action in {"buy", "sell", "hold"}


def test_atr_proxy_nonnegative():
    assert atr_proxy([100, 101, 99, 102]) > 0
    assert atr_proxy([100.0]) == 0.0


def test_rag_excerpts_flow_into_llm_context(monkeypatch):
    """With news on, filing excerpts are retrieved and reach the LLM context."""
    import agentic_finance.decision as decision_mod
    from agentic_finance.agents_v2.schemas import Signal

    captured = {}

    def _fake_llm_signal(context, model, source="news"):
        captured["context"] = context
        return Signal(source=source, direction="long", strength=0.5, confidence=0.6)

    monkeypatch.setattr(decision_mod, "get_headlines", lambda q, limit=8: ["Stock rallies"])
    monkeypatch.setattr(decision_mod, "ensure_ingested", lambda t: None)
    monkeypatch.setattr(decision_mod, "retrieve_filing_context",
                        lambda t, q, k=3: ["[10-K filed 2024-02-21] Revenue grew 126%."])
    monkeypatch.setattr(decision_mod, "get_chat_model", lambda: object())
    monkeypatch.setattr(decision_mod, "llm_signal", _fake_llm_signal)

    dec = agentic_decision("TEST", _uptrend(), _budget(), fetch_news=True,
                           fundamentals=False)
    assert dec.filings == ("[10-K filed 2024-02-21] Revenue grew 126%.",)
    assert "SEC filing excerpts" in captured["context"]
    assert "Revenue grew 126%" in captured["context"]


def test_rag_failure_never_breaks_decision(monkeypatch):
    """A ticker with no filings (or a RAG outage) proceeds without error."""
    import agentic_finance.decision as decision_mod

    def _boom(*a, **k):
        raise RuntimeError("no filings / network down")

    monkeypatch.setattr(decision_mod, "get_headlines", lambda q, limit=8: [])
    monkeypatch.setattr(decision_mod, "ensure_ingested", _boom)
    monkeypatch.setattr(decision_mod, "retrieve_filing_context", _boom)
    monkeypatch.setattr(decision_mod, "get_chat_model", _boom)

    dec = agentic_decision("RELIANCE.NS", _uptrend(), _budget(), fetch_news=True,
                           fundamentals=False)
    assert dec.filings == ()
    assert dec.order.action in {"buy", "sell", "hold"}


def test_rag_disabled_by_env(monkeypatch):
    import agentic_finance.decision as decision_mod

    def _fail(*a, **k):
        raise RuntimeError("skipped")

    monkeypatch.setenv("AGENTIC_RAG", "off")
    monkeypatch.setattr(decision_mod, "get_headlines", lambda q, limit=8: [])
    monkeypatch.setattr(decision_mod, "ensure_ingested",
                        lambda t: (_ for _ in ()).throw(AssertionError("RAG must be off")))
    monkeypatch.setattr(decision_mod, "get_chat_model", _fail)
    monkeypatch.setattr(decision_mod, "llm_signal", _fail)

    dec = agentic_decision("TEST", _uptrend(), _budget(), fetch_news=True,
                           fundamentals=False)
    assert dec.filings == ()


def test_compliance_gate_vetoes_restricted_ticker(monkeypatch):
    """A restricted instrument is vetoed to hold by the compliance gate."""
    monkeypatch.setenv("AGENTIC_RESTRICTED_TICKERS", "test, OTHER")
    dec = agentic_decision("TEST", _uptrend(), _budget(), fetch_news=False,
                           fundamentals=False)
    assert dec.order.action == "hold" and dec.order.vetoed
    assert dec.order.binding_constraint == "compliance_restricted"
    assert dec.compliance_violations


def test_portfolio_constraints_bind_with_full_book():
    """With a saturated book, the gross-exposure cap scales/vetoes the order."""
    from agentic_finance.risk_engine import Holding, PortfolioLimits

    full_book = [Holding(ticker="AAA", weight=0.6), Holding(ticker="BBB", weight=0.4)]
    dec = agentic_decision(
        "TEST", _uptrend(), _budget(), fetch_news=False, fundamentals=False,
        holdings=full_book, limits=PortfolioLimits(max_gross_exposure=1.0),
    )
    assert dec.order.action == "hold" and dec.order.vetoed
    assert dec.order.binding_constraint == "portfolio_gross"


def test_ladder_passes_clean_order_through():
    """No book, no restrictions -> the per-asset order stands unchanged."""
    dec = agentic_decision("TEST", _uptrend(), _budget(), fetch_news=False,
                           fundamentals=False)
    assert dec.order.action == "buy"
    assert dec.compliance_violations == ()
