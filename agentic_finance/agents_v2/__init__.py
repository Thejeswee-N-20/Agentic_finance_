"""Typed-signal agent layer (Agentic Finance redesign of TradingAgents).

Instead of TradingAgents' prose debates, each specialist agent emits a typed
``Signal{direction, strength, confidence, evidence}``. A signal-fusion layer
combines them into a ``PreDecision`` that feeds the quantitative risk engine.
This is cheaper for small language models, auditable, and SHAP-ready.

Concrete agents: ``technical_agent`` (deterministic, from prices),
``sentiment_agent`` (texts + a pluggable SLM backend), ``fundamental_agent``
(financial-health ratios), and ``fusion``.
"""

from agentic_finance.agents_v2.fundamental_agent import (
    FundamentalMetrics,
    fetch_fundamentals,
    fundamental_signal,
)
from agentic_finance.agents_v2.fusion import fuse_signals
from agentic_finance.agents_v2.llm_agent import llm_signal
from agentic_finance.agents_v2.pipeline import agent_direction_signal, technical_provider
from agentic_finance.agents_v2.schemas import Signal
from agentic_finance.agents_v2.sentiment_agent import sentiment_signal
from agentic_finance.agents_v2.technical_agent import technical_signal

__all__ = [
    "Signal",
    "technical_signal",
    "sentiment_signal",
    "fundamental_signal",
    "FundamentalMetrics",
    "fetch_fundamentals",
    "llm_signal",
    "fuse_signals",
    "technical_provider",
    "agent_direction_signal",
]
