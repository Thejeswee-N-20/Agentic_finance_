"""Agent pipeline: specialist signal providers -> fusion -> PreDecision.

Wires the typed-signal agents into a ``DirectionSignal`` the risk-budgeted
backtest can consume, so the framework's real (multi-indicator) signal can be
swapped in for the placeholder momentum rule and compared head-to-head.

Only price-derived providers are point-in-time inside a price-only backtest;
the technical agent qualifies. Sentiment / fundamental providers (which need
per-date external text/financials) are fused in the live path, where that data
is available as-of the decision date.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

from agentic_finance.agents_v2.fusion import fuse_signals
from agentic_finance.agents_v2.schemas import Signal
from agentic_finance.agents_v2.technical_agent import technical_signal
from agentic_finance.risk_engine.schemas import PreDecision

# A provider maps point-in-time prices to one agent Signal.
SignalProvider = Callable[[Sequence[float]], Signal]
DirectionSignal = Callable[[Sequence[float]], PreDecision]

__all__ = ["technical_provider", "agent_direction_signal", "SignalProvider"]


def technical_provider() -> SignalProvider:
    """Provider backed by the technical agent (prices -> Signal)."""
    return lambda prices: technical_signal(prices)


def agent_direction_signal(
    providers: Sequence[SignalProvider],
    weights: Optional[Mapping[str, float]] = None,
    deadband: float = 0.05,
) -> DirectionSignal:
    """Build a DirectionSignal that runs all providers and fuses their signals."""
    def signal(prices: Sequence[float]) -> PreDecision:
        signals = [provider(prices) for provider in providers]
        return fuse_signals(signals, weights=weights, deadband=deadband)
    return signal
