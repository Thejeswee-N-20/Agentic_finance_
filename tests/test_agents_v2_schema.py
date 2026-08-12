"""Unit tests for the typed Signal schema (the agent message contract)."""
import dataclasses

import pytest

from agentic_finance.agents_v2.schemas import Signal

pytestmark = pytest.mark.unit


def test_signal_construction_and_fields():
    s = Signal(source="technical", direction="long", strength=0.7, confidence=0.8,
               rationale="trend up", evidence=("sma20>sma50",))
    assert s.source == "technical"
    assert s.direction == "long"
    assert s.strength == 0.7
    assert s.confidence == 0.8
    assert s.evidence == ("sma20>sma50",)


def test_signal_is_frozen():
    s = Signal(source="x", direction="flat", strength=0.0, confidence=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.confidence = 1.0  # type: ignore[misc]


def test_signal_rejects_bad_direction():
    with pytest.raises(ValueError):
        Signal(source="x", direction="up", strength=0.5, confidence=0.5)


def test_signal_rejects_out_of_range_strength():
    with pytest.raises(ValueError):
        Signal(source="x", direction="long", strength=1.5, confidence=0.5)


def test_signal_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        Signal(source="x", direction="long", strength=0.5, confidence=-0.1)


def test_signed_strength_helper():
    assert Signal("x", "long", 0.6, 0.5).signed_strength == pytest.approx(0.6)
    assert Signal("x", "short", 0.6, 0.5).signed_strength == pytest.approx(-0.6)
    assert Signal("x", "flat", 0.6, 0.5).signed_strength == pytest.approx(0.0)
