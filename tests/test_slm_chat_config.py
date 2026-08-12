"""Unit tests for the env-driven SLM provider config.

The generative tier (and sentiment) must be switchable between Gemini and the
local Ollama + SLMs via an env var, with no network/keys required to *construct*
a backend (clients connect lazily on first use).
"""
import pytest

from agentic_finance.slm.config import (
    slm_provider,
    get_chat_model,
    sentiment_backend,
    get_sentiment_model,
)
from agentic_finance.slm.gemini import GeminiChatModel, GeminiSentimentModel
from agentic_finance.slm.ollama import OllamaChatModel
from agentic_finance.slm.heuristic import HeuristicSentimentModel

pytestmark = pytest.mark.unit


# --- provider selection ---------------------------------------------------

def test_explicit_provider_env(monkeypatch):
    monkeypatch.setenv("AGENTIC_SLM_PROVIDER", "ollama")
    assert slm_provider() == "ollama"
    monkeypatch.setenv("AGENTIC_SLM_PROVIDER", "gemini")
    assert slm_provider() == "gemini"


def test_provider_autodetect_from_google_key(monkeypatch):
    monkeypatch.delenv("AGENTIC_SLM_PROVIDER", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    assert slm_provider() == "gemini"
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert slm_provider() == "ollama"


# --- chat model factory ---------------------------------------------------

def test_get_chat_model_ollama():
    m = get_chat_model("ollama")
    assert isinstance(m, OllamaChatModel)
    assert m.name


def test_get_chat_model_gemini_constructs_without_network():
    m = get_chat_model("gemini")
    assert isinstance(m, GeminiChatModel)
    assert m.name


def test_get_chat_model_rejects_unknown():
    with pytest.raises(ValueError):
        get_chat_model("bogus")


def test_get_chat_model_uses_env_when_provider_none(monkeypatch):
    monkeypatch.setenv("AGENTIC_SLM_PROVIDER", "ollama")
    assert isinstance(get_chat_model(), OllamaChatModel)


def test_ollama_defaults_from_env(monkeypatch):
    monkeypatch.setenv("AGENTIC_OLLAMA_MODEL", "qwen2.5:1.5b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example:11434")
    m = get_chat_model("ollama")
    assert m.model == "qwen2.5:1.5b"
    assert "example" in m.base_url


def test_gemini_default_model(monkeypatch):
    monkeypatch.delenv("AGENTIC_GEMINI_MODEL", raising=False)
    m = get_chat_model("gemini")
    assert "gemini" in m.model


# --- sentiment backend factory -------------------------------------------

def test_sentiment_backend_default_is_heuristic(monkeypatch):
    monkeypatch.delenv("AGENTIC_SENTIMENT_BACKEND", raising=False)
    assert sentiment_backend() == "heuristic"


def test_get_sentiment_model_heuristic():
    assert isinstance(get_sentiment_model("heuristic"), HeuristicSentimentModel)


def test_get_sentiment_model_gemini_constructs_without_network():
    assert isinstance(get_sentiment_model("gemini"), GeminiSentimentModel)


def test_get_sentiment_model_rejects_unknown():
    with pytest.raises(ValueError):
        get_sentiment_model("bogus")
