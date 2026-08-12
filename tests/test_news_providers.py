"""Unit tests for the news provider abstraction (Tavily / Yahoo), offline.

Network calls are not exercised here; the response *parsers* and the env-driven
provider selection are, using canned payloads.
"""
import pytest

from agentic_finance.news.base import NewsArticle
from agentic_finance.news.tavily import TavilyNewsProvider
from agentic_finance.news.yahoo import YahooNewsProvider
from agentic_finance.news.config import news_provider, get_news_provider, get_headlines

pytestmark = pytest.mark.unit


# --- Tavily parsing -------------------------------------------------------

def test_tavily_parse_extracts_articles():
    data = {"results": [
        {"title": "Nvidia beats", "url": "https://x.com/a", "content": "strong demand",
         "published_date": "Thu, 23 May 2024 08:24:31 GMT"},
        {"title": "Preview", "url": "https://forbes.com/b", "content": "what to expect",
         "published_date": "Wed, 22 May 2024 18:43:14 GMT"},
    ]}
    arts = TavilyNewsProvider._parse(data)
    assert len(arts) == 2
    assert all(isinstance(a, NewsArticle) for a in arts)
    assert arts[0].title == "Nvidia beats"
    assert "x.com" in arts[0].source
    assert arts[0].published.startswith("Thu, 23 May 2024")


def test_tavily_parse_empty():
    assert TavilyNewsProvider._parse({"results": []}) == []
    assert TavilyNewsProvider._parse({}) == []


def test_tavily_build_payload_includes_date_range():
    p = TavilyNewsProvider(api_key="k")
    payload = p._payload("NVDA", "2024-05-01", "2024-05-31", 5)
    assert payload["topic"] == "news"
    assert payload["start_date"] == "2024-05-01"
    assert payload["end_date"] == "2024-05-31"
    assert payload["max_results"] == 5


def test_tavily_payload_omits_dates_when_none():
    p = TavilyNewsProvider(api_key="k")
    payload = p._payload("NVDA", None, None, 5)
    assert "start_date" not in payload and "end_date" not in payload


# --- Yahoo parsing (old + new shapes) ------------------------------------

def test_yahoo_parse_new_shape():
    items = [{"content": {"title": "AAPL rallies",
                          "canonicalUrl": {"url": "https://finance.yahoo.com/a"},
                          "pubDate": "2024-05-20T12:00:00Z"}}]
    arts = YahooNewsProvider._parse(items)
    assert arts[0].title == "AAPL rallies"
    assert "yahoo.com" in arts[0].source


def test_yahoo_parse_old_shape():
    items = [{"title": "Old style", "link": "https://news.com/x", "providerPublishTime": 1716200000}]
    arts = YahooNewsProvider._parse(items)
    assert arts[0].title == "Old style"


def test_yahoo_parse_skips_titleless():
    items = [{"content": {"canonicalUrl": {"url": "https://x.com/a"}}}, {"title": "Keep"}]
    arts = YahooNewsProvider._parse(items)
    assert [a.title for a in arts] == ["Keep"]


# --- config selection -----------------------------------------------------

def test_news_provider_explicit_env(monkeypatch):
    monkeypatch.setenv("AGENTIC_NEWS_PROVIDER", "yahoo")
    assert news_provider() == "yahoo"
    monkeypatch.setenv("AGENTIC_NEWS_PROVIDER", "tavily")
    assert news_provider() == "tavily"


def test_news_provider_autodetect(monkeypatch):
    monkeypatch.delenv("AGENTIC_NEWS_PROVIDER", raising=False)
    monkeypatch.setenv("TAVILY_NEWS_API_KEY", "tvly-x")
    assert news_provider() == "tavily"
    monkeypatch.delenv("TAVILY_NEWS_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert news_provider() == "yahoo"


def test_get_news_provider_types():
    assert isinstance(get_news_provider("yahoo"), YahooNewsProvider)
    assert isinstance(get_news_provider("tavily", api_key="k"), TavilyNewsProvider)
    with pytest.raises(ValueError):
        get_news_provider("bogus")


def test_get_headlines_uses_provider():
    class StubProvider:
        def fetch(self, query, start=None, end=None, limit=10):
            return [NewsArticle(title=f"{query} headline {i}", published="", url="", source="", snippet="")
                    for i in range(3)]
    titles = get_headlines("NVDA", provider=StubProvider())
    assert titles == ["NVDA headline 0", "NVDA headline 1", "NVDA headline 2"]


def test_yahoo_reduces_free_text_query_to_symbol():
    """Callers phrase queries for a search API; yf.Ticker needs the bare symbol."""
    from agentic_finance.news.yahoo import YahooNewsProvider

    assert YahooNewsProvider._symbol("AAPL stock") == "AAPL"
    assert YahooNewsProvider._symbol("RELIANCE.NS stock news") == "RELIANCE.NS"
    assert YahooNewsProvider._symbol("  MSFT  ") == "MSFT"
    assert YahooNewsProvider._symbol("AAPL") == "AAPL"
    assert YahooNewsProvider._symbol("") == ""
    assert YahooNewsProvider._symbol(None) == ""


def test_yahoo_empty_query_returns_no_articles():
    from agentic_finance.news.yahoo import YahooNewsProvider

    assert YahooNewsProvider().fetch("") == []
