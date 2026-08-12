"""News providers for the agent layer.

A small, env-switchable abstraction over news sources, mirroring the SLM tier:

- ``tavily`` : Tavily news API — supports **date ranges** (point-in-time news,
  the key capability for as-of backtesting). Needs ``TAVILY_NEWS_API_KEY``.
- ``yahoo``  : yfinance headlines — recent only (no historical date range).

Selected via ``AGENTIC_NEWS_PROVIDER`` (auto: tavily when a Tavily key is set,
else yahoo). Agents consume the ``NewsProvider`` protocol, never a concrete source.
"""

from agentic_finance.news.base import NewsArticle, NewsProvider
from agentic_finance.news.config import get_headlines, get_news_provider, news_provider

__all__ = [
    "NewsArticle",
    "NewsProvider",
    "news_provider",
    "get_news_provider",
    "get_headlines",
]
