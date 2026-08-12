"""Env-driven selection of the news provider.

``AGENTIC_NEWS_PROVIDER`` = ``tavily`` | ``yahoo``. If unset, auto-detect:
``tavily`` when a Tavily key is present (``TAVILY_NEWS_API_KEY`` /
``TAVILY_API_KEY``), else ``yahoo``.
"""
from __future__ import annotations

import os
from typing import List, Optional

from agentic_finance.news.base import NewsProvider
from agentic_finance.news.tavily import TavilyNewsProvider
from agentic_finance.news.yahoo import YahooNewsProvider

__all__ = ["news_provider", "get_news_provider", "get_headlines"]

_TAVILY = "tavily"
_YAHOO = "yahoo"


def _has_tavily_key() -> bool:
    return bool(os.environ.get("TAVILY_NEWS_API_KEY") or os.environ.get("TAVILY_API_KEY"))


def news_provider() -> str:
    """Resolve the news provider from env (with auto-detect)."""
    explicit = os.environ.get("AGENTIC_NEWS_PROVIDER", "").strip().lower()
    if explicit in (_TAVILY, _YAHOO):
        return explicit
    return _TAVILY if _has_tavily_key() else _YAHOO


def get_news_provider(provider: Optional[str] = None, api_key: Optional[str] = None) -> NewsProvider:
    """Construct the configured news provider."""
    provider = (provider or news_provider()).strip().lower()
    if provider == _TAVILY:
        return TavilyNewsProvider(api_key=api_key)
    if provider == _YAHOO:
        return YahooNewsProvider()
    raise ValueError(f"unknown news provider {provider!r}; use 'tavily' or 'yahoo'")


def get_headlines(
    query: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 8,
    provider: Optional[NewsProvider] = None,
) -> List[str]:
    """Convenience: fetch headline titles via the configured (or supplied) provider."""
    provider = provider or get_news_provider()
    return [a.title for a in provider.fetch(query, start=start, end=end, limit=limit)]
