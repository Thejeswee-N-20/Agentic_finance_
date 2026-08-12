"""Tavily news provider.

Supports an explicit ``start_date``/``end_date`` range, which is what enables
**point-in-time** news for as-of decisions/backtests. The HTTP call happens
lazily in ``fetch``; the payload builder and response parser are pure and unit-
tested. API key read from ``TAVILY_NEWS_API_KEY`` (or ``TAVILY_API_KEY``).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional
from urllib.parse import urlparse

from agentic_finance.news.base import NewsArticle

__all__ = ["TavilyNewsProvider"]

_ENDPOINT = "https://api.tavily.com/search"


class TavilyNewsProvider:
    """News via the Tavily search API (topic=news, with date range)."""

    name = "tavily"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0):
        self._api_key = api_key or os.environ.get("TAVILY_NEWS_API_KEY") or os.environ.get("TAVILY_API_KEY")
        self._timeout = timeout

    @staticmethod
    def _payload(query: str, start: Optional[str], end: Optional[str], limit: int) -> Dict:
        payload: Dict = {"query": query, "topic": "news", "max_results": limit}
        if start:
            payload["start_date"] = start
        if end:
            payload["end_date"] = end
        return payload

    @staticmethod
    def _parse(data: Dict) -> List[NewsArticle]:
        articles: List[NewsArticle] = []
        for r in (data or {}).get("results", []):
            title = r.get("title")
            if not title:
                continue
            url = r.get("url", "")
            source = urlparse(url).netloc or "tavily"
            articles.append(NewsArticle(
                title=title,
                published=r.get("published_date", ""),
                url=url,
                source=source,
                snippet=(r.get("content", "") or "")[:300],
            ))
        return articles

    def fetch(self, query: str, start: Optional[str] = None,
              end: Optional[str] = None, limit: int = 10) -> List[NewsArticle]:
        if not self._api_key:
            raise RuntimeError("Tavily API key missing (set TAVILY_NEWS_API_KEY).")
        import requests

        resp = requests.post(
            _ENDPOINT,
            json=self._payload(query, start, end, limit),
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return self._parse(resp.json())
