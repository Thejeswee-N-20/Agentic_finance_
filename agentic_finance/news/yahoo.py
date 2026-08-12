"""Yahoo (yfinance) news provider.

Returns recent headlines for a ticker. yfinance does **not** support an arbitrary
historical date range, so ``start``/``end`` are accepted for interface
compatibility but only best-effort filtering is applied — Yahoo is the
"recent news" option; use Tavily for point-in-time. Handles both the legacy and
current yfinance news payload shapes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from agentic_finance.news.base import NewsArticle

__all__ = ["YahooNewsProvider"]


class YahooNewsProvider:
    """Recent news via yfinance ``Ticker.news``."""

    name = "yahoo"

    @staticmethod
    def _parse(items) -> List[NewsArticle]:
        articles: List[NewsArticle] = []
        for item in items or []:
            content = item.get("content") if isinstance(item.get("content"), dict) else None
            if content:  # current shape
                title = content.get("title")
                url = (content.get("canonicalUrl") or {}).get("url", "")
                published = content.get("pubDate", "")
                source = (content.get("provider") or {}).get("displayName", "") if isinstance(
                    content.get("provider"), dict) else ""
            else:  # legacy shape
                title = item.get("title")
                url = item.get("link", "")
                ts = item.get("providerPublishTime")
                published = (
                    datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else ""
                )
                source = item.get("publisher", "")
            if not title:
                continue
            netloc = ""
            if url:
                from urllib.parse import urlparse
                netloc = urlparse(url).netloc
            articles.append(NewsArticle(title=title, published=published, url=url,
                                        source=source or netloc, snippet=""))
        return articles

    @staticmethod
    def _symbol(query: str) -> str:
        """Reduce a free-text news query to the bare ticker Yahoo expects.

        Callers phrase queries for a search API (``"AAPL stock"``), which Tavily
        handles but ``yf.Ticker`` does not — it needs the symbol alone, and
        silently returns no news for anything else. Taking the first token keeps
        both call styles working.
        """
        return (query or "").strip().split()[0] if (query or "").strip() else ""

    def fetch(self, query: str, start: Optional[str] = None,
              end: Optional[str] = None, limit: int = 10) -> List[NewsArticle]:
        import yfinance as yf

        symbol = self._symbol(query)
        if not symbol:
            return []
        try:
            items = yf.Ticker(symbol).news or []
        except Exception:
            items = []
        return self._parse(items)[:limit]
