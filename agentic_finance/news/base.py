"""News data contract.

A ``NewsArticle`` is the normalized unit every provider returns; ``NewsProvider``
is the protocol agents depend on. ``fetch`` takes an optional ``start``/``end``
date range (``YYYY-MM-DD``) so callers can request **point-in-time** news.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable

__all__ = ["NewsArticle", "NewsProvider"]


@dataclass(frozen=True)
class NewsArticle:
    """A single normalized news item."""

    title: str
    published: str   # raw provider timestamp (kept as-is; formats vary)
    url: str
    source: str
    snippet: str = ""


@runtime_checkable
class NewsProvider(Protocol):
    """Fetches news for a query/ticker, optionally within a date range."""

    name: str

    def fetch(self, query: str, start: Optional[str] = None,
              end: Optional[str] = None, limit: int = 10) -> List[NewsArticle]:
        ...
