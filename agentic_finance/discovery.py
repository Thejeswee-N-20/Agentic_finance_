"""Open, vast company discovery — search *any* listed company, find similar ones.

Removes the hard-coded-universe limitation: instead of a fixed list, this queries
Yahoo Finance's public (no-key) endpoints so a real user can look up any global
equity/ETF/index/crypto and get genuinely relevant peers:

- ``search_symbols`` -> Yahoo autocomplete (global symbols by name/ticker).
- ``similar_symbols`` -> Yahoo "recommendationsbysymbol" (co-moved / similar names).
- ``get_sector`` / ``company_name`` -> yfinance ``.info`` (crumb-handled), cached.

Network clients are imported lazily and every call degrades gracefully to an
empty/None result, so the rest of the app never breaks if discovery is offline.
Pure parse helpers (``_parse_search`` / ``_parse_similar``) are unit-tested on
canned payloads without any network — mirroring the news-provider pattern.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

__all__ = [
    "SymbolHit", "search_symbols", "similar_symbols", "get_sector", "company_name",
]

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AgenticFinance/1.0)"}
_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
_RECS_URL = "https://query1.finance.yahoo.com/v6/finance/recommendationsbysymbol/{sym}"
_TIMEOUT = 15.0

# Instrument types a user would actually want to trade/track.
_ALLOWED_TYPES = {"EQUITY", "ETF", "INDEX", "CRYPTOCURRENCY", "MUTUALFUND"}


@dataclass(frozen=True)
class SymbolHit:
    symbol: str
    name: str
    exchange: str
    quote_type: str

    @property
    def label(self) -> str:
        ex = f" · {self.exchange}" if self.exchange else ""
        return f"{self.name} ({self.symbol}){ex}"


def _parse_search(data: dict, limit: int = 15) -> List[SymbolHit]:
    """Parse a Yahoo autocomplete payload into de-duplicated SymbolHits (pure)."""
    hits: List[SymbolHit] = []
    seen = set()
    for q in (data or {}).get("quotes", []):
        sym = (q.get("symbol") or "").strip()
        qt = (q.get("quoteType") or "").upper()
        if not sym or sym.upper() in seen or (qt and qt not in _ALLOWED_TYPES):
            continue
        seen.add(sym.upper())
        name = (q.get("shortname") or q.get("longname") or q.get("shortName")
                or q.get("longName") or sym)
        exch = q.get("exchDisp") or q.get("exchange") or ""
        hits.append(SymbolHit(sym, str(name).strip(), str(exch).strip(), qt or "EQUITY"))
        if len(hits) >= limit:
            break
    return hits


def _parse_similar(data: dict) -> List[str]:
    """Parse a recommendationsbysymbol payload into a list of symbols (pure)."""
    try:
        result = (data or {})["finance"]["result"]
        if not result:
            return []
        recs = result[0].get("recommendedSymbols", [])
        return [r["symbol"] for r in recs if r.get("symbol")]
    except (KeyError, IndexError, TypeError):
        return []


def search_symbols(query: str, limit: int = 15) -> List[SymbolHit]:
    """Live global symbol search by company name or ticker (empty on failure)."""
    q = (query or "").strip()
    if not q:
        return []
    try:
        import requests
        r = requests.get(
            _SEARCH_URL,
            params={"q": q, "quotesCount": limit, "newsCount": 0, "listsCount": 0},
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return _parse_search(r.json(), limit=limit)
    except Exception:  # noqa: BLE001 - discovery is best-effort
        return []


def similar_symbols(symbol: str, limit: int = 12) -> List[str]:
    """Yahoo's 'similar / people also watch' symbols for ``symbol`` (empty on fail)."""
    s = (symbol or "").strip()
    if not s:
        return []
    try:
        import requests
        r = requests.get(_RECS_URL.format(sym=s), headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        out = _parse_similar(r.json())
        return [x for x in out if x.upper() != s.upper()][:limit]
    except Exception:  # noqa: BLE001
        return []


@lru_cache(maxsize=1024)
def _info(symbol: str) -> dict:
    try:
        import yfinance as yf
        return yf.Ticker(symbol).info or {}
    except Exception:  # noqa: BLE001
        return {}


def get_sector(symbol: str) -> Optional[str]:
    """Best-effort sector for a symbol (via yfinance .info, cached)."""
    return _info(symbol).get("sector")


def company_name(symbol: str) -> Optional[str]:
    """Best-effort human name for a symbol (via yfinance .info, cached)."""
    info = _info(symbol)
    return info.get("shortName") or info.get("longName")
