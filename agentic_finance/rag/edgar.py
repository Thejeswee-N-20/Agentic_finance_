"""SEC EDGAR client: free, official filings source (no API key).

Three endpoints, all public:

1. ``company_tickers.json`` — ticker -> CIK number
2. ``data.sec.gov/submissions/CIK##########.json`` — a company's filing index
3. ``www.sec.gov/Archives/...`` — the filing document itself

SEC policy requires a descriptive ``User-Agent`` and <= 10 requests/second —
both respected here. Indian (and other non-US) tickers are handled by
stripping the ``.NS``/``.BO`` suffix and matching the US ADR symbol where one
exists (e.g. ``INFY.NS`` -> INFY's 20-F filings); tickers with no SEC presence
simply resolve to ``None``. Every function degrades on failure — ``None``,
``[]``, or ``""`` — and never raises, so a filings outage can never break a
decision.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests

__all__ = ["Filing", "cik_for", "list_filings", "fetch_filing_text"]

_HEADERS = {"User-Agent": "AgenticFinance academic research (thejeshnaidu555@gmail.com)"}
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_TIMEOUT = 20
_MIN_INTERVAL = 0.12  # <= ~8 req/s, inside SEC's 10 req/s policy

DEFAULT_FORMS: Tuple[str, ...] = ("10-K", "10-Q", "20-F")

_session_obj: Optional[requests.Session] = None
_ticker_map: Optional[dict] = None
_last_request = 0.0


def _session() -> requests.Session:
    global _session_obj
    if _session_obj is None:
        _session_obj = requests.Session()
    return _session_obj


def _reset_caches() -> None:
    """Test hook: clear the in-memory ticker map."""
    global _ticker_map
    _ticker_map = None


def _get(url: str):
    """Rate-limited GET through the shared session."""
    global _last_request
    wait = _MIN_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()
    return _session().get(url, headers=_HEADERS, timeout=_TIMEOUT)


@dataclass(frozen=True)
class Filing:
    """One EDGAR filing reference."""

    accession: str
    form: str
    filing_date: int  # YYYYMMDD
    url: str


def _base_symbol(ticker: str) -> str:
    """Strip exchange suffixes (``.NS``/``.BO``/…) to try the US/ADR symbol."""
    return ticker.split(".")[0].upper()


def cik_for(ticker: str) -> Optional[str]:
    """Resolve a ticker to its 10-digit CIK, or ``None`` if not SEC-registered."""
    global _ticker_map
    try:
        if _ticker_map is None:
            resp = _get(_TICKER_MAP_URL)
            resp.raise_for_status()
            _ticker_map = {
                str(row["ticker"]).upper(): int(row["cik_str"])
                for row in resp.json().values()
            }
        cik = _ticker_map.get(_base_symbol(ticker))
        return f"{cik:010d}" if cik is not None else None
    except Exception:  # noqa: BLE001 - no SEC data must never break callers
        return None


def _date_int(iso: str) -> int:
    return int(iso.replace("-", "")) if iso else 0


def list_filings(cik: str, forms: Tuple[str, ...] = DEFAULT_FORMS,
                 max_filings: int = 3) -> List[Filing]:
    """List the most recent filings of the given forms (newest first)."""
    try:
        resp = _get(_SUBMISSIONS_URL.format(cik=cik))
        resp.raise_for_status()
        recent = resp.json()["filings"]["recent"]
        filings: List[Filing] = []
        for accession, form, filed, doc in zip(
            recent["accessionNumber"], recent["form"],
            recent["filingDate"], recent["primaryDocument"],
        ):
            if form not in forms or not doc:
                continue
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{accession.replace('-', '')}/{doc}"
            )
            filings.append(Filing(accession=accession, form=form,
                                  filing_date=_date_int(filed), url=url))
            if len(filings) >= max_filings:
                break
        return filings
    except Exception:  # noqa: BLE001
        return []


def fetch_filing_text(filing: Filing) -> str:
    """Download a filing and return its cleaned plain text ('' on any failure)."""
    from agentic_finance.rag.chunking import html_to_text

    try:
        resp = _get(filing.url)
        resp.raise_for_status()
        return html_to_text(resp.text)
    except Exception:  # noqa: BLE001
        return ""
