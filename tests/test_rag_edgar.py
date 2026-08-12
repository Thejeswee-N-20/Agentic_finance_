"""Tests for the EDGAR client (offline: all HTTP stubbed; failures degrade)."""
from __future__ import annotations

import agentic_finance.rag.edgar as edgar
from agentic_finance.rag.edgar import Filing, cik_for, fetch_filing_text, list_filings


class _Resp:
    def __init__(self, payload=None, text=""):
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


def _stub_get(payloads):
    """Return a fake requests.get keyed by URL substring."""
    def _get(url, headers=None, timeout=None):
        for key, value in payloads.items():
            if key in url:
                return value
        raise RuntimeError(f"unexpected URL {url}")
    return _get


TICKER_MAP = {"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
              "1": {"cik_str": 1067491, "ticker": "INFY", "title": "INFOSYS LTD"}}

SUBMISSIONS = {
    "cik": 1045810,
    "filings": {"recent": {
        "accessionNumber": ["0001045810-24-000029", "0001045810-24-000010", "0001045810-23-000017"],
        "form": ["10-K", "8-K", "10-Q"],
        "filingDate": ["2024-02-21", "2024-01-10", "2023-08-25"],
        "primaryDocument": ["nvda-20240128.htm", "nvda-8k.htm", "nvda-20230730.htm"],
    }},
}


class TestCikFor:
    def test_resolves_ticker(self, monkeypatch):
        monkeypatch.setattr(edgar._session(), "get", _stub_get({"company_tickers": _Resp(TICKER_MAP)}))
        edgar._reset_caches()
        assert cik_for("NVDA") == "0001045810"

    def test_strips_indian_suffix_to_adr(self, monkeypatch):
        monkeypatch.setattr(edgar._session(), "get", _stub_get({"company_tickers": _Resp(TICKER_MAP)}))
        edgar._reset_caches()
        assert cik_for("INFY.NS") == "0001067491"

    def test_unknown_ticker_returns_none(self, monkeypatch):
        monkeypatch.setattr(edgar._session(), "get", _stub_get({"company_tickers": _Resp(TICKER_MAP)}))
        edgar._reset_caches()
        assert cik_for("RELIANCE.NS") is None

    def test_network_failure_returns_none(self, monkeypatch):
        def _boom(url, headers=None, timeout=None):
            raise RuntimeError("offline")
        monkeypatch.setattr(edgar._session(), "get", _boom)
        edgar._reset_caches()
        assert cik_for("NVDA") is None


class TestListFilings:
    def test_parses_and_filters_forms(self, monkeypatch):
        monkeypatch.setattr(edgar._session(), "get", _stub_get({"submissions": _Resp(SUBMISSIONS)}))
        filings = list_filings("0001045810", forms=("10-K", "10-Q"))
        assert [f.form for f in filings] == ["10-K", "10-Q"]
        f = filings[0]
        assert f.accession == "0001045810-24-000029"
        assert f.filing_date == 20240221
        assert "nvda-20240128.htm" in f.url

    def test_max_filings_cap(self, monkeypatch):
        monkeypatch.setattr(edgar._session(), "get", _stub_get({"submissions": _Resp(SUBMISSIONS)}))
        assert len(list_filings("0001045810", forms=("10-K", "10-Q"), max_filings=1)) == 1

    def test_failure_returns_empty(self, monkeypatch):
        def _boom(url, headers=None, timeout=None):
            raise RuntimeError("offline")
        monkeypatch.setattr(edgar._session(), "get", _boom)
        assert list_filings("0001045810") == []


class TestFetchFilingText:
    def test_returns_cleaned_text(self, monkeypatch):
        html = "<html><body><p>Risk factors include demand.</p></body></html>"
        monkeypatch.setattr(edgar._session(), "get", _stub_get({"Archives": _Resp(text=html)}))
        filing = Filing(accession="x", form="10-K", filing_date=20240221,
                        url="https://www.sec.gov/Archives/edgar/data/x.htm")
        text = fetch_filing_text(filing)
        assert "Risk factors include demand." in text
        assert "<p>" not in text

    def test_failure_returns_empty_string(self, monkeypatch):
        def _boom(url, headers=None, timeout=None):
            raise RuntimeError("offline")
        monkeypatch.setattr(edgar._session(), "get", _boom)
        filing = Filing(accession="x", form="10-K", filing_date=20240221,
                        url="https://www.sec.gov/Archives/x.htm")
        assert fetch_filing_text(filing) == ""
