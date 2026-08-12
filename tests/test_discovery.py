"""Unit tests for discovery payload parsers (offline, canned Yahoo payloads)."""
from agentic_finance.discovery import SymbolHit, _parse_search, _parse_similar

_SEARCH_PAYLOAD = {
    "quotes": [
        {"symbol": "INFY", "shortname": "Infosys Limited", "exchDisp": "NYSE",
         "quoteType": "EQUITY"},
        {"symbol": "INFY.NS", "longname": "INFOSYS LIMITED", "exchDisp": "NSE",
         "quoteType": "EQUITY"},
        {"symbol": "INFY", "shortname": "dup should dedupe", "quoteType": "EQUITY"},
        {"symbol": "OPTX", "shortname": "some option", "quoteType": "OPTION"},  # filtered
        {"symbol": "SPY", "shortname": "S&P 500 ETF", "quoteType": "ETF"},
    ]
}

_SIMILAR_PAYLOAD = {
    "finance": {"result": [
        {"symbol": "AAPL", "recommendedSymbols": [
            {"symbol": "MSFT", "score": 0.2}, {"symbol": "GOOG", "score": 0.1},
            {"symbol": "AAPL", "score": 0.9},  # self, kept here (filtered in caller)
        ]}
    ]}
}


def test_parse_search_dedupes_and_filters_types():
    hits = _parse_search(_SEARCH_PAYLOAD)
    syms = [h.symbol for h in hits]
    assert syms == ["INFY", "INFY.NS", "SPY"]      # OPTION dropped, INFY deduped
    assert all(isinstance(h, SymbolHit) for h in hits)
    assert hits[0].name == "Infosys Limited"
    assert "NSE" in hits[1].label


def test_parse_search_respects_limit():
    assert len(_parse_search(_SEARCH_PAYLOAD, limit=2)) == 2


def test_parse_search_empty_and_malformed():
    assert _parse_search({}) == []
    assert _parse_search({"quotes": []}) == []
    assert _parse_search(None) == []


def test_parse_similar():
    assert _parse_similar(_SIMILAR_PAYLOAD) == ["MSFT", "GOOG", "AAPL"]


def test_parse_similar_malformed():
    assert _parse_similar({}) == []
    assert _parse_similar({"finance": {"result": []}}) == []
    assert _parse_similar(None) == []
