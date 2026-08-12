"""Unit tests for the company & sector registry (offline, no network)."""
from agentic_finance import universe


def test_search_by_name_and_ticker():
    assert any(c.ticker == "INFY.NS" for c in universe.search("infosys"))
    assert any(c.ticker == "AAPL" for c in universe.search("AAPL"))
    # prefix ranking: querying "app" surfaces Apple near the top
    res = universe.search("app")
    assert res and any(c.ticker == "AAPL" for c in res)


def test_resolve_name_ticker_and_raw():
    assert universe.resolve("Infosys") == "INFY.NS"
    assert universe.resolve("aapl") == "AAPL"
    # unknown raw US ticker passes through upper-cased
    assert universe.resolve("abcd") == "ABCD"
    # unknown Indian ticker keeps its suffix
    assert universe.resolve("XYZ.NS") == "XYZ.NS"


def test_sector_and_peers():
    assert universe.sector_of("INFY.NS") == "IT"
    peers = universe.peers("INFY.NS")
    tickers = {c.ticker for c in peers}
    assert "TCS.NS" in tickers          # an IT peer
    assert "INFY.NS" not in tickers     # self excluded
    assert all(c.sector == "IT" for c in peers)


def test_index_has_no_peers():
    assert universe.peers("SPY") == []


def test_region_and_currency():
    assert universe.region_of("INFY.NS") == "IN"
    assert universe.region_of("AAPL") == "US"
    assert universe.currency_symbol("RELIANCE.NS") == "₹"
    assert universe.currency_symbol("MSFT") == "$"
    assert universe.currency_code("TCS.NS") == "INR"
