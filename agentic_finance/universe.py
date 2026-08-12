"""Curated company & sector registry (US large-caps + NIFTY-50), source-agnostic.

A small, offline, dependency-free lookup that powers the UI's company picker and
the sector-based suggestion engine. It intentionally uses a *static* curated map
(rather than live yfinance ``.info`` calls) for speed and reliability during a
live demo. Indian tickers carry the ``.NS`` (NSE) suffix that yfinance expects;
US tickers are bare. A shared sector taxonomy lets peers span both markets.

Nothing here fetches data — it only maps names/tickers to sectors, regions, and
currencies. Any ticker not in the registry still works downstream (yfinance is
called with it directly); registry membership only enables name-search and peers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

__all__ = [
    "Company",
    "SECTORS",
    "COMPANIES",
    "search",
    "resolve",
    "get_company",
    "sector_of",
    "peers",
    "region_of",
    "currency_code",
    "currency_symbol",
]

# Shared taxonomy so a US stock and an Indian stock can be "sector peers".
SECTORS = (
    "IT", "Financials", "Energy", "Auto", "Healthcare",
    "Consumer", "Materials", "Communication", "Industrials", "Index",
)


@dataclass(frozen=True)
class Company:
    name: str
    ticker: str
    sector: str
    region: str      # "US" | "IN"
    currency: str    # "USD" | "INR"


def _us(name: str, ticker: str, sector: str) -> Company:
    return Company(name, ticker, sector, "US", "USD")


def _in(name: str, ticker: str, sector: str) -> Company:
    return Company(name, ticker, sector, "IN", "INR")


# --- US large-caps ---------------------------------------------------------
_US_COMPANIES = [
    _us("Apple", "AAPL", "IT"),
    _us("Microsoft", "MSFT", "IT"),
    _us("Alphabet (Google)", "GOOGL", "Communication"),
    _us("NVIDIA", "NVDA", "IT"),
    _us("Meta Platforms", "META", "Communication"),
    _us("Amazon", "AMZN", "Consumer"),
    _us("Advanced Micro Devices", "AMD", "IT"),
    _us("Intel", "INTC", "IT"),
    _us("Adobe", "ADBE", "IT"),
    _us("Salesforce", "CRM", "IT"),
    _us("Oracle", "ORCL", "IT"),
    _us("JPMorgan Chase", "JPM", "Financials"),
    _us("Bank of America", "BAC", "Financials"),
    _us("Goldman Sachs", "GS", "Financials"),
    _us("Visa", "V", "Financials"),
    _us("Mastercard", "MA", "Financials"),
    _us("Exxon Mobil", "XOM", "Energy"),
    _us("Chevron", "CVX", "Energy"),
    _us("Tesla", "TSLA", "Auto"),
    _us("General Motors", "GM", "Auto"),
    _us("Ford", "F", "Auto"),
    _us("Johnson & Johnson", "JNJ", "Healthcare"),
    _us("Pfizer", "PFE", "Healthcare"),
    _us("UnitedHealth", "UNH", "Healthcare"),
    _us("Coca-Cola", "KO", "Consumer"),
    _us("PepsiCo", "PEP", "Consumer"),
    _us("Procter & Gamble", "PG", "Consumer"),
    _us("Walmart", "WMT", "Consumer"),
    _us("McDonald's", "MCD", "Consumer"),
    _us("Netflix", "NFLX", "Communication"),
    _us("Caterpillar", "CAT", "Industrials"),
    _us("S&P 500 ETF", "SPY", "Index"),
    _us("Nasdaq-100 ETF", "QQQ", "Index"),
]

# --- NIFTY-50 (NSE, .NS suffix) -------------------------------------------
_IN_COMPANIES = [
    _in("Tata Consultancy Services", "TCS.NS", "IT"),
    _in("Infosys", "INFY.NS", "IT"),
    _in("HCL Technologies", "HCLTECH.NS", "IT"),
    _in("Wipro", "WIPRO.NS", "IT"),
    _in("Tech Mahindra", "TECHM.NS", "IT"),
    _in("HDFC Bank", "HDFCBANK.NS", "Financials"),
    _in("ICICI Bank", "ICICIBANK.NS", "Financials"),
    _in("State Bank of India", "SBIN.NS", "Financials"),
    _in("Kotak Mahindra Bank", "KOTAKBANK.NS", "Financials"),
    _in("Axis Bank", "AXISBANK.NS", "Financials"),
    _in("IndusInd Bank", "INDUSINDBK.NS", "Financials"),
    _in("Bajaj Finance", "BAJFINANCE.NS", "Financials"),
    _in("Bajaj Finserv", "BAJAJFINSV.NS", "Financials"),
    _in("HDFC Life Insurance", "HDFCLIFE.NS", "Financials"),
    _in("SBI Life Insurance", "SBILIFE.NS", "Financials"),
    _in("Reliance Industries", "RELIANCE.NS", "Energy"),
    _in("Oil & Natural Gas Corp", "ONGC.NS", "Energy"),
    _in("NTPC", "NTPC.NS", "Energy"),
    _in("Power Grid Corp", "POWERGRID.NS", "Energy"),
    _in("Coal India", "COALINDIA.NS", "Energy"),
    _in("Bharat Petroleum", "BPCL.NS", "Energy"),
    _in("Maruti Suzuki", "MARUTI.NS", "Auto"),
    _in("Tata Motors", "TATAMOTORS.NS", "Auto"),
    _in("Mahindra & Mahindra", "M&M.NS", "Auto"),
    _in("Bajaj Auto", "BAJAJ-AUTO.NS", "Auto"),
    _in("Eicher Motors", "EICHERMOT.NS", "Auto"),
    _in("Hero MotoCorp", "HEROMOTOCO.NS", "Auto"),
    _in("Sun Pharmaceutical", "SUNPHARMA.NS", "Healthcare"),
    _in("Cipla", "CIPLA.NS", "Healthcare"),
    _in("Dr. Reddy's Laboratories", "DRREDDY.NS", "Healthcare"),
    _in("Divi's Laboratories", "DIVISLAB.NS", "Healthcare"),
    _in("Apollo Hospitals", "APOLLOHOSP.NS", "Healthcare"),
    _in("Hindustan Unilever", "HINDUNILVR.NS", "Consumer"),
    _in("ITC", "ITC.NS", "Consumer"),
    _in("Nestle India", "NESTLEIND.NS", "Consumer"),
    _in("Britannia Industries", "BRITANNIA.NS", "Consumer"),
    _in("Tata Consumer Products", "TATACONSUM.NS", "Consumer"),
    _in("Titan Company", "TITAN.NS", "Consumer"),
    _in("Tata Steel", "TATASTEEL.NS", "Materials"),
    _in("JSW Steel", "JSWSTEEL.NS", "Materials"),
    _in("Hindalco Industries", "HINDALCO.NS", "Materials"),
    _in("UltraTech Cement", "ULTRACEMCO.NS", "Materials"),
    _in("Grasim Industries", "GRASIM.NS", "Materials"),
    _in("Asian Paints", "ASIANPAINT.NS", "Materials"),
    _in("Larsen & Toubro", "LT.NS", "Industrials"),
    _in("Adani Ports & SEZ", "ADANIPORTS.NS", "Industrials"),
    _in("Bharti Airtel", "BHARTIARTL.NS", "Communication"),
    _in("NIFTY 50 Index", "^NSEI", "Index"),
]

COMPANIES: List[Company] = _US_COMPANIES + _IN_COMPANIES

_BY_TICKER = {c.ticker.upper(): c for c in COMPANIES}


# --- lookups ---------------------------------------------------------------
def search(query: str, limit: int = 12) -> List[Company]:
    """Case-insensitive substring match on company name or ticker.

    Results are ranked: name-prefix, then ticker-prefix, then any substring.
    """
    q = (query or "").strip().lower()
    if not q:
        return COMPANIES[:limit]
    name_prefix, ticker_prefix, contains = [], [], []
    for c in COMPANIES:
        name, tick = c.name.lower(), c.ticker.lower()
        if name.startswith(q):
            name_prefix.append(c)
        elif tick.startswith(q):
            ticker_prefix.append(c)
        elif q in name or q in tick:
            contains.append(c)
    return (name_prefix + ticker_prefix + contains)[:limit]


def get_company(ticker: str) -> Optional[Company]:
    return _BY_TICKER.get((ticker or "").upper())


def resolve(name_or_ticker: str) -> str:
    """Return a canonical ticker for a name or ticker.

    If the input matches a registered company (by name or ticker) its ticker is
    returned; otherwise the input is treated as a raw ticker and upper-cased.
    """
    s = (name_or_ticker or "").strip()
    if not s:
        return s
    hit = _BY_TICKER.get(s.upper())
    if hit:
        return hit.ticker
    for c in COMPANIES:
        if c.name.lower() == s.lower():
            return c.ticker
    # Unknown -> assume a raw ticker (preserve any exchange suffix casing).
    return s.upper() if "." not in s else s


def sector_of(ticker: str) -> Optional[str]:
    c = get_company(ticker)
    return c.sector if c else None


def peers(ticker: str, exclude_self: bool = True) -> List[Company]:
    """Sector peers of ``ticker`` from the registry (empty if unknown/Index)."""
    sector = sector_of(ticker)
    if not sector or sector == "Index":
        return []
    out = [c for c in COMPANIES if c.sector == sector and c.sector != "Index"]
    if exclude_self:
        out = [c for c in out if c.ticker.upper() != (ticker or "").upper()]
    return out


# --- region / currency (works for any ticker via exchange suffix) ----------
def region_of(ticker: str) -> str:
    """Coarse region for benchmark selection: 'IN' for NSE/BSE, else 'US'."""
    t = (ticker or "").upper()
    return "IN" if t.endswith(".NS") or t.endswith(".BO") or t.startswith("^NSE") else "US"


# Exchange-suffix -> (currency code, symbol). Covers the common global venues so
# a searched foreign stock displays a sensible currency; defaults to USD.
_SUFFIX_CCY = {
    ".NS": ("INR", "₹"), ".BO": ("INR", "₹"),
    ".L": ("GBP", "£"),
    ".DE": ("EUR", "€"), ".F": ("EUR", "€"), ".PA": ("EUR", "€"),
    ".AS": ("EUR", "€"), ".MI": ("EUR", "€"), ".MC": ("EUR", "€"), ".BR": ("EUR", "€"),
    ".T": ("JPY", "¥"), ".HK": ("HKD", "HK$"),
    ".TO": ("CAD", "C$"), ".V": ("CAD", "C$"),
    ".AX": ("AUD", "A$"), ".NZ": ("NZD", "NZ$"),
    ".SW": ("CHF", "CHF "), ".SS": ("CNY", "¥"), ".SZ": ("CNY", "¥"),
    ".SA": ("BRL", "R$"), ".KS": ("KRW", "₩"), ".TW": ("TWD", "NT$"),
}


def _ccy(ticker: str):
    t = (ticker or "").upper()
    if t.startswith("^NSE"):
        return ("INR", "₹")
    for suffix, ccy in _SUFFIX_CCY.items():
        if t.endswith(suffix):
            return ccy
    return ("USD", "$")


def currency_code(ticker: str) -> str:
    return _ccy(ticker)[0]


def currency_symbol(ticker: str) -> str:
    return _ccy(ticker)[1]
