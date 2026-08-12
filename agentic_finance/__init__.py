"""Agentic Finance — multi-agent, risk-mitigated, explainable trading framework.

A typed-signal agent pipeline (specialist agents -> fusion -> quantitative risk
engine -> explainability) with configurable generative (Gemini/Ollama) and news
(Tavily/Yahoo) backends. See README.md and PROJECT_OVERVIEW.md.
"""
# Auto-load a .env file from the working directory tree, if python-dotenv is
# present, so API keys / AGENTIC_* settings are available without manual export.
try:  # pragma: no cover - environment-dependent
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except Exception:
    pass

# On Windows behind a TLS-scanning AV/proxy (e.g. Avast Web Shield), HTTPS certs are
# re-signed by a local root CA that certifi doesn't know about, breaking yfinance and
# Tavily. Trust the OS certificate store so live-data demos work without disabling
# verification. No-op elsewhere; respects any user-set cert env vars.
try:  # pragma: no cover - environment-dependent
    from agentic_finance.sslcompat import ensure_ca_bundle

    ensure_ca_bundle()
except Exception:
    pass

__version__ = "0.1.0"
