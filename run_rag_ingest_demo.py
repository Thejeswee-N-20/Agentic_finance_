"""Ingest SEC filings into the local RAG index for one or more tickers.

Usage:
    python run_rag_ingest_demo.py NVDA AAPL INFY.NS

Idempotent: already-ingested documents and already-attempted tickers are
skipped (see data/rag_index/manifest.json). Tickers without SEC filings
(e.g. purely domestic Indian names) are silently recorded as attempted.
The live decision pipeline also ingests on first use, so this script is a
convenience for pre-warming the index before a demo.
"""
import sys

from agentic_finance.rag.ingest import _main

if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
