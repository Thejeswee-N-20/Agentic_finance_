"""Leakage-safe, point-in-time RAG over company filings (SEC EDGAR).

Retrieval-Augmented Generation for the news-reasoning agent: 10-K/10-Q/20-F
filings are downloaded once from EDGAR, cleaned, chunked, embedded, and stored
in a small local vector index. At decision time the top-matching chunks — with
filing dates **on or before the decision date** — are appended to the agent's
context, so the LLM reasons over real filing text without look-ahead leakage.

Design mirrors the framework's provider philosophy: zero required heavy
dependencies (numpy vector store + deterministic hashing embedder by default;
sentence-transformers is an opt-in upgrade), idempotent ingestion (a manifest
records ingested documents and attempted tickers, so nothing is fetched twice),
and total graceful degradation — a ticker with no EDGAR filings, a network
outage, or a parse failure all yield an empty context, never an error.
"""

from agentic_finance.rag.chunking import chunk_text, html_to_text
from agentic_finance.rag.edgar import Filing, cik_for, fetch_filing_text, list_filings
from agentic_finance.rag.ingest import (
    IngestResult,
    ensure_ingested,
    ingest_ticker,
    retrieve_filing_context,
)
from agentic_finance.rag.store import HashingEmbedder, LocalVectorStore, RagChunk

__all__ = [
    "chunk_text",
    "html_to_text",
    "Filing",
    "cik_for",
    "list_filings",
    "fetch_filing_text",
    "IngestResult",
    "ingest_ticker",
    "ensure_ingested",
    "retrieve_filing_context",
    "HashingEmbedder",
    "LocalVectorStore",
    "RagChunk",
]
