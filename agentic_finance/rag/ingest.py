"""RAG ingestion + retrieval orchestration (idempotent, never raises).

``ingest_ticker`` pulls a ticker's recent filings from EDGAR into the local
vector store, skipping any document already recorded in the manifest.
``ensure_ingested`` is the runtime guard used by the decision pipeline: it
attempts ingestion **at most once per ticker** (the manifest remembers even
unsuccessful attempts, so a ticker with no SEC filings — e.g. a purely
domestic Indian name — is silently skipped forever after).
``retrieve_filing_context`` returns the top-matching excerpts with the
point-in-time (as-of) filter applied.

CLI: ``python -m agentic_finance.rag.ingest NVDA AAPL INFY.NS``
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from agentic_finance.rag.chunking import chunk_text
from agentic_finance.rag.edgar import (
    DEFAULT_FORMS,
    cik_for,
    fetch_filing_text,
    list_filings,
)
from agentic_finance.rag.store import (
    Embedder,
    LocalVectorStore,
    RagChunk,
    get_embedder,
)

__all__ = ["IngestResult", "ingest_ticker", "ensure_ingested",
           "retrieve_filing_context", "get_default_store"]

_DEFAULT_DIR = "data/rag_index"
_store_cache: dict = {}


def get_default_store() -> LocalVectorStore:
    """Shared store at ``AGENTIC_RAG_DIR`` (default ``data/rag_index``)."""
    path = Path(os.environ.get("AGENTIC_RAG_DIR", _DEFAULT_DIR))
    key = str(path.resolve())
    if key not in _store_cache:
        _store_cache[key] = LocalVectorStore(path)
    return _store_cache[key]


@dataclass(frozen=True)
class IngestResult:
    """Outcome of one ingestion pass (all zeros when nothing was available)."""

    ticker: str
    fetched: int = 0        # documents downloaded + embedded this run
    skipped: int = 0        # documents already in the index
    chunks_added: int = 0


def ingest_ticker(ticker: str, store: Optional[LocalVectorStore] = None,
                  embedder: Optional[Embedder] = None,
                  forms: Tuple[str, ...] = DEFAULT_FORMS,
                  max_filings: int = 3) -> IngestResult:
    """Ingest a ticker's recent filings; already-ingested documents are skipped.

    Never raises: no CIK, no filings, or any network/parse failure simply
    yields a zero/partial :class:`IngestResult`.
    """
    store = store or get_default_store()
    embedder = embedder or get_embedder()
    fetched = skipped = chunks_added = 0
    try:
        cik = cik_for(ticker)
        if cik is None:
            return IngestResult(ticker=ticker)
        for filing in list_filings(cik, forms=forms, max_filings=max_filings):
            if store.is_ingested(filing.accession):
                skipped += 1
                continue
            text = fetch_filing_text(filing)
            if not text:
                continue
            pieces = chunk_text(text)
            chunks = [
                RagChunk(
                    chunk_id=f"{filing.accession}-{i}", doc_id=filing.accession,
                    ticker=ticker.upper(), form=filing.form,
                    filing_date=filing.filing_date, text=piece, source="edgar",
                )
                for i, piece in enumerate(pieces)
            ]
            store.add(filing.accession, chunks,
                      embedder.embed([c.text for c in chunks]))
            fetched += 1
            chunks_added += len(chunks)
    except Exception:  # noqa: BLE001 - ingestion must never break a caller
        pass
    return IngestResult(ticker=ticker, fetched=fetched, skipped=skipped,
                        chunks_added=chunks_added)


def ensure_ingested(ticker: str, store: Optional[LocalVectorStore] = None,
                    embedder: Optional[Embedder] = None) -> Optional[IngestResult]:
    """Runtime guard: ingest a ticker at most once (attempts are remembered).

    Returns the :class:`IngestResult` when an ingestion ran, ``None`` when the
    ticker had already been attempted. Never raises.
    """
    store = store or get_default_store()
    try:
        if store.ticker_attempted(ticker):
            return None
        result = ingest_ticker(ticker, store=store, embedder=embedder)
        store.mark_ticker_attempted(ticker, filings=result.fetched)
        return result
    except Exception:  # noqa: BLE001
        return None


def retrieve_filing_context(ticker: str, query: str, as_of: Optional[int] = None,
                            k: int = 3, store: Optional[LocalVectorStore] = None,
                            embedder: Optional[Embedder] = None,
                            max_chars: int = 700) -> List[str]:
    """Top-``k`` filing excerpts for a query, labelled and as-of filtered.

    Returns ``[]`` for unknown tickers, empty stores, or any failure.
    """
    store = store or get_default_store()
    embedder = embedder or get_embedder()
    try:
        vec = embedder.embed([query])[0]
        hits = store.query(vec, ticker=ticker, as_of=as_of, k=k)
        out: List[str] = []
        for h in hits:
            d = str(h.filing_date)
            label = f"[{h.form} filed {d[:4]}-{d[4:6]}-{d[6:]}]"
            out.append(f"{label} {h.text[:max_chars]}")
        return out
    except Exception:  # noqa: BLE001
        return []


def _main(argv: List[str]) -> int:
    tickers = argv or ["NVDA"]
    store = get_default_store()
    embedder = get_embedder()
    for ticker in tickers:
        result = ingest_ticker(ticker, store=store, embedder=embedder)
        store.mark_ticker_attempted(ticker, filings=result.fetched)
        print(f"{ticker}: fetched={result.fetched} skipped={result.skipped} "
              f"chunks_added={result.chunks_added} (index total {store.count()})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main(sys.argv[1:]))
