"""Tests for RAG ingestion + retrieval orchestration (offline; EDGAR stubbed)."""
from __future__ import annotations

import agentic_finance.rag.ingest as ingest_mod
from agentic_finance.rag.edgar import Filing
from agentic_finance.rag.ingest import ensure_ingested, ingest_ticker, retrieve_filing_context
from agentic_finance.rag.store import HashingEmbedder, LocalVectorStore

_FILING = Filing(accession="0001045810-24-000029", form="10-K",
                 filing_date=20240221, url="https://sec.gov/Archives/nvda.htm")
_TEXT = ("Risk factors include competition and export controls. " * 30
         + "Data-center revenue grew significantly year over year. " * 30)


def _stub_edgar(monkeypatch, cik="0001045810", filings=(_FILING,), text=_TEXT):
    monkeypatch.setattr(ingest_mod, "cik_for", lambda t: cik)
    monkeypatch.setattr(ingest_mod, "list_filings",
                        lambda c, forms=None, max_filings=3: list(filings))
    monkeypatch.setattr(ingest_mod, "fetch_filing_text", lambda f: text)


class TestIngestTicker:
    def test_ingests_chunks(self, tmp_path, monkeypatch):
        _stub_edgar(monkeypatch)
        store = LocalVectorStore(tmp_path / "idx")
        result = ingest_ticker("NVDA", store=store, embedder=HashingEmbedder(dim=64))
        assert result.fetched == 1
        assert result.skipped == 0
        assert result.chunks_added > 0
        assert store.is_ingested(_FILING.accession)

    def test_second_run_skips_ingested_documents(self, tmp_path, monkeypatch):
        _stub_edgar(monkeypatch)
        store = LocalVectorStore(tmp_path / "idx")
        emb = HashingEmbedder(dim=64)
        first = ingest_ticker("NVDA", store=store, embedder=emb)
        second = ingest_ticker("NVDA", store=store, embedder=emb)
        assert first.fetched == 1
        assert second.fetched == 0
        assert second.skipped == 1
        assert store.count() == first.chunks_added  # no duplicates

    def test_no_cik_is_graceful_zero(self, tmp_path, monkeypatch):
        _stub_edgar(monkeypatch, cik=None)
        store = LocalVectorStore(tmp_path / "idx")
        result = ingest_ticker("RELIANCE.NS", store=store,
                               embedder=HashingEmbedder(dim=64))
        assert result.fetched == 0 and result.chunks_added == 0

    def test_empty_document_text_is_skipped_gracefully(self, tmp_path, monkeypatch):
        _stub_edgar(monkeypatch, text="")
        store = LocalVectorStore(tmp_path / "idx")
        result = ingest_ticker("NVDA", store=store, embedder=HashingEmbedder(dim=64))
        assert result.chunks_added == 0


class TestEnsureIngested:
    def test_first_call_ingests_then_marks(self, tmp_path, monkeypatch):
        _stub_edgar(monkeypatch)
        store = LocalVectorStore(tmp_path / "idx")
        ensure_ingested("NVDA", store=store, embedder=HashingEmbedder(dim=64))
        assert store.ticker_attempted("NVDA")
        assert store.count() > 0

    def test_second_call_does_not_refetch(self, tmp_path, monkeypatch):
        _stub_edgar(monkeypatch)
        store = LocalVectorStore(tmp_path / "idx")
        emb = HashingEmbedder(dim=64)
        ensure_ingested("NVDA", store=store, embedder=emb)

        def _boom(*a, **k):
            raise AssertionError("must not refetch an attempted ticker")
        monkeypatch.setattr(ingest_mod, "cik_for", _boom)
        ensure_ingested("NVDA", store=store, embedder=emb)  # no error, no refetch

    def test_ticker_without_filings_marked_and_silent(self, tmp_path, monkeypatch):
        _stub_edgar(monkeypatch, cik=None)
        store = LocalVectorStore(tmp_path / "idx")
        ensure_ingested("RELIANCE.NS", store=store, embedder=HashingEmbedder(dim=64))
        assert store.ticker_attempted("RELIANCE.NS")
        assert store.count() == 0


class TestRetrieveFilingContext:
    def test_returns_relevant_chunks(self, tmp_path, monkeypatch):
        _stub_edgar(monkeypatch)
        store = LocalVectorStore(tmp_path / "idx")
        emb = HashingEmbedder(dim=64)
        ingest_ticker("NVDA", store=store, embedder=emb)
        chunks = retrieve_filing_context("NVDA", "data-center revenue growth",
                                         store=store, embedder=emb, k=2)
        assert 0 < len(chunks) <= 2
        assert all(isinstance(c, str) for c in chunks)

    def test_as_of_excludes_future_filings(self, tmp_path, monkeypatch):
        _stub_edgar(monkeypatch)
        store = LocalVectorStore(tmp_path / "idx")
        emb = HashingEmbedder(dim=64)
        ingest_ticker("NVDA", store=store, embedder=emb)
        chunks = retrieve_filing_context("NVDA", "revenue", as_of=20230101,
                                         store=store, embedder=emb)
        assert chunks == []  # only filing is dated 2024-02-21

    def test_unknown_ticker_returns_empty_never_raises(self, tmp_path):
        store = LocalVectorStore(tmp_path / "idx")
        chunks = retrieve_filing_context("RELIANCE.NS", "anything",
                                         store=store, embedder=HashingEmbedder(dim=64))
        assert chunks == []
