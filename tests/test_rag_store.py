"""Tests for the local vector store + hashing embedder (offline)."""
from __future__ import annotations

from agentic_finance.rag.store import HashingEmbedder, LocalVectorStore, RagChunk


def _chunk(i: int, ticker: str = "NVDA", date: int = 20240215, text: str = "") -> RagChunk:
    return RagChunk(
        chunk_id=f"doc1-{i}", doc_id="doc1", ticker=ticker, form="10-K",
        filing_date=date, text=text or f"chunk {i} about data-center revenue growth",
        source="edgar",
    )


class TestHashingEmbedder:
    def test_deterministic_and_normalized(self):
        emb = HashingEmbedder(dim=64)
        v1 = emb.embed(["nvidia revenue growth"])[0]
        v2 = emb.embed(["nvidia revenue growth"])[0]
        assert v1 == v2
        assert len(v1) == 64
        norm = sum(x * x for x in v1) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    def test_similar_texts_are_closer_than_different(self):
        emb = HashingEmbedder(dim=256)
        a = emb.embed(["data center revenue grew strongly"])[0]
        b = emb.embed(["revenue from data center grew"])[0]
        c = emb.embed(["cryptocurrency mining ban in china"])[0]
        sim = lambda x, y: sum(p * q for p, q in zip(x, y))
        assert sim(a, b) > sim(a, c)

    def test_empty_text_is_safe(self):
        v = HashingEmbedder(dim=32).embed([""])[0]
        assert len(v) == 32


class TestLocalVectorStore:
    def test_add_and_query_roundtrip(self, tmp_path):
        store = LocalVectorStore(tmp_path / "idx")
        emb = HashingEmbedder(dim=64)
        chunks = [_chunk(i) for i in range(3)]
        store.add("doc1", chunks, emb.embed([c.text for c in chunks]))
        hits = store.query(emb.embed(["data-center revenue"])[0], ticker="NVDA", k=2)
        assert len(hits) == 2
        assert all(h.ticker == "NVDA" for h in hits)

    def test_is_ingested_and_idempotent_add(self, tmp_path):
        store = LocalVectorStore(tmp_path / "idx")
        emb = HashingEmbedder(dim=32)
        chunks = [_chunk(0)]
        assert not store.is_ingested("doc1")
        store.add("doc1", chunks, emb.embed([c.text for c in chunks]))
        assert store.is_ingested("doc1")
        # second add of the same doc is a no-op
        store.add("doc1", chunks, emb.embed([c.text for c in chunks]))
        assert store.count() == 1

    def test_as_of_filter_excludes_future_filings(self, tmp_path):
        store = LocalVectorStore(tmp_path / "idx")
        emb = HashingEmbedder(dim=32)
        old = _chunk(0, date=20230101, text="old filing revenue")
        new = RagChunk(chunk_id="doc2-0", doc_id="doc2", ticker="NVDA", form="10-K",
                       filing_date=20250101, text="new filing revenue", source="edgar")
        store.add("doc1", [old], emb.embed([old.text]))
        store.add("doc2", [new], emb.embed([new.text]))
        hits = store.query(emb.embed(["revenue"])[0], ticker="NVDA", as_of=20240101, k=5)
        assert [h.chunk_id for h in hits] == ["doc1-0"]

    def test_ticker_filter(self, tmp_path):
        store = LocalVectorStore(tmp_path / "idx")
        emb = HashingEmbedder(dim=32)
        nvda, aapl = _chunk(0), _chunk(0, ticker="AAPL")
        aapl = RagChunk(chunk_id="a-0", doc_id="a", ticker="AAPL", form="10-K",
                        filing_date=20240101, text="apple revenue", source="edgar")
        store.add("doc1", [nvda], emb.embed([nvda.text]))
        store.add("a", [aapl], emb.embed([aapl.text]))
        hits = store.query(emb.embed(["revenue"])[0], ticker="AAPL", k=5)
        assert [h.ticker for h in hits] == ["AAPL"]

    def test_query_empty_store_returns_empty(self, tmp_path):
        store = LocalVectorStore(tmp_path / "idx")
        emb = HashingEmbedder(dim=32)
        assert store.query(emb.embed(["anything"])[0], ticker="NVDA", k=3) == []

    def test_persistence_across_reopen(self, tmp_path):
        path = tmp_path / "idx"
        emb = HashingEmbedder(dim=32)
        store = LocalVectorStore(path)
        chunks = [_chunk(0)]
        store.add("doc1", chunks, emb.embed([c.text for c in chunks]))
        reopened = LocalVectorStore(path)
        assert reopened.is_ingested("doc1")
        assert reopened.count() == 1
        hits = reopened.query(emb.embed(["revenue"])[0], ticker="NVDA", k=1)
        assert hits and hits[0].chunk_id == "doc1-0"

    def test_ticker_attempt_marking(self, tmp_path):
        store = LocalVectorStore(tmp_path / "idx")
        assert not store.ticker_attempted("RELIANCE.NS")
        store.mark_ticker_attempted("RELIANCE.NS", filings=0)
        assert store.ticker_attempted("RELIANCE.NS")
        # persists
        assert LocalVectorStore(tmp_path / "idx").ticker_attempted("RELIANCE.NS")
