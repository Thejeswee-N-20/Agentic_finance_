"""Local vector store + embedders for the RAG layer.

No external vector database: chunks and their embeddings persist to a plain
directory (``chunks.jsonl`` + ``embeddings.npy`` + ``manifest.json``) and
similarity is brute-force cosine over numpy — milliseconds at the few-thousand
-chunk scale of this project, with nothing that can fail to install or start.

The manifest is what makes ingestion idempotent: it records every ingested
document (by EDGAR accession id) and every *attempted* ticker, so repeated
runs — or repeated live decisions on the same ticker — never re-download or
re-embed anything.

Embeddings default to :class:`HashingEmbedder` — a deterministic, dependency-
free signed feature-hashing embedder (lexical similarity, BM25-like in spirit).
Set ``AGENTIC_RAG_EMBEDDER=st`` to opt into sentence-transformers when
installed; any failure falls back to hashing silently.
"""
from __future__ import annotations

import json
import os
import zlib
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Protocol, Sequence

import numpy as np

__all__ = ["RagChunk", "Embedder", "HashingEmbedder", "LocalVectorStore", "get_embedder"]

_DEFAULT_DIM = 384


@dataclass(frozen=True)
class RagChunk:
    """One embedded excerpt of a source document (immutable)."""

    chunk_id: str
    doc_id: str
    ticker: str
    form: str
    filing_date: int  # YYYYMMDD — enables the as-of (point-in-time) filter
    text: str
    source: str = "edgar"


class Embedder(Protocol):
    """Anything that maps texts to fixed-dimension vectors."""

    dim: int

    def embed(self, texts: Sequence[str]) -> List[List[float]]: ...


class HashingEmbedder:
    """Deterministic signed feature-hashing embedder (zero dependencies).

    Words and word-bigrams are hashed (stable CRC32, not Python's randomised
    ``hash``) into ``dim`` signed buckets and L2-normalised. This gives lexical
    similarity — sufficient for retrieving filing sections that share vocabulary
    with the query — with perfect reproducibility and no model download.
    """

    def __init__(self, dim: int = _DEFAULT_DIM) -> None:
        self.dim = dim

    def _features(self, text: str) -> List[str]:
        words = [w for w in text.lower().split() if w]
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return words + bigrams

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float64)
            for feature in self._features(text):
                h = zlib.crc32(feature.encode("utf-8"))
                idx = h % self.dim
                sign = 1.0 if (h >> 16) & 1 else -1.0
                vec[idx] += sign
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            out.append(vec.tolist())
        return out


def get_embedder() -> Embedder:
    """Default embedder: hashing; ``AGENTIC_RAG_EMBEDDER=st`` opts into
    sentence-transformers (lazy import; silently falls back on any failure)."""
    if os.environ.get("AGENTIC_RAG_EMBEDDER", "").lower() in ("st", "sentence-transformers"):
        try:
            from sentence_transformers import SentenceTransformer  # lazy, optional

            model = SentenceTransformer("all-MiniLM-L6-v2")

            class _STEmbedder:
                dim = int(model.get_sentence_embedding_dimension())

                def embed(self, texts: Sequence[str]) -> List[List[float]]:
                    return model.encode(list(texts), normalize_embeddings=True).tolist()

            return _STEmbedder()
        except Exception:  # noqa: BLE001 - missing/broken optional dep -> fallback
            pass
    return HashingEmbedder()


class LocalVectorStore:
    """Persistent local index: manifest + chunk metadata + embedding matrix."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._manifest_file = self.path / "manifest.json"
        self._chunks_file = self.path / "chunks.jsonl"
        self._emb_file = self.path / "embeddings.npy"
        self._manifest = self._load_manifest()
        self._chunks: List[RagChunk] = self._load_chunks()
        self._emb: Optional[np.ndarray] = self._load_embeddings()

    # -- persistence -------------------------------------------------------
    def _load_manifest(self) -> dict:
        if self._manifest_file.exists():
            try:
                return json.loads(self._manifest_file.read_text())
            except Exception:  # noqa: BLE001 - corrupt manifest -> start clean
                pass
        return {"documents": {}, "tickers": {}, "dim": None}

    def _load_chunks(self) -> List[RagChunk]:
        if not self._chunks_file.exists():
            return []
        chunks: List[RagChunk] = []
        try:
            for line in self._chunks_file.read_text().splitlines():
                if line.strip():
                    chunks.append(RagChunk(**json.loads(line)))
        except Exception:  # noqa: BLE001 - corrupt row -> keep what parsed
            pass
        return chunks

    def _load_embeddings(self) -> Optional[np.ndarray]:
        if not self._emb_file.exists():
            return None
        try:
            return np.load(self._emb_file)
        except Exception:  # noqa: BLE001
            return None

    def _save(self) -> None:
        self._manifest_file.write_text(json.dumps(self._manifest, indent=1))
        self._chunks_file.write_text(
            "\n".join(json.dumps(asdict(c)) for c in self._chunks)
        )
        if self._emb is not None:
            np.save(self._emb_file, self._emb)

    # -- idempotency bookkeeping ------------------------------------------
    def is_ingested(self, doc_id: str) -> bool:
        return doc_id in self._manifest["documents"]

    def ticker_attempted(self, ticker: str) -> bool:
        return ticker.upper() in self._manifest["tickers"]

    def mark_ticker_attempted(self, ticker: str, filings: int = 0) -> None:
        self._manifest["tickers"][ticker.upper()] = {
            "date": date.today().isoformat(), "filings": filings,
        }
        self._save()

    def count(self) -> int:
        return len(self._chunks)

    # -- write / read ------------------------------------------------------
    def add(self, doc_id: str, chunks: Sequence[RagChunk],
            embeddings: Sequence[Sequence[float]]) -> None:
        """Add one document's chunks. A no-op if the document was already
        ingested or the embedding dimension conflicts with the index."""
        if self.is_ingested(doc_id) or not chunks:
            return
        matrix = np.asarray(embeddings, dtype=np.float64)
        if self._manifest["dim"] is None:
            self._manifest["dim"] = int(matrix.shape[1])
        elif int(matrix.shape[1]) != self._manifest["dim"]:
            return  # incompatible embedder; refuse silently rather than corrupt
        self._chunks.extend(chunks)
        self._emb = matrix if self._emb is None else np.vstack([self._emb, matrix])
        self._manifest["documents"][doc_id] = {
            "ticker": chunks[0].ticker, "form": chunks[0].form,
            "filing_date": chunks[0].filing_date, "chunks": len(chunks),
        }
        self._save()

    def query(self, embedding: Sequence[float], ticker: Optional[str] = None,
              as_of: Optional[int] = None, k: int = 5) -> List[RagChunk]:
        """Top-k cosine matches, filtered by ticker and as-of date (YYYYMMDD)."""
        if self._emb is None or not self._chunks:
            return []
        vec = np.asarray(embedding, dtype=np.float64)
        if vec.shape[0] != self._emb.shape[1]:
            return []
        mask = np.ones(len(self._chunks), dtype=bool)
        for i, chunk in enumerate(self._chunks):
            if ticker is not None and chunk.ticker.upper() != ticker.upper():
                mask[i] = False
            elif as_of is not None and chunk.filing_date > as_of:
                mask[i] = False
        if not mask.any():
            return []
        scores = self._emb @ vec
        scores[~mask] = -np.inf
        order = np.argsort(-scores)[:k]
        return [self._chunks[i] for i in order if np.isfinite(scores[i])]
