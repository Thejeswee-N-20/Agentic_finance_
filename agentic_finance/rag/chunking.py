"""Document cleaning and chunking for RAG ingestion.

Filings arrive as (often messy) HTML; ``html_to_text`` reduces them to plain
text and ``chunk_text`` splits that into overlapping, embedding-sized pieces.
Both are pure functions with no dependencies beyond the standard library.
"""
from __future__ import annotations

import html as html_lib
import re
from typing import List

__all__ = ["html_to_text", "chunk_text"]

_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

# Chunks smaller than this fold into the previous chunk rather than standing alone.
_MIN_CHUNK_CHARS = 50


def html_to_text(html: str) -> str:
    """Strip HTML to readable plain text (tags, scripts, styles, entities)."""
    if not html:
        return ""
    text = _SCRIPT_STYLE.sub(" ", html)
    text = _TAG.sub(" ", text)
    text = html_lib.unescape(text)
    return _WS.sub(" ", text).strip()


def chunk_text(text: str, chunk_chars: int = 2000, overlap: int = 200) -> List[str]:
    """Split text into overlapping windows of ``chunk_chars`` characters.

    The final fragment is merged into the previous chunk when it is too small
    to be a meaningful retrieval unit on its own.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    step = max(1, chunk_chars - max(0, overlap))
    chunks: List[str] = []
    for start in range(0, len(text), step):
        piece = text[start:start + chunk_chars].strip()
        if not piece:
            continue
        if len(piece) < _MIN_CHUNK_CHARS and chunks:
            chunks[-1] = chunks[-1] + piece
        else:
            chunks.append(piece)
        if start + chunk_chars >= len(text):
            break
    return chunks
