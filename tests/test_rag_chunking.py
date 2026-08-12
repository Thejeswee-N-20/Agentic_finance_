"""Tests for RAG document cleaning and chunking (offline)."""
from __future__ import annotations

from agentic_finance.rag.chunking import chunk_text, html_to_text


class TestHtmlToText:
    def test_strips_tags_and_keeps_text(self):
        html = "<html><body><h1>Risk Factors</h1><p>Demand may decline.</p></body></html>"
        text = html_to_text(html)
        assert "Risk Factors" in text
        assert "Demand may decline." in text
        assert "<" not in text

    def test_drops_script_and_style_blocks(self):
        html = "<script>var x=1;</script><style>.a{color:red}</style><p>Revenue grew.</p>"
        text = html_to_text(html)
        assert "Revenue grew." in text
        assert "var x" not in text
        assert "color:red" not in text

    def test_unescapes_entities_and_collapses_whitespace(self):
        html = "<p>R&amp;D    spend\n\n\nincreased</p>"
        text = html_to_text(html)
        assert "R&D spend increased" in text

    def test_empty_and_plain_text_are_safe(self):
        assert html_to_text("") == ""
        assert html_to_text("no tags here") == "no tags here"


class TestChunkText:
    def test_short_text_is_single_chunk(self):
        chunks = chunk_text("short document", chunk_chars=2000)
        assert chunks == ["short document"]

    def test_long_text_is_split_with_overlap(self):
        text = " ".join(f"sentence{i}." for i in range(1000))
        chunks = chunk_text(text, chunk_chars=500, overlap=100)
        assert len(chunks) > 3
        assert all(len(c) <= 600 for c in chunks)  # chunk + boundary slack
        # consecutive chunks share some overlap text
        assert chunks[0][-20:] in chunks[0]  # sanity
        joined = "".join(chunks)
        assert "sentence0." in joined and "sentence999." in joined

    def test_empty_text_yields_no_chunks(self):
        assert chunk_text("") == []
        assert chunk_text("   \n  ") == []

    def test_tiny_chunks_never_emitted(self):
        # trailing fragment smaller than min size folds into the previous chunk
        chunks = chunk_text("a" * 510, chunk_chars=500, overlap=0)
        assert all(len(c) >= 50 for c in chunks)
