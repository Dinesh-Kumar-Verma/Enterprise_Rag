"""
Tests for EnterpriseRAG core components.
Run: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.ingestion.chunker import ChunkingPipeline
from src.ingestion.loaders import DocumentLoader


# ── Chunking Tests ────────────────────────────────────────────────────────────

class TestChunkingPipeline:
    def setup_method(self):
        self.pipeline = ChunkingPipeline(chunk_size=200, chunk_overlap=20)

    def _make_doc(self, text: str, source: str = "test") -> Document:
        return Document(
            page_content=text,
            metadata={"source_name": source, "doc_hash": "abc123", "source_type": "file"},
        )

    def test_basic_chunking(self):
        long_text = "This is a test sentence. " * 50
        doc = self._make_doc(long_text)
        chunks = self.pipeline.chunk_documents([doc])
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 200 * 1.5

    def test_metadata_enrichment(self):
        doc = self._make_doc("Short test text with enough content to pass the minimum length filter for testing.")
        chunks = self.pipeline.chunk_documents([doc])
        assert len(chunks) >= 1
        chunk = chunks[0]
        assert "chunk_id" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert "word_count" in chunk.metadata
        assert "char_count" in chunk.metadata

    def test_deduplication(self):
        text = "This is a repeated document. " * 20
        doc1 = self._make_doc(text, "source1")
        doc2 = self._make_doc(text, "source1")
        chunks = self.pipeline.chunk_documents([doc1, doc2])
        hashes = [c.metadata["chunk_hash"] for c in chunks]
        assert len(hashes) == len(set(hashes))

    def test_short_doc_filtered(self):
        doc = self._make_doc("Hi")
        chunks = self.pipeline.chunk_documents([doc])
        assert len(chunks) == 0

    def test_empty_input(self):
        chunks = self.pipeline.chunk_documents([])
        assert chunks == []

    def test_chunk_index_continuity(self):
        text = "Paragraph content here. " * 100
        doc = self._make_doc(text)
        chunks = self.pipeline.chunk_documents([doc])
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))


# ── Loader Tests ──────────────────────────────────────────────────────────────

class TestDocumentLoader:
    def setup_method(self):
        self.loader = DocumentLoader()

    def test_load_text(self):
        text = "This is a test document with meaningful content for loading."
        docs = self.loader.load_text(text, source_name="test_source")
        assert len(docs) == 1
        assert docs[0].page_content == text
        assert docs[0].metadata["source_name"] == "test_source"
        assert docs[0].metadata["source_type"] == "text"
        assert "doc_hash" in docs[0].metadata

    def test_load_text_hash_consistency(self):
        text = "Consistent content"
        docs1 = self.loader.load_text(text)
        docs2 = self.loader.load_text(text)
        assert docs1[0].metadata["doc_hash"] == docs2[0].metadata["doc_hash"]

    @patch("requests.get")
    def test_load_url(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = "<html><body><p>Test content from web page</p></body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        docs = self.loader.load_url("https://example.com/test")
        assert len(docs) == 1
        assert "Test content from web page" in docs[0].page_content
        assert docs[0].metadata["source_type"] == "web"

    @patch("requests.get")
    def test_load_url_failure(self, mock_get):
        mock_get.side_effect = Exception("Connection error")
        docs = self.loader.load_url("https://unreachable.example.com")
        assert docs == []

    def test_unsupported_file_type(self, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            self.loader.load_file(bad_file)


# ── Context Builder Tests ─────────────────────────────────────────────────────

class TestContextBuilder:
    def setup_method(self):
        from src.retrieval.reranker import ContextBuilder
        self.builder = ContextBuilder()

    def _make_doc(self, text: str, score: float = 0.8) -> Document:
        return Document(
            page_content=text,
            metadata={
                "chunk_id": f"chunk_{hash(text)}",
                "source_name": "test_source",
                "source_type": "file",
                "rerank_score": score,
            },
        )

    def test_basic_context_build(self):
        docs = [self._make_doc(f"Relevant content about topic {i}. " * 5) for i in range(3)]
        context, sources = self.builder.build(docs, "test query")
        assert len(sources) == 3
        assert "Source 1" in context
        assert "Source 2" in context

    def test_relevance_filtering(self):
        high_docs = [self._make_doc("High relevance content. " * 5, score=0.9) for _ in range(2)]
        low_docs = [self._make_doc("Low relevance content. " * 5, score=0.1) for _ in range(3)]
        context, sources = self.builder.build(high_docs + low_docs, "test")
        source_scores = [s["relevance_score"] for s in sources]
        assert all(s >= self.builder.threshold for s in source_scores)

    def test_token_budget_enforcement(self):
        long_docs = [self._make_doc("x " * 2000, score=0.9) for _ in range(10)]
        context, sources = self.builder.build(long_docs, "test")
        estimated_tokens = len(context) // 4
        assert estimated_tokens <= self.builder.max_tokens * 1.1
