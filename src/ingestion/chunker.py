"""
Chunking pipeline with semantic splitting, metadata enrichment, and deduplication.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


class ChunkingPipeline:
    """
    Two-pass chunking strategy:
    1. Split by semantic boundaries (headings, paragraphs)
    2. Secondary split for oversized chunks
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_length: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

        self.primary_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

        self.secondary_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap // 2,
            separators=[". ", "! ", "? ", " ", ""],
            length_function=len,
        )

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF]", "", text)
        return text.strip()

    def _enrich_metadata(
        self, chunk: Document, parent_doc: Document, chunk_idx: int, total_chunks: int
    ) -> Document:
        chunk_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
        chunk.metadata.update(
            {
                **parent_doc.metadata,
                "chunk_id": f"{parent_doc.metadata.get('doc_hash', 'unknown')}_{chunk_idx}",
                "chunk_index": chunk_idx,
                "total_chunks": total_chunks,
                "chunk_hash": chunk_hash,
                "char_count": len(chunk.page_content),
                "word_count": len(chunk.page_content.split()),
            }
        )
        return chunk

    def _deduplicate(self, chunks: list[Document]) -> list[Document]:
        seen: set[str] = set()
        unique = []
        for chunk in chunks:
            h = chunk.metadata.get("chunk_hash", "")
            if h not in seen:
                seen.add(h)
                unique.append(chunk)
        removed = len(chunks) - len(unique)
        if removed:
            logger.debug(f"Deduplication removed {removed} duplicate chunks")
        return unique

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        all_chunks: list[Document] = []

        for doc in documents:
            doc.page_content = self._clean_text(doc.page_content)
            if len(doc.page_content) < self.min_chunk_length:
                logger.debug(f"Skipping short document: {len(doc.page_content)} chars")
                continue

            raw_chunks = self.primary_splitter.split_documents([doc])

            refined: list[Document] = []
            for c in raw_chunks:
                if len(c.page_content) > self.chunk_size * 1.5:
                    refined.extend(self.secondary_splitter.split_documents([c]))
                else:
                    refined.append(c)

            filtered = [
                c for c in refined if len(c.page_content.strip()) >= self.min_chunk_length
            ]

            total = len(filtered)
            enriched = [
                self._enrich_metadata(chunk, doc, idx, total)
                for idx, chunk in enumerate(filtered)
            ]
            all_chunks.extend(enriched)

        deduped = self._deduplicate(all_chunks)
        logger.info(
            f"Chunked {len(documents)} docs → {len(deduped)} chunks "
            f"(avg {sum(c.metadata['char_count'] for c in deduped) // max(len(deduped),1)} chars)"
        )
        return deduped
