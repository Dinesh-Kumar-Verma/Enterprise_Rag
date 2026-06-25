"""
Two-stage post-retrieval:
  1. Cross-encoder reranking (ms-marco-MiniLM via HuggingFace)
  2. Relevance grading + token-budget context building
"""

from __future__ import annotations

import math

from langchain_core.documents import Document
from loguru import logger
from sentence_transformers import CrossEncoder

from config.settings import get_settings

settings = get_settings()

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self):
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            logger.info(f"Loading cross-encoder: {RERANKER_MODEL}")
            self._model = CrossEncoder(RERANKER_MODEL, max_length=512)
        return self._model

    def rerank(
        self, query: str, documents: list[Document], top_k: int | None = None
    ) -> list[Document]:
        if not documents:
            return []

        top_k = top_k or settings.top_k_rerank
        pairs = [(query, doc.page_content) for doc in documents]
        raw_scores = self.model.predict(pairs)

        # Sigmoid normalise to [0,1]
        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        scored = sorted(
            zip(documents, raw_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        reranked = []
        for doc, score in scored[:top_k]:
            doc.metadata["rerank_score"] = round(sigmoid(float(score)), 4)
            reranked.append(doc)

        logger.info(
            f"Reranked {len(documents)} → top {len(reranked)} | "
            f"top score: {reranked[0].metadata['rerank_score'] if reranked else 'N/A'}"
        )
        return reranked


class ContextBuilder:
    """
    Filters by relevance threshold, enforces token budget,
    and formats context with source attribution.
    """

    CHARS_PER_TOKEN = 4  # rough approximation

    def __init__(self):
        self.max_tokens = settings.max_context_tokens
        self.threshold = settings.relevance_threshold

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // self.CHARS_PER_TOKEN

    def build(
        self, documents: list[Document], query: str
    ) -> tuple[str, list[dict]]:
        """
        Returns:
            context_str: formatted context string for LLM
            sources: list of source attribution dicts
        """
        relevant = [
            doc for doc in documents
            if doc.metadata.get("rerank_score", doc.metadata.get("hybrid_score", 0.0))
               >= self.threshold
        ]

        if not relevant:
            logger.warning(f"No docs passed relevance threshold {self.threshold}")
            # Return EMPTY context — the pipeline will handle this gracefully
            # instead of forcing irrelevant docs into the answer
            return "", []

        budget = self.max_tokens
        selected: list[Document] = []
        for doc in relevant:
            tokens = self._estimate_tokens(doc.page_content)
            if tokens <= budget:
                selected.append(doc)
                budget -= tokens
            if budget <= 0:
                break

        context_parts = []
        sources = []

        for i, doc in enumerate(selected, 1):
            source_name = doc.metadata.get("source_name", "Unknown")
            source_type = doc.metadata.get("source_type", "unknown")
            score = doc.metadata.get("rerank_score", doc.metadata.get("hybrid_score", 0.0))

            context_parts.append(
                f"[Source {i}: {source_name}]\n{doc.page_content.strip()}"
            )
            sources.append(
                {
                    "index": i,
                    "source_name": source_name,
                    "source_type": source_type,
                    "relevance_score": score,
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "url": doc.metadata.get("url", ""),
                    "preview": doc.page_content[:150].strip() + "...",
                }
            )

        context_str = "\n\n---\n\n".join(context_parts)
        logger.info(
            f"Context built: {len(selected)} chunks | "
            f"~{self._estimate_tokens(context_str)} tokens | "
            f"{len(relevant) - len(selected)} chunks dropped (budget)"
        )
        return context_str, sources
