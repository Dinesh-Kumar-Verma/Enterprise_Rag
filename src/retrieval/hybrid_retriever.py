"""
Advanced retriever combining:
  - HyDE (Hypothetical Document Embeddings) for query expansion
  - Hybrid search: BM25 (lexical) + dense vector
  - MMR-style diversity via score blending
"""

from __future__ import annotations

import math
from collections import defaultdict

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from loguru import logger
from rank_bm25 import BM25Okapi

from config.settings import get_settings
from src.ingestion.vector_store import VectorStoreManager

settings = get_settings()

HYDE_PROMPT = """You are a technical documentation expert. Given the user question below, 
write a short hypothetical document passage (3-5 sentences) that would perfectly answer it.
Write only the passage, no preamble.

Question: {query}
Hypothetical passage:"""


class HybridRetriever:
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        self._llm: ChatGroq | None = None
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[Document] = []

    @property
    def llm(self) -> ChatGroq:
        if self._llm is None:
            self._llm = ChatGroq(
                model=settings.groq_model,
                api_key=settings.groq_api_key,
                temperature=0.3,
            )
        return self._llm

    def _hyde_expand(self, query: str) -> str:
        """Generate a hypothetical document to improve query embedding."""
        try:
            prompt = HYDE_PROMPT.format(query=query)
            response = self.llm.invoke(prompt)
            hypothetical = response.content.strip()
            logger.debug(f"HyDE expansion: {hypothetical[:80]}...")
            return hypothetical
        except Exception as e:
            logger.warning(f"HyDE expansion failed, using raw query: {e}")
            return query

    def _build_bm25(self, docs: list[Document]) -> None:
        """Build BM25 index from retrieved dense candidates."""
        tokenized = [doc.page_content.lower().split() for doc in docs]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_docs = docs

    def _bm25_scores(self, query: str) -> dict[str, float]:
        if not self._bm25 or not self._bm25_docs:
            return {}
        tokens = query.lower().split()
        raw_scores = self._bm25.get_scores(tokens)
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        return {
            doc.metadata["chunk_id"]: float(score / max_score)
            for doc, score in zip(self._bm25_docs, raw_scores)
        }

    def _reciprocal_rank_fusion(
        self,
        dense_docs: list[Document],
        bm25_scores: dict[str, float],
        alpha: float,
        k: int = 60,
    ) -> list[Document]:
        """
        Blend dense vector scores + BM25 scores via weighted combination.
        alpha=1.0 → pure dense; alpha=0.0 → pure BM25
        """
        score_map: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs):
            cid = doc.metadata["chunk_id"]
            dense_score = doc.metadata.get("vector_score", 0.0)
            bm25_score = bm25_scores.get(cid, 0.0)
            fused = alpha * dense_score + (1.0 - alpha) * bm25_score
            score_map[cid] = fused
            doc_map[cid] = doc

        sorted_ids = sorted(score_map, key=score_map.__getitem__, reverse=True)
        result = []
        for cid in sorted_ids:
            doc = doc_map[cid]
            doc.metadata["hybrid_score"] = round(score_map[cid], 4)
            result.append(doc)
        return result

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        use_hyde: bool = True,
        alpha: float | None = None,
    ) -> list[Document]:
        top_k = top_k or settings.top_k_retrieval
        alpha = alpha if alpha is not None else settings.hybrid_alpha

        expanded = self._hyde_expand(query) if use_hyde else query
        query_embedding = self.vector_store.embed_query(expanded)

        dense_docs = self.vector_store.query(query_embedding, top_k=top_k)

        if not dense_docs:
            logger.warning("No documents retrieved from vector store")
            return []

        self._build_bm25(dense_docs)
        bm25_scores = self._bm25_scores(query)

        fused = self._reciprocal_rank_fusion(dense_docs, bm25_scores, alpha)

        logger.info(
            f"Retrieved {len(fused)} docs | HyDE={'on' if use_hyde else 'off'} | alpha={alpha}"
        )
        return fused[:top_k]
