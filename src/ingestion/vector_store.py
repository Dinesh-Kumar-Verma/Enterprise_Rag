"""
ChromaDB vector store manager with BGE-M3 embeddings + embedding cache.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings
from src.utils.cache import get_embedding_cache

settings = get_settings()


class VectorStoreManager:
    def __init__(self):
        self._embeddings: HuggingFaceEmbeddings | None = None
        self._client: chromadb.HttpClient | None = None
        self._collection: chromadb.Collection | None = None
        self._embed_cache = get_embedding_cache()

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            )
        return self._embeddings

    @property
    def client(self) -> chromadb.HttpClient:
        if self._client is None:
            self._client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"Using collection '{settings.chroma_collection}' "
                f"({self._collection.count()} existing docs)"
            )
        return self._collection

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with cache — only calls HuggingFace for cache misses."""
        cached, missing_indices = self._embed_cache.get_batch(texts)

        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            logger.debug(
                f"Embedding {len(missing_texts)} texts "
                f"({len(texts) - len(missing_indices)} from cache)"
            )
            fresh_embeddings = self.embeddings.embed_documents(missing_texts)
            self._embed_cache.set_batch(missing_texts, fresh_embeddings)
            for idx, emb in zip(missing_indices, fresh_embeddings):
                cached[idx] = emb
        else:
            logger.debug(f"All {len(texts)} embeddings served from cache")

        return cached  # type: ignore[return-value]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string with cache."""
        cached = self._embed_cache.get_embedding(query)
        if cached is not None:
            return cached
        emb = self.embeddings.embed_query(query)
        self._embed_cache.set_embedding(query, emb)
        return emb

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def add_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0

        ids = [doc.metadata["chunk_id"] for doc in documents]
        texts = [doc.page_content for doc in documents]
        metadatas = [
            {k: str(v) for k, v in doc.metadata.items() if v is not None}
            for doc in documents
        ]
        embeddings = self.embed_texts(texts)

        batch_size = 100
        added = 0
        for i in range(0, len(ids), batch_size):
            s = slice(i, i + batch_size)
            self.collection.upsert(
                ids=ids[s],
                documents=texts[s],
                embeddings=embeddings[s],
                metadatas=metadatas[s],
            )
            added += len(ids[s])

        logger.info(f"Added/updated {added} chunks in vector store")
        return added

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        where: dict | None = None,
    ) -> list[Document]:
        total_docs = self.collection.count()
        if total_docs == 0:
            logger.warning("Query ignored: Vector store is empty")
            return []
    
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total_docs),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = []
        for doc_text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - distance
            doc = Document(
                page_content=doc_text,
                metadata={**metadata, "vector_score": round(similarity, 4)},
            )
            documents.append(doc)
        return documents

    def delete_by_source(self, source_name: str) -> int:
        results = self.collection.get(where={"source_name": source_name})
        ids = results.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks from source: {source_name}")
        return len(ids)

    def get_stats(self) -> dict:
        return {
            "collection": settings.chroma_collection,
            "total_chunks": self.collection.count(),
            "embedding_model": settings.embedding_model,
            "embedding_cache": self._embed_cache.stats,
        }
