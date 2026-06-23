"""
EnterpriseRAG orchestrator — wires ingestion, retrieval, generation, observability.
Includes input sanitization and two-layer caching (query + embedding).
"""

from __future__ import annotations

from ast import If
import time
from pathlib import Path

from langsmith.run_helpers import traceable
from loguru import logger

from config.settings import get_settings
from src.ingestion.chunker import ChunkingPipeline
from src.ingestion.loaders import DocumentLoader
from src.ingestion.vector_store import VectorStoreManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import ContextBuilder, Reranker
from src.generation.rag_pipeline import RAGPipeline
from src.observability.tracking import LangSmithTracker, setup_langsmith, track_latency
from src.utils.sanitizer import (
    SanitizationError,
    sanitize_file,
    sanitize_query,
    sanitize_text,
    sanitize_url,
)
from src.utils.cache import get_query_cache
from src.security.guardrails import RAGGuardrails

settings = get_settings()


class EnterpriseRAG:
    def __init__(self):
        setup_langsmith()

        self.loader = DocumentLoader()
        self.chunker = ChunkingPipeline()
        self.vector_store = VectorStoreManager()
        self.retriever = HybridRetriever(self.vector_store)
        self.reranker = Reranker()
        self.context_builder = ContextBuilder()
        self.generator = RAGPipeline()
        self.tracker = LangSmithTracker()
        self.query_cache = get_query_cache()
        self.guardrails = RAGGuardrails()

        logger.info(f"EnterpriseRAG initialised | env={settings.app_env}")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_file(self, path: str | Path, original_filename: str | None = None) -> dict:
        filename = original_filename or Path(path).name
        file_size = Path(path).stat().st_size
        sanitize_file(filename, file_size)   # raises SanitizationError if invalid

        start = time.perf_counter()
        docs = self.loader.load_file(path)
        chunks = self.chunker.chunk_documents(docs)
        added = self.vector_store.add_documents(chunks)
        duration = time.perf_counter() - start

        self.query_cache.invalidate_all()   # new docs → invalidate stale cached answers
        source_type = docs[0].metadata.get("source_type", "file") if docs else "file"
        self.tracker.log_ingestion(str(path), source_type, len(docs), added, duration)
        return {"documents": len(docs), "chunks": added, "duration": round(duration, 2)}

    def ingest_url(self, url: str) -> dict:
        url = sanitize_url(url)   # raises SanitizationError on localhost/private IPs/bad scheme

        start = time.perf_counter()
        docs = self.loader.load_url(url)
        chunks = self.chunker.chunk_documents(docs)
        added = self.vector_store.add_documents(chunks)
        duration = time.perf_counter() - start

        self.query_cache.invalidate_all()
        self.tracker.log_ingestion(url, "web", len(docs), added, duration)
        return {"documents": len(docs), "chunks": added, "duration": round(duration, 2)}

    def ingest_text(self, text: str, source_name: str = "manual") -> dict:
        text, source_name = sanitize_text(text, source_name)

        docs = self.loader.load_text(text, source_name)
        chunks = self.chunker.chunk_documents(docs)
        added = self.vector_store.add_documents(chunks)
        self.query_cache.invalidate_all()
        return {"documents": 1, "chunks": added}

    def ingest_directory(self, directory: str | Path) -> dict:
        start = time.perf_counter()
        docs = self.loader.load_directory(directory)
        chunks = self.chunker.chunk_documents(docs)
        added = self.vector_store.add_documents(chunks)
        duration = time.perf_counter() - start
        self.query_cache.invalidate_all()
        return {"documents": len(docs), "chunks": added, "duration": round(duration, 2)}

    # ── Query ─────────────────────────────────────────────────────────────────

    @traceable(name="rag_query", run_type="chain")
    def query(self, query: str, use_hyde: bool = True) -> dict:
        query = sanitize_query(query)   # raises SanitizationError on injection/bad input

        # Check input guardrails
        blocked_msg = self.guardrails.check_input_sync(query)
        if blocked_msg:
            logger.warning(f"Query blocked by NeMo Guardrails: {query[:80]}")
            return {
                "answer": blocked_msg,
                "sources": [],
                "query_type": "OUT_OF_SCOPE",
                "is_grounded": True,
                "latencies": {"guardrails": 0.0},
                "cached": False,
            }

        # Check query cache first
        cached = self.query_cache.get_result(query, use_hyde)
        if cached is not None:
            logger.info(f"Query cache HIT — skipping retrieval+generation")
            return {**cached, "cached": True}

        latencies: dict[str, float] = {}

        with track_latency("retrieval"):
            t0 = time.perf_counter()
            candidates = self.retriever.retrieve(query, use_hyde=use_hyde)
            latencies["retrieval"] = time.perf_counter() - t0

        with track_latency("reranking"):
            t0 = time.perf_counter()
            reranked = self.reranker.rerank(query, candidates)
            latencies["reranking"] = time.perf_counter() - t0

        context, sources = self.context_builder.build(reranked, query)

        with track_latency("generation"):
            t0 = time.perf_counter()
            result = self.generator.run(query, context, sources)
            latencies["generation"] = time.perf_counter() - t0

        # Check output guardrails
        final_answer = self.guardrails.check_output_sync(query, result["answer"])

        self.tracker.log_query(
            query=query,
            answer=final_answer,
            sources=sources,
            latencies=latencies,
            metadata={
                "use_hyde": use_hyde,
                "query_type": result["query_type"],
                "is_grounded": result["is_grounded"],
            },
        )

        output = {
            "answer": final_answer,
            "sources": sources,
            "query_type": result["query_type"],
            "is_grounded": result["is_grounded"],
            "latencies": {k: round(v, 3) for k, v in latencies.items()},
            "cached": False,
        }

        # Only cache grounded answers
        if result["is_grounded"]:
            self.query_cache.set_result(query, use_hyde, output)

        return output

    @traceable(name="rag_stream", run_type="chain")
    async def astream(self, query: str, use_hyde: bool = True):
        query = sanitize_query(query)

        # Check input guardrails
        blocked_msg = await self.guardrails.check_input(query)
        if blocked_msg:
            logger.warning(f"Query blocked by NeMo Guardrails: {query[:80]}")
            yield {"type": "sources", "data": []}
            yield {"type": "token", "data": blocked_msg}
            yield {"type": "done"}
            return

        candidates = self.retriever.retrieve(query, use_hyde=use_hyde)
        reranked = self.reranker.rerank(query, candidates)
        context, sources = self.context_builder.build(reranked, query)

        yield {"type": "sources", "data": sources}

        async for token in self.generator.stream(query, context, sources):
            yield {"type": "token", "data": token}

        yield {"type": "done"}

    def log_feedback(self, run_id: str, score: float, comment: str = "") -> None:
        self.tracker.log_feedback(run_id, score, comment)

    def get_stats(self) -> dict:
        return {
            "vector_store": self.vector_store.get_stats(),
            "query_cache": self.query_cache.stats,
            "settings": {
                "embedding_model": settings.embedding_model,
                "llm_model": settings.gemini_model,
                "top_k_retrieval": settings.top_k_retrieval,
                "top_k_rerank": settings.top_k_rerank,
                "relevance_threshold": settings.relevance_threshold,
                "langsmith_project": settings.langsmith_project,
            },
        }


def main():
    file_path = r"C:\Users\Dinesh Verma\Downloads\Two Pointer Pattern.pdf"
    EnterpriseRAG().ingest_file(file_path)
    
if __name__ == "__main__":
    main()