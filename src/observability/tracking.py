"""
Observability via LangSmith — tracing, latency, token usage, feedback.
Replaces MLflow + Prometheus + Grafana.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any

from langsmith import Client
from langsmith.run_helpers import traceable
from loguru import logger

from config.settings import get_settings

settings = get_settings()


def setup_langsmith() -> None:
    """Configure LangSmith environment variables."""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    logger.info(f"LangSmith tracing enabled → project: {settings.langsmith_project}")


class LangSmithTracker:
    """
    Wraps LangSmith client for logging query runs, ingestion events,
    and user feedback. All traces visible at smith.langchain.com.
    """

    def __init__(self):
        self._client: Client | None = None

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(
                api_key=settings.langsmith_api_key,
                api_url="https://api.smith.langchain.com",
            )
        return self._client

    def log_query(
        self,
        query: str,
        answer: str,
        sources: list[dict],
        latencies: dict[str, float],
        ragas_scores: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Log a completed RAG query run to LangSmith."""
        try:
            run_id = self.client.create_run(
                name="rag_query",
                run_type="chain",
                inputs={"query": query},
                outputs={"answer": answer},
                extra={
                    "metadata": {
                        **(metadata or {}),
                        "num_sources": len(sources),
                        "latencies": latencies,
                        **(ragas_scores or {}),
                    }
                },
            )
            logger.debug(f"LangSmith run logged: {run_id}")
            return str(run_id)
        except Exception as e:
            logger.warning(f"LangSmith logging failed (non-critical): {e}")
            return None

    def log_feedback(
        self,
        run_id: str,
        score: float,
        comment: str = "",
        key: str = "user_rating",
    ) -> None:
        """Log user thumbs up/down feedback on a run."""
        try:
            self.client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                comment=comment,
            )
            logger.debug(f"Feedback logged for run {run_id}: {score}")
        except Exception as e:
            logger.warning(f"Feedback logging failed: {e}")

    def log_ingestion(
        self,
        source_name: str,
        source_type: str,
        num_docs: int,
        num_chunks: int,
        duration: float,
    ) -> None:
        """Log ingestion event as a LangSmith run."""
        try:
            self.client.create_run(
                name="ingestion",
                run_type="tool",
                inputs={"source_name": source_name, "source_type": source_type},
                outputs={"num_docs": num_docs, "num_chunks": num_chunks},
                extra={"metadata": {"duration_seconds": round(duration, 2)}},
            )
        except Exception as e:
            logger.warning(f"Ingestion logging failed: {e}")


@contextmanager
def track_latency(stage: str):
    """Simple context manager to measure and log stage latency."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.debug(f"[{stage}] {elapsed:.3f}s")
