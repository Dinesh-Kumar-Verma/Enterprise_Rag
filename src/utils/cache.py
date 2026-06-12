"""
Two-layer caching:
  1. Query cache     — exact-match cache for repeated identical queries
  2. Embedding cache — avoids re-embedding the same text strings
Both use in-memory LRU with TTL. No Redis required (free tier friendly).
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from threading import Lock
from typing import Any

from loguru import logger


# ── LRU Cache with TTL ────────────────────────────────────────────────────────

class TTLCache:
    """
    Thread-safe LRU cache with per-entry TTL expiry.
    Evicts entries when capacity is exceeded (LRU policy)
    or when TTL expires on access.
    """

    def __init__(self, maxsize: int = 256, ttl_seconds: int = 3600):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        hashed = self._make_key(key)
        with self._lock:
            if hashed not in self._cache:
                self._misses += 1
                return None

            value, timestamp = self._cache[hashed]

            if time.time() - timestamp > self.ttl:
                del self._cache[hashed]
                self._misses += 1
                logger.debug(f"Cache TTL expired for key: {key[:40]}")
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(hashed)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        hashed = self._make_key(key)
        with self._lock:
            if hashed in self._cache:
                self._cache.move_to_end(hashed)
            self._cache[hashed] = (value, time.time())

            if len(self._cache) > self.maxsize:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug(f"Cache evicted LRU entry (size={self.maxsize})")

    def invalidate(self, key: str) -> bool:
        hashed = self._make_key(key)
        with self._lock:
            if hashed in self._cache:
                del self._cache[hashed]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "ttl_seconds": self.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
            }


# ── Specialized Caches ────────────────────────────────────────────────────────

class QueryCache(TTLCache):
    """
    Caches full RAG query results keyed on (query + use_hyde flag).
    TTL: 1 hour by default — short enough to pick up new ingested docs.
    """

    def __init__(self):
        super().__init__(maxsize=512, ttl_seconds=3600)

    def make_key(self, query: str, use_hyde: bool) -> str:
        return f"{query.strip().lower()}|hyde={use_hyde}"

    def get_result(self, query: str, use_hyde: bool) -> dict | None:
        key = self.make_key(query, use_hyde)
        result = self.get(key)
        if result is not None:
            logger.debug(f"Query cache HIT: {query[:60]}")
        return result

    def set_result(self, query: str, use_hyde: bool, result: dict) -> None:
        key = self.make_key(query, use_hyde)
        self.set(key, result)
        logger.debug(f"Query cache SET: {query[:60]}")

    def invalidate_all(self) -> None:
        """Call after new documents are ingested to avoid stale answers."""
        self.clear()
        logger.info("Query cache invalidated (new documents ingested)")


class EmbeddingCache(TTLCache):
    """
    Caches embedding vectors keyed on text content.
    TTL: 24 hours — embeddings don't change unless model changes.
    Avoids redundant HuggingFace inference calls.
    """

    def __init__(self):
        super().__init__(maxsize=2048, ttl_seconds=86400)

    def get_embedding(self, text: str) -> list[float] | None:
        return self.get(text)

    def set_embedding(self, text: str, embedding: list[float]) -> None:
        self.set(text, embedding)

    def get_batch(self, texts: list[str]) -> tuple[list[list[float] | None], list[int]]:
        """
        Returns (cached_results, missing_indices).
        cached_results has None at positions where cache missed.
        missing_indices are the positions that need fresh embedding.
        """
        results: list[list[float] | None] = []
        missing: list[int] = []
        for i, text in enumerate(texts):
            emb = self.get_embedding(text)
            results.append(emb)
            if emb is None:
                missing.append(i)
        return results, missing

    def set_batch(self, texts: list[str], embeddings: list[list[float]]) -> None:
        for text, emb in zip(texts, embeddings):
            self.set_embedding(text, emb)


# ── Singleton instances ───────────────────────────────────────────────────────

_query_cache: QueryCache | None = None
_embedding_cache: EmbeddingCache | None = None


def get_query_cache() -> QueryCache:
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache()
    return _query_cache


def get_embedding_cache() -> EmbeddingCache:
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache
