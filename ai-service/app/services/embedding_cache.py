"""
Embedding cache service to reduce redundant API calls.
"""
import hashlib
import json
from typing import List, Optional
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """
    Thread-safe in-memory cache for embedding results.
    Uses LRU eviction policy when cache is full.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of cached embeddings
        """
        self._cache: dict = {}
        self._access_order: dict = {}  # Track access order for LRU
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[List[float]]:
        """
        Get cached embedding if available.

        Args:
            text: Input text
            model: Model name

        Returns:
            Cached embedding or None
        """
        key = self._get_cache_key(text, model)

        if key in self._cache:
            # Update access order
            self._access_order[key] = self._hits + self._misses
            self._hits += 1
            logger.debug(f"Embedding cache hit: text_length={len(text)}, model={model}")
            return self._cache[key]

        self._misses += 1
        return None

    def set(self, text: str, model: str, embedding: List[float]):
        """
        Cache embedding result.

        Args:
            text: Input text
            model: Model name
            embedding: Embedding vector
        """
        key = self._get_cache_key(text, model)

        # Check if cache is full
        if len(self._cache) >= self._max_size and key not in self._cache:
            self._evict_lru()

        self._cache[key] = embedding
        self._access_order[key] = self._hits + self._misses
        logger.debug(f"Embedding cached: text_length={len(text)}, model={model}, cache_size={len(self._cache)}")

    def _evict_lru(self):
        """Evict least recently used items (10% of cache)."""
        # Sort by access order (oldest first)
        sorted_keys = sorted(self._access_order.keys(), key=lambda k: self._access_order[k])

        # Remove oldest 10%
        num_to_remove = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:num_to_remove]:
            del self._cache[key]
            del self._access_order[key]

        logger.info(f"LRU eviction completed: evicted_count={num_to_remove}, remaining_count={len(self._cache)}")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.2%}",
            "cache_size": len(self._cache),
            "max_size": self._max_size
        }

    def clear(self):
        """Clear all cached embeddings."""
        self._cache.clear()
        self._access_order.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Embedding cache cleared")


# Global singleton instance
embedding_cache = EmbeddingCache(max_size=1000)
