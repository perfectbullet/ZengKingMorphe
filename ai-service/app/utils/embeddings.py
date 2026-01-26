"""
Custom embedding implementations for the Digital Employee AI Service.
"""

from typing import List, Optional
import requests
from langchain_core.embeddings import Embeddings
import numpy as np
from app.core.logging import get_logger
from app.services.embedding_cache import embedding_cache

logger = get_logger(__name__)


class ChromaEmbeddingWrapper:
    """Adapter to satisfy Chroma's EmbeddingFunction.__call__ signature."""

    def __init__(self, embedder: Embeddings):
        self.embedder = embedder

    def __call__(self, input: List[str]) -> List[List[float]]:  # type: ignore[override]
        return self.embedder.embed_documents(list(input))


class OllamaEmbeddings(Embeddings):
    """Ollama embedding implementation using /api/embeddings endpoint."""

    def __init__(
        self, model: str, base_url: str, batch_size: int = 32, max_tokens: int = 8192
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        # Conservative character limit for safety
        # BGE-M3 supports 8192 tokens, use max_tokens // 2 for character limit
        self.max_chars = max_tokens // 2

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limit."""
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars - 3] + "..."

    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text using Ollama API."""

        # Check cache first
        cached = embedding_cache.get(text, self.model)
        if cached is not None:
            return cached

        # Truncate if needed
        truncated_text = self._truncate_text(text)

        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": truncated_text,
            "keep_alive": -1  # Keep model loaded indefinitely
        }

        try:
            response = requests.post(url, json=payload, timeout=30.0)
            response.raise_for_status()
            result = response.json()

            if "embedding" not in result:
                raise ValueError(f"No embedding in response: {result}")

            embedding = result["embedding"]

            # Cache the result
            embedding_cache.set(text, self.model, embedding)

            return embedding
        except Exception as e:
            logger.error(f"Ollama embedding failed: url={url}, model={self.model}, error={str(e)}", exc_info=True)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        # Check for truncation
        truncated_count = sum(1 for text in texts if len(text) > self.max_chars)
        if truncated_count > 0:
            logger.warning(f"Ollama: Truncating {truncated_count}/{len(texts)} texts to fit {self.max_tokens} token limit")

        logger.info(f"Ollama embedding request: url={self.base_url}, model={self.model}, texts_count={len(texts)}")

        embeddings = []
        cache_hits = 0
        for text in texts:
            # Check if cached (already done in _embed_single, but track stats here)
            cached = embedding_cache.get(text, self.model)
            if cached is not None:
                cache_hits += 1
                embeddings.append(np.array(cached, dtype=float))
            else:
                embedding = self._embed_single(text)
                embeddings.append(np.array(embedding, dtype=float))

        logger.info(f"Ollama request successful: received {len(embeddings)} embeddings, cache_hits={cache_hits}")

        # Log cache stats periodically
        stats = embedding_cache.get_stats()
        if int(stats["hits"]) % 100 == 0:  # Every 100 cache hits
            logger.info("Embedding cache statistics", **stats)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.embed_documents([text])[0]


class SiliconFlowEmbeddings(Embeddings):
    """SiliconFlow embedding implementation."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        batch_size: int = 32,
        max_tokens: int = 512,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        # Conservative character limit: assume 1 char = 2 tokens for safety
        self.max_chars = max_tokens // 2

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limit."""
        if len(text) <= self.max_chars:
            return text
        # Truncate and add ellipsis
        return text[: self.max_chars - 3] + "..."

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Truncate texts to fit token limit
        truncated_texts = [self._truncate_text(text) for text in texts]

        payload = {"model": self.model, "input": truncated_texts}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Log truncation stats
        truncated_count = sum(
            1 for orig, trunc in zip(texts, truncated_texts) if len(orig) > len(trunc)
        )
        if truncated_count > 0:
            logger.warning(f"Truncated {truncated_count}/{len(texts)} texts to fit {self.max_tokens} token limit")

        logger.info(f"SiliconFlow embedding request: url={self.base_url}, model={self.model}, texts_count={len(truncated_texts)}")

        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            result = response.json()
            if result.get("code") not in (None, 0):
                raise ValueError(f"Embedding request failed: {result}")
            data = result.get("data")
            if not data:
                raise ValueError(f"No embedding data returned: {result}")

            logger.info(f"SiliconFlow request successful: received {len(data)} embeddings")
            embeddings = [np.array(item["embedding"], dtype=float) for item in data]
            return embeddings
        except Exception as e:
            logger.error(f"SiliconFlow embedding failed: url={self.base_url}, result={result}, error={str(e)}", exc_info=True)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings.extend(self._embed_batch(batch))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class OpenAIStyleEmbeddings(Embeddings):
    """适配 OpenAI /v1/embeddings 风格接口的嵌入实现"""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        payload = {"input": list(texts), "model": self.model}
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "digital-employee-service/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/v1/embeddings"

        # Add debug logging
        logger.info(f"Sending embedding request: url={url}, model={self.model}, texts_count={len(texts)}, has_api_key={bool(self.api_key)}")

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()

            data = result.get("data")
            if not data:
                raise ValueError(f"Embedding service returned no data: {result}")

            logger.info(f"Embedding request successful: received {len(data)} embeddings")
            embeddings = [np.array(item["embedding"], dtype=float) for item in data]
            return embeddings
        except Exception as e:
            logger.error(f"Embedding request failed: url={url}, error={str(e)}", exc_info=True)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]


def get_embedding(settings=None) -> Embeddings:
    """
    根据配置创建 Embedding 实例的工厂函数。

    默认使用 Ollama embeddings（EMBEDDING_OLLAMA_MODEL）。

    Args:
        settings: 应用配置对象（可选，默认使用全局配置）

    Returns:
        Embeddings 实例
    """
    from app.core.config import settings as app_settings

    if not settings:
        settings = app_settings

    # Default to Ollama embeddings
    logger.info(f"Creating Ollama embeddings: model={settings.embedding_ollama_model}")
    return OllamaEmbeddings(
        model=settings.embedding_ollama_model,
        base_url=settings.ollama_base_url,
        max_tokens=8192  # BGE-M3 supports 8192 tokens
    )
