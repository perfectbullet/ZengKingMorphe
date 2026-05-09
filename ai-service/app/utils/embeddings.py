"""
Custom embedding implementations for the Digital Employee AI Service.

Note: embedding_cache has been removed since llama-rag-sdk handles caching internally.
"""

from typing import List, Optional
import requests
import time
from langchain_core.embeddings import Embeddings
import numpy as np
from app.core.logging import get_logger

logger = get_logger(__name__)

# Simple in-memory cache for non-RAG embeddings (FAQ, sensitive words, etc.)
# RAG embeddings are cached by llama-rag-sdk
_embedding_cache: dict[tuple[str, str], List[float]] = {}


def _get_cache_key(text: str, model: str) -> tuple[str, str]:
    """Get cache key for text and model."""
    return (text, model)


def _get_from_cache(text: str, model: str) -> List[float] | None:
    """Get embedding from cache."""
    return _embedding_cache.get(_get_cache_key(text, model))


def _set_cache(text: str, model: str, embedding: List[float]) -> None:
    """Set embedding in cache."""
    _embedding_cache[_get_cache_key(text, model)] = embedding


class TextTruncator:
    """Shared text truncator with intelligent fallback levels."""

    # 智能降级截断限制
    # bge-large-zh-v1.5-2k 模型对中文的实际字符限制约为 400
    # 使用更保守的限制来确保兼容性
    TRUNCATE_LIMITS = [384, 320, 256]

    def __init__(self, max_tokens: int = 1024):
        self.max_tokens = max_tokens
        self.max_chars = max_tokens

    def truncate(self, text: str) -> tuple[str, int, int]:
        """
        Truncate text with intelligent fallback levels.

        Args:
            text: Original text to truncate

        Returns:
            tuple[str, int, int]: (truncated_text, level, original_length)
                - truncated_text: Truncated text (may be original if no truncation needed)
                - level: Truncation level (0=512, 1=384, -1=none)
                - original_length: Original text length in chars
        """
        original_length = len(text)

        for level, limit in enumerate(self.TRUNCATE_LIMITS):
            if len(text) <= limit:
                return text, -1, original_length

        # Need to truncate
        for level, limit in enumerate(self.TRUNCATE_LIMITS):
            truncated = text[:limit - 3] + "..."
            if len(truncated) <= limit:
                return truncated, level, original_length

        # Final fallback
        raise ValueError(
            f"Text still too long after {len(self.TRUNCATE_LIMITS)} truncation attempts. "
            f"Original length: {original_length} chars"
        )


class ChromaEmbeddingWrapper:
    """Adapter to satisfy Chroma's EmbeddingFunction.__call__ signature."""

    def __init__(self, embedder: Embeddings):
        self.embedder = embedder

    def __call__(self, input: List[str]) -> List[List[float]]:  # type: ignore[override]
        return self.embedder.embed_documents(list(input))


class OllamaEmbeddings(Embeddings):
    """Ollama embedding implementation using /api/embeddings endpoint."""

    def __init__(
        self, model: str, base_url: str, batch_size: int = 32, max_tokens: int = 1024,
        max_chars: int = 384, enable_fallback: bool = True,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        # 固定字符限制，支持智能降级
        self.max_chars = max_chars
        self.enable_fallback = enable_fallback
        # 使用共享的文本截断器
        self.truncator = TextTruncator(max_tokens=max_tokens)
        # Request delay to avoid overwhelming Ollama server
        self.request_delay = 0.5  # 500ms delay between requests (further increased for stability)
        # Retry config
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds

    def _truncate_text(self, text: str, level: int = 0) -> tuple[str, int]:
        """
        智能截断文本，支持降级策略。

        Args:
            text: 原始文本
            level: 降级级别 (0=512, 1=384, 2+=直接抛异常)

        Returns:
            tuple[str, int]: (截断后的文本, 截断级别)
        """
        truncated_text, trunc_level, original_length = self.truncator.truncate(text)

        # 添加截断日志
        if trunc_level >= 0:
            logger.info(
                f"[OllamaEmbedding] Text truncated from {original_length} to {len(truncated_text)} chars "
                f"(level={trunc_level}, limit={self.truncator.TRUNCATE_LIMITS[trunc_level]})"
            )

        return truncated_text, trunc_level

    def _embed_single(self, text: str) -> List[float]:
        """
        Embed a single text using Ollama API with intelligent fallback.

        智能降级策略: 512字符 → 384字符 → 抛异常
        当 API 调用失败时，自动降低截断限制重试。
        """

        # Check cache first
        cached = _get_from_cache(text, self.model)
        if cached is not None:
            return cached

        original_length = len(text)

        # 智能降级循环: 尝试不同的截断级别
        for level in range(len(self.truncator.TRUNCATE_LIMITS)):
            truncated_text, trunc_level = self._truncate_text(text, level=level)
            text_length = len(truncated_text)

            url = f"{self.base_url}/api/embed"
            payload = {
                "model": self.model,
                "input": truncated_text,  # String, not array
                "keep_alive": 0  # 用完立即卸载，避免与其他模型冲突
            }

            # 尝试 API 调用（带重试机制）
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(url, json=payload, timeout=60.0)
                    response.raise_for_status()
                    result = response.json()

                    # /api/embed returns "embeddings" array, /api/embeddings returns "embedding"
                    if "embeddings" in result:
                        embedding = result["embeddings"][0]
                    elif "embedding" in result:
                        embedding = result["embedding"]
                    else:
                        raise ValueError(f"No embedding in response: {result}")

                    # Validate embedding dimension
                    if not embedding or len(embedding) == 0:
                        raise ValueError(f"Empty embedding returned: {result}")

                    # Cache the result
                    _set_cache(text, self.model, embedding)

                    logger.info(
                        f"Ollama embedding successful: level={level}, "
                        f"text_length={text_length} chars, vector_dim={len(embedding)}"
                    )
                    return embedding

                except requests.exceptions.HTTPError as e:
                    last_error = e
                    status_code = e.response.status_code if e.response is not None else 0

                    # Don't retry on 4xx errors (except 429 rate limit)
                    if 400 <= status_code < 500 and status_code != 429:
                        logger.error(
                            f"Ollama embedding failed (client error): url={url}, "
                            f"status={status_code}, error={str(e)}"
                        )
                        # 对于 4xx 错误，尝试下一级降级
                        break

                    # Retry on 5xx errors or timeouts
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(
                            f"Ollama embedding failed (attempt {attempt + 1}/{self.max_retries}), "
                            f"retrying in {wait_time}s: status={status_code}"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"Ollama embedding failed after {self.max_retries} attempts: "
                            f"status={status_code}"
                        )
                        # 尝试下一级降级
                        break

                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(
                            f"Ollama embedding timeout/connection error (attempt {attempt + 1}/{self.max_retries}), "
                            f"retrying in {wait_time}s"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Ollama embedding failed after {self.max_retries} attempts")
                        # 尝试下一级降级
                        break

                except Exception as e:
                    last_error = e
                    logger.error(
                        f"Ollama embedding failed with unexpected error: {str(e)}", exc_info=True
                    )
                    # 尝试下一级降级
                    break

            # 如果所有重试都失败，继续下一级降级
            if level < len(self.truncator.TRUNCATE_LIMITS) - 1:
                logger.warning(
                    f"Ollama embedding failed at level {level}, trying level {level + 1} "
                    f"with limit {self.truncator.TRUNCATE_LIMITS[level + 1]} chars"
                )

        # 所有降级级别都失败
        raise ValueError(
            f"Ollama embedding failed after {len(self.truncator.TRUNCATE_LIMITS)} truncation attempts. "
            f"Original length: {original_length} chars. "
            f"Attempted limits: {self.truncator.TRUNCATE_LIMITS}. "
            f"Last error: {str(last_error) if last_error else 'Unknown'}"
        )

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts using individual requests.

        Ollama API does NOT support batch array input for /api/embed.
        Each text must be embedded individually.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out cached texts
        uncached_indices = []
        uncached_texts = []
        cached_results = [None] * len(texts)

        for i, text in enumerate(texts):
            cached = _get_from_cache(text, self.model)
            if cached is not None:
                cached_results[i] = np.array(cached, dtype=float)
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # If all cached, return immediately
        if not uncached_texts:
            return cached_results

        # Truncate uncached texts
        truncated_texts = [self._truncate_text(t)[0] for t in uncached_texts]

        # Log batch info
        total_chars = sum(len(t) for t in truncated_texts)
        logger.info(
            f"Ollama batch embedding: batch_size={len(truncated_texts)}, "
            f"total_chars={total_chars}, max_chars={self.max_chars}, cached={len(texts) - len(uncached_texts)}"
        )

        # Process each text individually (Ollama doesn't support batch array input)
        url = f"{self.base_url}/api/embed"
        batch_embeddings = []

        for i, (idx, text, truncated_text) in enumerate(zip(uncached_indices, uncached_texts, truncated_texts)):
            # Add delay between requests to avoid overwhelming Ollama server
            if i > 0:
                time.sleep(self.request_delay)

            payload = {
                "model": self.model,
                "input": truncated_text,  # String, not array
                "keep_alive": 0  # 用完立即卸载，避免与其他模型冲突
            }

            last_error = None
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(url, json=payload, timeout=60.0)
                    response.raise_for_status()
                    result = response.json()

                    if "embeddings" in result:
                        embedding = result["embeddings"][0]
                    elif "embedding" in result:
                        embedding = result["embedding"]
                    else:
                        raise ValueError(f"No embedding in response: {result}")

                    # Validate embedding dimension
                    if not embedding or len(embedding) == 0:
                        raise ValueError(f"Empty embedding returned: {result}")

                    # Cache the result
                    _set_cache(text, self.model, embedding)
                    batch_embeddings.append(embedding)
                    cached_results[idx] = np.array(embedding, dtype=float)

                    break

                except requests.exceptions.HTTPError as e:
                    last_error = e
                    status_code = e.response.status_code if e.response is not None else 0

                    # Don't retry on 4xx errors (except 429 rate limit)
                    if 400 <= status_code < 500 and status_code != 429:
                        logger.error(
                            f"Ollama embedding failed (client error): status={status_code}, error={str(e)}"
                        )
                        raise

                    # Retry on 5xx errors or timeouts
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(
                            f"Ollama embedding failed (attempt {attempt + 1}/{self.max_retries}), "
                            f"retrying in {wait_time}s: status={status_code}"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Ollama embedding failed after {self.max_retries} attempts: status={status_code}")

                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(
                            f"Ollama embedding timeout/connection error (attempt {attempt + 1}/{self.max_retries}), "
                            f"retrying in {wait_time}s"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Ollama embedding failed after {self.max_retries} attempts")

                except Exception as e:
                    last_error = e
                    logger.error(
                        f"Ollama embedding failed with unexpected error: {str(e)}", exc_info=True
                    )
                    raise

        logger.info(
            f"[OllamaEmbedding] Batch embedding completed | "
            f"model={self.model} | processed={len(batch_embeddings)} | "
            f"embedding_dim={len(batch_embeddings[0]) if batch_embeddings else 0}"
        )

        return cached_results

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents using batch requests.

        Uses batch_size=8 for optimal quality (batch_size >= 16 causes quality degradation).
        """
        # Check for truncation
        truncated_count = sum(1 for text in texts if len(text) > self.max_chars)
        if truncated_count > 0:
            logger.warning(f"Ollama: Truncating {truncated_count}/{len(texts)} texts to fit {self.max_tokens} token limit")

        logger.info(f"Ollama embedding request: url={self.base_url}, model={self.model}, texts_count={len(texts)}")

        embeddings = []
        cache_hits = 0
        request_count = 0

        # Process in batches of 8 (optimal for quality, see GitHub issue #6262)
        BATCH_SIZE = 8
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]

            # Check cache for entire batch
            uncached_count = sum(1 for t in batch if _get_from_cache(t, self.model) is None)

            if uncached_count == 0:
                # All cached
                cache_hits += len(batch)
                for text in batch:
                    cached = _get_from_cache(text, self.model)
                    embeddings.append(np.array(cached, dtype=float))
            else:
                # Use batch request
                if request_count > 0:
                    time.sleep(self.request_delay)

                batch_embeddings = self._embed_batch(batch)
                embeddings.extend(batch_embeddings)
                request_count += 1

                # Track cache hits from batch result
                cache_hits += sum(1 for t in batch if _get_from_cache(t, self.model) is not None)

        logger.info(f"Ollama request successful: received {len(embeddings)} embeddings, cache_hits={cache_hits}, batch_requests={request_count}")

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

    使用 SDK 的 vLLM embeddings 配置（VLLM_EMBEDDING_BASE_URL 等）。

    Args:
        settings: 应用配置对象（可选，默认使用全局配置）

    Returns:
        Embeddings 实例
    """
    import os
    from app.core.config import settings as app_settings

    if not settings:
        settings = app_settings

    # 使用 SDK 的 vLLM embeddings 配置
    base_url = os.getenv("VLLM_EMBEDDING_BASE_URL", settings.embedding_base_url)
    model = os.getenv("VLLM_EMBEDDING_MODEL", settings.embedding_model)
    api_key = os.getenv("VLLM_API_KEY", settings.embedding_api_key or "not-needed")

    logger.info(f"Creating vLLM embeddings: model={model}, base_url={base_url}")
    return OpenAIStyleEmbeddings(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=30.0,
    )
