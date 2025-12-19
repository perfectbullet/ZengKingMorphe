"""
Custom embedding implementations for the Digital Employee AI Service.
"""
from typing import List, Optional
import requests
from langchain_core.embeddings import Embeddings
import numpy as np

class ChromaEmbeddingWrapper:
    """Adapter to satisfy Chroma's EmbeddingFunction.__call__ signature."""

    def __init__(self, embedder: Embeddings):
        self.embedder = embedder

    def __call__(self, input: List[str]) -> List[List[float]]:  # type: ignore[override]
        return self.embedder.embed_documents(list(input))


class SiliconFlowEmbeddings(Embeddings):
    """SiliconFlow embedding implementation."""
    
    def __init__(self, model: str, api_key: str, base_url: str, batch_size: int = 32, max_tokens: int = 512):
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
        return text[:self.max_chars - 3] + "..."

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Truncate texts to fit token limit
        truncated_texts = [self._truncate_text(text) for text in texts]
        
        payload = {"model": self.model, "input": truncated_texts}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Add debug logging
        import logging
        logger = logging.getLogger(__name__)
        
        # Log truncation stats
        truncated_count = sum(1 for orig, trunc in zip(texts, truncated_texts) if len(orig) > len(trunc))
        if truncated_count > 0:
            logger.warning(
                f"Truncated {truncated_count}/{len(texts)} texts to fit {self.max_tokens} token limit"
            )
        
        logger.info(
            f"SiliconFlow embedding request: url={self.base_url}, "
            f"model={self.model}, texts_count={len(truncated_texts)}"
        )
        
        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            result = response.json()
            if result.get("code") not in (None, 0):
                raise ValueError(f"Embedding request failed: {result}")
            data = result.get("data")
            if not data:
                raise ValueError(f"No embedding data returned: {result}")
            
            logger.info(f"SiliconFlow request successful: received {len(data)} embeddings")
            return [item["embedding"] for item in data]
        except Exception as e:
            logger.error(
                f"SiliconFlow embedding failed: url={self.base_url}, error={str(e)}",
                exc_info=True
            )
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
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"Sending embedding request: url={url}, model={self.model}, "
            f"texts_count={len(texts)}, has_api_key={bool(self.api_key)}"
        )
        
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
        
            return [item["embedding"] for item in data]
        except Exception as e:
            logger.error(
                f"Embedding request failed: url={url}, error={str(e)}",
                exc_info=True
            )
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]
