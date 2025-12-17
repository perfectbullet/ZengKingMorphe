"""
Custom embedding implementations for the Digital Employee AI Service.
"""
from typing import List, Optional
import requests
from langchain_core.embeddings import Embeddings


class ChromaEmbeddingWrapper:
    """Adapter to satisfy Chroma's EmbeddingFunction.__call__ signature."""

    def __init__(self, embedder: Embeddings):
        self.embedder = embedder

    def __call__(self, input: List[str]) -> List[List[float]]:  # type: ignore[override]
        return self.embedder.embed_documents(list(input))


class SiliconFlowEmbeddings(Embeddings):
    """SiliconFlow embedding implementation."""
    
    def __init__(self, model: str, api_key: str, base_url: str, batch_size: int = 32):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.batch_size = batch_size

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self.model, "input": texts}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.base_url, json=payload, headers=headers)
        result = response.json()
        if result.get("code") not in (None, 0):
            raise ValueError(f"Embedding request failed: {result}")
        data = result.get("data")
        if not data:
            raise ValueError(f"No embedding data returned: {result}")
        return [item["embedding"] for item in data]

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

        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()

        data = result.get("data")
        if not data:
            raise ValueError(f"Embedding service returned no data: {result}")
        return [item["embedding"] for item in data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]
