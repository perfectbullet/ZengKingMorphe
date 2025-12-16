import os
from typing import List, Optional

import requests
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama, ChatOpenAI
from langchain_core.embeddings import Embeddings

load_dotenv()
# ==================== Embedding 实现 ====================
class SiliconFlowEmbeddings(Embeddings):
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
        print(response)
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
            "User-Agent": "crag-service/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            print(f"Using API Key for Embedding service authentication.{self.api_key}")

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

# ==================== Embedding 用法 ====================
embedding = OpenAIStyleEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"),
            base_url=os.getenv("EMBEDDING_BASE_URL", "http://localhost:50009"),
        )
        
embedding = SiliconFlowEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"),
    base_url=os.getenv("EMBEDDING_API_URL", "http://localhost:50009"),
    api_key=os.environ.get("SILICONFLOW_API_KEY", None),
)






# ==================== OpenAI SDK 调用示例 ====================
if os.environ["USE_OLLAMA"]:
    # 初始化 LLM
    llm = ChatOllama(
        base_url=os.environ["OLLAMA_BASE_URL"],
        model=os.environ["OLLAMA_MODEL"],
        temperature=0,
        streaming=True,
    )

    grader_llm = ChatOllama(
        base_url=os.environ["OLLAMA_BASE_URL"],
        model=os.environ.get("OLLAMA_GRADER_MODEL", "qwen2.5:7b"),  # 小模型
        temperature=0,  #
        format="json",  # 强制 JSON
    )
else:
    llm = ChatOpenAI(
        base_url="https://api.siliconflow.cn/v1",
        api_key=os.environ["SILICONFLOW_API_KEY"],
        model="deepseek-ai/DeepSeek-V3.1-Terminus",
        temperature=0,
        streaming=True,
    )
    grader_llm = ChatOpenAI(
        base_url="https://api.siliconflow.cn/v1",
        api_key=os.environ["SILICONFLOW_API_KEY"],
        model="deepseek-ai/DeepSeek-V3",  # 非推理版本，快速评分
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )