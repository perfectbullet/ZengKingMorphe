"""
检索模块

提供混合检索 + Rerank + 查询扩展流水线
"""

from src.retrieval.base import RetrievalStrategy, RetrievedDocument
from src.retrieval.strategies import HybridRerankRetrieval
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import BGERerankerClient
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.llm_client import (
    LLMClient,
    LLMProvider,
    OllamaLLMClient,
    OpenAICompatibleLLMClient,
    create_llm_client,
)

__all__ = [
    "RetrievalStrategy",
    "RetrievedDocument",
    "HybridRerankRetrieval",
    "Retriever",
    "BGERerankerClient",
    "QueryExpander",
    "LLMClient",
    "LLMProvider",
    "OllamaLLMClient",
    "OpenAICompatibleLLMClient",
    "create_llm_client",
]
