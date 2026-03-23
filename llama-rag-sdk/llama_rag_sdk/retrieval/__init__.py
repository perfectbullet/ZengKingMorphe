"""
检索模块

提供混合检索 + Rerank + 查询扩展 + 上下文扩展流水线
"""

from llama_rag_sdk.retrieval.base import RetrievalStrategy, RetrievedDocument
from llama_rag_sdk.retrieval.strategies import HybridRerankRetrieval
from llama_rag_sdk.retrieval.retriever import Retriever
from llama_rag_sdk.retrieval.reranker import BGERerankerClient
from llama_rag_sdk.retrieval.query_expansion import QueryExpander
from llama_rag_sdk.retrieval.context_expander import ContextExpander, AutoMergingRetriever
from llama_rag_sdk.retrieval.llm_client import (
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
    "ContextExpander",
    "AutoMergingRetriever",
    "LLMClient",
    "LLMProvider",
    "OllamaLLMClient",
    "OpenAICompatibleLLMClient",
    "create_llm_client",
]
