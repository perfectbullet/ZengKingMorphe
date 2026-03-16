"""
检索器

提供统一的检索接口
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from src.retrieval.base import RetrievedDocument
from src.retrieval.strategies import (
    VectorRetrieval,
    HybridRetrieval,
    RerankRetrieval
)
from src.document_indexer.storage import VectorStore
from src.config import settings

try:
    from llama_index.embeddings.ollama import OllamaEmbedding
except ImportError:
    logger.warning("llama-index-embeddings-ollama 未安装")
    OllamaEmbedding = None


class Retriever:
    """统一检索器"""

    def __init__(
        self,
        vector_store: VectorStore,
        use_hybrid: bool = False,
        use_rerank: bool = False,
        embedding_model: Optional[Any] = None
    ):
        """
        初始化检索器

        Args:
            vector_store: 向量存储实例
            use_hybrid: 是否使用混合检索
            use_rerank: 是否使用重排序
            embedding_model: Embedding 模型
        """
        self.vector_store = vector_store
        self.use_hybrid = use_hybrid or settings.use_hybrid_retrieval
        self.use_rerank = use_rerank or settings.use_rerank
        self.embedding_model = embedding_model

        # 创建检索策略
        self.strategy = self._create_strategy()

    def _create_embedding_model(self) -> Optional[Any]:
        """创建 Embedding 模型"""
        if self.embedding_model is not None:
            return self.embedding_model

        if OllamaEmbedding is None:
            logger.warning("OllamaEmbedding 不可用")
            return None

        try:
            return OllamaEmbedding(
                model_name=settings.ollama_embedding_model,
                base_url=settings.ollama_base_url
            )
        except Exception as e:
            logger.error(f"初始化 Embedding 模型失败: {e}")
            return None

    def _create_strategy(self):
        """创建检索策略"""
        embed_model = self._create_embedding_model()

        # 基础向量检索
        vector_retrieval = VectorRetrieval(
            vector_store=self.vector_store,
            embedding_model=embed_model
        )

        # 根据配置选择策略
        if self.use_rerank:
            # 重排序检索
            strategy = RerankRetrieval(
                base_retrieval=vector_retrieval
            )
            logger.info("使用重排序检索策略")
        elif self.use_hybrid:
            # 混合检索
            strategy = HybridRetrieval(
                vector_retrieval=vector_retrieval
            )
            logger.info("使用混合检索策略")
        else:
            # 纯向量检索
            strategy = vector_retrieval
            logger.info("使用向量检索策略")

        return strategy

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        执行检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            检索到的文档列表
        """
        if top_k is None:
            top_k = settings.top_k

        logger.debug(f"检索: query='{query}', top_k={top_k}, filters={filters}")

        # 使用配置的策略执行检索
        documents = await self.strategy.retrieve(
            query=query,
            top_k=top_k,
            filters=filters
        )

        return documents

    async def retrieve_multiple(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        deduplicate: bool = True
    ) -> List[RetrievedDocument]:
        """
        执行多次检索

        Args:
            queries: 查询文本列表
            top_k: 每次查询返回结果数量
            filters: 过滤条件
            deduplicate: 是否去重

        Returns:
            检索到的文档列表
        """
        if top_k is None:
            top_k = settings.top_k

        all_documents = []

        for query in queries:
            docs = await self.retrieve(query, top_k, filters)
            all_documents.extend(docs)

        if deduplicate:
            # 按 chunk_id 去重
            seen = set()
            unique_documents = []
            for doc in all_documents:
                if doc.chunk_id and doc.chunk_id not in seen:
                    seen.add(doc.chunk_id)
                    unique_documents.append(doc)
            all_documents = unique_documents

        logger.info(f"多次检索完成: {len(queries)} 个查询 -> {len(all_documents)} 个结果")
        return all_documents
