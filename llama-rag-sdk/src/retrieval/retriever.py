"""
检索器

提供统一的检索接口，使用混合检索 + Rerank 流水线
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from src.retrieval.base import RetrievedDocument
from src.retrieval.strategies import HybridRerankRetrieval
from src.retrieval.reranker import BGERerankerClientError
from src.document_indexer.storage import VectorStore
from src.config import settings
from src.embedding_factory import EmbeddingFactory


class Retriever:
    """统一检索器（混合检索 + Rerank）"""

    def __init__(
        self,
        vector_store: VectorStore,
        candidate_multiplier: Optional[int] = None,
        embedding_model: Optional[Any] = None
    ):
        """
        初始化检索器

        Args:
            vector_store: 向量存储实例
            candidate_multiplier: 候选数量倍数（默认从配置读取）
            embedding_model: Embedding 模型

        Raises:
            RuntimeError: 当 Reranker 服务不可用时抛出异常
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model

        # 创建检索策略
        self.strategy = self._create_strategy(candidate_multiplier)

    def _create_embedding_model(self) -> Optional[Any]:
        """创建 Embedding 模型（使用工厂模式）"""
        return EmbeddingFactory.create_embedding_model(
            model_name=self.embedding_model
        ) if self.embedding_model else EmbeddingFactory.create_embedding_model()
        if self.embedding_model is not None:
            return self.embedding_model

        if OpenAIEmbedding is None:
            logger.warning("OpenAIEmbedding 不可用")
            return None

        try:
            return OpenAIEmbedding(
                model_name=settings.vllm_embedding_model,
                api_base=settings.vllm_embedding_base_url,
                api_key=settings.vllm_api_key,
                embed_batch_size=32,
                timeout=300,
            )
        except Exception as e:
            logger.error(f"初始化 Embedding 模型失败: {e}")
            return None

    def _create_strategy(self, candidate_multiplier: Optional[int] = None) -> HybridRerankRetrieval:
        """
        创建检索策略 - 统一使用混合+Rerank

        Args:
            candidate_multiplier: 候选数量倍数

        Returns:
            检索策略实例

        Raises:
            RuntimeError: 当 Reranker 服务不可用时抛出异常
        """
        # 创建 Reranker 客户端
        from src.retrieval.reranker import BGERerankerClient

        try:
            rerank_client = BGERerankerClient(
                base_url=settings.rerank_base_url,
                api_key=settings.vllm_api_key,
                model=settings.rerank_model,
                timeout=settings.rerank_timeout,
            )
            logger.info(f"BGE Reranker 客户端已创建: {settings.rerank_base_url}")
        except BGERerankerClientError as e:
            raise RuntimeError(
                f"无法连接到 BGE Reranker 服务: {e}\n"
                f"请确认服务已启动: {settings.rerank_base_url}"
            )

        embed_model = self._create_embedding_model()

        # 统一使用混合+Rerank策略
        strategy = HybridRerankRetrieval(
            vector_store=self.vector_store,
            embedding_model=embed_model,
            rerank_client=rerank_client,
            candidate_multiplier=candidate_multiplier or settings.rerank_candidate_multiplier,
        )

        logger.info("使用混合检索 + Rerank 策略")
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

        Raises:
            BGERerankerClientError: Rerank 请求失败时抛出异常
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

        Raises:
            BGERerankerClientError: Rerank 请求失败时抛出异常
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
