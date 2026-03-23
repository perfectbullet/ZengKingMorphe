"""
检索器

提供统一的检索接口，使用混合检索 + Rerank + 查询扩展流水线
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

from llama_rag_sdk.config import settings
from llama_rag_sdk.document_indexer.storage import VectorStore
from llama_rag_sdk.embedding_factory import EmbeddingFactory
from llama_rag_sdk.retrieval.base import RetrievedDocument
from llama_rag_sdk.retrieval.reranker import BGERerankerClientError
from llama_rag_sdk.retrieval.strategies import HybridRerankRetrieval

if TYPE_CHECKING:
    from llama_rag_sdk.retrieval.query_expansion import QueryExpander


class Retriever:
    """统一检索器（混合检索 + Rerank + 查询扩展）"""

    def __init__(
        self,
        vector_store: VectorStore,
        candidate_multiplier: Optional[int] = None,
        embedding_model: Optional[Any] = None,
        enable_query_expansion: Optional[bool] = None,
    ):
        """
        初始化检索器

        Args:
            vector_store: 向量存储实例
            candidate_multiplier: 候选数量倍数（默认从配置读取）
            embedding_model: Embedding 模型
            enable_query_expansion: 是否启用查询扩展（默认从配置读取）

        Raises:
            RuntimeError: 当 Reranker 服务不可用时抛出异常
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model

        # 创建查询扩展器
        self.query_expander = self._create_query_expander()

        # 创建检索策略
        self.strategy = self._create_strategy(
            candidate_multiplier,
            enable_query_expansion=enable_query_expansion
        )

    def _create_embedding_model(self) -> Optional[Any]:
        """创建 Embedding 模型（使用工厂模式）"""
        return EmbeddingFactory.create_embedding_model(
            model_name=self.embedding_model
        ) if self.embedding_model else EmbeddingFactory.create_embedding_model()

    def _create_query_expander(self) -> Optional["QueryExpander"]:
        """
        创建查询扩展器（仅 LLM 语义扩展）

        Returns:
            QueryExpander 实例，如果配置禁用则返回 None
        """
        # 检查配置是否启用查询扩展
        try:
            enabled = getattr(settings, 'query_expansion_enabled', False)
        except Exception:
            enabled = False

        if not enabled:
            logger.debug("查询扩展未启用")
            return None

        try:
            from llama_rag_sdk.retrieval.query_expansion import QueryExpander
            from llama_rag_sdk.retrieval.llm_client import create_llm_client

            # 创建 LLM 客户端
            llm_client = create_llm_client(
                provider=getattr(settings, 'llm_provider', 'ollama'),
                base_url=getattr(settings, 'llm_base_url', 'http://localhost:11434'),
                model=getattr(settings, 'llm_model', 'qwen2.5:14b'),
                api_key=getattr(settings, 'llm_api_key', 'not-needed'),
                timeout=getattr(settings, 'llm_timeout', 60)
            )

            # 获取最大扩展数量
            max_expansions = getattr(
                settings,
                'query_expansion_max_expansions',
                3
            )

            expander = QueryExpander(
                llm_client=llm_client,
                max_total_expansions=max_expansions
            )

            logger.info("查询扩展器已创建（仅 LLM 语义扩展）")
            return expander

        except Exception as e:
            logger.warning(f"创建查询扩展器失败: {e}，将不使用查询扩展")
            return None

    def _create_strategy(
        self,
        candidate_multiplier: Optional[int] = None,
        enable_query_expansion: Optional[bool] = None
    ) -> HybridRerankRetrieval:
        """
        创建检索策略 - 统一使用混合+Rerank+查询扩展

        Args:
            candidate_multiplier: 候选数量倍数
            enable_query_expansion: 是否启用查询扩展

        Returns:
            检索策略实例

        Raises:
            RuntimeError: 当 Reranker 服务不可用时抛出异常
        """
        # 创建 Reranker 客户端
        from llama_rag_sdk.retrieval.reranker import BGERerankerClient

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

        # 确定是否启用查询扩展
        use_qe = enable_query_expansion
        if use_qe is None:
            try:
                use_qe = settings.query_expansion_enabled
            except Exception:
                use_qe = False

        # 统一使用混合+Rerank+查询扩展策略
        strategy = HybridRerankRetrieval(
            vector_store=self.vector_store,
            embedding_model=embed_model,
            rerank_client=rerank_client,
            candidate_multiplier=candidate_multiplier or settings.rerank_candidate_multiplier,
            query_expander=self.query_expander,
            enable_query_expansion=use_qe and self.query_expander is not None,
        )

        qe_status = "启用" if use_qe else "未启用"
        logger.info(f"使用混合检索 + Rerank 策略，查询扩展: {qe_status}")
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
