"""
检索策略实现
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from src.retrieval.base import RetrievalStrategy, RetrievedDocument
from src.document_indexer.storage import VectorStore
from src.config import settings

try:
    from llama_index.embeddings.ollama import OllamaEmbedding
except ImportError:
    logger.warning("llama-index-embeddings-ollama 未安装")
    OllamaEmbedding = None


class VectorRetrieval(RetrievalStrategy):
    """纯向量检索策略"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: Optional[Any] = None
    ):
        """
        初始化向量检索

        Args:
            vector_store: 向量存储实例
            embedding_model: Embedding 模型
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def _create_embedding(self, text: str) -> List[float]:
        """
        生成查询的嵌入向量

        Args:
            text: 查询文本

        Returns:
            嵌入向量
        """
        if self.embedding_model is None:
            if OllamaEmbedding is None:
                raise RuntimeError("Embedding 模型不可用")

            self.embedding_model = OllamaEmbedding(
                model_name=settings.ollama_embedding_model,
                base_url=settings.ollama_base_url
            )

        try:
            return self.embedding_model.get_text_embedding(text)
        except Exception as e:
            logger.error(f"生成查询嵌入向量失败: {e}")
            raise

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        执行向量检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            检索到的文档列表
        """
        logger.info(f"执行向量检索: query='{query}', top_k={top_k}")

        # 生成查询嵌入
        query_embedding = self._create_embedding(query)

        # 查询向量存储
        results = self.vector_store.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )

        # 转换为 RetrievedDocument
        documents = []
        ids = results.get('ids', [[]])[0]
        texts = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]

        for i, doc_id in enumerate(ids):
            if i < len(texts):
                score = 1.0 - distances[i] if i < len(distances) else 0.0

                # 应用相似度阈值
                if score >= settings.similarity_threshold:
                    doc = RetrievedDocument(
                        text=texts[i],
                        metadata=metadatas[i] if i < len(metadatas) else {},
                        score=score,
                        source=metadatas[i].get('source', '') if i < len(metadatas) else '',
                        chunk_id=metadatas[i].get('chunk_id', '') if i < len(metadatas) else ''
                    )
                    documents.append(doc)

        logger.info(f"检索完成: 返回 {len(documents)} 个结果")
        return documents


class HybridRetrieval(RetrievalStrategy):
    """
    混合检索策略（向量 + 关键词）

    注意：完整实现需要 BM25 索引支持
    这里提供简化版本，仅使用向量检索
    """

    def __init__(
        self,
        vector_retrieval: VectorRetrieval,
        alpha: float = 0.7
    ):
        """
        初始化混合检索

        Args:
            vector_retrieval: 向量检索实例
            alpha: 向量检索权重（0-1）
        """
        self.vector_retrieval = vector_retrieval
        self.alpha = alpha

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        执行混合检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            检索到的文档列表
        """
        logger.info(f"执行混合检索: query='{query}', alpha={self.alpha}")

        # 当前实现仅使用向量检索
        # 完整实现需要 BM25 索引
        documents = await self.vector_retrieval.retrieve(query, top_k, filters)

        # 这里可以添加关键词匹配逻辑
        # 例如：基于简单的文本匹配

        return documents


class RerankRetrieval(RetrievalStrategy):
    """
    重排序检索策略

    对检索结果进行重排序，提高相关性
    """

    def __init__(
        self,
        base_retrieval: RetrievalStrategy,
        rerank_model: Optional[str] = None
    ):
        """
        初始化重排序检索

        Args:
            base_retrieval: 基础检索策略
            rerank_model: 重排序模型名称
        """
        self.base_retrieval = base_retrieval
        self.rerank_model = rerank_model or settings.rerank_model

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        执行重排序检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            检索到的文档列表
        """
        logger.info(f"执行重排序检索: query='{query}', top_k={top_k}")

        # 先获取更多候选结果
        candidates = await self.base_retrieval.retrieve(
            query,
            top_k=top_k * 2,  # 获取更多候选
            filters=filters
        )

        if not candidates:
            return []

        # 如果有重排序模型，使用它进行重排序
        # 这里提供简化的基于分数的重排序
        # 完整实现需要集成 BGE-Reranker 等模型

        # 按分数排序
        reranked = sorted(
            candidates,
            key=lambda x: x.score,
            reverse=True
        )

        # 返回 top_k 结果
        return reranked[:top_k]
