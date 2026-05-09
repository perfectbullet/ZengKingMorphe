"""
检索策略实现

统一的混合检索 + Rerank + 查询扩展流水线
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

from llama_rag_sdk.config import settings
from llama_rag_sdk.document_indexer.storage import VectorStore
from llama_rag_sdk.embedding_factory import EmbeddingFactory
from llama_rag_sdk.retrieval.base import RetrievedDocument, RetrievalStrategy
from llama_rag_sdk.retrieval.reranker import BGERerankerClientError

if TYPE_CHECKING:
    from llama_rag_sdk.retrieval.query_expansion import QueryExpander


class HybridRerankRetrieval(RetrievalStrategy):
    """
    混合检索 + Rerank + 查询扩展流水线

    流程: 查询文本 → 查询扩展 → 向量检索获取候选 → BGE Rerank 重排序 → 返回结果
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: Optional[Any] = None,
        rerank_client: Optional[Any] = None,
        candidate_multiplier: int = None,
        query_expander: Optional["QueryExpander"] = None,
        enable_query_expansion: bool = False,
    ):
        """
        初始化混合检索 + Rerank + 查询扩展策略

        Args:
            vector_store: 向量存储实例
            embedding_model: Embedding 模型
            rerank_client: BGE Reranker 客户端（必需）
            candidate_multiplier: 候选数量倍数（候选 = top_k * multiplier）
            query_expander: 查询扩展器实例
            enable_query_expansion: 是否启用查询扩展

        Raises:
            ValueError: rerank_client 为 None 时抛出
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.rerank_client = rerank_client
        self.candidate_multiplier = candidate_multiplier or settings.rerank_candidate_multiplier
        self.query_expander = query_expander
        self.enable_query_expansion = enable_query_expansion

        if self.rerank_client is None:
            raise ValueError("rerank_client 是必需参数，不能为 None")

        logger.info(
            f"初始化混合检索 + Rerank 策略: "
            f"candidate_multiplier={self.candidate_multiplier}, "
            f"query_expansion={self.enable_query_expansion}"
        )

    def _create_embedding(self, text: str) -> List[float]:
        """
        生成查询的嵌入向量（使用工厂模式）

        Args:
            text: 查询文本

        Returns:
            嵌入向量
        """
        if self.embedding_model is None:
            self.embedding_model = EmbeddingFactory.create_embedding_model()

        if self.embedding_model is None:
            raise RuntimeError("Embedding 模型不可用")

        try:
            return self.embedding_model.get_text_embedding(text)
        except Exception as e:
            logger.error(f"生成查询嵌入向量失败: {e}")
            raise

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
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
        logger.debug(f"执行向量检索: query='{query}', top_k={top_k}")

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

                doc = RetrievedDocument(
                    text=texts[i],
                    metadata=metadatas[i] if i < len(metadatas) else {},
                    score=score,
                    source=metadatas[i].get('source', '') if i < len(metadatas) else '',
                    chunk_id=metadatas[i].get('chunk_id', '') if i < len(metadatas) else ''
                )
                documents.append(doc)

        return documents

    async def _rerank_documents(
        self,
        query: str,
        candidates: List[RetrievedDocument],
    ) -> List[RetrievedDocument]:
        """
        使用 BGE Rerank 重排序候选文档

        Args:
            query: 查询文本
            candidates: 候选文档列表

        Returns:
            重排序后的文档列表

        Raises:
            BGERerankerClientError: Rerank 请求失败时抛出异常
        """
        if not candidates:
            return []

        # 提取文档文本
        documents = [doc.text for doc in candidates]

        # 调用 Reranker API
        try:
            ranked_results = self.rerank_client.rerank(query, documents, top_n=len(documents))
        except BGERerankerClientError as e:
            logger.error(f"Rerank 调用失败: {e}")
            raise

        # 按重排序结果重新组织文档
        reranked = []
        for idx, _, score in ranked_results:
            if 0 <= idx < len(candidates):
                # 更新分数为 Reranker 分数
                doc = candidates[idx]
                # 归一化分数到 0-1 范围 (使用配置常量)
                normalized_score = (
                    (score - settings.rerank_score_min) /
                    (settings.rerank_score_max - settings.rerank_score_min)
                    if score is not None else doc.score
                )
                doc.score = max(0.0, min(1.0, normalized_score))
                reranked.append(doc)

        logger.debug(
            f"Rerank 完成: 原始数量={len(candidates)}, "
            f"重排序后={len(reranked)}"
        )

        return reranked

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        执行混合检索 + Rerank + 查询扩展

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            检索到的文档列表
        """
        logger.info(f"执行混合检索 + Rerank: query='{query}', top_k={top_k}")

        # 1. 查询扩展
        queries = [query]
        if self.enable_query_expansion and self.query_expander:
            logger.info(f"查询扩展已启用，正在处理原始查询...")
            queries = await self.query_expander.expand(query)
            logger.info(f"查询扩展结果: {queries}")
        else:
            logger.info(f"查询扩展未启用 (enable={self.enable_query_expansion}, expander={'已创建' if self.query_expander else '未创建'})")

        # 2. 对每个扩展查询执行检索
        all_candidates = []
        for expanded_query in queries:
            candidate_count = top_k * self.candidate_multiplier
            candidates = await self._vector_search(expanded_query, candidate_count, filters)
            all_candidates.extend(candidates)
            logger.debug(f"查询 '{expanded_query[:30]}...' 检索到 {len(candidates)} 个候选")

        if not all_candidates:
            logger.info("未检索到候选文档")
            return []

        # 3. 去重（按 chunk_id）
        unique_candidates = self._deduplicate_documents(all_candidates)
        logger.debug(f"去重后候选数: {len(all_candidates)} → {len(unique_candidates)}")

        # 4. BGE Rerank 重排序（使用原始查询）
        reranked = await self._rerank_documents(query, unique_candidates)

        # 5. 返回前 top_k 结果
        results = reranked[:top_k]

        logger.info(
            f"检索完成: 扩展查询数={len(queries)}, "
            f"总候选={len(all_candidates)}, "
            f"去重后={len(unique_candidates)}, "
            f"重排序后={len(reranked)}, "
            f"返回={len(results)}"
        )

        return results

    def _deduplicate_documents(
        self,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        按 chunk_id 去重

        Args:
            documents: 文档列表

        Returns:
            去重后的文档列表
        """
        seen = set()
        unique = []
        for doc in documents:
            if doc.chunk_id and doc.chunk_id not in seen:
                seen.add(doc.chunk_id)
                unique.append(doc)
            elif not doc.chunk_id:
                # 如果没有 chunk_id，根据文本内容去重
                text_hash = hash(doc.text)
                if text_hash not in seen:
                    seen.add(text_hash)
                    unique.append(doc)
        return unique
