"""
RAG retrieval service using llama-rag-sdk.

使用 llama-rag-sdk 进行向量检索和 FAQ 检索。
"""
from typing import List, Dict, Any, Optional

from app.core.logging import get_logger
from llama_rag_sdk.rag_system import RAGSystem

logger = get_logger(__name__)


class RAGRetrieval:
    """
    RAG 检索服务（完全使用 llama-rag-sdk）

    使用 SDK 的 RAGSystem 进行：
    - 标准文档检索（向量 + 内置 Reranker）
    - FAQ 检索
    """

    def __init__(self):
        """Initialize RAGRetrieval."""
        self._rag_system = None
        self._faq_system = None

    @property
    def rag_system(self):
        """文档检索用 RAGSystem（延迟初始化）"""
        if self._rag_system is None:
            
            self._rag_system = RAGSystem(
                collection_name="rag_documents",
                enable_image_description=False,
                enable_summarization=True,
            )
        return self._rag_system

    @property
    def faq_system(self):
        """FAQ 检索用 RAGSystem（独立 collection，延迟初始化）"""
        if self._faq_system is None:
            from llama_rag_sdk.rag_system import RAGSystem
            self._faq_system = RAGSystem(
                collection_name="rag_faq",
                enable_image_description=False,
                enable_summarization=False,
            )
        return self._faq_system

    async def search(
        self,
        query: str,
        kb_ids: Optional[List[str]] = None,
        top_k: int = 5,
        use_hybrid: bool = True,  # 忽略，SDK 只用向量搜索
        enable_rerank: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        使用 SDK 进行文档检索（向量 + 内置 Reranker）

        Args:
            query: 搜索查询
            kb_ids: 知识库 ID 列表
            top_k: 返回结果数量
            use_hybrid: 忽略（SDK 只使用向量搜索）
            enable_rerank: 忽略（SDK 内部使用 Reranker）

        Returns:
            相关文档列表
        """
        try:
            # 构建过滤条件
            filters = {"kb_id": {"$in": kb_ids}} if kb_ids else None

            # 调用 SDK 检索（向量 + Reranker）
            results = await self.rag_system.retrieve(
                query=query,
                top_k=top_k,
                filters=filters,
            )

            # 转换为 ai-service 格式
            return [
                {
                    "content": doc.text,
                    "score": doc.score,
                    "doc_id": doc.metadata.get("doc_id"),
                    "kb_id": doc.metadata.get("kb_id"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "content_type": doc.metadata.get("content_type", "text"),
                    "context_text": doc.metadata.get("context_text", ""),
                }
                for doc in results
            ]

        except Exception as e:
            logger.error(f"SDK RAG search failed: query={query[:50]}, error={e}", exc_info=True)
            raise

    async def faq_search(
        self,
        query: str,
        employee_id: str,
        faq_top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        FAQ 搜索（使用 SDK）

        Args:
            query: 搜索查询
            employee_id: 员工 ID
            faq_top_k: 返回结果数量

        Returns:
            FAQ 列表，每项包含 faq_id, question_name, combined_text, score
        """
        try:
            logger.info(
                f"FAQ search: query={query[:100]}, employee_id={employee_id}, top_k={faq_top_k}"
            )

            results = await self.faq_system.retrieve(
                query=query,
                top_k=faq_top_k,
                filters={"employee_id": employee_id},
            )

            faq_results = [
                {
                    "faq_id": doc.metadata.get("faq_id"),
                    "question_name": doc.metadata.get("question_name"),
                    "combined_text": doc.text,
                    "score": doc.score,  # BGE reranker score
                }
                for doc in results
            ]

            logger.info(
                f"FAQ search completed: results={len(faq_results)}, "
                f"best_score={faq_results[0]['score'] if faq_results else 0}"
            )

            return faq_results

        except Exception as e:
            logger.error(f"FAQ search failed: query={query[:50]}, error={e}", exc_info=True)
            return []


# Global RAG retrieval instance
rag_retrieval = RAGRetrieval()
