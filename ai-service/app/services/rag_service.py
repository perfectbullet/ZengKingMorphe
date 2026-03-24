"""
RAG retrieval service using llama-rag-sdk.

使用 llama-rag-sdk 进行向量检索和 FAQ 检索。

架构说明：
- 每个 kb_id 使用独立的 ChromaDB 集合（rag_documents_<kb_id>）
- 支持多知识库联合查询（分别查询后合并结果）
- FAQ 仍使用共享集合（rag_faq）
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

    集合架构：
    - 文档检索：每个 kb_id 独立集合（rag_documents_<kb_id>）
    - FAQ 检索：共享集合（rag_faq）
    """

    def __init__(self):
        """Initialize RAGRetrieval."""
        self._rag_systems: Dict[str, RAGSystem] = {}  # kb_id -> RAGSystem 缓存
        self._faq_system = None

    def _get_rag_system(self, kb_id: str) -> RAGSystem:
        """
        获取或创建指定 kb_id 的 RAGSystem

        每个 kb_id 使用独立的 ChromaDB 集合（rag_documents_<kb_id>）

        Args:
            kb_id: 知识库 ID

        Returns:
            对应的 RAGSystem 实例
        """
        if kb_id not in self._rag_systems:
            logger.info(f"创建 RAGSystem for kb_id={kb_id}, 集合名=rag_documents_{kb_id}")
            self._rag_systems[kb_id] = RAGSystem(
                kb_id=kb_id,  # 自动生成集合名 rag_documents_<kb_id>
                enable_image_description=False,
                enable_summarization=True,
            )
        return self._rag_systems[kb_id]

    @property
    def faq_system(self):
        """FAQ 检索用 RAGSystem（共享 collection，延迟初始化）"""
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

        支持多知识库联合查询：为每个 kb_id 分别检索，然后合并 + rerank

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
            if not kb_ids:
                logger.warning("RAG search called without kb_ids, returning empty")
                return []

            # 多知识库联合查询
            if len(kb_ids) == 1:
                # 单个知识库，直接查询
                kb_id = kb_ids[0]
                rag_system = self._get_rag_system(kb_id)
                results = await rag_system.retrieve(
                    query=query,
                    top_k=top_k,
                    filters=None,  # 独立集合无需过滤
                )
            else:
                # 多个知识库，分别查询后合并
                logger.info(f"多知识库联合查询: kb_ids={kb_ids}")
                all_results = []
                for kb_id in kb_ids:
                    rag_system = self._get_rag_system(kb_id)
                    kb_results = await rag_system.retrieve(
                        query=query,
                        top_k=top_k,
                        filters=None,
                    )
                    all_results.extend(kb_results)

                # 按分数排序并限制数量
                all_results.sort(key=lambda x: x.score, reverse=True)
                results = all_results[:top_k]
                logger.info(f"多知识库查询结果: 总数={len(all_results)}, 返回={len(results)}")

            # 转换为 ai-service 格式
            return [
                {
                    "content": doc.text,
                    "score": doc.score,
                    "doc_id": doc.metadata.get("doc_id"),
                    "kb_id": doc.metadata.get("kb_id"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "content_type": doc.metadata.get("content_type", "text"),
                    "context_text": doc.metadata.get("context_text", ""),
                    "teaching_script_tts": doc.metadata.get("teaching_script_tts"),
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

    async def delete_kb_collection(self, kb_id: str) -> bool:
        """
        删除指定知识库的 ChromaDB 集合

        Args:
            kb_id: 知识库 ID

        Returns:
            是否删除成功
        """
        try:
            if kb_id in self._rag_systems:
                await self._rag_systems[kb_id].clear_collection()
                del self._rag_systems[kb_id]
                logger.info(f"已删除知识库集合: rag_documents_{kb_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"删除知识库集合失败: kb_id={kb_id}, error={e}", exc_info=True)
            return False


# Global RAG retrieval instance
rag_retrieval = RAGRetrieval()
