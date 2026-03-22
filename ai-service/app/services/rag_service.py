"""
RAG retrieval service using llama-rag-sdk.

使用 llama-rag-sdk 进行向量检索，保留 ES 用于 FAQ 搜索。
"""
from typing import List, Dict, Any, Optional
import sys

# 添加 llama-rag-sdk 到路径
sys.path.insert(0, '/home/zj/ZengKingMorphe/llama-rag-sdk')

from app.core.logging import get_logger
from app.core.chroma import chroma_db
from app.core.elasticsearch import es_db
from app.services.sdk_adapter.config import setup_sdk_env

# 设置 SDK 环境变量
setup_sdk_env()

logger = get_logger(__name__)


class RAGRetrieval:
    """
    RAG 检索服务（使用 llama-rag-sdk）

    主要检索使用 SDK 向量搜索，FAQ 搜索保留原有的 ES 实现
    """

    def __init__(self):
        """Initialize RAGRetrieval."""
        self._rag_system = None

    @property
    def rag_system(self):
        """延迟初始化 llama-rag-sdk RAGSystem"""
        if self._rag_system is None:
            from src.rag_system import RAGSystem
            self._rag_system = RAGSystem(
                collection_name="rag_documents",
                enable_image_description=False,
                enable_summarization=True,
            )
        return self._rag_system

    async def search(
        self,
        query: str,
        kb_ids: Optional[List[str]] = None,
        top_k: int = 5,
        use_hybrid: bool = True,  # 忽略，SDK 只用向量搜索
        enable_rerank: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        使用 llama-rag-sdk 进行向量检索

        Args:
            query: 搜索查询
            kb_ids: 知识库 ID 列表
            top_k: 返回结果数量
            use_hybrid: 忽略，SDK 只使用向量搜索
            enable_rerank: 忽略，SDK 内部使用 Reranker

        Returns:
            相关文档列表（兼容旧格式）
        """
        try:
            # 构建过滤条件
            filters = None
            if kb_ids:
                filters = {"kb_id": {"$in": kb_ids}}

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
                    "content_type": doc.metadata.get("content_type", "unknown"),
                    "context_text": doc.metadata.get("context_text", ""),
                    "source": "sdk_vector",
                    # 保留其他元数据
                    **{k: v for k, v in doc.metadata.items()
                       if k not in ["doc_id", "kb_id", "chunk_index", "content_type", "context_text"]}
                }
                for doc in results
            ]

        except Exception as e:
            logger.error(f"SDK RAG search failed: query={query}, error={str(e)}", exc_info=True)
            raise

    # ========== FAQ 搜索（保留原有 ES 实现）==========

    async def faq_hybrid_search(
        self,
        query: str,
        employee_id: str,
        faq_sim_threshold: float = 0.0,
        faq_top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        FAQ 混合搜索（向量+关键词双路召回+RRF融合）

        保留原有的 ES 实现，不使用 SDK
        """
        try:
            logger.info(f"FAQ hybrid search: query={query[:100]}, employee_id={employee_id}, threshold={faq_sim_threshold}, top_k={faq_top_k}")

            # Step 1: Vector search in ChromaDB
            vector_results = await self._faq_vector_search(query, employee_id, faq_top_k * 2)

            # Step 2: Keyword search in ElasticSearch
            keyword_results = await self._faq_keyword_search(query, employee_id, faq_top_k * 2)

            # Step 3: RRF fusion
            fused_results = self._faq_rrf_fusion(vector_results, keyword_results, k=60)

            # Step 4: Filter by similarity threshold
            filtered_results = [
                result for result in fused_results
                if (
                    (result.get("rrf_score", 0.0) >= faq_sim_threshold) or
                    (result.get("vector_score", 0.0) >= faq_sim_threshold) or
                    (
                        result.get("keyword_score", 0.0) >= faq_sim_threshold and
                        result.get("vector_score", 0.0) >= faq_sim_threshold * 0.7
                    )
                )
            ]

            logger.info(f"FAQ hybrid search completed: total_fused={len(fused_results)}, above_threshold={len(filtered_results)}, returning_top_k={min(faq_top_k, len(filtered_results))}")

            return filtered_results[:faq_top_k]

        except Exception as e:
            logger.error(f"FAQ hybrid search failed: query={query}, error={str(e)}", exc_info=True)
            return []

    async def _faq_vector_search(
        self,
        query: str,
        employee_id: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """FAQ 向量搜索（ChromaDB）"""
        try:
            where_filter = {"employee_id": employee_id}

            results = await chroma_db.query_documents(
                collection_name="faq",
                query_texts=[query],
                n_results=top_k,
                where=where_filter
            )

            faq_results = []
            if results and results.get("documents") and len(results["documents"]) > 0:
                for i, doc_text in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i] if results.get("distances") else 2.0
                    similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))

                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}

                    faq_results.append({
                        "faq_id": metadata.get("faq_id"),
                        "question_name": metadata.get("question_name"),
                        "combined_text": doc_text,
                        "score": similarity,
                        "source": "vector"
                    })

            return faq_results

        except Exception as e:
            logger.error(f"FAQ vector search failed: query={query}, error={str(e)}", exc_info=True)
            return []

    async def _faq_keyword_search(
        self,
        query: str,
        employee_id: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """FAQ 关键词搜索（ElasticSearch）"""
        try:
            es_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["question_name^3", "similar_questions^2", "combined_text"],
                                    "type": "best_fields"
                                }
                            }
                        ],
                        "filter": [
                            {"term": {"employee_id": employee_id}},
                            {"term": {"is_enable": 1}}
                        ]
                    }
                }
            }

            results = await es_db.search(
                index="faq",
                query=es_query,
                size=top_k
            )

            faq_results = []
            if results and results.get("hits"):
                for hit in results["hits"]["hits"]:
                    source = hit["_source"]
                    score = hit["_score"]
                    normalized_score = min(1.0, score / 10.0)

                    faq_results.append({
                        "faq_id": source.get("faq_id"),
                        "question_name": source.get("question_name"),
                        "combined_text": source.get("combined_text"),
                        "score": normalized_score,
                        "source": "keyword"
                    })

            return faq_results

        except Exception as e:
            logger.error(f"FAQ keyword search failed: query={query}, error={str(e)}", exc_info=True)
            return []

    def _create_faq_fusion_entry(
        self,
        result: Dict[str, Any],
        score: float,
        rank: int,
        source_type: str
    ) -> Dict[str, Any]:
        """Create a new entry for FAQ RRF fusion."""
        entry = {
            "faq_id": result["faq_id"],
            "question_name": result["question_name"],
            "combined_text": result["combined_text"],
            "rrf_score": 0.0,
            "vector_rank": None,
            "keyword_rank": None
        }
        if source_type == "vector":
            entry["vector_score"] = score
            entry["keyword_score"] = 0.0
            entry["vector_rank"] = rank
        else:
            entry["vector_score"] = 0.0
            entry["keyword_score"] = score
            entry["keyword_rank"] = rank
        return entry

    def _faq_rrf_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """FAQ 结果的 RRF 融合"""
        faq_scores: Dict[str, Dict[str, Any]] = {}

        for rank, result in enumerate(vector_results, start=1):
            faq_id = result["faq_id"]
            if faq_id not in faq_scores:
                faq_scores[faq_id] = self._create_faq_fusion_entry(result, result["score"], rank, "vector")
            faq_scores[faq_id]["rrf_score"] += 1.0 / (k + rank)

        for rank, result in enumerate(keyword_results, start=1):
            faq_id = result["faq_id"]
            if faq_id not in faq_scores:
                faq_scores[faq_id] = self._create_faq_fusion_entry(result, result["score"], rank, "keyword")
            else:
                faq_scores[faq_id]["keyword_score"] = result["score"]
                faq_scores[faq_id]["keyword_rank"] = rank
            faq_scores[faq_id]["rrf_score"] += 1.0 / (k + rank)

        sorted_faqs = sorted(
            faq_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        return sorted_faqs


# Global RAG retrieval instance
rag_retrieval = RAGRetrieval()
