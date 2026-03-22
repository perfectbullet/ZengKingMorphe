"""
Mathematics Textbook Retrieval Service.

数学教材检索服务（使用标准 RAG 检索）
"""
from typing import List, Dict, Any, Optional
from app.core.logging import get_logger
from app.services.rag_service import rag_retrieval

logger = get_logger(__name__)

# Math textbook knowledge base ID
MATH_KB_ID = "kb_9abcbe4aa557"


async def search_math_textbook(
    query: str,
    top_k: int = 5,
    enable_rerank: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """
    数学教材检索（使用标准 RAG 检索）

    Args:
        query: 搜索查询
        top_k: 返回结果数量
        enable_rerank: 是否启用重排序

    Returns:
        检索结果列表
    """
    try:
        logger.info(f"Math textbook search: query={query[:100]}, top_k={top_k}")

        # 使用标准 RAG 检索
        results = await rag_retrieval.search(
            query=query,
            kb_ids=[MATH_KB_ID],
            top_k=top_k,
            use_hybrid=True,
            enable_rerank=enable_rerank,
        )

        # 添加数学教材特定的元数据字段（如果有）
        for result in results:
            result.setdefault("book_title", "")
            result.setdefault("chapter_title", "")
            result.setdefault("section_title", "")

        logger.info(
            f"Math textbook search completed: query={query[:100]}, results_count={len(results)}"
        )

        return results

    except Exception as e:
        logger.error(f"Math textbook search failed: query={query}, error={str(e)}", exc_info=True)
        return []


# 全局实例（向后兼容）
math_textbook_retrieval = None  # 不再使用单独的实例，直接用函数
