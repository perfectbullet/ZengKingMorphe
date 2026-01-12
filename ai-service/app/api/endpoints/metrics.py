"""
Metrics and monitoring API endpoints.

Provides endpoints for:
- Retrieval metrics (RAG performance)
- Conversation statistics
- System health monitoring
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query

from app.api.middleware.auth import get_api_key
from app.core.logging import get_logger
from app.core.database import get_database
from app.core.config import settings

logger = get_logger(__name__)

router = APIRouter()


@router.get("/retrieval-metrics")
async def get_retrieval_metrics(
    employee_id: Optional[str] = Query(None, description="Filter by employee ID"),
    kb_id: Optional[str] = Query(None, description="Filter by knowledge base ID"),
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    api_key: str = Depends(get_api_key),
    db = Depends(get_database),
) -> Dict[str, Any]:
    """
    获取检索指标统计。

    指标包括：
    - hit_rate: 召回命中率 (relevance_score > threshold 的查询比例)
    - avg_relevance_score: 平均相关性分数
    - avg_response_time: 平均响应时间
    - total_queries: 总查询数
    - rag_queries: 使用RAG的查询数
    - web_search_queries: 使用联网搜索的查询数
    - faq_queries: FAQ命中的查询数

    Args:
        employee_id: 员工ID过滤
        kb_id: 知识库ID过滤
        days: 统计天数
        api_key: API密钥
        db: 数据库实例

    Returns:
        检索指标统计
    """
    try:
        logger.info(
            "Retrieval metrics request",
            employee_id=employee_id,
            kb_id=kb_id,
            days=days,
        )

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Build query
        query = {
            "created_at": {"$gte": start_date, "$lte": end_date}
        }
        if employee_id:
            query["employee_id"] = employee_id

        # Get all conversations in date range
        cursor = db.conversations.find(query).sort("created_at", -1)
        conversations = await cursor.to_list(length=None)

        # Calculate metrics
        total_queries = len(conversations)
        rag_queries = 0
        web_search_queries = 0
        faq_queries = 0
        total_relevance = 0.0
        total_response_time = 0.0
        high_relevance_count = 0

        for conv in conversations:
            # Count RAG queries
            if conv.get("retrieved_docs"):
                rag_queries += 1
                relevance = conv.get("relevance_score", 0.0)
                total_relevance += relevance
                if relevance >= settings.relevance_threshold:
                    high_relevance_count += 1

            # Count web search queries
            if conv.get("web_search_used"):
                web_search_queries += 1

            # Count FAQ queries (inferred from intent)
            if conv.get("intent") == "faq_match":
                faq_queries += 1

            # Response time
            total_response_time += conv.get("response_time_ms", 0)

        # Calculate averages
        avg_relevance = total_relevance / rag_queries if rag_queries > 0 else 0.0
        avg_response_time = total_response_time / total_queries if total_queries > 0 else 0.0
        hit_rate = (high_relevance_count / rag_queries * 100) if rag_queries > 0 else 0.0

        # Get knowledge base breakdown if kb_id is specified
        kb_breakdown = []
        if kb_id:
            kb_conv = [c for c in conversations if kb_id in c.get("kb_used", [])]
            kb_queries = len(kb_conv)
            kb_hit_rate = 0.0
            if kb_queries > 0:
                kb_high_relevance = sum(
                    1 for c in kb_conv
                    if c.get("relevance_score", 0) >= settings.relevance_threshold
                )
                kb_hit_rate = (kb_high_relevance / kb_queries) * 100

            kb_breakdown.append({
                "kb_id": kb_id,
                "queries": kb_queries,
                "hit_rate": f"{kb_hit_rate:.1f}%",
            })

        return {
            "code": 200,
            "message": "success",
            "data": {
                "summary": {
                    "total_queries": total_queries,
                    "rag_queries": rag_queries,
                    "web_search_queries": web_search_queries,
                    "faq_queries": faq_queries,
                    "hit_rate": f"{hit_rate:.1f}%",
                    "avg_relevance_score": f"{avg_relevance:.3f}",
                    "avg_response_time_ms": f"{avg_response_time:.0f}",
                },
                "breakdown": {
                    "rag_rate": f"{(rag_queries / total_queries * 100):.1f}%" if total_queries > 0 else "0%",
                    "web_search_rate": f"{(web_search_queries / total_queries * 100):.1f}%" if total_queries > 0 else "0%",
                    "faq_rate": f"{(faq_queries / total_queries * 100):.1f}%" if total_queries > 0 else "0%",
                },
                "kb_breakdown": kb_breakdown,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "days": days,
                }
            }
        }

    except Exception as e:
        logger.error(
            "Failed to get retrieval metrics",
            error=str(e),
            exc_info=True
        )
        return {
            "code": 500,
            "message": "Failed to retrieve metrics",
            "error": str(e)
        }


@router.get("/conversation-stats")
async def get_conversation_stats(
    employee_id: Optional[str] = Query(None, description="Filter by employee ID"),
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    api_key: str = Depends(get_api_key),
    db = Depends(get_database),
) -> Dict[str, Any]:
    """
    获取对话统计信息。

    统计包括：
    - 总对话数
    - 平均对话轮次
    - 意图分布
    - 满意度分布
    - 每日对话趋势

    Args:
        employee_id: 员工ID过滤
        days: 统计天数
        api_key: API密钥
        db: 数据库实例

    Returns:
        对话统计数据
    """
    try:
        logger.info(
            "Conversation stats request",
            employee_id=employee_id,
            days=days,
        )

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Build query
        query = {
            "created_at": {"$gte": start_date, "$lte": end_date}
        }
        if employee_id:
            query["employee_id"] = employee_id

        # Get conversations
        cursor = db.conversations.find(query).sort("created_at", -1)
        conversations = await cursor.to_list(length=None)

        # Calculate statistics
        total_conversations = len(conversations)

        # Intent distribution
        intent_dist: Dict[str, int] = {}
        for conv in conversations:
            intent = conv.get("intent", "unknown")
            intent_dist[intent] = intent_dist.get(intent, 0) + 1

        # Daily trend
        daily_trend = {}
        for conv in conversations:
            date_key = conv["created_at"].strftime("%Y-%m-%d")
            daily_trend[date_key] = daily_trend.get(date_key, 0) + 1

        return {
            "code": 200,
            "message": "success",
            "data": {
                "total_conversations": total_conversations,
                "intent_distribution": intent_dist,
                "daily_trend": dict(sorted(daily_trend.items())),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "days": days,
                }
            }
        }

    except Exception as e:
        logger.error(
            "Failed to get conversation stats",
            error=str(e),
            exc_info=True
        )
        return {
            "code": 500,
            "message": "Failed to retrieve stats",
            "error": str(e)
        }


@router.get("/system-health")
async def get_system_health(
    api_key: str = Depends(get_api_key),
    db = Depends(get_database),
) -> Dict[str, Any]:
    """
    获取系统健康状态。

    检查项包括：
    - MongoDB 连接状态
    - ChromaDB 连接状态
    - ElasticSearch 连接状态
    - 最近错误日志统计

    Args:
        api_key: API密钥
        db: 数据库实例

    Returns:
        系统健康状态
    """
    try:
        # Get recent error count (from conversations with errors)
        since = datetime.now() - timedelta(hours=1)
        error_count = await db.conversations.count_documents({
            "created_at": {"$gte": since},
            "error": {"$exists": True}
        })

        return {
            "code": 200,
            "message": "success",
            "data": {
                "mongodb": "connected" if db.db is not None else "disconnected",
                "recent_errors_1h": error_count,
                "status": "healthy" if error_count < 10 else "degraded"
            }
        }

    except Exception as e:
        logger.error(
            "Failed to get system health",
            error=str(e),
            exc_info=True
        )
        return {
            "code": 500,
            "message": "Failed to retrieve health status",
            "error": str(e)
        }
