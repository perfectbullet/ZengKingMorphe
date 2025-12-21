"""
Conversation record API endpoints.
"""
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, Query
from app.api.middleware.auth import get_api_key
from app.core.logging import get_logger
from app.core.database import mongodb

logger = get_logger(__name__)

router = APIRouter()


@router.get("/records")
async def get_conversation_records(
    api_key: str = Depends(get_api_key),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    user_id: Optional[str] = Query(None, description="User ID"),
    employee_id: Optional[str] = Query(None, description="Employee ID"),
    session_id: Optional[str] = Query(None, description="Session ID"),
    keyword: Optional[str] = Query(None, description="Search keyword"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size")
):
    """获取对话记录。
    
    支持按日期范围、用户、员工、会话、关键词筛选，支持分页查询。
    
    Args:
        api_key: API key for authentication
        start_date: Start date (format: YYYY-MM-DD)
        end_date: End date (format: YYYY-MM-DD)
        user_id: User ID filter
        employee_id: Employee ID filter
        session_id: Session ID filter
        keyword: Search keyword (searches in user_query and ai_response)
        page: Page number (starting from 1)
        page_size: Records per page (1-100)

    Returns:
        dict: Response containing conversation records and pagination info
    """
    try:
        # 构建查询条件
        query_filter: Dict[str, Any] = {}
        
        # 日期范围筛选
        if start_date or end_date:
            date_filter = {}
            if start_date:
                try:
                    start_dt = datetime.fromisoformat(start_date)
                    date_filter["$gte"] = start_dt
                except ValueError:
                    return {
                        "code": 400,
                        "message": "Invalid start_date format. Use YYYY-MM-DD"
                    }
            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date)
                    # 结束日期加上一天，以包含当天所有记录
                    from datetime import timedelta
                    end_dt = end_dt + timedelta(days=1)
                    date_filter["$lt"] = end_dt
                except ValueError:
                    return {
                        "code": 400,
                        "message": "Invalid end_date format. Use YYYY-MM-DD"
                    }
            query_filter["created_at"] = date_filter
        
        # 用户ID筛选
        if user_id:
            query_filter["user_id"] = user_id
        
        # 员工ID筛选
        if employee_id:
            query_filter["employee_id"] = employee_id
        
        # 会话ID筛选
        if session_id:
            query_filter["session_id"] = session_id
        
        # 关键词搜索（在用户问题和AI回复中搜索）
        if keyword:
            query_filter["$or"] = [
                {"user_query": {"$regex": keyword, "$options": "i"}},
                {"ai_response": {"$regex": keyword, "$options": "i"}}
            ]
        
        # 计算总数
        total = await mongodb.db.conversations.count_documents(query_filter)
        
        # 计算分页参数
        skip = (page - 1) * page_size
        total_pages = (total + page_size - 1) // page_size
        
        # 查询数据
        cursor = mongodb.db.conversations.find(query_filter).sort(
            "created_at", -1
        ).skip(skip).limit(page_size)
        
        records = await cursor.to_list(length=page_size)
        
        # 格式化数据（移除MongoDB的_id字段，格式化时间）
        formatted_records = []
        for record in records:
            record.pop("_id", None)
            if "created_at" in record and isinstance(record["created_at"], datetime):
                record["created_at"] = record["created_at"].isoformat()
            if "updated_at" in record and isinstance(record["updated_at"], datetime):
                record["updated_at"] = record["updated_at"].isoformat()
            formatted_records.append(record)
        
        logger.info(
            "Retrieved conversation records",
            total=total,
            page=page,
            page_size=page_size,
            filters=query_filter
        )
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "records": formatted_records,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "total_pages": total_pages
                }
            }
        }
        
    except Exception as e:
        logger.error("Failed to retrieve conversation records", error=str(e), exc_info=True)
        return {
            "code": 500,
            "message": f"Failed to retrieve records: {str(e)}"
        }


@router.get("/statistics")
async def get_conversation_statistics(
    api_key: str = Depends(get_api_key),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    dimension: Optional[str] = Query(None, description="Statistics dimension: time/user/employee/intent/kb"),
    employee_id: Optional[str] = Query(None, description="Employee ID filter")
):
    """获取对话统计数据。
    
    支持多维度统计分析：时间维度、用户维度、员工维度、意图维度、知识库维度。
    
    Args:
        api_key: API key for authentication
        start_date: Start date (format: YYYY-MM-DD)
        end_date: End date (format: YYYY-MM-DD)
        dimension: Statistics dimension (time/user/employee/intent/kb)
        employee_id: Employee ID filter (optional)

    Returns:
        dict: Response containing statistical data
    """
    try:
        # 解析日期范围
        try:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            from datetime import timedelta
            end_dt = end_dt + timedelta(days=1)  # 包含结束日期当天
        except ValueError:
            return {
                "code": 400,
                "message": "Invalid date format. Use YYYY-MM-DD"
            }
        
        # 构建基础查询条件
        base_filter = {
            "created_at": {
                "$gte": start_dt,
                "$lt": end_dt
            }
        }
        
        if employee_id:
            base_filter["employee_id"] = employee_id
        
        # 基础统计数据
        total_conversations = await mongodb.db.conversations.count_documents(base_filter)
        
        # 计算平均置信度和响应时间
        pipeline = [
            {"$match": base_filter},
            {"$group": {
                "_id": None,
                "avg_confidence": {"$avg": "$confidence"},
                "avg_response_time": {"$avg": "$response_time_ms"},
                "web_search_count": {"$sum": {"$cond": ["$web_search_used", 1, 0]}},
                "realtime_query_count": {"$sum": {"$cond": ["$is_realtime_query", 1, 0]}}
            }}
        ]
        
        agg_result = await mongodb.db.conversations.aggregate(pipeline).to_list(1)
        basic_stats = agg_result[0] if agg_result else {}
        
        statistics = {
            "summary": {
                "total_conversations": total_conversations,
                "avg_confidence": round(basic_stats.get("avg_confidence", 0.0), 2),
                "avg_response_time_ms": int(basic_stats.get("avg_response_time", 0)),
                "web_search_usage": basic_stats.get("web_search_count", 0),
                "realtime_query_count": basic_stats.get("realtime_query_count", 0),
                "date_range": {
                    "start_date": start_date,
                    "end_date": end_date
                }
            }
        }
        
        # 按维度统计
        if dimension == "time":
            # 按日期分组统计
            time_pipeline = [
                {"$match": base_filter},
                {"$project": {
                    "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                    "confidence": 1,
                    "response_time_ms": 1
                }},
                {"$group": {
                    "_id": "$date",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence"},
                    "avg_response_time": {"$avg": "$response_time_ms"}
                }},
                {"$sort": {"_id": 1}}
            ]
            time_stats = await mongodb.db.conversations.aggregate(time_pipeline).to_list(None)
            statistics["time_distribution"] = [
                {
                    "date": item["_id"],
                    "count": item["count"],
                    "avg_confidence": round(item["avg_confidence"], 2),
                    "avg_response_time_ms": int(item["avg_response_time"])
                }
                for item in time_stats
            ]
        
        elif dimension == "user":
            # 按用户分组统计
            user_pipeline = [
                {"$match": base_filter},
                {"$group": {
                    "_id": "$user_id",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence"}
                }},
                {"$sort": {"count": -1}},
                {"$limit": 50}  # 限制返回前50名用户
            ]
            user_stats = await mongodb.db.conversations.aggregate(user_pipeline).to_list(None)
            statistics["user_distribution"] = [
                {
                    "user_id": item["_id"],
                    "conversation_count": item["count"],
                    "avg_confidence": round(item["avg_confidence"], 2)
                }
                for item in user_stats
            ]
        
        elif dimension == "employee":
            # 按员工分组统计
            employee_pipeline = [
                {"$match": base_filter},
                {"$group": {
                    "_id": "$employee_id",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence"},
                    "employee_name": {"$first": "$employee_name"}
                }},
                {"$sort": {"count": -1}}
            ]
            employee_stats = await mongodb.db.conversations.aggregate(employee_pipeline).to_list(None)
            statistics["employee_distribution"] = [
                {
                    "employee_id": item["_id"],
                    "employee_name": item.get("employee_name", ""),
                    "conversation_count": item["count"],
                    "avg_confidence": round(item["avg_confidence"], 2)
                }
                for item in employee_stats
            ]
        
        elif dimension == "intent":
            # 按意图分组统计
            intent_pipeline = [
                {"$match": base_filter},
                {"$group": {
                    "_id": "$intent",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence"}
                }},
                {"$sort": {"count": -1}}
            ]
            intent_stats = await mongodb.db.conversations.aggregate(intent_pipeline).to_list(None)
            statistics["intent_distribution"] = [
                {
                    "intent": item["_id"] or "unknown",
                    "count": item["count"],
                    "avg_confidence": round(item["avg_confidence"], 2)
                }
                for item in intent_stats
            ]
        
        elif dimension == "kb":
            # 按知识库分组统计
            kb_pipeline = [
                {"$match": base_filter},
                {"$unwind": {"path": "$kb_used", "preserveNullAndEmptyArrays": True}},
                {"$group": {
                    "_id": "$kb_used",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}}
            ]
            kb_stats = await mongodb.db.conversations.aggregate(kb_pipeline).to_list(None)
            statistics["kb_distribution"] = [
                {
                    "kb_id": item["_id"] or "none",
                    "usage_count": item["count"]
                }
                for item in kb_stats
            ]
        
        logger.info(
            "Retrieved conversation statistics",
            dimension=dimension,
            total_conversations=total_conversations,
            date_range=f"{start_date} to {end_date}"
        )
        
        return {
            "code": 200,
            "message": "success",
            "data": statistics
        }
        
    except Exception as e:
        logger.error("Failed to retrieve conversation statistics", error=str(e), exc_info=True)
        return {
            "code": 500,
            "message": f"Failed to retrieve statistics: {str(e)}"
        }
