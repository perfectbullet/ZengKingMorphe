"""
Conversation record API endpoints.
"""
from fastapi import APIRouter, Depends
from app.api.middleware.auth import get_api_key
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/records")
async def get_conversation_records(
    api_key: str = Depends(get_api_key)
):
    """获取对话记录 - Phase 4 实现。"""
    return {
        "code": 501,
        "message": "To be implemented in Phase 4"
    }


@router.get("/statistics")
async def get_conversation_statistics(
    api_key: str = Depends(get_api_key)
):
    """获取对话统计数据 - Phase 4 实现。"""
    return {
        "code": 501,
        "message": "To be implemented in Phase 4"
    }
