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
    """Get conversation records - to be implemented in Phase 4."""
    return {
        "code": 501,
        "message": "To be implemented in Phase 4"
    }


@router.get("/statistics")
async def get_conversation_statistics(
    api_key: str = Depends(get_api_key)
):
    """Get conversation statistics - to be implemented in Phase 4."""
    return {
        "code": 501,
        "message": "To be implemented in Phase 4"
    }
