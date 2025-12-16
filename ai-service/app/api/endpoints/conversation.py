"""
Conversation record API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from app.api.middleware.auth import get_current_user_optional
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/records")
async def get_conversation_records(
    current_user: dict = Depends(get_current_user_optional)
):
    """Get conversation records - to be implemented in Phase 4."""
    return {
        "code": 501,
        "message": "To be implemented in Phase 4"
    }


@router.get("/statistics")
async def get_conversation_statistics(
    current_user: dict = Depends(get_current_user_optional)
):
    """Get conversation statistics - to be implemented in Phase 4."""
    return {
        "code": 501,
        "message": "To be implemented in Phase 4"
    }
