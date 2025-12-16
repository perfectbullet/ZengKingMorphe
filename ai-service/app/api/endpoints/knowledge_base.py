"""
Knowledge base management API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from app.api.middleware.auth import get_current_user_optional
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/create")
async def create_knowledge_base(
    current_user: dict = Depends(get_current_user_optional)
):
    """Create knowledge base - to be implemented in Phase 2."""
    return {
        "code": 501,
        "message": "To be implemented in Phase 2"
    }


@router.get("/list")
async def list_knowledge_bases(
    current_user: dict = Depends(get_current_user_optional)
):
    """List knowledge bases - to be implemented in Phase 2."""
    return {
        "code": 501,
        "message": "To be implemented in Phase 2"
    }
