"""
Session management API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from app.models.schemas import SessionResponse
from app.api.middleware.auth import get_api_key
from app.core.database import get_database
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str = Path(..., description="Session ID"),
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    获取会话信息。
    
    \nArgs:
        \n- session_id: Session ID
        \n- api_key: API key from auth
        \n- db: Database instance
        
    \nReturns:
        \n- Session information
    """
    try:
        logger.info("Get session request", session_id=session_id)
        
        # Get session from database
        session = await db.sessions.find_one({"session_id": session_id})
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Convert MongoDB document to dict
        session.pop("_id", None)
        session["created_at"] = session["created_at"].isoformat() + "Z"
        session["last_activity"] = session["last_activity"].isoformat() + "Z"
        if session.get("ended_at"):
            session["ended_at"] = session["ended_at"].isoformat() + "Z"
        
        return SessionResponse(
            code=200,
            message="success",
            data=session
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get session error", session_id=session_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session"
        )


@router.delete("/{session_id}")
async def end_session(
    session_id: str = Path(..., description="Session ID"),
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    结束会话。
    
    \nArgs:
        \n- session_id: Session ID
        \n- api_key: API key from auth
        \n- db: Database instance
        
    \nReturns:
        \n- Success message
    """
    try:
        logger.info("End session request", session_id=session_id)
        
        # Update session status
        from datetime import datetime
        result = await db.sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "status": "ended",
                    "ended_at": datetime.utcnow()
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        return {
            "code": 200,
            "message": "Session ended successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("End session error", session_id=session_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to end session"
        )


@router.get("/{session_id}/conversations")
async def get_session_conversations(
    session_id: str = Path(..., description="Session ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Page size"),
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    获取会话下的所有对话记录。
    
    按时间顺序返回该会话中的所有问答记录，支持分页查询。
    
    Args:
        - session_id: Session ID
        - page: Page number (starting from 1)
        - page_size: Records per page (1-200)
        - api_key: API key from auth
        - db: Database instance
        
    Returns:
        - Conversation records for the session with pagination info
    """
    try:
        logger.info("Get session conversations request", session_id=session_id, page=page, page_size=page_size)
        
        # First check if session exists
        session = await db.sessions.find_one({"session_id": session_id})
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Build query filter
        query_filter = {"session_id": session_id}
        
        # Get total count
        total = await db.conversations.count_documents(query_filter)
        
        # Calculate pagination
        skip = (page - 1) * page_size
        total_pages = (total + page_size - 1) // page_size
        
        # Query conversations sorted by created_at (ascending order to show conversation flow)
        cursor = db.conversations.find(query_filter).sort(
            "created_at", 1  # 1 for ascending (oldest first)
        ).skip(skip).limit(page_size)
        
        conversations = await cursor.to_list(length=page_size)
        
        # Format data (remove MongoDB _id field, format datetime)
        formatted_conversations = []
        for conv in conversations:
            conv.pop("_id", None)
            if "created_at" in conv and hasattr(conv["created_at"], "isoformat"):
                conv["created_at"] = conv["created_at"].isoformat()
            if "updated_at" in conv and conv.get("updated_at") and hasattr(conv["updated_at"], "isoformat"):
                conv["updated_at"] = conv["updated_at"].isoformat()
            formatted_conversations.append(conv)
        
        logger.info(
            "Retrieved session conversations",
            session_id=session_id,
            total=total,
            page=page,
            returned=len(formatted_conversations)
        )
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "session_id": session_id,
                "conversations": formatted_conversations,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "total_pages": total_pages
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get session conversations error", session_id=session_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session conversations"
        )
