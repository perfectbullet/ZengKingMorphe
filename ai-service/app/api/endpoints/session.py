"""
Session management API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Path
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
