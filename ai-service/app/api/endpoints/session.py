"""
Session management API endpoints.
"""
import hashlib
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from app.models.schemas import SessionResponse, CreateSessionRequest
from app.api.middleware.auth import get_api_key
from app.core.database import get_database
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_session(
    request: CreateSessionRequest,
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    创建新会话。
    
    Args:
        - request: Session creation request
        - api_key: API key from auth
        - db: Database instance
        
    Returns:
        - Created session information
    """
    try:
        logger.info("Create session request", user_id=request.user_id, employee_id=request.employee_id)
        
        # Check if employee exists
        employee = await db.employee_configs.find_one({"employee_id": request.employee_id})
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Employee {request.employee_id} not found"
            )
        
        # Generate session ID
        timestamp = datetime.utcnow().timestamp()
        session_id = f"sess_{hashlib.md5(f'{request.user_id}_{timestamp}'.encode()).hexdigest()[:12]}"
        
        # Create session document
        session_doc = {
            "session_id": session_id,
            "user_id": request.user_id,
            "employee_id": request.employee_id,
            "status": "active",
            "message_count": 0,
            "context_messages": [],
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "ended_at": None,
            "metadata": request.metadata
        }
        
        # Insert into database
        await db.sessions.insert_one(session_doc)
        
        logger.info("Session created", session_id=session_id, user_id=request.user_id)
        
        # Format response
        session_doc.pop("_id", None)
        session_doc["created_at"] = session_doc["created_at"].isoformat() + "Z"
        session_doc["last_activity"] = session_doc["last_activity"].isoformat() + "Z"
        
        return {
            "code": 201,
            "message": "Session created successfully",
            "data": session_doc
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create session", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session"
        )


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
