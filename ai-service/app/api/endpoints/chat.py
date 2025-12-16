"""
Chat API endpoints.
"""
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse
import json
from datetime import datetime

from app.models.schemas import ChatRequest, ChatResponse
from app.api.middleware.auth import get_current_user_optional
from app.api.middleware.rate_limit import rate_limit_middleware
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/message", response_model=ChatResponse)
async def chat_message(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user_optional)
):
    """
    Synchronous chat endpoint.
    
    Args:
        request: Chat request
        current_user: Current user from auth
        
    Returns:
        Chat response
    """
    try:
        # Rate limiting
        await rate_limit_middleware(
            request=None,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        logger.info(
            "Chat message request",
            user_id=request.user_id,
            employee_id=request.employee_id,
            query=request.query[:100]
        )
        
        # TODO: Implement chat logic with LangGraph workflow
        # This is a placeholder response
        response_data = {
            "conversation_id": f"conv_{datetime.utcnow().timestamp()}",
            "session_id": request.session_id or f"sess_{datetime.utcnow().timestamp()}",
            "answer": "This is a placeholder response. The full implementation will be completed in Phase 3.",
            "intent": "general_query",
            "confidence": 0.85,
            "kb_used": [],
            "web_search_used": False,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        return ChatResponse(
            code=200,
            message="success",
            data=response_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Chat message error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat message"
        )


async def generate_stream_response(request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Generate streaming response.
    
    Args:
        request: Chat request
        
    Yields:
        SSE formatted messages
    """
    try:
        # Start event
        yield f"data: {json.dumps({'type': 'start', 'session_id': request.session_id})}\n\n"
        
        # TODO: Implement streaming chat logic with LangGraph workflow
        # This is a placeholder
        placeholder_text = "This is a placeholder streaming response. The full implementation will be completed in Phase 3."
        
        for char in placeholder_text.split():
            yield f"data: {json.dumps({'type': 'token', 'content': char + ' '})}\n\n"
        
        # Done event
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': f'conv_{datetime.utcnow().timestamp()}'})}\n\n"
        
    except Exception as e:
        logger.error("Stream generation error", error=str(e), exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': 'Stream generation failed'})}\n\n"


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user_optional)
):
    """
    Streaming chat endpoint (SSE).
    
    Args:
        request: Chat request
        current_user: Current user from auth
        
    Returns:
        SSE stream
    """
    try:
        # Rate limiting
        await rate_limit_middleware(
            request=None,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        logger.info(
            "Chat stream request",
            user_id=request.user_id,
            employee_id=request.employee_id,
            query=request.query[:100]
        )
        
        return EventSourceResponse(generate_stream_response(request))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Chat stream error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start chat stream"
        )
