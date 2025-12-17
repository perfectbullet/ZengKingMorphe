"""
Chat API endpoints.
"""
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse
import json
from datetime import datetime
import hashlib

from app.models.schemas import ChatRequest, ChatResponse
from app.api.middleware.auth import get_api_key
from app.api.middleware.rate_limit import rate_limit_middleware
from app.core.logging import get_logger
from app.services.conversation_service import conversation_workflow, ConversationState

logger = get_logger(__name__)

router = APIRouter()


@router.post("/message", response_model=ChatResponse)
async def chat_message(
    request: ChatRequest,
    api_key: str = Depends(get_api_key)
):
    """
    同步对话接口。
    
    Args:
        request: Chat request
        api_key: API key from auth
        
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
        
        # Generate session ID if not provided
        session_id = request.session_id
        if not session_id:
            session_id = f"sess_{hashlib.md5(f'{request.user_id}_{datetime.utcnow().timestamp()}'.encode()).hexdigest()[:12]}"
        
        # Build initial state
        initial_state: ConversationState = {
            "messages": [],
            "user_query": request.query,
            "user_id": request.user_id,
            "session_id": session_id,
            "employee_id": request.employee_id,
            "employee_config": {},
            "is_realtime_query": False,
            "realtime_category": "",
            "realtime_detect_reason": "",
            "intent": "",
            "entities": {},
            "retrieved_docs": [],
            "relevance_score": 0.0,
            "web_search_results": [],
            "final_answer": "",
            "confidence": 0.0,
            "context": request.context or {},
            "has_sensitive": False,
            "error": None,
            "kb_used": [],
            "web_search_used": False,
            "conversation_id": "",
            "response_time_ms": 0
        }
        
        # Run workflow
        result = await conversation_workflow.run(initial_state)
        
        # Build response
        response_data = {
            "conversation_id": result["conversation_id"],
            "session_id": session_id,
            "answer": result["final_answer"],
            "intent": result.get("intent", "general_query"),
            "confidence": result.get("confidence", 0.0),
            "kb_used": result.get("kb_used", []),
            "web_search_used": result.get("web_search_used", False),
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
    api_key: str = Depends(get_api_key)
):
    """
    流式对话接口 (SSE)。
    
    Args:
        request: Chat request
        api_key: API key from auth
        
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
