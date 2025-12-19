"""
Chat API endpoints.
"""
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse
import json
from datetime import datetime
import hashlib

from app.models.schemas import ChatRequest, ChatResponse, OpenAIChatRequest
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
            - request: Chat request
            - api_key: API key from auth
        
        Returns:
            - Chat response
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
            session_id = f"sess_{hashlib.md5(f'{request.user_id}_{datetime.now().timestamp()}'.encode()).hexdigest()[:12]}"
        
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
            "web_search_used": True,
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
            "timestamp": datetime.now().isoformat() + "Z"
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
    Generate streaming response integrated with LangGraph workflow.
    
    Args:
        request: Chat request
        
    Yields:
        SSE formatted messages
    """
    try:
        import hashlib
        
        # Generate session_id
        session_id = request.session_id or f"sess_{hashlib.md5(f'{request.user_id}_{datetime.now().timestamp()}'.encode()).hexdigest()[:12]}"
        
        # Send start event
        yield json.dumps({'type': 'start', 'session_id': session_id})
        
        # Build initial state
        initial_state = {
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
            "faq_matched": None,
            "kb_used": [],
            "web_search_used": False,
            "conversation_id": "",
            "response_time_ms": 0
        }
        
        # Stream workflow execution (node by node)
        node_count = 0
        async for event in conversation_workflow.workflow.astream(initial_state):
            node_name = list(event.keys())[0]
            state_update = event[node_name]
            node_count += 1
            
            # Send node progress
            yield json.dumps({'type': 'progress', 'node': node_name, 'step': node_count})
            
            # If generate_answer node and has answer, stream tokens
            if node_name == "generate_answer" and state_update.get("final_answer"):
                answer = state_update["final_answer"]
                # Stream by sentences
                sentences = answer.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')
                for sentence in sentences:
                    if sentence.strip():
                        yield json.dumps({'type': 'token', 'content': sentence})
        
        # Get final state
        final_state = state_update
        
        # Send done event
        yield json.dumps({
            'type': 'done',
            'conversation_id': final_state.get('conversation_id', ''),
            'confidence': final_state.get('confidence', 0.0),
            'kb_used': final_state.get('kb_used', []),
            'web_search_used': final_state.get('web_search_used', False),
            'faq_matched': final_state.get('faq_matched')
        })
        
    except Exception as e:
        logger.error("Stream generation error", error=str(e), exc_info=True)
        yield json.dumps({'type': 'error', 'message': str(e)})


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


async def generate_openai_stream_response(request: OpenAIChatRequest) -> AsyncGenerator[str, None]:
    """
    Generate OpenAI-style streaming response.
    
    Args:
        request: OpenAI chat request
        
    Yields:
        OpenAI-formatted SSE messages
    """
    try:
        import time
        
        # Generate IDs
        session_id = request.session_id or f"sess_{hashlib.md5(f'{request.user_id}_{datetime.now().timestamp()}'.encode()).hexdigest()[:12]}"
        chat_id = f"chatcmpl-{hashlib.md5(f'{session_id}_{time.time()}'.encode()).hexdigest()[:12]}"
        created = int(time.time())
        
        # Extract user query from messages
        user_query = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_query = msg.content
                break
        
        if not user_query:
            user_query = request.messages[-1].content if request.messages else ""
        
        # Build initial state
        initial_state = {
            "messages": [],
            "user_query": user_query,
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
            "context": {},
            "has_sensitive": False,
            "error": None,
            "faq_matched": None,
            "kb_used": [],
            "web_search_used": False,
            "conversation_id": "",
            "response_time_ms": 0
        }
        
        # Send initial role chunk
        yield json.dumps({
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None
            }]
        })
        
        # Stream workflow execution
        content_sent = False
        async for event in conversation_workflow.workflow.astream(initial_state):
            node_name = list(event.keys())[0]
            state_update = event[node_name]
            
            # Stream answer tokens when available
            if node_name == "generate_answer" and state_update.get("final_answer"):
                answer = state_update["final_answer"]
                
                # Stream by characters or small chunks for smoother output
                chunk_size = 10  # Characters per chunk
                for i in range(0, len(answer), chunk_size):
                    chunk = answer[i:i+chunk_size]
                    yield json.dumps({
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }]
                    })
                    content_sent = True
        
        # Get final state
        final_state = state_update
        
        # Send finish chunk
        yield json.dumps({
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(user_query),
                "completion_tokens": len(final_state.get("final_answer", "")),
                "total_tokens": len(user_query) + len(final_state.get("final_answer", ""))
            },
            "metadata": {
                "conversation_id": final_state.get("conversation_id", ""),
                "confidence": final_state.get("confidence", 0.0),
                "kb_used": final_state.get("kb_used", []),
                "web_search_used": final_state.get("web_search_used", False)
            }
        })
        
        # Send [DONE] marker
        yield "[DONE]"
        
    except Exception as e:
        logger.error("OpenAI stream generation error", error=str(e), exc_info=True)
        # Send error in OpenAI format
        yield json.dumps({
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": "internal_error"
            }
        })


@router.post("/openai/chat/completions")
async def openai_chat_completions(
    request: OpenAIChatRequest,
    api_key: str = Depends(get_api_key)
):
    """
    OpenAI-compatible chat completions endpoint.
    
    Supports both streaming and non-streaming modes.
    Compatible with OpenAI SDK and API format.
    
    Args:
        request: OpenAI-style chat request
        api_key: API key from auth
        
    Returns:
        OpenAI-formatted response or SSE stream
    """
    try:
        # Rate limiting
        await rate_limit_middleware(
            request=None,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        logger.info(
            "OpenAI chat completion request",
            user_id=request.user_id,
            employee_id=request.employee_id,
            stream=request.stream,
            model=request.model
        )
        
        if request.stream:
            # Return streaming response
            return EventSourceResponse(generate_openai_stream_response(request))
        else:
            # Non-streaming response
            import time
            
            # Extract user query
            user_query = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    user_query = msg.content
                    break
            
            session_id = request.session_id or f"sess_{hashlib.md5(f'{request.user_id}_{datetime.now().timestamp()}'.encode()).hexdigest()[:12]}"
            chat_id = f"chatcmpl-{hashlib.md5(f'{session_id}_{time.time()}'.encode()).hexdigest()[:12]}"
            created = int(time.time())
            
            # Build initial state
            initial_state: ConversationState = {
                "messages": [],
                "user_query": user_query,
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
                "context": {},
                "has_sensitive": False,
                "error": None,
                "faq_matched": None,
                "kb_used": [],
                "web_search_used": False,
                "conversation_id": "",
                "response_time_ms": 0
            }
            
            # Run workflow
            result = await conversation_workflow.run(initial_state)
            
            # Return OpenAI-formatted response
            return {
                "id": chat_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["final_answer"]
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_query),
                    "completion_tokens": len(result["final_answer"]),
                    "total_tokens": len(user_query) + len(result["final_answer"])
                },
                "metadata": {
                    "conversation_id": result["conversation_id"],
                    "confidence": result.get("confidence", 0.0),
                    "kb_used": result.get("kb_used", []),
                    "web_search_used": result.get("web_search_used", False)
                }
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("OpenAI chat completion error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat completion"
        )
