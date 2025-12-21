"""
Chat API endpoints.
"""

import time
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


def format_sources(
    retrieved_docs: list, web_search_results: list, max_content_length: int = 200
) -> dict:
    """
    Format RAG documents and web search results for source attribution.

    Args:
        retrieved_docs: List of retrieved document chunks from RAG
        web_search_results: List of web search results from Tavily
        max_content_length: Maximum content snippet length (default: 200 chars)

    Returns:
        Dict with rag_sources and web_sources lists
    """
    sources = {"rag_sources": [], "web_sources": []}

    # Format RAG document sources (top 3)
    for idx, doc in enumerate(retrieved_docs[:3], 1):
        content_snippet = doc.get("content", "")[:max_content_length]
        if len(doc.get("content", "")) > max_content_length:
            content_snippet += "..."

        rag_source = {
            "rank": idx,
            "doc_id": doc.get("doc_id", ""),
            "kb_id": doc.get("kb_id", ""),
            "content_snippet": content_snippet,
            "score": round(doc.get("rrf_score", doc.get("score", 0.0)), 4),
        }

        # Add chunk_index if available
        if "chunk_index" in doc:
            rag_source["chunk_index"] = doc["chunk_index"]

        sources["rag_sources"].append(rag_source)

    # Format web search sources (top 5)
    for result in web_search_results[:5]:
        web_source = {
            "rank": result.get("rank", 0),
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "score": round(result.get("score", 0.0), 4),
        }
        sources["web_sources"].append(web_source)

    return sources


@router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest, api_key: str = Depends(get_api_key)):
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
            request=None, user_id=request.user_id, session_id=request.session_id
        )

        logger.info(
            "Chat message request",
            user_id=request.user_id,
            employee_id=request.employee_id,
            query=request.query[:100],
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
            "response_time_ms": 0,
        }

        # Run workflow
        result = await conversation_workflow.run(initial_state)

        # Format source attribution
        sources = format_sources(
            retrieved_docs=result.get("retrieved_docs", []),
            web_search_results=result.get("web_search_results", []),
        )

        # Build response
        response_data = {
            "conversation_id": result["conversation_id"],
            "session_id": session_id,
            "answer": result["final_answer"],
            "intent": result.get("intent", "general_query"),
            "confidence": result.get("confidence", 0.0),
            "kb_used": result.get("kb_used", []),
            "web_search_used": result.get("web_search_used", False),
            "sources": sources,
            "timestamp": datetime.now().isoformat() + "Z",
        }

        return ChatResponse(code=200, message="success", data=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Chat message error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat message",
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

        # Generate session_id
        session_id = (
            request.session_id
            or f"sess_{hashlib.md5(f'{request.user_id}_{datetime.now().timestamp()}'.encode()).hexdigest()[:12]}"
        )

        # Send start event
        yield json.dumps({"type": "start", "session_id": session_id})

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
            "response_time_ms": 0,
        }

        # Stream workflow execution (node by node)
        node_count = 0
        async for event in conversation_workflow.workflow.astream(initial_state):
            node_name = list(event.keys())[0]
            state_update = event[node_name]
            node_count += 1

            # Send node progress
            yield json.dumps(
                {"type": "progress", "node": node_name, "step": node_count}
            )

            # If generate_answer node and has answer, stream tokens
            if node_name == "generate_answer" and state_update.get("final_answer"):
                answer = state_update["final_answer"]
                # Stream by sentences
                sentences = (
                    answer.replace("。", "。\n")
                    .replace("！", "！\n")
                    .replace("？", "？\n")
                    .split("\n")
                )
                for sentence in sentences:
                    if sentence.strip():
                        yield json.dumps({"type": "token", "content": sentence})

        # Get final state
        final_state = state_update

        # Format source attribution
        sources = format_sources(
            retrieved_docs=final_state.get("retrieved_docs", []),
            web_search_results=final_state.get("web_search_results", []),
        )

        # Send done event
        yield json.dumps(
            {
                "type": "done",
                "conversation_id": final_state.get("conversation_id", ""),
                "confidence": final_state.get("confidence", 0.0),
                "kb_used": final_state.get("kb_used", []),
                "web_search_used": final_state.get("web_search_used", False),
                "faq_matched": final_state.get("faq_matched"),
                "sources": sources,
            }
        )

    except Exception as e:
        logger.error("Stream generation error", error=str(e), exc_info=True)
        yield json.dumps({"type": "error", "message": str(e)})


@router.post("/stream")
async def chat_stream(request: ChatRequest, api_key: str = Depends(get_api_key)):
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
            request=None, user_id=request.user_id, session_id=request.session_id
        )

        logger.info(
            "Chat stream request",
            user_id=request.user_id,
            employee_id=request.employee_id,
            query=request.query[:100],
        )

        return EventSourceResponse(generate_stream_response(request))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Chat stream error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start chat stream",
        )


async def generate_openai_stream_response(
    request: OpenAIChatRequest,
) -> AsyncGenerator[str, None]:
    """
    Generate OpenAI-style streaming response.

    Args:
        request: OpenAI chat request

    Yields:
        OpenAI-formatted SSE messages
    """
    try:

        # Generate IDs
        session_id = (
            request.session_id
            or f"sess_{hashlib.md5(f'{request.user_id}_{datetime.now().timestamp()}'.encode()).hexdigest()[:12]}"
        )
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
            "response_time_ms": 0,
        }

        # Send initial role chunk
        yield json.dumps(
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
        )

        # Stream workflow execution and monitor for generate stage
        should_generate = False
        final_state = None
        full_answer = ""

        async for event in conversation_workflow.workflow.astream(
            initial_state, stream_mode="values"
        ):
            # 通过 async for event in conversation_workflow.workflow.astream(
            # initial_state, stream_mode="values")，逐步异步执行 LangGraph 的每个节点
            # （如 config/session/FAQ/RAG/web_search/grade/generate）。
            # 每到一个节点，event 就是当前节点执行后的最新 state，可以随时中断、分支或提前生成。

            # Check if we've reached generation stage
            if (
                "confidence" in event
                and event.get("confidence", 0) > 0
                and not should_generate
            ):
                should_generate = True
                final_state = event

            # When ready to generate, do REAL streaming
            if should_generate and final_state:
                should_generate = False  # Only generate once

                # Build messages for LLM
                messages = conversation_workflow.build_generation_messages(final_state)

                # TRUE token-level streaming from LLM
                async for chunk in conversation_workflow.llm.astream(messages):
                    token = chunk.content
                    if token:
                        full_answer += token
                        yield json.dumps(
                            {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": token},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )

                # Update state with generated answer
                final_state["final_answer"] = full_answer

        # If no final_state yet, use last event
        if final_state is None:
            final_state = event

        # Format source attribution
        sources = format_sources(
            retrieved_docs=final_state.get("retrieved_docs", []),
            web_search_results=final_state.get("web_search_results", []),
        )

        # Send finish chunk
        yield json.dumps(
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": len(user_query),
                    "completion_tokens": len(final_state.get("final_answer", "")),
                    "total_tokens": len(user_query)
                    + len(final_state.get("final_answer", "")),
                },
                "metadata": {
                    "conversation_id": final_state.get("conversation_id", ""),
                    "confidence": final_state.get("confidence", 0.0),
                    "kb_used": final_state.get("kb_used", []),
                    "web_search_used": final_state.get("web_search_used", False),
                    "sources": sources,
                },
            }
        )

        # Send [DONE] marker
        yield "[DONE]"

    except Exception as e:
        logger.error("OpenAI stream generation error", error=str(e), exc_info=True)
        # Send error in OpenAI format
        yield json.dumps(
            {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "internal_error",
                }
            }
        )

@router.get("/v1")
async def chat_v1_health_check():
    """
    用于API Key验证或健康检查的占位接口。
    """
    return {"status": "ok", "message": "API Key valid (mocked)"}


@router.post("/v1/chat/completions")
async def openai_chat_completions(
    request: OpenAIChatRequest, api_key: str = Depends(get_api_key)
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
            request=None, user_id=request.user_id, session_id=request.session_id
        )

        logger.info(
            "OpenAI chat completion request",
            user_id=request.user_id,
            employee_id=request.employee_id,
            stream=request.stream,
            model=request.model,
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

            session_id = (
                request.session_id
                or f"sess_{hashlib.md5(f'{request.user_id}_{datetime.now().timestamp()}'.encode()).hexdigest()[:12]}"
            )
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
                "response_time_ms": 0,
            }

            # Run workflow
            result = await conversation_workflow.run(initial_state)

            # Format source attribution
            sources = format_sources(
                retrieved_docs=result.get("retrieved_docs", []),
                web_search_results=result.get("web_search_results", []),
            )

            # Return OpenAI-formatted response
            return {
                "id": chat_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result["final_answer"],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(user_query),
                    "completion_tokens": len(result["final_answer"]),
                    "total_tokens": len(user_query) + len(result["final_answer"]),
                },
                "metadata": {
                    "conversation_id": result["conversation_id"],
                    "confidence": result.get("confidence", 0.0),
                    "kb_used": result.get("kb_used", []),
                    "web_search_used": result.get("web_search_used", False),
                    "sources": sources,
                },
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("OpenAI chat completion error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat completion",
        )
