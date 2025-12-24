"""
Chat API endpoints.
"""

import time
from typing import AsyncGenerator, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sse_starlette.sse import EventSourceResponse
import json
from datetime import datetime
import hashlib

from app.models.schemas import ChatRequest, ChatResponse, OpenAIChatRequest, StreamChunkResponse
from app.models.database import StreamChunkModel
from app.api.middleware.auth import get_api_key
from app.api.middleware.rate_limit import rate_limit_middleware
from app.core.logging import get_logger
from app.core.database import get_database
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
        # Get database instance
        db = await get_database()
        
        # Generate IDs
        session_id = (
            request.session_id
            or f"sess_{hashlib.md5(f'{request.user_id}_{datetime.now().timestamp()}'.encode()).hexdigest()[:12]}"
        )
        chat_id = f"chatcmpl-{hashlib.md5(f'{session_id}_{time.time()}'.encode()).hexdigest()[:12]}"
        created = int(time.time())
        
        # Chunk sequence counter
        chunk_sequence = 0

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

        # Save user query chunk to DB
        chunk_sequence += 1
        user_query_chunk_data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "user_message": user_query,
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages]
        }
        user_query_chunk_record = StreamChunkModel(
            chunk_id=f"{chat_id}_chunk_{chunk_sequence}",
            conversation_id=None,  # Will be updated later
            session_id=session_id,
            user_id=request.user_id,
            employee_id=request.employee_id,
            chat_id=chat_id,
            chunk_type="user_query",
            chunk_data=user_query_chunk_data,
            sequence=chunk_sequence,
            timestamp=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        await db.stream_chunks.insert_one(user_query_chunk_record.model_dump())
        
        # Send initial role chunk
        role_chunk_data = {
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
        
        # Save initial role chunk to DB
        chunk_sequence += 1
        role_chunk_record = StreamChunkModel(
            chunk_id=f"{chat_id}_chunk_{chunk_sequence}",
            conversation_id=None,  # Will be updated later
            session_id=session_id,
            user_id=request.user_id,
            employee_id=request.employee_id,
            chat_id=chat_id,
            chunk_type="role",
            chunk_data=role_chunk_data,
            sequence=chunk_sequence,
            timestamp=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        await db.stream_chunks.insert_one(role_chunk_record.model_dump())
        
        yield json.dumps(role_chunk_data)

        # Stream workflow execution and monitor for generate stage
        should_generate = False
        final_state = None
        full_answer = ""

        async for event in conversation_workflow.workflow.astream(
            initial_state, stream_mode="updates"
        ):
            # 使用 stream_mode="updates" 来获取节点级别的更新
            # event 格式: {node_name: state_update}
            node_name = list(event.keys())[0] if event else None
            state_update = event.get(node_name, {}) if node_name else {}
            
            # 检测 knowledge_retrieval 节点并发送状态提示
            if node_name == "knowledge_retrieval":
                status_token = "正在查询知识库\n"
                status_chunk_data = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "knowledge_retrieval",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": status_token},
                            "finish_reason": None,
                        }
                    ],
                }
                
                # 保存状态 chunk 到数据库
                chunk_sequence += 1
                status_chunk_record = StreamChunkModel(
                    chunk_id=f"{chat_id}_chunk_{chunk_sequence}",
                    conversation_id=None,
                    session_id=session_id,
                    user_id=request.user_id,
                    employee_id=request.employee_id,
                    chat_id=chat_id,
                    chunk_type="token",
                    chunk_data=status_chunk_data,
                    sequence=chunk_sequence,
                    timestamp=datetime.utcnow(),
                    created_at=datetime.utcnow()
                )
                await db.stream_chunks.insert_one(status_chunk_record.model_dump())
                
                yield json.dumps(status_chunk_data)
                
            # 检测 web_search 节点并发送状态提示
            elif node_name == "web_search":
                status_token = "正在网络搜索\n"
                status_chunk_data = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "web_search",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": status_token},
                            "finish_reason": None,
                        }
                    ],
                }
                
                # 保存状态 chunk 到数据库
                chunk_sequence += 1
                status_chunk_record = StreamChunkModel(
                    chunk_id=f"{chat_id}_chunk_{chunk_sequence}",
                    conversation_id=None,
                    session_id=session_id,
                    user_id=request.user_id,
                    employee_id=request.employee_id,
                    chat_id=chat_id,
                    chunk_type="token",
                    chunk_data=status_chunk_data,
                    sequence=chunk_sequence,
                    timestamp=datetime.utcnow(),
                    created_at=datetime.utcnow()
                )
                await db.stream_chunks.insert_one(status_chunk_record.model_dump())
                
                yield json.dumps(status_chunk_data)

            # Check if we've reached generation stage
            if (
                "confidence" in state_update
                and state_update.get("confidence", 0) > 0
                and not should_generate
            ):
                should_generate = True
                final_state = state_update
            
            # When ready to generate, do REAL streaming
            if should_generate and final_state:
                should_generate = False  # Only generate once

                # Build messages for LLM
                messages = conversation_workflow.build_generation_messages(final_state)

                # TRUE token-level streaming from LLM
                # 这里是用 conversation_workflow.llm.astream 的流式输出，
                async for chunk in conversation_workflow.llm.astream(messages):
                    token = chunk.content
                    if token:
                        full_answer += token
                        
                        token_chunk_data = {
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
                        
                        # Save token chunk to DB
                        chunk_sequence += 1
                        token_chunk_record = StreamChunkModel(
                            chunk_id=f"{chat_id}_chunk_{chunk_sequence}",
                            conversation_id=final_state.get("conversation_id"),
                            session_id=session_id,
                            user_id=request.user_id,
                            employee_id=request.employee_id,
                            chat_id=chat_id,
                            chunk_type="token",
                            chunk_data=token_chunk_data,
                            sequence=chunk_sequence,
                            timestamp=datetime.utcnow(),
                            created_at=datetime.utcnow()
                        )
                        await db.stream_chunks.insert_one(token_chunk_record.model_dump())
                        
                        yield json.dumps(token_chunk_data)

                # Update state with generated answer
                final_state["final_answer"] = full_answer
                
                # Save conversation before breaking to ensure history is recorded
                await conversation_workflow.save_conversation(final_state)
                
                # Break out of workflow loop to prevent duplicate generation
                break

        # If no final_state yet, use last event
        if final_state is None:
            final_state = event

        # Format source attribution
        sources = format_sources(
            retrieved_docs=final_state.get("retrieved_docs", []),
            web_search_results=final_state.get("web_search_results", []),
        )

        # Send finish chunk
        finish_chunk_data = {
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
        
        # Save finish chunk to DB
        chunk_sequence += 1
        finish_chunk_record = StreamChunkModel(
            chunk_id=f"{chat_id}_chunk_{chunk_sequence}",
            conversation_id=final_state.get("conversation_id", ""),
            session_id=session_id,
            user_id=request.user_id,
            employee_id=request.employee_id,
            chat_id=chat_id,
            chunk_type="done",
            chunk_data=finish_chunk_data,
            sequence=chunk_sequence,
            timestamp=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        await db.stream_chunks.insert_one(finish_chunk_record.model_dump())
        
        yield json.dumps(finish_chunk_data)

        # Send [DONE] marker
        yield "[DONE]"

    except Exception as e:
        logger.error("OpenAI stream generation error", error=str(e), exc_info=True)
        
        # Send error in OpenAI format
        error_chunk_data = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": "internal_error",
            }
        }
        
        # Try to save error chunk to DB
        try:
            db = await get_database()
            chunk_sequence += 1
            error_chunk_record = StreamChunkModel(
                chunk_id=f"{chat_id}_chunk_{chunk_sequence}",
                conversation_id=None,
                session_id=session_id,
                user_id=request.user_id,
                employee_id=request.employee_id,
                chat_id=chat_id,
                chunk_type="error",
                chunk_data=error_chunk_data,
                sequence=chunk_sequence,
                timestamp=datetime.utcnow(),
                created_at=datetime.utcnow()
            )
            await db.stream_chunks.insert_one(error_chunk_record.model_dump())
        except Exception as db_error:
            logger.error("Failed to save error chunk to DB", error=str(db_error), exc_info=True)
        
        yield json.dumps(error_chunk_data)

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
    兼容 OpenAI 的对话补全接口

    支持流式传输和非流式传输两种模式

    与 OpenAI 软件开发工具包（SDK）及应用程序接口（API）格式完全兼容

    参数说明:

        request: 符合 OpenAI 格式的对话请求

        api_key: 来自身份验证的应用程序接口密钥

    返回:
    
        符合 OpenAI 格式的响应或服务器发送事件（SSE）流
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
            # Non-streaming response (not implemented in this snippet)
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="没有实现非流式响应",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("OpenAI chat completion error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat completion",
        )


@router.get("/stream/chunks", response_model=StreamChunkResponse)
async def query_stream_chunks(
    user_id: Optional[str] = Query(None, description="User ID"),
    employee_id: Optional[str] = Query(None, description="Employee ID"),
    session_id: Optional[str] = Query(None, description="Session ID"),
    chat_id: Optional[str] = Query(None, description="Chat completion ID"),
    chunk_type: Optional[str] = Query(None, description="Chunk type filter"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Page size"),
    api_key: str = Depends(get_api_key)
):
    """
    按多条件查询流式输出的chunk数据。

    支持的过滤条件：
    - user_id：用户ID
    - employee_id：数字员工ID
    - session_id：会话ID
    - chat_id：OpenAI风格的chat completion ID
    - chunk_type：chunk类型（user_query/role/token/done/error等）
    - start_date/end_date：日期范围过滤

    查询结果按创建时间倒序分页返回。

    参数说明：
    - user_id：可选，用户ID过滤
    - employee_id：可选，员工ID过滤
    - session_id：可选，会话ID过滤
    - chat_id：可选，chat completion ID过滤
    - chunk_type：可选，chunk类型过滤
    - start_date：可选，起始日期（YYYY-MM-DD）
    - end_date：可选，结束日期（YYYY-MM-DD）
    - page：页码（从1开始）
    - page_size：每页数量（最大200）
    - api_key：API密钥

    返回：
        分页的chunk列表及分页信息
    """
    try:
        logger.info(
            "Query stream chunks",
            user_id=user_id,
            employee_id=employee_id,
            session_id=session_id,
            chat_id=chat_id,
            page=page,
            page_size=page_size
        )
        
        # Get database instance
        db = await get_database()
        
        # Build query filters
        query_filter = {}
        
        if user_id:
            query_filter["user_id"] = user_id
        if employee_id:
            query_filter["employee_id"] = employee_id
        if session_id:
            query_filter["session_id"] = session_id
        if chat_id:
            query_filter["chat_id"] = chat_id
        if chunk_type:
            query_filter["chunk_type"] = chunk_type
            
        # Date range filter
        if start_date or end_date:
            date_filter = {}
            if start_date:
                try:
                    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
                    date_filter["$gte"] = start_datetime
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid start_date format. Use YYYY-MM-DD"
                    )
            if end_date:
                try:
                    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
                    # Add one day to include the entire end_date
                    from datetime import timedelta
                    end_datetime = end_datetime + timedelta(days=1)
                    date_filter["$lt"] = end_datetime
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid end_date format. Use YYYY-MM-DD"
                    )
            if date_filter:
                query_filter["created_at"] = date_filter
        
        # Calculate skip for pagination
        skip = (page - 1) * page_size
        
        # Query total count
        total_count = await db.stream_chunks.count_documents(query_filter)
        
        # Query chunks with pagination
        cursor = db.stream_chunks.find(query_filter).sort("created_at", -1).skip(skip).limit(page_size)
        chunks = await cursor.to_list(length=page_size)
        
        # Format response
        chunks_data = []
        for chunk in chunks:
            # Remove MongoDB _id field
            chunk.pop("_id", None)
            
            # Convert datetime to ISO string
            if "timestamp" in chunk and isinstance(chunk["timestamp"], datetime):
                chunk["timestamp"] = chunk["timestamp"].isoformat() + "Z"
            if "created_at" in chunk and isinstance(chunk["created_at"], datetime):
                chunk["created_at"] = chunk["created_at"].isoformat() + "Z"
                
            chunks_data.append(chunk)
        
        # Calculate pagination info
        total_pages = (total_count + page_size - 1) // page_size
        has_next = page < total_pages
        has_prev = page > 1
        
        response_data = {
            "chunks": chunks_data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev
            }
        }
        
        return StreamChunkResponse(code=200, message="success", data=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query stream chunks error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query stream chunks"
        )
