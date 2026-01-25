"""
Chat API endpoints.
"""

import random
import time
from typing import AsyncGenerator, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi import Query
from sse_starlette.sse import EventSourceResponse
import json
from datetime import datetime
import hashlib

from app.models.schemas import (
    OpenAIChatRequest,
    StreamChunkResponse,
)
from app.models.database import StreamChunkModel
from app.api.middleware.auth import get_api_key
from app.api.middleware.rate_limit import rate_limit_middleware
from app.core.logging import get_logger
from app.core.database import get_database
from app.services.conversation_service import conversation_workflow

logger = get_logger(__name__)

router = APIRouter()


# Status message variations for better UX
STATUS_TOKENS: List[str] = [
    "正在查询资料。",
    "正在检索知识库...",
    "正在查找相关信息...",
    "正在搜索知识库...",
    "正在阅读文档...",
    "正在分析问题...",
    "正在查询相关资料...",
    "正在检索数据库...",
    "正在查找答案...",
    "正在阅读相关内容...",
]

SEARCH_TOKENS: List[str] = [
    "正在进行网络搜索...",
    "正在联网查找...",
    "正在搜索网络资料...",
    "正在获取最新信息...",
    "正在查询网络数据...",
    "正在在线搜索...",
    "正在检索互联网信息...",
    "正在查找网络资源...",
    "正在获取实时信息...",
    "正在搜索网络...",
]


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

        # Normalize RRF score to 0-1 range for display
        # RRF score range: 0 ~ 2/k (when k=60, max ~0.033)
        raw_rrf_score = doc.get("rrf_score", doc.get("score", 0.0))
        rrf_k = 60  # Must match the k value used in _rrf_fusion
        max_possible_rrf = 2.0 / rrf_k
        normalized_score = (raw_rrf_score / max_possible_rrf) if max_possible_rrf > 0 else 0.0
        normalized_score = max(0.0, min(1.0, normalized_score))

        rag_source = {
            "rank": idx,
            "doc_id": doc.get("doc_id", ""),
            "kb_id": doc.get("kb_id", ""),
            "content_snippet": content_snippet,
            "score": round(normalized_score, 4),
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


async def save_stream_chunk(
    db,
    chat_id: str,
    chunk_sequence: int,
    session_id: str,
    user_id: str,
    employee_id: str,
    chunk_type: str,
    chunk_data: dict,
    conversation_id: Optional[str] = None,
) -> None:
    """
    Save a stream chunk to MongoDB.

    Args:
        db: Database instance
        chat_id: Chat completion ID
        chunk_sequence: Chunk sequence number
        session_id: Session ID
        user_id: User ID
        employee_id: Employee ID
        chunk_type: Type of chunk (user_query, role, token, done, error, status)
        chunk_data: Chunk data to save
        conversation_id: Optional conversation ID
    """
    chunk_record = StreamChunkModel(
        chunk_id=f"{chat_id}_chunk_{chunk_sequence}",
        conversation_id=conversation_id,
        session_id=session_id,
        user_id=user_id,
        employee_id=employee_id,
        chat_id=chat_id,
        chunk_type=chunk_type,
        chunk_data=chunk_data,
        sequence=chunk_sequence,
        timestamp=datetime.utcnow(),
        created_at=datetime.utcnow(),
    )
    await db.stream_chunks.insert_one(chunk_record.model_dump())


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

        # Build initial state with all parameters from request
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
            # Performance monitoring
            "workflow_start_time": time.time(),
            "node_timings": {},
            "ttfb_ms": None,
            # LLM parameters from OpenAI request
            "llm_temperature": request.temperature,
            "llm_top_p": request.top_p,
            "llm_max_tokens": request.max_tokens,
            "llm_presence_penalty": request.presence_penalty,
            "llm_frequency_penalty": request.frequency_penalty,
            "llm_seed": request.seed,
            "llm_n": request.n,
            "llm_tools": [tool.model_dump() for tool in request.tools] if request.tools else None,
            # Additional context
            "channel_name": request.channel_name,
            "team_id": request.team_id,
        }

        # Save user query chunk to DB
        chunk_sequence += 1
        user_query_chunk_data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "user_message": user_query,
            "messages": [
                {"role": msg.role, "content": msg.content} for msg in request.messages
            ],
        }
        await save_stream_chunk(
            db, chat_id, chunk_sequence, session_id, request.user_id,
            request.employee_id, "user_query", user_query_chunk_data
        )

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
        await save_stream_chunk(
            db, chat_id, chunk_sequence, session_id, request.user_id,
            request.employee_id, "role", role_chunk_data
        )

        yield json.dumps(role_chunk_data)

        # Quick check for realtime query BEFORE workflow starts
        # This allows immediate feedback to user before slow web search
        query_lower = user_query.lower()
        is_likely_realtime = any(keyword in query_lower for keyword in
                                  ['天气', '气温', '温度', '下雨', '下雪', '刮风',
                                   '股价', '股票', '汇率', '金价', '银价',
                                   '新闻', '今日', '最新', '实时'])

        if is_likely_realtime:
            # Send immediate search status indicator
            search_token = random.choice(SEARCH_TOKENS)
            search_chunk_data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "status",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": search_token},
                        "finish_reason": None,
                    }
                ],
            }

            # Save search status chunk to DB
            chunk_sequence += 1
            await save_stream_chunk(
                db, chat_id, chunk_sequence, session_id, request.user_id,
                request.employee_id, "status", search_chunk_data
            )

            yield json.dumps(search_chunk_data)

        # Stream workflow execution and monitor for generate stage
        should_generate = False
        final_state = None
        current_state = initial_state.copy()  # Accumulate state across nodes
        full_answer = ""
        model_name = request.model  # Initialize with requested model

        async for event in conversation_workflow.workflow.astream(
            initial_state, stream_mode="updates"
        ):
            # 使用 stream_mode="updates" 来获取节点级别的更新
            # event 格式: {node_name: state_update}
            node_name = list(event.keys())[0] if event else None
            state_update = event.get(node_name, {}) if node_name else {}

            # Accumulate state updates
            if state_update:
                current_state.update(state_update)

            # 检测 knowledge_retrieval 节点并发送状态提示
            if node_name == "knowledge_retrieval":
                status_token = random.choice(STATUS_TOKENS)
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
                await save_stream_chunk(
                    db, chat_id, chunk_sequence, session_id, request.user_id,
                    request.employee_id, "token", status_chunk_data
                )

                yield json.dumps(status_chunk_data)

            # Check if we've reached generation stage
            if (
                "confidence" in state_update
                and state_update.get("confidence", 0) > 0
                and not should_generate
            ):
                should_generate = True
                final_state = current_state  # Use accumulated state

            # When ready to generate, do REAL streaming
            if should_generate and final_state:
                should_generate = False  # Only generate once

                # Build messages for LLM
                messages = conversation_workflow.build_generation_messages(final_state)

                # Get appropriate LLM for streaming based on hybrid routing
                streaming_llm, model_name = conversation_workflow.get_streaming_llm(final_state)
                logger.info(
                    f"Streaming with LLM: {model_name} | "
                    f"intent={final_state.get('intent')} | "
                    f"faq_matched={bool(final_state.get('faq_matched'))} | "
                    f"web_search_used={final_state.get('web_search_used', False)}"
                )

                # TRUE token-level streaming from LLM
                first_token_received = False
                async for chunk in streaming_llm.astream(messages):
                    token = chunk.content
                    if token:
                        # Track TTFB on first token
                        if not first_token_received:
                            first_token_received = True
                            ttfb_ms = int((time.time() - initial_state["workflow_start_time"]) * 1000)
                            final_state["ttfb_ms"] = ttfb_ms
                            logger.info(f"First token received | ttfb_ms={ttfb_ms}")

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
                        await save_stream_chunk(
                            db, chat_id, chunk_sequence, session_id, request.user_id,
                            request.employee_id, "token", token_chunk_data,
                            final_state.get("conversation_id")
                        )

                        yield json.dumps(token_chunk_data)

                # Log workflow completion time
                workflow_end_time = time.time()
                total_time_ms = int((workflow_end_time - initial_state["workflow_start_time"]) * 1000)
                logger.info(f"Workflow completed | total_time_ms={total_time_ms} | ttfb_ms={final_state.get('ttfb_ms')}")

                # Update state with generated answer
                final_state["final_answer"] = full_answer

                # Save conversation before breaking to ensure history is recorded
                await conversation_workflow.save_conversation(final_state)

                # Break out of workflow loop to prevent duplicate generation
                break

        # If no final_state yet, use accumulated current_state
        if final_state is None:
            final_state = current_state

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
                "intent": final_state.get("intent", ""),
                "is_realtime_query": final_state.get("is_realtime_query", False),
                "realtime_category": final_state.get("realtime_category", ""),
                "model": model_name,
            },
        }

        # Save finish chunk to DB
        chunk_sequence += 1
        await save_stream_chunk(
            db, chat_id, chunk_sequence, session_id, request.user_id,
            request.employee_id, "done", finish_chunk_data,
            final_state.get("conversation_id", "")
        )

        yield json.dumps(finish_chunk_data)

        # Send [DONE] marker
        yield "[DONE]"

    except Exception as e:
        logger.error(f"OpenAI stream generation error | error={str(e)}", exc_info=True)

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
            await save_stream_chunk(
                db, chat_id, chunk_sequence, session_id, request.user_id,
                request.employee_id, "error", error_chunk_data
            )
        except Exception as db_error:
            logger.error(f"Failed to save error chunk to DB | error={str(db_error)}", exc_info=True)

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

    支持流式传输模式

    与 OpenAI 软件开发工具包（SDK）及应用程序接口（API）格式完全兼容

    参数说明:

        request: 符合 OpenAI 格式的对话请求，包含:
            - model: 模型名称
            - messages: 对话消息列表
            - stream: 是否启用流式传输
            - temperature: 采样温度 (0.0-2.0)
            - top_p: 核采样参数 (0.0-1.0)
            - max_tokens: 最大生成token数
            - presence_penalty: 存在惩罚 (-2.0-2.0)
            - frequency_penalty: 频率惩罚 (-2.0-2.0)
            - seed: 随机种子
            - n: 生成候选数量
            - tools: 工具/函数调用列表
            - employee_id: 数字员工ID
            - user_id: 用户ID
            - session_id: 会话ID
            - channel_name: 渠道名称
            - team_id: 团队ID

        api_key: 来自身份验证的应用程序接口密钥

    返回:

        符合 OpenAI 格式的响应或服务器发送事件（SSE）流
    """
    try:
        # 处理 extra_body 参数（OpenAI SDK 通过 extra_body 传递非标准参数）
        effective_team_id = request.team_id
        effective_user_id = request.user_id
        effective_employee_id = request.employee_id
        effective_channel_name = request.channel_name

        # 只有当 team_id/user_id/employee_id 不存在时，才从 channel_name 解析
        if request.extra_body and "channel_name" in request.extra_body:
            channel_name = request.extra_body["channel_name"]
            if channel_name and not effective_channel_name:
                effective_channel_name = channel_name
                # 检查是否需要解析（参数缺失时）
                need_parse = not effective_team_id or not effective_user_id or not effective_employee_id

                if need_parse:
                    logger.info(f"Received channel_name from extra_body: {channel_name}")
                    # 解析 channel_name: employee_<team_id>_<user_id>_<employee_id>
                    parts = channel_name.split('_')
                    if len(parts) >= 4 and parts[0] == "employee":
                        try:
                            parsed_team_id = parts[1]
                            parsed_user_id = parts[2]
                            parsed_employee_id = parts[3]

                            # 只覆盖缺失的值
                            if not effective_team_id:
                                effective_team_id = parsed_team_id
                            if not effective_user_id:
                                effective_user_id = parsed_user_id
                            if not effective_employee_id:
                                effective_employee_id = parsed_employee_id

                            logger.info(
                                f"Parsed from channel_name: team_id={effective_team_id}, "
                                f"user_id={effective_user_id}, employee_id={effective_employee_id}"
                            )
                        except (ValueError, IndexError) as e:
                            logger.error(f"Failed to parse channel_name '{channel_name}': {e}")
                    else:
                        logger.error(f"Invalid channel_name format: '{channel_name}', expected 'employee_<team_id>_<user_id>_<employee_id>'")

        # extra_body 中的直接参数优先级最高（覆盖所有其他来源）
        if request.extra_body:
            if "session_id" in request.extra_body and request.extra_body["session_id"]:
                request.session_id = request.extra_body["session_id"]
            if "team_id" in request.extra_body and request.extra_body["team_id"]:
                effective_team_id = request.extra_body["team_id"]
            if "user_id" in request.extra_body and request.extra_body["user_id"]:
                effective_user_id = request.extra_body["user_id"]
            if "employee_id" in request.extra_body and request.extra_body["employee_id"]:
                effective_employee_id = request.extra_body["employee_id"]
            if "channel_name" in request.extra_body and request.extra_body["channel_name"] and not effective_channel_name:
                effective_channel_name = request.extra_body["channel_name"]

        # Rate limiting
        await rate_limit_middleware(
            request=None, user_id=effective_user_id, session_id=request.session_id
        )

        # 提取最后一条用户消息用于日志
        last_user_message = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                last_user_message = msg.content
                break

        logger.info(
            f"OpenAI chat completion request | "
            f"model={request.model} | "
            f"user_id={effective_user_id} | "
            f"employee_id={effective_employee_id} | "
            f"session_id={request.session_id} | "
            f"stream={request.stream} | "
            f"temperature={request.temperature} | "
            f"top_p={request.top_p} | "
            f"max_tokens={request.max_tokens} | "
            f"presence_penalty={request.presence_penalty} | "
            f"frequency_penalty={request.frequency_penalty} | "
            f"seed={request.seed} | "
            f"n={request.n} | "
            f"has_tools={request.tools is not None} | "
            f"channel_name={effective_channel_name} | "
            f"team_id={effective_team_id} | "
            f"extra_body_provided={request.extra_body is not None} | "
            f"messages_count={len(request.messages)} | "
            f"last_user_message={last_user_message[:200] if last_user_message else ''}"
        )

        if request.stream:
            # Return streaming response with effective parameters
            # 创建一个包含解析后参数的请求副本
            stream_request = request.model_copy(
                update={
                    "user_id": effective_user_id,
                    "employee_id": effective_employee_id,
                    "team_id": effective_team_id,
                    "channel_name": effective_channel_name,
                }
            )
            return EventSourceResponse(generate_openai_stream_response(stream_request))
        else:
            # Non-streaming response (not implemented in this snippet)
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="没有实现非流式响应",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OpenAI chat completion error | error={str(e)}", exc_info=True)
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
    api_key: str = Depends(get_api_key),
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
            f"Query stream chunks | "
            f"user_id={user_id} | "
            f"employee_id={employee_id} | "
            f"session_id={session_id} | "
            f"chat_id={chat_id} | "
            f"page={page} | "
            f"page_size={page_size}"
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
                        detail="Invalid start_date format. Use YYYY-MM-DD",
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
                        detail="Invalid end_date format. Use YYYY-MM-DD",
                    )
            if date_filter:
                query_filter["created_at"] = date_filter

        # Calculate skip for pagination
        skip = (page - 1) * page_size

        # Query total count
        total_count = await db.stream_chunks.count_documents(query_filter)

        # Query chunks with pagination
        cursor = (
            db.stream_chunks.find(query_filter)
            .sort("created_at", -1)
            .skip(skip)
            .limit(page_size)
        )
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
                "has_prev": has_prev,
            },
        }

        return StreamChunkResponse(code=200, message="success", data=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query stream chunks error | error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query stream chunks",
        )
