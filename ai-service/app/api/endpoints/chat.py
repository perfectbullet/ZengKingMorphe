"""
Chat API endpoints.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi import Query
from sse_starlette.sse import EventSourceResponse
from datetime import datetime

from app.models.schemas import (
    OpenAIChatRequest,
    StreamChunkResponse,
)
from app.api.middleware.auth import get_api_key
from app.api.middleware.rate_limit import rate_limit_middleware
from app.core.logging import get_logger
from app.core.database import get_database

# Import stream generators for v1 and v2 endpoints
from app.api.endpoints.chat_stream_v1 import generate_openai_stream_v1
from app.api.endpoints.chat_stream_v2 import generate_openai_stream_v2

logger = get_logger(__name__)

router = APIRouter()


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
            f"user_id={effective_user_id} | "
            f"employee_id={effective_employee_id} | "
            f"session_id={request.session_id} | "
            f"stream={request.stream} | "
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
            return EventSourceResponse(generate_openai_stream_v1(stream_request))
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


@router.post("/v2/chat/completions")
async def openai_chat_completions_v2(
    request: OpenAIChatRequest
):
    """
    兼容 OpenAI 的对话补全接口 (v2 - 原版保留)

    v2版本保留原始行为，不做任何修改。

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
            - team_id: 团队id
            - user_name: 用户名称
            - head_url: 用户头像

        api_key: 来自身份验证的应用程序接口密钥

    返回:

        符合 OpenAI 格式的响应或服务器发送事件（SSE）流
    """
    # 处理 extra_body 参数（OpenAI SDK 通过 extra_body 传递非标准参数）
    effective_team_id = request.team_id
    effective_user_id = request.user_id
    effective_employee_id = request.employee_id
    effective_channel_name = request.channel_name
    effective_user_name = request.user_name
    effective_head_url = request.head_url

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
                        parsed_user_name = parts[4]
                        parsed_head_url = parts[5]

                        # 只覆盖缺失的值
                        if not effective_team_id:
                            effective_team_id = parsed_team_id
                        if not effective_user_id:
                            effective_user_id = parsed_user_id
                        if not effective_employee_id:
                            effective_employee_id = parsed_employee_id
                        if not effective_user_name:
                            effective_user_name = parsed_user_name
                        if not effective_head_url:
                            effective_head_url = parsed_head_url

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
        if "user_name" in request.extra_body and request.extra_body["user_name"]:
            effective_user_name = request.extra_body["user_name"]
        if "head_url" in request.extra_body and request.extra_body["head_url"]:
            effective_head_url = request.extra_body["head_url"]

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
        f"OpenAI v2 chat completion request | "
        f"model={request.model} | "
        f"user_id={effective_user_id} | "
        f"user_name={effective_user_name} | "
        f"head_url={effective_head_url} | "
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
        stream_request = request.model_copy(
            update={
                "user_id": effective_user_id,
                "user_name": effective_user_name,
                "head_url": effective_head_url,
                "employee_id": effective_employee_id,
                "team_id": effective_team_id,
                "channel_name": effective_channel_name,
            }
        )
        return EventSourceResponse(generate_openai_stream_v2(stream_request))
    else:
        # Non-streaming response (not implemented)
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="没有实现非流式响应",
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
