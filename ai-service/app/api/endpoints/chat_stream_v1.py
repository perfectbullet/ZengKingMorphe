"""
Chat stream response generator for /v1/chat/completions endpoint.

This version can be modified for custom behavior specific to v1 API.
"""
import os
import asyncio
import hashlib
import json
import re
import time
from datetime import datetime
import random
from typing import AsyncGenerator, Optional, Any

from langchain_openai import ChatOpenAI

from app.models.schemas import OpenAIChatRequest
from app.models.database import StreamChunkModel
from app.core.logging import get_logger
from app.core.database import get_database
from app.services.conversation_service import conversation_workflow

from app.services.revise_llm import (
    get_revise_llm,
    convert_formula_to_voice,
    convert_math_sentence_to_voice,
)
from app.utils.latex import normalize_latex_formulas
from app.utils.sentence_buffer import SentenceBuffer, has_latex_formula
from app.utils.think_tag_buffer import ThinkTagBuffer
from app.utils.tts_formatter import strip_markdown_for_tts
from app.utils.text_mapping import map_english_to_chinese
from app.utils.common import sanitize_filename
from app.services.raganything_wrapper import get_raganything_stream
logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

SENTENCE_BUFFER_MAX_CHARS = 100
SENTENCE_BUFFER_MAX_WAIT_SECONDS = 1
SENTENCE_BUFFER_COMMA_SPLIT_THRESHOLD = 30

RRF_K = 60
MAX_CONTENT_LENGTH = 200
CHUNK_SAVE_DELAY_SECONDS = 0.01

MAX_RAG_SOURCES = 3
MAX_WEB_SOURCES = 5
LOG_TRUNCATE_LENGTH = 100

CHUNK_TYPE_USER_QUERY = "user_query"
CHUNK_TYPE_ROLE = "role"
CHUNK_TYPE_TOKEN = "token"
CHUNK_TYPE_DONE = "done"
CHUNK_TYPE_ERROR = "error"

# =============================================================================
# Status messages for UX
# =============================================================================

# Note: STATUS_TOKENS removed - no longer needed with RAGAnything integration

# =============================================================================
# Utility Functions
# =============================================================================

def _clean_user_query(text: str) -> str:
    """
    Clean user query by removing leading punctuation.

    Args:
        text: User query text

    Returns:
        Cleaned text with leading punctuation removed
    """
    text = re.sub(r'^[，。！？、；：,.?!;:\s]+', '', text)
    return text.lstrip()

def _build_history_prefix_for_query(
    context_messages: list,
    prefer_zh_output: bool,
    max_turns: int = 6,
    max_chars: int = 1200,
) -> str:
    """
    为 RAG 查询构建精简对话历史前缀，支持多轮上下文承接
    用于处理指代、追问、纠错类需求
    """
    if not context_messages:
        return ""

    # 取最近 N 轮对话
    recent = context_messages[-max_turns:]
    lines: list[str] = []
    for m in recent:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        # 截断过长内容
        if len(content) > 300:
            content = content[:300] + "…"
        # 角色标签（中英适配）
        role_label = "User" if role == "user" else "Assistant" if role == "assistant" else role or "Message"
        if prefer_zh_output:
            role_label = "用户" if role == "user" else "助手" if role == "assistant" else role_label
        lines.append(f"{role_label}: {content}")

    history_text = "\n".join(lines).strip()
    if not history_text:
        return ""
    # 总长度截断
    if len(history_text) > max_chars:
        history_text = history_text[-max_chars:]

    # 带指令的对话历史前缀（中英）
    if prefer_zh_output:
        return (
            "【对话历史（用于承接上下文）】\n"
            f"{history_text}\n\n"
            "要求：如果用户追问里出现“它/这个/为什么/结果不对/再算一遍”等指代或纠错，请优先回指上文的题目、条件、结论、关键变量、公式与定义来回答；"
            "若上文信息仍不足，再向用户追问缺失条件。\n\n"
        )
    return (
        "[Conversation history (for context)]\n"
        f"{history_text}\n\n"
        "Requirement: If the user uses pronouns or follow-ups like \"it/this/why/the result seems wrong/recalculate\", "
        "resolve them by referring to the prior problem statement, conditions, conclusion, key variables, formulas, definitions, "
        "and your previous steps. If information is still missing, ask for the missing details.\n\n"
    )

def _prefer_zh_output(user_query: str) -> bool:
    """判断输出语言：含中文→中文，含英文→英文，其余默认中文"""
    if not user_query:
        return True
    if re.search(r"[\u4e00-\u9fff]", user_query):
        return True
    if re.search(r"[A-Za-z]", user_query):
        return False
    return True

# =============================================================================
# Source Attribution
# =============================================================================

def format_sources(
    retrieved_docs: list[dict],
    web_search_results: list[dict],
    max_content_length: int = MAX_CONTENT_LENGTH,
) -> dict:
    """
    Format RAG documents and web search results for source attribution.

    Args:
        retrieved_docs: List of retrieved document chunks from RAG
        web_search_results: List of web search results from Tavily
        max_content_length: Maximum content snippet length

    Returns:
        Dict with rag_sources and web_sources lists
    """
    sources = {"rag_sources": [], "web_sources": []}

    for idx, doc in enumerate(retrieved_docs[:MAX_RAG_SOURCES], 1):
        content = doc.get("content", "")
        content_snippet = content[:max_content_length]
        if len(content) > max_content_length:
            content_snippet += "..."

        raw_rrf_score = doc.get("rrf_score", doc.get("score", 0.0))
        max_possible_rrf = 2.0 / RRF_K
        normalized_score = (
            (raw_rrf_score / max_possible_rrf) if max_possible_rrf > 0 else 0.0
        )
        normalized_score = max(0.0, min(1.0, normalized_score))

        rag_source = {
            "rank": idx,
            "doc_id": doc.get("doc_id", ""),
            "kb_id": doc.get("kb_id", ""),
            "content_snippet": content_snippet,
            "score": round(normalized_score, 4),
        }

        if "chunk_index" in doc:
            rag_source["chunk_index"] = doc["chunk_index"]

        sources["rag_sources"].append(rag_source)

    for result in web_search_results[:MAX_WEB_SOURCES]:
        web_source = {
            "rank": result.get("rank", 0),
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "score": round(result.get("score", 0.0), 4),
        }
        sources["web_sources"].append(web_source)

    return sources

# =============================================================================
# Formula to Voice Conversion
# =============================================================================

def _has_math_symbols_simple(text: str) -> bool:
    """
    Check if text contains math symbols without LaTeX delimiters.

    Args:
        text: Text to check

    Returns:
        True if text contains math symbols
    """
    math_symbol_pattern = re.compile(r'[∈∉⊂⊃⊆⊇∪∩∅∨∧¬∀∃→⇒⇐⇔≡≠≤≥≈≪≫√∞²³°π∏∑∫∂∇Δ]')
    return math_symbol_pattern.search(text) is not None

async def _process_segment_for_output(
    segment: str,
    revise_llm: ChatOpenAI,
    log_prefix: str = "",
    prefer_zh_output: bool = True,
    enable_math_sentence_conversion: bool = False,
) -> tuple[str, str]:
    """
    Process a text segment for output.

    This function:
    1. Normalizes LaTeX delimiters
    2. Converts LaTeX formulas to voice-friendly text using LLM
    3. Converts sentences with math symbols to voice-friendly text using LLM

    Args:
        segment: Text segment to process
        revise_llm: LLM for formula-to-voice conversion
        log_prefix: Prefix for log messages

    Returns:
        Tuple of (display_content, voice_content)
    """
    if '$$' in segment:
        logger.warning(
            f"[{_process_segment_for_output.__name__}] Received segment with formula: "
            f"len={len(segment)}, starts_with_$$={segment.startswith('$$')}, ends_with_$$={segment.endswith('$$')}, "
            f"preview={repr(segment[:50])}...{repr(segment[-10:])}"
        )

    display_content = normalize_latex_formulas(segment)

    # 中文提问时，英文结果转中文（避免“英文问中文答”）
    if prefer_zh_output:
        display_content = map_english_to_chinese(display_content)

    if has_latex_formula(display_content):
        logger.info(f"[{log_prefix} 公式转换] 转换前长度={len(display_content)}, 转换前={repr(display_content)}")
        voice_content = await convert_formula_to_voice(display_content, revise_llm)
        logger.info(f"[{log_prefix} 公式转换] 转换后长度={len(voice_content)}, 转换后={repr(voice_content)}")
    elif enable_math_sentence_conversion and _has_math_symbols_simple(display_content):
        logger.info(f"[{log_prefix} 数学句子转换] 转换前长度={len(display_content)}, 转换前={repr(display_content)}")
        voice_content = await convert_math_sentence_to_voice(display_content, revise_llm)
        logger.info(f"[{log_prefix} 数学句子转换] 转换后长度={len(voice_content)}, 转换后={repr(voice_content)}")
    else:
        logger.info(f"{log_prefix}没有公式: {display_content}")
        voice_content = display_content

    # Strip markdown formatting from voice_content for TTS
    # (display_content retains original markdown formatting for display)
    logger.info(f"[{log_prefix} 语音voice_content markdown清理前: {repr(voice_content)}")
    voice_content = strip_markdown_for_tts(voice_content)
    logger.info(f"[{log_prefix} 语音voice_content markdown清理后: {repr(voice_content)}")
    return display_content, voice_content

def _build_token_chunk_data(
    chat_id: str,
    created: int,
    model: str,
    content: str,
) -> dict:
    """Build token chunk data for SSE output."""
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content},
            "finish_reason": None,
        }],
    }

async def _stream_segment_with_formula_conversion(
    segment: str,
    revise_llm: ChatOpenAI,
    chat_id: str,
    created: int,
    model: str,
    db: Any,
    chunk_sequence: int,
    session_id: str,
    user_id: str,
    employee_id: str,
    conversation_id: Optional[str],
    log_prefix: str = "",
    prefer_zh_output: bool = True,
    enable_math_sentence_conversion: bool = False,
) -> tuple[int, dict]:
    """
    Process a text segment and handle streaming with formula conversion.

    Args:
        segment: Text segment to process
        revise_llm: LLM for formula-to-voic，e conversion
        chat_id: Chat completion ID
        created: Creation timestamp
        model: Model name
        db: Database instance
        chunk_sequence: Current chunk sequence number
        session_id: Session ID
        user_id: User ID
        employee_id: Employee ID
        conversation_id: Optional conversation ID
        log_prefix: Prefix for log messages

    Returns:
        Tuple of (updated_sequence, voice_chunk_data_for_yielding)
    """
    display_content, voice_content = await _process_segment_for_output(
        segment,
        revise_llm,
        log_prefix,
        prefer_zh_output=prefer_zh_output,
        enable_math_sentence_conversion=enable_math_sentence_conversion,
    )

    voice_chunk_data = _build_token_chunk_data(chat_id, created, model, voice_content)

    new_sequence = chunk_sequence + 1
    await asyncio.sleep(CHUNK_SAVE_DELAY_SECONDS)
    await save_stream_chunk(
        db, chat_id, new_sequence, session_id, user_id, employee_id,
        "token",
        {**voice_chunk_data, "choices": [{
            **voice_chunk_data["choices"][0],
            "delta": {"content": display_content},
        }]},
        conversation_id
    )

    return new_sequence, voice_chunk_data

# =============================================================================
# Main Stream Generator
# =============================================================================

def _extract_user_query(messages: list) -> str:
    """Extract the last user message from the messages list."""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return messages[-1].content if messages else ""

def _build_initial_state(request: OpenAIChatRequest, session_id: str, user_query: str) -> dict:
    """Build the initial state for the conversation workflow."""
    return {
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
        "web_search_error": None,
        "conversation_id": "",
        "response_time_ms": 0,
        "workflow_start_time": time.time(),
        "node_timings": {},
        "ttfb_ms": None,
        "llm_temperature": request.temperature,
        "llm_top_p": request.top_p,
        "llm_max_tokens": request.max_tokens,
        "llm_presence_penalty": request.presence_penalty,
        "llm_frequency_penalty": request.frequency_penalty,
        "llm_seed": request.seed,
        "llm_n": request.n,
        "llm_tools": [tool.model_dump() for tool in request.tools] if request.tools else None,
        "channel_name": request.channel_name,
        "team_id": request.team_id,
        # Streaming output configuration
        "streaming_type": None,
        "streaming_llm": None,
        "streaming_messages": None,
        "raganything_query": None,
        "raganything_mode": None,
        "sources": [],
        # LLM-based classification (from QueryClassifier)
        "classification_label": None,
        "classification_confidence": None,
        "classification_reason": None,
    }

def _build_finish_chunk_data(
    chat_id: str,
    created: int,
    model: str,
    user_query: str,
) -> dict:
    """Build the finish chunk data template."""
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": len(user_query),
            "completion_tokens": len(user_query),
            "total_tokens": len(user_query),
        },
        "metadata": {
            "conversation_id": "",
            "confidence": 1,
            "kb_used": [],
            "web_search_used": False,
            "sources": {},
            "intent": "",
            "is_realtime_query": False,
            "realtime_category": False,
            "model": model,
        },
    }

def _update_finish_chunk_metadata(
    finish_chunk_data: dict,
    final_state: dict,
    user_query: str,
    model_name: str,
    sources: list,
) -> None:
    """Update the finish chunk data with final state information."""
    finish_chunk_data["usage"] = {
        "prompt_tokens": len(user_query),
        "completion_tokens": len(final_state.get("final_answer", "")),
        "total_tokens": len(user_query) + len(final_state.get("final_answer", "")),
    }
    finish_chunk_data["metadata"] = {
        "conversation_id": final_state.get("conversation_id", ""),
        "confidence": final_state.get("confidence", 0.0),
        "kb_used": final_state.get("kb_used", []),
        "web_search_used": final_state.get("web_search_used", False),
        "sources": sources,
        "intent": final_state.get("intent", ""),
        "is_realtime_query": final_state.get("is_realtime_query", False),
        "realtime_category": final_state.get("realtime_category", ""),
        "model": model_name,
    }

# =============================================================================
# MongoDB Storage
# =============================================================================

async def save_stream_chunk(
    db: Any,
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
    # 打印所有参数用于调试
    chunk_data_preview = str(chunk_data)[:200] if chunk_data else None
    logger.info(
        f"[save_stream_chunk] chat_id={chat_id}, seq={chunk_sequence}, "
        f"session_id={session_id}, user_id={user_id}, employee_id={employee_id}, "
        f"chunk_type={chunk_type}, conversation_id={conversation_id}, "
        f"chunk_data_keys={list(chunk_data.keys()) if chunk_data else []}, "
        f"chunk_data_preview={chunk_data_preview}"
    )

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

async def generate_openai_stream_v1(
    request: OpenAIChatRequest,
) -> AsyncGenerator[str, None]:
    """
    Generate OpenAI-style streaming response for v1 API.

    This version can be modified for custom behavior specific to v1 endpoint.

    Args:
        request: OpenAI chat request

    Yields:
        OpenAI-formatted SSE messages
    """

    db = await get_database()

    session_id = (
        request.session_id
        or f"sess_{hashlib.md5(f'{request.user_id}_{datetime.now().timestamp()}'.encode()).hexdigest()[:12]}"
    )
    chat_id = f"chatcmpl-{hashlib.md5(f'{session_id}_{time.time()}'.encode()).hexdigest()[:12]}"
    created = int(time.time())

    user_query = _clean_user_query(_extract_user_query(request.messages))
    prefer_zh_output = _prefer_zh_output(user_query)

    initial_state = _build_initial_state(request, session_id, user_query)
    # 工作流全局输出语言偏好
    initial_state["prefer_zh_output"] = prefer_zh_output

    sentence_buffer = SentenceBuffer(
        max_chars=SENTENCE_BUFFER_MAX_CHARS,
        max_wait_seconds=SENTENCE_BUFFER_MAX_WAIT_SECONDS,
        # comma_split_threshold=SENTENCE_BUFFER_COMMA_SPLIT_THRESHOLD
    )
    think_tag_buffer = ThinkTagBuffer()  # 用于过滤 think 标签

    finish_chunk_data = _build_finish_chunk_data(chat_id, created, request.model, user_query)

    chunk_sequence = 0

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

    chunk_sequence += 1
    await save_stream_chunk(
        db, chat_id, chunk_sequence, session_id, request.user_id,
        request.employee_id, "role", role_chunk_data
    )

    yield json.dumps(role_chunk_data)

    final_state = None
    current_state = initial_state.copy()
    model_name = request.model

    async for event in conversation_workflow.workflow.astream(initial_state, stream_mode="updates"):
        node_name = list(event.keys())[0] if event else None
        state_update = event.get(node_name, {}) if node_name else {}

        # DEBUG: 打印事件结构
        logger.debug(f"Event | node={node_name} | state_update_keys={list(state_update.keys())}")

        if state_update:
            current_state.update(state_update)

        # 检测敏感词并提前终止
        if node_name == "input_validation" and state_update.get("has_sensitive"):
            # 发送拒绝消息
            reject_message = (
                "抱歉，您的问题包含敏感内容，请规范用语后再试。"
                if prefer_zh_output
                else "Sorry, your question contains sensitive content. Please rephrase and try again."
            )

            reject_chunk_data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "status",
                "choices": [{"index": 0, "delta": {"content": reject_message}, "finish_reason": "sensitive"}],
            }
            chunk_sequence += 1
            await save_stream_chunk(
                db, chat_id, chunk_sequence, session_id, request.user_id,
                request.employee_id, "done", reject_chunk_data
            )
            yield json.dumps(reject_chunk_data)
            logger.info(f"Sensitive word detected | reject_message sent | breaking workflow")
            break  # 跳出循环，终止后续处理
            
        # 当到达 generate_answer 节点时，开始流式输出
        if node_name == "generate_answer":
            streaming_type = current_state.get("streaming_type")
            revise_llm = await get_revise_llm()

            # DEBUG: 打印当前状态中的关键字段
            logger.debug(
                f"generate_answer state | streaming_type={streaming_type} | "
                f"has_streaming_llm={current_state.get('streaming_llm') is not None} | "
                f"intent={current_state.get('intent')} | "
                f"all_keys={list(current_state.keys())}"
            )

            logger.info(
                f"Streaming configured | type={streaming_type} | "
                f"intent={current_state.get('intent')} | "
                f"web_search_used={current_state.get('web_search_used', False)}"
            )

            # 检查是否有预生成的答案（direct_match 或 noise 响应）
            existing_answer = current_state.get("final_answer", "")
            streaming_type = current_state.get("streaming_type")

            if existing_answer and streaming_type == "text":
                # 噪声输入或其他预设响应，直接返回
                ttfb_ms = int((time.time() - initial_state["workflow_start_time"]) * 1000)
                current_state["ttfb_ms"] = ttfb_ms
                logger.info(f"Using preset answer | length={len(existing_answer)} | ttfb_ms={ttfb_ms}")

                # 流式返回预设答案
                for char in existing_answer:
                    token_chunk_data = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": char},
                            "finish_reason": None,
                        }],
                    }
                    chunk_sequence += 1
                    await save_stream_chunk(
                        db, chat_id, chunk_sequence, session_id, request.user_id,
                        request.employee_id, "token", token_chunk_data,
                        current_state.get("conversation_id")
                    )
                    yield json.dumps(token_chunk_data)

                # 发送结束标记
                chunk_sequence += 1
                finish_chunk_data["metadata"]["conversation_id"] = current_state.get("conversation_id", "")
                finish_chunk_data["metadata"]["sources"] = current_state.get("sources", [])
                finish_chunk_data["metadata"]["intent"] = current_state.get("intent", "")
                finish_chunk_data["metadata"]["is_realtime_query"] = current_state.get("is_realtime_query", False)
                finish_chunk_data["metadata"]["realtime_category"] = current_state.get("realtime_category", "")
                finish_chunk_data["metadata"]["confidence"] = 0.99
                await save_stream_chunk(
                    db, chat_id, chunk_sequence, session_id, request.user_id,
                    request.employee_id, "done", finish_chunk_data,
                    current_state.get("conversation_id", "")
                )
                yield "[DONE]"
                await conversation_workflow.save_conversation(current_state)
                continue

            first_token_received = False
            full_answer = ""
            TALKING_POINTS: list = (
                [
                    "好的，我正在梳理您的问题要点…",
                    "等我一小下下······",
                ]
                if prefer_zh_output
                else [
                    "Got it—let me think for a moment…",
                    "One sec, I'm putting this together…",
                ]
            )
            
            # 发送统一过渡话术，按输入语言适配
            preface = TALKING_POINTS[0] + ("\n" if prefer_zh_output else "\n")
            enable_math_sentence_conversion = bool(current_state.get("is_math_problem", False))
            chunk_sequence, chunk_data = await _stream_segment_with_formula_conversion(
                preface, revise_llm, chat_id, created, request.model,
                db, chunk_sequence, session_id, request.user_id,
                request.employee_id, current_state.get("conversation_id"),
                prefer_zh_output=prefer_zh_output,
                enable_math_sentence_conversion=enable_math_sentence_conversion,
                log_prefix="Preface"
            )
            yield json.dumps(chunk_data)

            if not first_token_received:
                first_token_received = True
                ttfb_ms = int((time.time() - initial_state["workflow_start_time"]) * 1000)
                current_state["ttfb_ms"] = ttfb_ms
                logger.info(f"First token received | ttfb_ms={ttfb_ms}")
            model_name = streaming_type or model_name
            # 根据 streaming_type 选择不同的流式输出方式
            if streaming_type == "raganything_stream":
                # RAGAnything 流式输出
                query = current_state.get("raganything_query", current_state.get("user_query", ""))
                # 按输入语言追加回答指令，避免中英文不匹配
                if prefer_zh_output:
                    query = f"请用中文回答。\n\n{query}"
                else:
                    query = f"Please answer in English.\n\n{query}"

                # 拼接最近对话历史，解决“多轮失忆/无法承接/指代词无法回指”
                context_messages = (current_state.get("context") or {}).get("messages") or []
                history_prefix = _build_history_prefix_for_query(
                    context_messages=context_messages,
                    prefer_zh_output=prefer_zh_output,
                )
                if history_prefix:
                    query = history_prefix + query
                mode = current_state.get("raganything_mode", "hybrid")
                logger.info(f"Using RAGAnything stream | query={query[:50]} | mode={mode}")
                async for chunk in get_raganything_stream(query, mode=mode, prefer_zh_output=prefer_zh_output):
                    chunk_type = chunk.get("type")
                    content = chunk.get("content")
                    if chunk_type == "chunk":
                        # 跳过 None 内容
                        if content is None:
                            continue
                        full_answer += content
                        segment = sentence_buffer.add(content)
                        if segment:
                            chunk_sequence, chunk_data = await _stream_segment_with_formula_conversion(
                                segment, revise_llm, chat_id, created, request.model,
                                db, chunk_sequence, session_id, request.user_id,
                                request.employee_id, current_state.get("conversation_id"),
                                prefer_zh_output=prefer_zh_output,
                                enable_math_sentence_conversion=enable_math_sentence_conversion,
                                log_prefix="RAGAnything"
                            )
                            yield json.dumps(chunk_data)
                    elif chunk_type == "sources_info":
                        content: dict
                        logger.info(f"📊 检索到: {content['entities_count']} 个实体, "
                           f"{content['relationships_count']} 个关系, "
                           f"{content['chunks_count']} 个文档块")
                    elif chunk_type == "sources":
                        # 捕获 RAGAnything 返回的 sources
                        sources = content
                        
                        # 【调试代码】
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            os.makedirs("json格式数据", exist_ok=True)
                            json_file_path = f"json格式数据/{timestamp}_sources.json"
                            with open(json_file_path, 'w', encoding='utf-8') as f:
                                json.dump(sources, f, ensure_ascii=False, indent=2)
                            print(f"📁 来源数据已保存到: {json_file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to save sources debug json: error={e}")
                        # 【调试代码】
                        
                        # 处理实体引用
                        if sources.get("entities"):
                            entities = sources["entities"]
                            current_state["sources"].append({
                                "type": "entity",
                                "from": "【知识图谱】",
                                "entities": sources["entities"]
                            })
                            logger.info(f"RAGAnything entities | count={len(entities)} | 添加到 state['sources']")
                        # 处理文档块引用
                        if sources.get("chunks"):
                            chunks = sources["chunks"]
                            current_state["sources"].append({
                                "type": "chunk",
                                "from": "【文档块】",
                                "chunks": chunks
                            })
                            logger.info(f"RAGAnything chunks | count={len(chunks)} | 添加到 state['sources']")
                    elif chunk_type == "error":
                        logger.error(f"RAGAnything error | {chunk['content']}")

                # 刷新 buffer 中剩余内容
                final_segment = await sentence_buffer.flush(is_final=True)
                if final_segment:
                    chunk_sequence, chunk_data = await _stream_segment_with_formula_conversion(
                        final_segment.content, revise_llm, chat_id, created, request.model,
                        db, chunk_sequence, session_id, request.user_id,
                        request.employee_id, current_state.get("conversation_id"),
                        prefer_zh_output=prefer_zh_output,
                        enable_math_sentence_conversion=enable_math_sentence_conversion,
                        log_prefix="RAGAnything FinalSegment"
                    )
                    yield json.dumps(chunk_data)

                final_state = current_state
                final_state["final_answer"] = full_answer

            # 直接文本输出
            elif streaming_type == "direct_text":
                # 获取直接回答内容（如时间、固定回答）
                direct_text = current_state.get("direct_text_answer", "")
                if not direct_text:
                    direct_text = current_state.get("final_answer", "")

                # 存在直接文本则进行流式输出
                if direct_text:
                    full_answer += direct_text
                    # 调用流式输出工具，分段发送文本并转换公式格式
                    chunk_sequence, chunk_data = await _stream_segment_with_formula_conversion(
                        direct_text, revise_llm, chat_id, created, request.model,
                        db, chunk_sequence, session_id, request.user_id,
                        request.employee_id, current_state.get("conversation_id"),
                        enable_math_sentence_conversion=enable_math_sentence_conversion,
                        log_prefix="DirectText"
                    )
                    yield json.dumps(chunk_data)

                # 更新最终状态，保存完整答案
                final_state = current_state
                final_state["final_answer"] = full_answer


            elif streaming_type == "phi4_math":
                # Phi-4 数学推理流式输出
                streaming_llm = current_state.get("streaming_llm")
                messages = current_state.get("streaming_messages")
                if not streaming_llm or not messages:
                    logger.error("streaming_llm or messages not configured for phi4_math type")
                    continue
                logger.info(f"Using Phi-4 math stream | model={model_name}")

                # 使用真正的流式输出
                logger.info("Starting streaming response with astream")
                first_token_received = False
                full_answer = ""

                async for chunk in streaming_llm.astream(messages):
                    token = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    if token:
                        # 过滤 think 标签
                        filtered_token = think_tag_buffer.add(token)
                        if not filtered_token:
                            # logger.info(f'跟踪但不输出: {token!r}') # 本行日志疯狂打印，不要随意开启
                            full_answer += token  # 跟踪但不输出
                        else:
                            full_answer += filtered_token
                            segment = sentence_buffer.add(filtered_token)
                            if segment:
                                chunk_sequence, chunk_data = await _stream_segment_with_formula_conversion(
                                    segment, revise_llm, chat_id, created, request.model,
                                    db, chunk_sequence, session_id, request.user_id,
                                    request.employee_id, current_state.get("conversation_id"),
                                    prefer_zh_output=prefer_zh_output,
                                    enable_math_sentence_conversion=True,
                                    log_prefix="Phi-4-Math-Stream"
                                )
                                yield json.dumps(chunk_data)

                                if not first_token_received:
                                    first_token_received = True
                                    ttfb_ms = int((time.time() - initial_state["workflow_start_time"]) * 1000)
                                    current_state["ttfb_ms"] = ttfb_ms
                                    logger.info(f"First token received | ttfb_ms={ttfb_ms}")

                # 刷新 buffer 中剩余内容
                final_segment = await sentence_buffer.flush(is_final=True)
                if final_segment:
                    chunk_sequence, chunk_data = await _stream_segment_with_formula_conversion(
                        final_segment.content, revise_llm, chat_id, created, request.model,
                        db, chunk_sequence, session_id, request.user_id,
                        request.employee_id, current_state.get("conversation_id"),
                        prefer_zh_output=prefer_zh_output,
                        enable_math_sentence_conversion=True,
                        log_prefix="Phi-4-Math FinalSegment"
                    )
                    yield json.dumps(chunk_data)

                final_state = current_state
                final_state["final_answer"] = full_answer

            elif streaming_type == "langchain_llm":              
                # LangChain LLM 流式输出（原有逻辑）, 可以不使用话术                
                streaming_llm = current_state.get("streaming_llm")
                messages = current_state.get("streaming_messages")
                if not streaming_llm or not messages:
                    logger.error("streaming_llm or messages not configured for langchain_llm type")
                    continue
                logger.info(f"Using LangChain LLM stream | model={model_name}")
                async for chunk in streaming_llm.astream(messages):
                    token = chunk.content
                    if token:
                        full_answer += token
                        segment = sentence_buffer.add(token)
                        token_len = len(token)
                        buffer_len = sentence_buffer.get_buffer_length()
                        if segment or ('$$' in token[:10]):
                            logger.warning(
                                f"[STREAMING] token_len={token_len}, buffer_len={buffer_len}, "
                                f"has_segment={bool(segment)}, token_preview={repr(token[:50])}, "
                                f"buffer_start={repr(sentence_buffer.buffer[:30])}, buffer_end={repr(sentence_buffer.buffer[-30:])}"
                            )
                        if segment:
                            chunk_sequence, chunk_data = await _stream_segment_with_formula_conversion(
                                segment, revise_llm, chat_id, created, request.model,
                                db, chunk_sequence, session_id, request.user_id,
                                request.employee_id, current_state.get("conversation_id"),
                                prefer_zh_output=prefer_zh_output,
                                enable_math_sentence_conversion=enable_math_sentence_conversion,
                                log_prefix=""
                            )
                            yield json.dumps(chunk_data)
                final_segment = await sentence_buffer.flush(is_final=True)
                if final_segment:
                    chunk_sequence, chunk_data = await _stream_segment_with_formula_conversion(
                        final_segment.content, revise_llm, chat_id, created, request.model,
                        db, chunk_sequence, session_id, request.user_id,
                        request.employee_id, current_state.get("conversation_id"),
                        prefer_zh_output=prefer_zh_output,
                        enable_math_sentence_conversion=enable_math_sentence_conversion,
                        log_prefix="FinalSegment"
                    )
                    yield json.dumps(chunk_data)

                final_state = current_state
                final_state["final_answer"] = full_answer

    if final_state is None:
        final_state = current_state

    sources = final_state.get("sources", [])

    _update_finish_chunk_metadata(finish_chunk_data, final_state, user_query, model_name, sources)

    chunk_sequence += 1
    await save_stream_chunk(
        db, chat_id, chunk_sequence, session_id, request.user_id,
        request.employee_id, "done", finish_chunk_data,
        final_state.get("conversation_id", "")
    )

    yield json.dumps(finish_chunk_data)

    # 保存 final_state 用于调试
    save_dir = "finish_chunk_data"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = sanitize_filename(user_query)
    filename = f"{timestamp}_{session_id}_{safe_query}_finish_chunk_data.json"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(finish_chunk_data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"[调试] 保存 conversation_state 到 {filepath}")

    # 流式结束保存会话，保证上下文记忆
    try:
        await conversation_workflow.save_conversation(final_state)
    except Exception as e:
        logger.error(f"Failed to persist streaming conversation at end: {e}", exc_info=True)

    yield "[DONE]"
