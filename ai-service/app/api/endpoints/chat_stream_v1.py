"""
Chat stream response generator for /v1/chat/completions endpoint.

This version can be modified for custom behavior specific to v1 API.
"""
import asyncio
import hashlib
import json
import re
import time
from datetime import datetime
import random
from typing import AsyncGenerator, Optional, Any

from langchain_community.chat_models import ChatOllama
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
from app.utils.tts_formatter import strip_markdown_for_tts
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
    revise_llm: ChatOllama | ChatOpenAI,
    log_prefix: str = "",
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

    if has_latex_formula(display_content):
        logger.info(f"[{log_prefix}公式转换] 转换前长度={len(display_content)}, 转换前={repr(display_content)}")
        voice_content = await convert_formula_to_voice(display_content, revise_llm)
        logger.info(f"[{log_prefix}公式转换] 转换后长度={len(voice_content)}, 转换后={repr(voice_content)}")
    elif _has_math_symbols_simple(display_content):
        logger.info(f"[{log_prefix}数学句子转换] 转换前长度={len(display_content)}, 转换前={repr(display_content)}")
        voice_content = await convert_math_sentence_to_voice(display_content, revise_llm)
        logger.info(f"[{log_prefix}数学句子转换] 转换后长度={len(voice_content)}, 转换后={repr(voice_content)}")
    else:
        voice_content = display_content

    # Strip markdown formatting from voice_content for TTS
    # (display_content retains original markdown formatting for display)
    voice_content = strip_markdown_for_tts(voice_content)
    logger.info(f"[{log_prefix}markdown清理] , markdown清理={repr(voice_content)}")
    
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
    revise_llm: ChatOllama | ChatOpenAI,
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
) -> tuple[int, dict]:
    """
    Process a text segment and handle streaming with formula conversion.

    Args:
        segment: Text segment to process
        revise_llm: LLM for formula-to-voice conversion
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
        segment, revise_llm, log_prefix
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
    sources: dict,
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

    initial_state = _build_initial_state(request, session_id, user_query)

    sentence_buffer = SentenceBuffer(
        max_chars=SENTENCE_BUFFER_MAX_CHARS,
        max_wait_seconds=SENTENCE_BUFFER_MAX_WAIT_SECONDS,
        # comma_split_threshold=SENTENCE_BUFFER_COMMA_SPLIT_THRESHOLD
    )

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
            reject_message = "抱歉，您的问题包含敏感内容，请规范用语后再试。"
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

            # 检查是否有预生成的答案（direct_match）
            existing_answer = current_state.get("final_answer", "")
            direct_match = current_state.get("direct_match")
            # 先不做预生产答案


            first_token_received = False
            full_answer = ""
            TALKING_POINTS: list = [
                "好的，我正在梳理您的问题要点…",
                "等我一小下下······",
            ]
            talking_point = random.choice(TALKING_POINTS)
            full_answer += full_answer
            chunk_sequence, chunk_data = await _stream_segment_with_formula_conversion(
                talking_point, revise_llm, chat_id, created, request.model,
                db, chunk_sequence, session_id, request.user_id,
                request.employee_id, current_state.get("conversation_id"),
                log_prefix="RAGAnything"
            )
            yield json.dumps(chunk_data)
            if not first_token_received:
                first_token_received = True
                ttfb_ms = int((time.time() - initial_state["workflow_start_time"]) * 1000)
                current_state["ttfb_ms"] = ttfb_ms
                logger.info(f"First token received | ttfb_ms={ttfb_ms}")
                
            # 根据 streaming_type 选择不同的流式输出方式
            if streaming_type == "raganything_stream":
                # RAGAnything 流式输出
                query = current_state.get("raganything_query", current_state.get("user_query", ""))
                mode = current_state.get("raganything_mode", "hybrid")
                logger.info(f"Using RAGAnything stream | query={query[:50]} | mode={mode}")
                async for chunk in get_raganything_stream(query, mode=mode):
                    if chunk["type"] == "chunk":
                        content = chunk["content"]
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
                                log_prefix="RAGAnything"
                            )
                            yield json.dumps(chunk_data)

                    elif chunk["type"] == "sources_info":
                        logger.info(f"RAGAnything sources info | {chunk['content']}")
                    elif chunk["type"] == "sources":
                        logger.info(f"RAGAnything sources received | entities={len(chunk['content'].get('entities', []))}")
                    elif chunk["type"] == "error":
                        logger.error(f"RAGAnything error | {chunk['content']}")

                # 刷新 buffer 中剩余内容
                final_segment = await sentence_buffer.flush(is_final=True)
                if final_segment:
                    chunk_sequence, chunk_data = await _stream_segment_with_formula_conversion(
                        final_segment.content, revise_llm, chat_id, created, request.model,
                        db, chunk_sequence, session_id, request.user_id,
                        request.employee_id, current_state.get("conversation_id"),
                        log_prefix="RAGAnything FinalSegment"
                    )
                    yield json.dumps(chunk_data)

                final_state = current_state
                final_state["final_answer"] = full_answer

            elif streaming_type == "langchain_llm":
                # LangChain LLM 流式输出（原有逻辑）
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
                                log_prefix=""
                            )
                            yield json.dumps(chunk_data)
                final_segment = await sentence_buffer.flush(is_final=True)
                if final_segment:
                    chunk_sequence, chunk_data = await _stream_segment_with_formula_conversion(
                        final_segment.content, revise_llm, chat_id, created, request.model,
                        db, chunk_sequence, session_id, request.user_id,
                        request.employee_id, current_state.get("conversation_id"),
                        log_prefix="FinalSegment"
                    )
                    yield json.dumps(chunk_data)

                final_state = current_state
                final_state["final_answer"] = full_answer

    if final_state is None:
        final_state = current_state

    sources = format_sources(
        retrieved_docs=final_state.get("retrieved_docs", []),
        web_search_results=final_state.get("web_search_results", []),
    )

    _update_finish_chunk_metadata(finish_chunk_data, final_state, user_query, model_name, sources)

    chunk_sequence += 1
    await save_stream_chunk(
        db, chat_id, chunk_sequence, session_id, request.user_id,
        request.employee_id, "done", finish_chunk_data,
        final_state.get("conversation_id", "")
    )

    yield json.dumps(finish_chunk_data)
    yield "[DONE]"
