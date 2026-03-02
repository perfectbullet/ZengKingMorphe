"""
Chat stream response generator for /v1/chat/completions endpoint.

This version can be modified for custom behavior specific to v1 API.
"""
import asyncio
import hashlib
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional, Any, Callable

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

STATUS_TOKENS: list[str] = [
    "好的，我正在梳理您的问题要点…",
    "这个我知道······",
    "等我一下······",
]

SEARCH_TOKENS = STATUS_TOKENS

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
    try:
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

        should_generate = False
        finish_sent = False
        final_state = None
        current_state = initial_state.copy()
        model_name = request.model

        async for event in conversation_workflow.workflow.astream(initial_state, stream_mode="updates"):
            node_name = list(event.keys())[0] if event else None
            state_update = event.get(node_name, {}) if node_name else {}

            if state_update:
                current_state.update(state_update)

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

                chunk_sequence += 1
                await save_stream_chunk(
                    db, chat_id, chunk_sequence, session_id, request.user_id,
                    request.employee_id, "token", status_chunk_data
                )

                yield json.dumps(status_chunk_data)

            if (
                "confidence" in state_update
                and state_update.get("confidence", 0) > 0
                and not should_generate
            ):
                should_generate = True
                final_state = current_state

            log_state = None
            if final_state:
                log_state = {
                    k: v for k, v in final_state.items()
                    if k not in ["retrieved_docs", "web_search_results", "context"]
                }
            logger.info(f'final_state: {log_state}')

            if should_generate and final_state:
                should_generate = False

                existing_answer = final_state.get("final_answer", "")
                direct_match = final_state.get("direct_match")

                if existing_answer and direct_match and not final_state.get("faq_matched"):
                    existing_answer = normalize_latex_formulas(existing_answer)

                    ttfb_ms = int((time.time() - initial_state["workflow_start_time"]) * 1000)
                    final_state["ttfb_ms"] = ttfb_ms

                    logger.info(
                        "Revising pre-generated answer for voice output | "
                        f"content_type={direct_match.get('content_type')} | "
                        f"rerank_score={direct_match.get('rerank_score')} | "
                        f"original_length={len(existing_answer)} | ttfb_ms={ttfb_ms}"
                    )

                    revise_llm = await get_revise_llm()

                    for char in existing_answer:
                        segment = sentence_buffer.add(char)
                        logger.info(f"segment={segment!r}")
                        if segment:
                            chunk_sequence, chunk_data = await _stream_segment_with_formula_conversion(
                                segment, revise_llm, chat_id, created, request.model,
                                db, chunk_sequence, session_id, request.user_id,
                                request.employee_id, final_state.get("conversation_id"),
                                log_prefix="DirectMatch"
                            )
                            yield json.dumps(chunk_data)

                    final_segment = await sentence_buffer.flush(is_final=True)
                    if final_segment:
                        chunk_sequence, chunk_data = await _stream_segment_with_formula_conversion(
                            final_segment.content, revise_llm, chat_id, created, request.model,
                            db, chunk_sequence, session_id, request.user_id,
                            request.employee_id, final_state.get("conversation_id"),
                            log_prefix="DirectMatch FinalSegment"
                        )
                        yield json.dumps(chunk_data)

                    chunk_sequence += 1
                    await save_stream_chunk(
                        db, chat_id, chunk_sequence, session_id, request.user_id,
                        request.employee_id, "done", finish_chunk_data,
                        final_state.get("conversation_id", "")
                    )
                    logger.info(f'save_stream_chunk finish_chunk_data is {finish_chunk_data}')

                    yield "[DONE]"

                    final_state["final_answer"] = existing_answer

                    finish_sent = True

                    await conversation_workflow.save_conversation(final_state)
                    break

                logger.info('No pre-generated answer, using LLM streaming')
                messages = conversation_workflow.build_generation_messages(final_state)
                streaming_llm, model_name = conversation_workflow.get_streaming_llm(final_state)
                revise_llm = await get_revise_llm()

                logger.info(
                    f"Streaming with LLM: {model_name} | "
                    f"intent={final_state.get('intent')} | "
                    f"faq_matched={bool(final_state.get('faq_matched'))} | "
                    f"web_search_used={final_state.get('web_search_used', False)}"
                )

                first_token_received = False
                full_answer = ""
                async for chunk in streaming_llm.astream(messages):
                    token = chunk.content
                    if token:
                        if not first_token_received:
                            first_token_received = True
                            ttfb_ms = int((time.time() - initial_state["workflow_start_time"]) * 1000)
                            final_state["ttfb_ms"] = ttfb_ms
                            logger.info(f"First token received | ttfb_ms={ttfb_ms}")
                        full_answer += token

                        segment = sentence_buffer.add(token)
                        logger.info(f"segment={segment!r}")

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
                                request.employee_id, final_state.get("conversation_id"),
                                log_prefix=""
                            )
                            yield json.dumps(chunk_data)

                final_segment = await sentence_buffer.flush(is_final=True)
                if final_segment:
                    chunk_sequence, chunk_data = await _stream_segment_with_formula_conversion(
                        final_segment.content, revise_llm, chat_id, created, request.model,
                        db, chunk_sequence, session_id, request.user_id,
                        request.employee_id, final_state.get("conversation_id"),
                        log_prefix="FinalSegment"
                    )
                    yield json.dumps(chunk_data)

                workflow_end_time = time.time()
                total_time_ms = int((workflow_end_time - initial_state["workflow_start_time"]) * 1000)
                logger.info(f"Workflow completed | total_time_ms={total_time_ms} | ttfb_ms={final_state.get('ttfb_ms')}")
                final_state["final_answer"] = full_answer

                chunk_sequence += 1
                await save_stream_chunk(
                    db, chat_id, chunk_sequence, session_id, request.user_id,
                    request.employee_id, "done", finish_chunk_data,
                    final_state.get("conversation_id", "")
                )
                logger.info(f'save_stream_chunk finish_chunk_data is {finish_chunk_data}')
                yield "[DONE]"

                finish_sent = True

                await conversation_workflow.save_conversation(final_state)
                break

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

        if not finish_sent:
            yield json.dumps(finish_chunk_data)
            yield "[DONE]"

    except Exception as e:
        logger.error(f"OpenAI stream v1 generation error | error={str(e)}", exc_info=True)

        error_chunk_data = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": "internal_error",
            }
        }

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
