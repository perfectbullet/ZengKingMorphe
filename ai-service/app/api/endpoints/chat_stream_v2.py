"""
Chat stream response generator for /v2/chat/completions endpoint.

This version preserves the original behavior without modifications.
"""
import os
import re
import random
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

from app.models.schemas import OpenAIChatRequest
from app.models.database import StreamChunkModel
from app.core.logging import get_logger
from app.core.database import get_database
from app.services.conversation_service import conversation_workflow
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

logger = get_logger(__name__)


# Status message variations for better UX
STATUS_TOKENS: list = [
    "好的，我正在梳理您的问题要点…",
    "这个我知道······",
    "等我一小下下······",
]

SEARCH_TOKENS: list = [
    "好的，我正在梳理您的问题要点…",
    "这个我知道······",
    "等我一小下下······",
]


def _clean_user_query(text: str) -> str:
    """
    清理用户查询，删除前导标点符号。
    Args:
        text: 用户输入的查询文本
    Returns:
        清理后的文本
    """
    # 删除前导标点符号（中文和英文）
    text = re.sub(r'^[，。！？、；：,.?!;:\s]+', '', text)

    # 删除前导空白字符
    text = text.lstrip()

    return text


def _load_revise_prompt() -> str:
    """Load the system prompt for math formula voice explanation."""
    # chat_stream_v2.py is at: app/api/endpoints/chat_stream_v2.py
    # prompts dir is at: prompts/ (from ai-service root)
    # So we need: app/api/endpoints/ -> app/api/ -> app/ -> ai-service/ -> prompts/
    prompt_path = Path(__file__).parent.parent.parent.parent / "prompts" / "数学公式口语化讲解.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning(f"Prompt file not found: {prompt_path}, using default prompt")
        return "你是一个数学公式口语化讲解专家。请将用户输入的数学公式和概念，用纯粹、流畅、易于理解的自然语言解释，完全不含任何数学符号或特殊格式，专为语音播报场景设计。"


def _get_revise_llm():
    """
    Get LLM instance for text revision (voice-friendly output).

    Supports:
    - siliconflow: SiliconFlow API (recommended)
    - ollama: Local Ollama

    Environment Variables:
    - REVISE_PROVIDER: Provider type (siliconflow or ollama), default siliconflow
    - OPENAI_API_KEY: SiliconFlow API key
    - OPENAI_API_BASE: SiliconFlow API base URL
    - OPENAI_REVISE_MODEL: SiliconFlow model name (default: deepseek-ai/DeepSeek-V3)
    - OLLAMA_BASE_URL: Ollama base URL (default: http://localhost:11434)
    - OLLAMA_REVISE_MODEL: Ollama model name (default: qwen2.5:7b)
    """
    # Get provider from env, default siliconflow
    provider = os.getenv("REVISE_PROVIDER", "siliconflow").lower()

    if provider == "siliconflow":
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
        model = os.getenv("OPENAI_REVISE_MODEL",
                         os.getenv("OPENAI_MODEL", "deepseek-ai/DeepSeek-V3"))

        if not api_key:
            logger.warning("OPENAI_API_KEY not set for SiliconFlow")

        logger.info(f"[Revise LLM] SiliconFlow | API_BASE={api_base} | MODEL={model}")

        return ChatOpenAI(
            base_url=api_base,
            api_key=api_key or "",  # Allow empty, let API handle error
            model=model,
            temperature=0.7,
            streaming=True,
        )
    else:
        # Use Ollama
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_REVISE_MODEL",
                                 os.getenv("OLLAMA_MODEL", "qwen2.5:7b"))

        logger.info(f"[Revise LLM] Ollama | BASE_URL={ollama_base_url} | MODEL={ollama_model}")

        return ChatOllama(
            base_url=ollama_base_url,
            model=ollama_model,
            temperature=0.7,
            streaming=True,
            keep_alive=-1
        )

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
        raw_rrf_score = doc.get("rrf_score", doc.get("score", 0.0))
        rrf_k = 60
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


async def generate_openai_stream_v2(
    request: OpenAIChatRequest,
) -> AsyncGenerator[str, None]:
    """
    Generate OpenAI-style streaming response for v2 API.

    This version preserves the original behavior without modifications.

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

        # 清理用户查询：删除前导标点符号
        user_query = _clean_user_query(user_query)

        # Build initial state with all parameters from request
        initial_state = {
            "messages": [],
            "user_query": user_query,
            "user_id": request.user_id,
            "user_name": request.user_name,
            "head_url": request.head_url,
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
        current_state = initial_state.copy()
        full_answer = ""
        model_name = request.model
        
        # 结束 chunk data
        finish_chunk_data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
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
                "model": model_name,
            },
        }

        async for event in conversation_workflow.workflow.astream(initial_state, stream_mode="updates"):
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
                final_state = current_state

            logger.info(f'final_state： {final_state}')

            # When ready to generate, do TRUE streaming
            if should_generate and final_state:
                should_generate = False

                # 检查是否已有预生成的答案（仅数学教材知识库的 direct match）
                existing_answer = final_state.get("final_answer", "")
                direct_match = final_state.get("direct_match")

                if existing_answer and direct_match and not final_state.get("faq_matched"):
                    # 规范化 LaTeX 公式：定界符、空格清理、反斜杠转义
                    from app.utils.latex import normalize_latex_formulas
                    existing_answer = normalize_latex_formulas(existing_answer)

                    # 直接流式返回预生成的答案，跳过 LLM 生成
                    ttfb_ms = int((time.time() - initial_state["workflow_start_time"]) * 1000)
                    final_state["ttfb_ms"] = ttfb_ms

                    logger.info(
                        "Using pre-generated answer (math textbook direct match) | "
                        f"content_type={direct_match.get('content_type')} | "
                        f"rerank_score={direct_match.get('rerank_score')} | "
                        f"length={len(existing_answer)} | ttfb_ms={ttfb_ms}"
                    )
                    # 按中文标点符号切分流式返回答案
                    segments = re.split(r'([。！？\n])', existing_answer)
                    for segment in segments:
                        token_chunk_data = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": segment},
                                "finish_reason": None,
                            }],
                        }
                        chunk_sequence += 1
                        # Save finish chunk to DB， 通过 ws 发送
                        await save_stream_chunk(
                            db, chat_id, chunk_sequence, session_id, request.user_id,
                            request.employee_id, "token", token_chunk_data,
                            final_state.get("conversation_id")
                        )
                    # 保存结束块
                    chunk_sequence += 1
                    await save_stream_chunk(
                        db, chat_id, chunk_sequence, session_id, request.user_id,
                        request.employee_id, "done", finish_chunk_data,
                        final_state.get("conversation_id", "")
                    )
                    logger.info(f'save_stream_chunk finish_chunk_data is {finish_chunk_data}')
                    # Send [DONE] marker
                    yield "[DONE]"
                    await conversation_workflow.save_conversation(final_state)

                    # 这里的输出会发生给语音合成服务
                    # 优先使用 teaching_script_tts，如果为空或查询不到则走 LLM 转换逻辑
                    chunk_id = direct_match.get("chunk_id")
                    teaching_script_tts = None
                    # 从 MongoDB 查询 teaching_script_tts
                    if chunk_id:
                        try:
                            tts_chunk = await db.document_chunks.find_one(
                                {"chunk_id": chunk_id},
                                {"teaching_script_tts": 1}
                            )
                            if tts_chunk:
                                teaching_script_tts = tts_chunk.get("teaching_script_tts")
                                logger.info(f"Found teaching_script_tts for chunk_id={chunk_id}, length={len(teaching_script_tts) if teaching_script_tts else 0}")
                                logger.info(f"teaching_script_tts content: {teaching_script_tts}")
                        except Exception as e:
                            logger.warning(f"Failed to query teaching_script_tts: {e}")
                    # 如果 teaching_script_tts 存在且非空，直接流式输出；否则走 LLM 转换
                    if teaching_script_tts and teaching_script_tts.strip():
                        # 直接输出 teaching_script_tts
                        logger.info(f"Using teaching_script_tts directly, length={len(teaching_script_tts)}")
                        tst_ls = re.split(r'([。！？\n])', teaching_script_tts)
                        for tst_token in tst_ls:
                            token_chunk_data = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": tst_token},
                                    "finish_reason": None,
                                }],
                            }
                            chunk_sequence += 1
                            yield json.dumps(token_chunk_data)
                        logger.info(f"teaching_script_tts output completed: {len(teaching_script_tts)} chars")
                    else:
                        # teaching_script_tts 为空或查询不到，走 LLM 转换逻辑
                        logger.info("teaching_script_tts not found, using LLM to revise for voice output")
                        system_prompt = _load_revise_prompt()
                        revise_llm = _get_revise_llm()
                        revise_messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": existing_answer}
                        ]
                        revised_answer = ""
                        async for chunk in revise_llm.astream(revise_messages):
                            token = chunk.content
                            if token:
                                revised_answer += token
                                token_chunk_data = {
                                    "id": chat_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": request.model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": token},
                                        "finish_reason": None,
                                    }],
                                }
                                chunk_sequence += 1
                                yield json.dumps(token_chunk_data)

                        logger.info("Answer revised for voice output: ")
                        logger.info(f"original={existing_answer}")
                        logger.info(f"revised_answer={revised_answer}")
                    # 标记结束
                    yield json.dumps(finish_chunk_data)
                    # Break out of workflow loop
                    break

                # Build messages for LLM
                messages = conversation_workflow.build_generation_messages(final_state)

                # Get appropriate LLM for streaming
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
                chunk_sequence += 1
                await save_stream_chunk(
                    db, chat_id, chunk_sequence, session_id, request.user_id,
                    request.employee_id, "done", finish_chunk_data,
                    final_state.get("conversation_id", "")
                )
                logger.info(f'save_stream_chunk finish_chunk_data is {finish_chunk_data}')
                # Send [DONE] marker
                yield "[DONE]"
                # Save conversation
                await conversation_workflow.save_conversation(final_state)
                # Break out of workflow loop
                break

        # If no final_state yet, use accumulated current_state
        if final_state is None:
            final_state = current_state

        # Format source attribution
        sources: dict = format_sources(
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
        logger.error(f"OpenAI stream v2 generation error | error={str(e)}", exc_info=True)

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


