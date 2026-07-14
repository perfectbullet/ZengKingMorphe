"""OpenAI-compatible SSE endpoint with one general LLM streaming executor."""

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Optional

from fastapi import Request

from app.core.database import get_database
from app.core.logging import get_logger
from app.models.database import RawTokenModel, StreamChunkModel
from app.models.schemas import OpenAIChatRequest
from app.services.conversation.conversation_state import ConversationState
from app.services.conversation_service import conversation_workflow
from app.services.revise_llm import convert_formula_to_voice
from app.utils.latex import normalize_latex_formulas
from app.utils.sentence_buffer import SentenceBuffer, has_latex_formula
from app.utils.think_tag_buffer import ThinkTagBuffer
from app.utils.tts_formatter import strip_markdown_for_tts

logger = get_logger(__name__)
SERVER_MODEL = os.getenv("LLM_MODEL", "")
CHUNK_SAVE_DELAY_SECONDS = 0.01


def _extract_user_query(messages: list) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return messages[-1].content if messages else ""


_STATE_DEFAULTS: ConversationState = {
    "messages": [], "user_query": "", "user_id": "", "user_name": "", "head_url": "",
    "session_id": "", "employee_id": "", "employee_config": {}, "is_realtime_query": False,
    "realtime_category": "", "realtime_detect_reason": "", "intent": "", "entities": {},
    "web_search_results": [], "web_search_used": False, "web_search_error": None,
    "final_answer": "", "confidence": 0.0, "context": {}, "has_sensitive": False,
    "error": None, "faq_matched": None, "conversation_id": "", "response_time_ms": 0,
    "workflow_start_time": 0.0, "node_timings": {}, "ttfb_ms": None, "rewritten_query": "",
    "query_rewritten": False, "answer_verified": False, "verification_result": None,
    "channel_name": None, "team_id": None, "streaming_llm": None, "streaming_messages": None,
    "sources": [], "classification_label": None, "classification_confidence": None,
    "classification_reason": None, "target_year": None, "context_dependence": None,
    "context_dependence_reason": None, "prefer_zh_output": True,
    "raw_classification_label": None, "raw_classification_confidence": None,
    "raw_classification_reason": None, "context_resolution_mode": None,
    "context_resolution_skipped_reason": None, "effective_query": None,
}
_missing = set(ConversationState.__annotations__) - set(_STATE_DEFAULTS)
assert not _missing, f"_STATE_DEFAULTS 缺少 ConversationState 字段: {_missing}"


def _build_initial_state(request: OpenAIChatRequest, session_id: str, user_query: str) -> ConversationState:
    return {**_STATE_DEFAULTS, "user_query": user_query, "user_id": request.user_id,
            "session_id": session_id, "employee_id": request.employee_id,
            "channel_name": request.channel_name, "team_id": request.team_id,
            "workflow_start_time": time.time()}


def _chunk(chat_id: str, created: int, model: str, content: str = "", finish_reason=None) -> dict:
    return {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": ({"content": content} if content else {}),
                         "finish_reason": finish_reason}]}


def _finish_chunk(chat_id: str, created: int, model: str, state: ConversationState, query: str) -> dict:
    answer = state.get("final_answer", "")
    payload = _chunk(chat_id, created, model, finish_reason="stop")
    payload["usage"] = {"prompt_tokens": len(query), "completion_tokens": len(answer),
                        "total_tokens": len(query) + len(answer)}
    payload["metadata"] = {"conversation_id": state.get("conversation_id", ""),
                           "confidence": state.get("confidence", 0.0),
                           "web_search_used": state.get("web_search_used", False),
                           "sources": state.get("sources", []), "intent": state.get("intent", ""),
                           "is_realtime_query": state.get("is_realtime_query", False),
                           "realtime_category": state.get("realtime_category", ""), "model": model}
    return payload


async def save_stream_chunk(db: Any, chat_id: str, sequence: int, session_id: str, user_id: str,
                            employee_id: str, chunk_type: str, chunk_data: dict,
                            conversation_id: Optional[str] = None) -> None:
    record = StreamChunkModel(chunk_id=f"{chat_id}_chunk_{sequence}", conversation_id=conversation_id,
                              session_id=session_id, user_id=user_id, employee_id=employee_id, chat_id=chat_id,
                              chunk_type=chunk_type, chunk_data=chunk_data, sequence=sequence,
                              timestamp=datetime.utcnow(), created_at=datetime.utcnow())
    await db.stream_chunks.insert_one(record.model_dump())


async def save_raw_token(db: Any, chat_id: str, session_id: str, user_id: str, employee_id: str,
                         token: str, index: int, conversation_id: Optional[str] = None) -> None:
    try:
        record = RawTokenModel(token_id=f"{chat_id}_raw_{index}", chat_id=chat_id, session_id=session_id,
                               user_id=user_id, employee_id=employee_id, conversation_id=conversation_id,
                               token_text=token, token_index=index, streaming_source="general_llm")
        await db.raw_stream_tokens.insert_one(record.model_dump())
    except Exception as exc:
        logger.warning("Failed to save raw token: %s", exc)


async def _segment_for_output(segment: str) -> tuple[str, str]:
    display = normalize_latex_formulas(segment)
    try:
        voice = await convert_formula_to_voice(display) if has_latex_formula(display) else display
    except Exception:
        logger.warning("Formula voice conversion failed", exc_info=True)
        voice = display
    return display, strip_markdown_for_tts(voice)


async def _emit_segment(segment: str, *, chat_id: str, created: int, model: str, db: Any,
                        sequence: int, request: OpenAIChatRequest, session_id: str,
                        conversation_id: Optional[str]) -> tuple[int, str]:
    display, voice = await _segment_for_output(segment)
    sequence += 1
    await asyncio.sleep(CHUNK_SAVE_DELAY_SECONDS)
    persisted = _chunk(chat_id, created, model, display)
    await save_stream_chunk(db, chat_id, sequence, session_id, request.user_id, request.employee_id,
                            "token", persisted, conversation_id)
    emitted = _chunk(chat_id, created, model, display)
    emitted["choices"][0]["delta"]["voice_content"] = voice
    return sequence, json.dumps(emitted, ensure_ascii=False)


def _safety_text(state: ConversationState) -> str:
    rules = (state.get("employee_config") or {}).get("safe_rule") or {}
    return rules.get("reject_answer") or "抱歉，您的问题包含敏感内容，请规范用语后再试。"


async def emit_safety_rejection(*, state: ConversationState, chat_id: str, created: int, model: str,
                                db: Any, sequence: int, request: OpenAIChatRequest,
                                session_id: str) -> tuple[int, str]:
    text = _safety_text(state)
    state["final_answer"] = text
    sequence, event = await _emit_segment(text, chat_id=chat_id, created=created, model=model, db=db,
                                          sequence=sequence, request=request, session_id=session_id,
                                          conversation_id=state.get("conversation_id"))
    return sequence, event


async def _stream_llm_response(*, streaming_llm: Any, messages: list, state: ConversationState,
                               chat_id: str, created: int, model: str, db: Any,
                               sequence: int, raw_index: int, request: OpenAIChatRequest,
                               session_id: str, http_request: Optional[Request]) -> AsyncGenerator[tuple[int, int, str], None]:
    """The sole normal LLM token loop for this endpoint."""
    if streaming_llm is None or not messages:
        raise RuntimeError("General streaming LLM or messages were not configured")
    buffer = SentenceBuffer(max_chars=100, max_wait_seconds=1)
    think = ThinkTagBuffer()
    visible_parts: list[str] = []
    first_visible = False
    async for chunk in streaming_llm.astream(messages):
        if http_request is not None and await http_request.is_disconnected():
            raise asyncio.CancelledError()
        token = getattr(chunk, "content", None) or str(chunk)
        if not token:
            continue
        raw_index += 1
        await save_raw_token(db, chat_id, session_id, request.user_id, request.employee_id, token, raw_index,
                             state.get("conversation_id"))
        visible = think.add(token)
        if not visible:
            continue
        if not first_visible:
            state["ttfb_ms"] = int((time.time() - state["workflow_start_time"]) * 1000)
            first_visible = True
        visible_parts.append(visible)
        segment = buffer.add(visible)
        if segment:
            sequence, event = await _emit_segment(segment.content, chat_id=chat_id, created=created, model=model,
                                                  db=db, sequence=sequence, request=request, session_id=session_id,
                                                  conversation_id=state.get("conversation_id"))
            yield sequence, raw_index, event
    trailing = think.flush()
    if trailing:
        visible_parts.append(trailing)
        buffer.add(trailing)
    final_segment = await buffer.flush(is_final=True)
    if final_segment:
        sequence, event = await _emit_segment(final_segment.content, chat_id=chat_id, created=created, model=model,
                                              db=db, sequence=sequence, request=request, session_id=session_id,
                                              conversation_id=state.get("conversation_id"))
        yield sequence, raw_index, event
    state["final_answer"] = "".join(visible_parts)


async def generate_openai_stream_v1(request: OpenAIChatRequest, http_request: Optional[Request] = None) -> AsyncGenerator[str, None]:
    db = await get_database()
    session_id = request.session_id or f"sess_{hashlib.md5(f'{request.user_id}_{time.time()}'.encode()).hexdigest()[:12]}"
    chat_id = f"chatcmpl-{hashlib.md5(f'{session_id}_{time.time()}'.encode()).hexdigest()[:12]}"
    created = int(time.time())
    original_query = _extract_user_query(request.messages)
    state = _build_initial_state(request, session_id, original_query)
    sequence = 0
    raw_index = 0
    user_saved = False
    final_state = state
    model_name = SERVER_MODEL

    async def save_user_query_once() -> None:
        nonlocal sequence, user_saved
        if user_saved:
            return
        sequence += 1
        payload = _chunk(chat_id, created, model_name, state.get("user_query") or original_query)
        await save_stream_chunk(db, chat_id, sequence, session_id, request.user_id, request.employee_id,
                                "user_query", payload, state.get("conversation_id"))
        user_saved = True

    role_payload = _chunk(chat_id, created, model_name)
    role_payload["choices"][0]["delta"] = {"role": "assistant"}
    sequence += 1
    await save_stream_chunk(db, chat_id, sequence, session_id, request.user_id, request.employee_id, "role",
                            role_payload, None)
    yield json.dumps(role_payload, ensure_ascii=False)

    try:
        async for event in conversation_workflow.workflow.astream(state, stream_mode="updates"):
            if not isinstance(event, dict):
                continue
            for node_name, update in event.items():
                if isinstance(update, dict):
                    state.update(update)
                    final_state = state
                if node_name == "preprocess_query":
                    await save_user_query_once()
                if state.get("has_sensitive"):
                    await save_user_query_once()
                    sequence, safety_event = await emit_safety_rejection(
                        state=state, chat_id=chat_id, created=created, model=model_name, db=db,
                        sequence=sequence, request=request, session_id=session_id,
                    )
                    yield safety_event
                    break
                if node_name == "generate_answer":
                    llm = state.get("streaming_llm")
                    messages = state.get("streaming_messages")
                    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or SERVER_MODEL
                    async for sequence, raw_index, output in _stream_llm_response(
                        streaming_llm=llm, messages=messages, state=state, chat_id=chat_id, created=created,
                        model=model_name, db=db, sequence=sequence, raw_index=raw_index, request=request,
                        session_id=session_id, http_request=http_request,
                    ):
                        yield output
                    break
            if state.get("final_answer"):
                break
        if not user_saved:
            await save_user_query_once()
        try:
            # The endpoint consumes the workflow only through generate_answer so that
            # it can own the single token loop. Persist after that loop, before the
            # final SSE chunk, so the client receives the conversation id as well.
            await conversation_workflow.save_conversation(final_state)
        except Exception:
            logger.error("Failed to persist streaming conversation", exc_info=True)
        finish = _finish_chunk(chat_id, created, model_name, final_state, state.get("user_query") or original_query)
        sequence += 1
        await save_stream_chunk(db, chat_id, sequence, session_id, request.user_id, request.employee_id, "done",
                                finish, final_state.get("conversation_id"))
        yield json.dumps(finish, ensure_ascii=False)
        yield "[DONE]"
    except asyncio.CancelledError:
        logger.info("Streaming cancelled | chat_id=%s", chat_id)
        raise
    except Exception as exc:
        logger.error("Streaming failed: %s", exc, exc_info=True)
        error = _chunk(chat_id, created, model_name)
        error["choices"][0]["delta"] = {"error": "生成回答时发生错误"}
        sequence += 1
        await save_stream_chunk(db, chat_id, sequence, session_id, request.user_id, request.employee_id, "error",
                                error, final_state.get("conversation_id"))
        yield json.dumps(error, ensure_ascii=False)
