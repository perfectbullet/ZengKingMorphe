"""Small shared helpers for the common LLM conversation path."""

import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import settings
from app.core.logging import get_logger
from app.services.conversation.conversation_state import ConversationState
from app.utils.common import detect_dominant_language

logger = get_logger(__name__)


def clean_user_query(text: str) -> str:
    return re.sub(r"^[，。！？、；：,.?!;:\s]+", "", text or "").lstrip()


def prefer_zh_output(user_query: str) -> bool:
    if re.search(r"[一-鿿]", user_query or ""):
        return True
    return not bool(re.search(r"[A-Za-z]", user_query or ""))


def resolve_prefer_zh_output(state: ConversationState) -> bool:
    explicit = state.get("prefer_zh_output")
    if explicit is not None:
        return bool(explicit)
    return detect_dominant_language(state.get("user_query") or "") != "en"


def resolve_target_year_from_query(query: str, now: datetime | None = None) -> int | None:
    now = now or datetime.now()
    text = (query or "").lower()
    if re.search(r"(今年|本年|this\s+year)", text):
        return now.year
    if re.search(r"(明年|下一年|next\s+year)", text):
        return now.year + 1
    if re.search(r"(去年|上一年|last\s+year)", text):
        return now.year - 1
    return None


@asynccontextmanager
async def time_node(node_name: str, state: ConversationState, llm_instance=None):
    started = time.time()
    try:
        yield
    finally:
        state.setdefault("node_timings", {})[node_name] = int((time.time() - started) * 1000)


def select_llm(state: ConversationState, local_llm, remote_llm) -> Tuple[Any, str]:
    """Select a normal general-purpose LLM; no intent-specific model exists."""
    model_name = os.getenv("LLM_MODEL") or getattr(settings, "ollama_model", "")
    routing_mode = getattr(settings, "llm_routing_mode", "local_only")
    if routing_mode == "remote_only":
        return remote_llm, model_name
    return local_llm, model_name


def _history(state: ConversationState) -> list:
    messages = []
    for item in (state.get("context") or {}).get("messages") or []:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if item.get("role") == "assistant":
            messages.append(AIMessage(content=content))
        elif item.get("role") == "user":
            messages.append(HumanMessage(content=content))
    return messages[-10:]


def _web_context(results: list[dict]) -> str:
    blocks = []
    for result in results[:5]:
        if not isinstance(result, dict):
            continue
        title = str(result.get("title") or "")
        text = str(result.get("content") or result.get("snippet") or "")
        if title or text:
            blocks.append(f"{title}\n{text}".strip())
    return "\n\n".join(blocks)


def build_generation_messages(state: ConversationState) -> List:
    """Build one general LLM prompt for greeting, realtime and general queries."""
    config = state.get("employee_config") or {}
    query = (state.get("rewritten_query") or state.get("user_query") or "").strip()
    chinese = resolve_prefer_zh_output(state)
    name = config.get("name") or ("AI助手" if chinese else "AI assistant")
    role = config.get("role") or ("AI助手" if chinese else "helpful assistant")
    description = config.get("description") or ""
    if chinese:
        system = f"你是{name}，角色是{role}。{description}\n请使用简体中文回答。"
        if state.get("classification_label") == "noise":
            system += "\n当前输入可能不完整；若无法判断用户意图，请简洁请用户补充信息，不要猜测麦克风或语音识别故障。"
        if state.get("is_realtime_query"):
            if state.get("web_search_used"):
                system += "\n这是实时问题。只能依据下方联网检索结果回答，不要编造当前数据。"
            else:
                system += "\n这是实时问题，但当前联网检索失败或没有结果。不得编造当前数据；请简洁说明暂时无法确认。"
    else:
        system = f"You are {name}, a {role}. {description}\nReply in English."
        if state.get("classification_label") == "noise":
            system += "\nThe input may be incomplete. Ask for clarification concisely without claiming a microphone or ASR failure."
        if state.get("is_realtime_query"):
            system += (
                "\nThis is a real-time query. Use only the web context below."
                if state.get("web_search_used")
                else "\nThis is a real-time query but web retrieval failed or returned no result. Do not invent current data; say it cannot be confirmed."
            )
    if state.get("is_realtime_query") and state.get("web_search_used"):
        system += "\n\nWeb context:\n" + _web_context(state.get("web_search_results") or [])
    return [SystemMessage(content=system), *_history(state), HumanMessage(content=query)]
