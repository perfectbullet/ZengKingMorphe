"""
/v1/chat/completions 端点的聊天流式响应生成器。

本版本可针对 v1 API 的自定义行为进行修改。
"""

import os
import asyncio
import hashlib
import json
import re
import time
from datetime import datetime

from typing import AsyncGenerator, Optional, Any
from copy import deepcopy

from fastapi import Request

from app.models.schemas import OpenAIChatRequest
from app.models.database import StreamChunkModel, RawTokenModel
from app.core.logging import get_logger
from app.core.database import get_database
from app.services.conversation_service import conversation_workflow
from app.services.conversation.conversation_state import ConversationState

from app.services.revise_llm import (
    convert_formula_to_voice,
    convert_math_sentence_to_voice,
)
from app.utils.latex import normalize_latex_formulas
from app.utils.sentence_buffer import SentenceBuffer, has_latex_formula
from app.utils.think_tag_buffer import ThinkTagBuffer
from app.utils.tts_formatter import strip_markdown_for_tts
from app.utils.text_mapping import map_english_to_chinese, replace_en_math_verbs
from app.utils.common import sanitize_filename, detect_dominant_language
from app.services.rag_stream_wrapper import get_rag_stream
from app.services.math_debug_dump import (
    build_base_debug_payload,
    create_math_debug_file,
    is_math_debug_dump_enabled,
    mark_cancelled,
    mark_completed,
    mark_error,
    safe_write_math_debug,
)

logger = get_logger(__name__)
# =============================================================================
# 常量
# =============================================================================

# 服务端实际使用的模型名（覆盖客户端传入的 SERVER_MODEL）
SERVER_MODEL = os.getenv("LLM_MODEL", "")

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


def _training_rag_general_fallback_enabled() -> bool:
    return os.getenv("TRAINING_RAG_GENERAL_FALLBACK", "false").lower() in (
        "true",
        "1",
        "yes",
    )


def _training_rag_include_history() -> bool:
    return os.getenv("TRAINING_RAG_INCLUDE_HISTORY", "false").lower() in (
        "true",
        "1",
        "yes",
    )


def _training_rag_empty_message(prefer_zh_output: bool) -> str:
    if prefer_zh_output:
        return "当前工训知识库暂时没有召回到足够资料，请换一种问法，或确认知识库是否已完成导入。"
    return (
        "The training knowledge base did not retrieve enough supporting material. "
        "Please rephrase the question or check whether the knowledge base has been imported."
    )


def _short_text(text: Any, max_len: int = 400) -> str:
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len] + "..."


def _build_chunk_citations_from_chunks(
    chunks: list[dict] | None, *, max_text_len: int = 400
) -> list[dict]:
    citations: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks or []:
        if not isinstance(chunk, dict):
            continue
        file_path = str(chunk.get("file_path") or chunk.get("source") or "").strip()
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        reference_id = str(chunk.get("reference_id") or "").strip()
        dedupe_key = (file_path, chunk_id or reference_id)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        citation_source = file_path or chunk_id or reference_id
        citation = {
            "reference_id": reference_id,
            "source": citation_source,
            "file_path": file_path,
            "doc_id": chunk_id,
            "content": "",
        }
        if "rerank_score" in chunk:
            citation["rerank_score"] = chunk.get("rerank_score")
        citations.append(citation)
    return citations


def _build_chunk_citations_from_sources(
    sources: list[dict] | None, *, max_text_len: int = 400
) -> list[dict]:
    citations: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        for citation in _build_chunk_citations_from_chunks(
            source.get("chunks") or [], max_text_len=max_text_len
        ):
            dedupe_key = (
                str(citation.get("file_path") or ""),
                str(citation.get("chunk_id") or citation.get("reference_id") or ""),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            citations.append(citation)
    return citations


def _attach_citations_to_lightrag_sources(
    sources: list[dict] | None, *, max_text_len: int = 400
) -> list[dict]:
    normalized: list[dict] = []
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        item = dict(source)
        if item.get("type") == "chunk":
            citations = _build_chunk_citations_from_chunks(
                item.get("chunks") or [], max_text_len=max_text_len
            )
            item["citations"] = citations
            if citations and not item.get("text"):
                item["text"] = citations[0].get("text") or ""
            else:
                item.setdefault("text", None)
        else:
            item.setdefault("citations", [])
            item.setdefault("text", None)
        normalized.append(item)
    return normalized


# =============================================================================
# 工具函数
# =============================================================================


def _build_history_prefix_for_query(
    context_messages: list,
    prefer_zh_output: bool,
    max_turns: int = 6,
    max_chars: int = 1200,
    context_dependence: Optional[str] = None,
) -> str:
    """
    为 RAG 查询构建精简对话历史前缀，支持多轮上下文承接
    用于处理指代、追问、纠错类需求

    语言一致性过滤（双向对称）：
       “本轮输出语言只跟当前问句的语言相关”——若历史里夹杂了与本轮不同语言的
       消息（典型场景：上一轮中文路况问答 → 本轮英文提问），把这段异种语言原文
       塞给小模型（qwen3:14b 等）会显著拉偏“该用什么语言回答”的判断，导致出现
       “英文问、混合中英文答”这类污染。这里在构造历史前缀时，统一丢弃主导语言
       与 ``prefer_zh_output`` 不一致的消息；纯数字/标点等无法判定语言的消息保留。

    动态上下文记忆：
       当上游 classify_query_type 把本轮判定为 ``unrelated``（与历史无关）时，
       统一返回空字符串——RAG 检索与最终生成都不再看到任何历史前缀，等价于
       "全新问题"。这是"动态上下文记忆"规则的 RAG 路径落地点。
    """
    if context_dependence == "unrelated":
        logger.info(
            "[history_prefix] dropped due to dynamic context memory: "
            "context_dependence=unrelated"
        )
        return ""

    if not context_messages:
        return ""

    target_lang = "zh" if prefer_zh_output else "en"
    filtered: list = []
    skipped = 0
    for m in context_messages:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        msg_lang = detect_dominant_language(content)
        # 主导语言可识别且与本轮目标语言不一致 → 丢弃，防止语言污染。
        # 不可识别（纯数字/符号/空）→ 保留。
        if msg_lang and msg_lang != target_lang:
            skipped += 1
            continue
        filtered.append(m)

    if skipped > 0:
        logger.info(
            f"[history_prefix] language filter: target={target_lang}, "
            f"skipped {skipped} cross-lang msg(s), kept {len(filtered)}"
        )

    if not filtered:
        return ""

    recent = filtered[-max_turns:]
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
        role_label = (
            "User"
            if role == "user"
            else "Assistant"
            if role == "assistant"
            else role or "Message"
        )
        if prefer_zh_output:
            role_label = (
                "用户"
                if role == "user"
                else "助手"
                if role == "assistant"
                else role_label
            )
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
        'Requirement: If the user uses pronouns or follow-ups like "it/this/why/the result seems wrong/recalculate", '
        "resolve them by referring to the prior problem statement, conditions, conclusion, key variables, formulas, definitions, "
        "and your previous steps. If information is still missing, ask for the missing details.\n\n"
    )


# =============================================================================
# 来源归因
# =============================================================================


def format_sources(
    retrieved_docs: list[dict],
    web_search_results: list[dict],
    max_content_length: int = MAX_CONTENT_LENGTH,
) -> dict:
    """
    格式化 RAG 文档和联网搜索结果，用于来源归因。

    Args:
        retrieved_docs: RAG 检索到的文档块列表
        web_search_results: Tavily 联网搜索结果列表
        max_content_length: 内容摘要最大长度

    Returns:
        包含 rag_sources 和 web_sources 列表的字典
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
# 公式转语音
# =============================================================================


def _has_math_symbols_simple(text: str) -> bool:
    """
    检查文本是否包含非 LaTeX 定界符的数学符号。

    Args:
        text: 待检查文本

    Returns:
        若包含数学符号则返回 True
    """
    math_symbol_pattern = re.compile(r"[∈∉⊂⊃⊆⊇∪∩∅∨∧¬∀∃→⇒⇐⇔≡≠≤≥≈≪≫√∞²³°π∏∑∫∂∇Δ]")
    return math_symbol_pattern.search(text) is not None


async def _process_segment_for_output(
    segment: str,
    log_prefix: str = "",
    prefer_zh_output: bool = True,
    enable_math_sentence_conversion: bool = False,
) -> tuple[str, str]:
    """
    处理文本段用于输出。

    流程：
    1. 规范化 LaTeX 定界符
    2. 将 LaTeX 公式转换为语音友好文本（内部自管 LLM）
    3. 将含数学符号的句子转换为语音友好文本（内部自管 LLM）

    语音转换为辅助能力：失败时 ``logger.exception`` 并降级为展示文本，不中断主输出。

    Args:
        segment: 待处理的文本段
        log_prefix: 日志前缀

    Returns:
        (display_content, voice_content) 元组
    """
    display_content = normalize_latex_formulas(segment)
    # 中文提问时，英文结果转中文（避免“英文问中文答”）
    if prefer_zh_output:
        display_content = map_english_to_chinese(display_content)
        display_content = replace_en_math_verbs(display_content)
    try:
        if has_latex_formula(display_content):
            logger.info(
                f"[{log_prefix} 公式转换] 转换前长度={len(display_content)}, 转换前={repr(display_content)}"
            )
            voice_content = await convert_formula_to_voice(display_content)
            logger.info(
                f"[{log_prefix} 公式转换] 转换后长度={len(voice_content)}, 转换后={repr(voice_content)}"
            )
        elif enable_math_sentence_conversion and _has_math_symbols_simple(
            display_content
        ):
            logger.info(
                f"[{log_prefix} 数学句子转换] 转换前长度={len(display_content)}, 转换前={repr(display_content)}"
            )
            voice_content = await convert_math_sentence_to_voice(display_content)
            logger.info(
                f"[{log_prefix} 数学句子转换] 转换后长度={len(voice_content)}, 转换后={repr(voice_content)}"
            )
        else:
            logger.info(f"{log_prefix}没有公式: {display_content}")
            voice_content = display_content
    except asyncio.CancelledError:
        logger.info(f"[{log_prefix} 语音转换被取消]")
        raise
    except Exception:
        logger.exception(
            f"[{log_prefix} 语音转换失败，降级使用展示文本] display_content={display_content[:500]!r}"
        )
        voice_content = display_content
    # 从 voice_content 中移除 markdown 格式，用于 TTS
    # （display_content 保留原始 markdown 格式用于显示）
    logger.info(
        f"[{log_prefix} 语音voice_content markdown清理前: {repr(voice_content)}"
    )
    voice_content = strip_markdown_for_tts(voice_content)
    logger.info(
        f"[{log_prefix} 语音voice_content markdown清理后: {repr(voice_content)}"
    )
    return display_content, voice_content


def _build_token_chunk_data(
    chat_id: str,
    created: int,
    model: str,
    content: str,
) -> dict:
    """构建 SSE 输出的 token 块数据。"""
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }


async def _stream_segment_with_formula_conversion(
    segment: str,
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
    处理文本段并处理流式公式转换。

    Args:
        segment: 待处理的文本段
        chat_id: 聊天完成 ID
        created: 创建时间戳
        model: 模型名
        db: 数据库实例
        chunk_sequence: 当前块序号
        session_id: 会话 ID
        user_id: 用户 ID
        employee_id: 数字员工 ID
        conversation_id: 可选的对话 ID
        log_prefix: 日志前缀

    Returns:
        (updated_sequence, voice_chunk_data_for_yielding) 元组
    """
    display_content, voice_content = await _process_segment_for_output(
        segment,
        log_prefix,
        prefer_zh_output=prefer_zh_output,
        enable_math_sentence_conversion=enable_math_sentence_conversion,
    )

    voice_chunk_data = _build_token_chunk_data(chat_id, created, model, voice_content)

    new_sequence = chunk_sequence + 1
    await asyncio.sleep(CHUNK_SAVE_DELAY_SECONDS)
    await save_stream_chunk(
        db,
        chat_id,
        new_sequence,
        session_id,
        user_id,
        employee_id,
        "token",
        {
            **voice_chunk_data,
            "choices": [
                {
                    **voice_chunk_data["choices"][0],
                    "delta": {"content": display_content},
                }
            ],
        },
        conversation_id,
    )

    return new_sequence, voice_chunk_data


# =============================================================================
# 主流式生成器
# =============================================================================


def _extract_user_query(messages: list) -> str:
    """从消息列表中提取最后一条用户消息。"""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return messages[-1].content if messages else ""


# ConversationState 全字段的默认值表。
# 新增字段到 ConversationState TypedDict 后，必须同步更新此表；
# 启动时会自动校验，遗漏则断言失败。
_STATE_DEFAULTS: ConversationState = {
    # ── 用户/会话标识 ──
    "messages": [],
    "user_query": "",
    "user_id": "",
    "user_name": "",
    "head_url": "",
    "session_id": "",
    "employee_id": "",
    "employee_config": {},
    # ── 查询分类 ──
    "is_realtime_query": False,
    "realtime_category": "",
    "realtime_detect_reason": "",
    "intent": "",
    "entities": {},
    "classification_label": None,
    "classification_confidence": None,
    "classification_reason": None,
    "is_math_problem": False,
    "answer_mode": None,
    # ── 检索 & 联网 ──
    "retrieved_docs": [],
    "relevance_score": 0.0,
    "web_search_results": [],
    "kb_used": [],
    "web_search_used": False,
    "web_search_error": None,
    "sources": [],
    # ── 答案 & 验证 ──
    "final_answer": "",
    "confidence": 0.0,
    "direct_match": None,
    "direct_text_answer": None,
    "query_rewritten": False,
    "rewritten_query": "",
    "compressed_context": None,
    "answer_verified": False,
    "verification_result": None,
    # ── 上下文 & 敏感词 ──
    "context": {},
    "has_sensitive": False,
    "error": None,
    "faq_matched": None,
    "context_dependence": None,
    "context_dependence_reason": None,
    # ── 对话持久化 ──
    "conversation_id": "",
    # ── 性能指标 ──
    "response_time_ms": 0,
    "workflow_start_time": 0.0,
    "node_timings": {},
    "ttfb_ms": None,
    "complexity_score": 0.0,
    "complexity_reason": "",
    "target_year": None,
    # ── 流式输出配置 ──
    "streaming_type": None,
    "streaming_llm": None,
    "streaming_messages": None,
    "rag_query": None,
    "rag_mode": None,
    "rag_backend": None,
    "raganything_query": None,
    "raganything_mode": None,
    # ── 客户端附加上下文 ──
    "channel_name": None,
    "team_id": None,
    # ── 输出语言偏好 ──
    "prefer_zh_output": True,
    # ── 工训术语归一化及后置兼容状态 ──
    "industrial_term_normalized": False,
    "industrial_term_before": None,
    "industrial_term_after": None,
    "industrial_term_matches": [],
    "asr_latex_should_run": False,
    "asr_latex_converted": False,
    "query_preprocessed": False,
    "asr_latex_before": None,
    "asr_latex_after": None,
    # ── 分类 / 上下文消歧拆分后的中间状态 ──
    "raw_classification_label": None,
    "raw_classification_confidence": None,
    "raw_classification_reason": None,
    "context_resolution_mode": None,
    "context_resolution_skipped_reason": None,
    "math_context_used": False,
    "math_context_text": None,
    "effective_query": None,
    # ── 人工概念检索状态 ──
    "concept_retrieval_enabled": False,
    "concept_retrieval_hit": False,
    "concept_retrieval_reason": None,
    "concept_context": None,
    "concept_context_source": None,
}

# 启动时校验：_STATE_DEFAULTS 必须覆盖 ConversationState 的全部字段
_missing = set(ConversationState.__annotations__) - set(_STATE_DEFAULTS)
assert not _missing, f"_STATE_DEFAULTS 缺少 ConversationState 字段: {_missing}"


def _build_initial_state(
    request: OpenAIChatRequest, session_id: str, user_query: str
) -> ConversationState:
    """基于 _STATE_DEFAULTS 构建初始状态，覆盖请求相关字段。"""
    return {
        **deepcopy(_STATE_DEFAULTS),
        "user_query": user_query,
        "user_id": request.user_id,
        "session_id": session_id,
        "employee_id": request.employee_id,
        "channel_name": request.channel_name,
        "team_id": request.team_id,
        "workflow_start_time": time.time(),
    }


def _enforce_target_year_consistency(text: str, target_year: int | None) -> str:
    """统一回答年份：过滤含冲突年份的句子，兜底返回原文"""
    # 无文本/无目标年份，直接返回
    if not text or target_year is None:
        return text
    # 提取文中所有20xx年份，识别冲突年份
    years = set(re.findall(r"\b(20\d{2})\b", text))
    conflict_years = {y for y in years if int(y) != int(target_year)}
    # 无冲突直接返回
    if not conflict_years:
        return text
    # 按句子拆分，过滤含冲突年份的句子
    parts = re.split(r"([。！？!?])", text)
    rebuilt: list[str] = []
    for i in range(0, len(parts), 2):
        seg = parts[i]
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        if not seg.strip():
            continue
        # 丢弃含冲突年份的句子
        if any(y in seg for y in conflict_years):
            continue
        rebuilt.append(seg + punct)
    # 拼接结果，为空则返回原文
    cleaned = "".join(rebuilt).strip()
    return cleaned or text


def _build_finish_chunk_data(
    chat_id: str,
    created: int,
    model: str,
    user_query: str,
) -> dict:
    """构建结束块数据模板。"""
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
    """用最终状态信息更新结束块数据。"""
    normalized_sources = _attach_citations_to_lightrag_sources(sources)
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
        "sources": normalized_sources,
        "citations": _build_chunk_citations_from_sources(normalized_sources),
        "intent": final_state.get("intent", ""),
        "is_realtime_query": final_state.get("is_realtime_query", False),
        "realtime_category": final_state.get("realtime_category", ""),
        "model": model_name,
    }


# =============================================================================
# MongoDB 存储
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
    保存流式块到 MongoDB。

    Args:
        db: 数据库实例
        chat_id: 聊天完成 ID
        chunk_sequence: 块序号
        session_id: 会话 ID
        user_id: 用户 ID
        employee_id: 数字员工 ID
        chunk_type: 块类型（user_query, role, token, done, error, status）
        chunk_data: 待保存的块数据
        conversation_id: 可选的对话 ID
    """
    # 打印所有参数用于调试
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


async def save_raw_token(
    db: Any,
    chat_id: str,
    session_id: str,
    user_id: str,
    employee_id: str,
    token_text: str,
    token_index: int,
    streaming_source: str,
    conversation_id: Optional[str] = None,
) -> None:
    """保存原始流式 token 到 MongoDB。"""
    try:
        record = RawTokenModel(
            token_id=f"{chat_id}_raw_{token_index}",
            chat_id=chat_id,
            session_id=session_id,
            user_id=user_id,
            employee_id=employee_id,
            conversation_id=conversation_id,
            token_text=token_text,
            token_index=token_index,
            streaming_source=streaming_source,
        )
        await db.raw_stream_tokens.insert_one(record.model_dump())
    except Exception as e:
        logger.warning(f"Failed to save raw token: index={token_index}, error={e}")


async def generate_openai_stream_v1(
    request: OpenAIChatRequest,
    http_request: Optional[Request] = None,
) -> AsyncGenerator[str, None]:
    """
    生成 OpenAI 风格的 v1 API 流式响应。
    本版本可针对 v1 端点的自定义行为进行修改。
    Args:
        request: OpenAI 聊天请求
    Yields:
        OpenAI 格式的 SSE 消息
    """

    db = await get_database()

    session_id = (
        request.session_id
        or f"sess_{hashlib.md5(f'{request.user_id}_{datetime.now().timestamp()}'.encode()).hexdigest()[:12]}"
    )
    chat_id = f"chatcmpl-{hashlib.md5(f'{session_id}_{time.time()}'.encode()).hexdigest()[:12]}"
    created = int(time.time())

    original_user_query = _extract_user_query(request.messages)

    initial_state = _build_initial_state(request, session_id, original_user_query)

    # 输出语言偏好：默认中文，preprocess_query 节点会根据用户查询更新 state 中的值；
    # 每次事件循环更新 current_state 后同步刷新此局部变量，下游统一使用。
    prefer_zh_output = True

    sentence_buffer = SentenceBuffer(
        max_chars=SENTENCE_BUFFER_MAX_CHARS,
        max_wait_seconds=SENTENCE_BUFFER_MAX_WAIT_SECONDS,
        # comma_split_threshold=SENTENCE_BUFFER_COMMA_SPLIT_THRESHOLD
    )
    think_tag_buffer = ThinkTagBuffer()  # 用于过滤 think 标签

    # 结束块模板：usage 字段会在最终发送 done 前由 _update_finish_chunk_metadata
    # 用 final_user_query（完成前置归一化后的文本）重新计算，此处仅用原文占位。
    finish_chunk_data = _build_finish_chunk_data(
        chat_id, created, SERVER_MODEL, original_user_query
    )

    chunk_sequence = 0
    raw_token_index = 0

    # user_query chunk 延迟到 post_classification_preprocess 节点之后保存，确保
    # 返回的是完成前置归一化后的文本；若 workflow 在此前退出（敏感词 break / 异常），
    # 由收尾兜底保存原文，保证前端一定能收到 user_query chunk。
    user_query_chunk_saved = False

    role_chunk_data = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": SERVER_MODEL,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }

    async def client_disconnected() -> bool:
        """检查客户端是否已断开；http_request 缺失时视为未断开。

        长时间推理（如数学题 TIR 185s）期间客户端可能已关闭，这里主动探测，
        避免对已断开的连接继续做昂贵的后处理（boxed→语音转换 LLM 调用等）。
        """
        if http_request is None:
            return False
        try:
            return await http_request.is_disconnected()
        except Exception:
            logger.exception("[Stream] Failed to check client disconnect state")
            return False

    async def stop_if_disconnected(stage: str) -> bool:
        """客户端已断开则记录日志并返回 True，调用方据此尽早 return 结束生成器。"""
        if await client_disconnected():
            logger.info(
                f"[Stream] Client disconnected, stop streaming | stage={stage} | "
                f"chat_id={chat_id} | session_id={session_id}"
            )
            return True
        return False

    chunk_sequence += 1
    await save_stream_chunk(
        db,
        chat_id,
        chunk_sequence,
        session_id,
        request.user_id,
        request.employee_id,
        "role",
        role_chunk_data,
    )

    yield json.dumps(role_chunk_data)

    final_state = None
    current_state = initial_state.copy()
    model_name = SERVER_MODEL

    async def save_user_query_chunk_once(
        display_user_query: str, query_preprocessed: bool
    ) -> None:
        """保存 user_query chunk（仅一次）。

        display_user_query 为 post_classification_preprocess 之后的展示文本；
        original_user_message 始终保留请求中的原始用户文本；
        query_preprocessed 标识是否发生过实际预处理。
        """
        nonlocal chunk_sequence, user_query_chunk_saved

        if user_query_chunk_saved:
            return

        updated_messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]
        # 最后一条 user 消息替换为转换后的展示文本
        for msg in reversed(updated_messages):
            if msg.get("role") == "user":
                msg["content"] = display_user_query
                break

        chunk_sequence += 1
        user_query_chunk_data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": SERVER_MODEL,
            "user_message": display_user_query,
            "original_user_message": original_user_query,
            "query_preprocessed": query_preprocessed,
            "messages": updated_messages,
        }
        await save_stream_chunk(
            db,
            chat_id,
            chunk_sequence,
            session_id,
            request.user_id,
            request.employee_id,
            "user_query",
            user_query_chunk_data,
            current_state.get("conversation_id"),
        )
        user_query_chunk_saved = True

    async for event in conversation_workflow.workflow.astream(
        initial_state, stream_mode="updates"
    ):
        node_name = list(event.keys())[0] if event else None
        state_update = event.get(node_name, {}) if node_name else {}

        # 调试：打印事件结构
        logger.info(f"Event node={node_name}")

        if state_update:
            current_state.update(state_update)
            # 同步输出语言偏好（preprocess_query 节点会设置此值）
            prefer_zh_output = current_state.get("prefer_zh_output", prefer_zh_output)

        # post_classification_preprocess 完成后保存 user_query chunk（含前置术语归一化结果）。
        # 必须在 current_state.update(state_update) 之后，确保取到最终 user_query。
        # chunk 保存位于此处，保证前端拿到完成前置归一化后的 user_query。
        if node_name == "post_classification_preprocess" and not user_query_chunk_saved:
            display_user_query = current_state.get("user_query") or original_user_query
            query_preprocessed = bool(
                current_state.get("query_preprocessed")
                or display_user_query != original_user_query
            )
            await save_user_query_chunk_once(
                display_user_query=display_user_query,
                query_preprocessed=query_preprocessed,
            )

        # 检测敏感词并提前终止：命中后不再进入 preprocess_query /
        # classify_query_type / generate_answer，但仍按正常流式协议返回
        # assistant role → content → finish chunk，并保存拒答对话。
        if node_name == "input_validation" and state_update.get("has_sensitive"):
            reject_message = (
                "抱歉，您的问题包含敏感内容，请规范用语后再试。"
                if prefer_zh_output
                else "Sorry, your question contains sensitive content. Please rephrase and try again."
            )

            # 写入 final_answer 等，保证收尾 save_conversation 落库的是拒答话术，
            # 而不是空字符串。
            current_state["final_answer"] = reject_message
            current_state["direct_text_answer"] = reject_message
            current_state["confidence"] = 1.0
            current_state["has_sensitive"] = True
            current_state["streaming_type"] = "direct_text"

            # 命中敏感词时尚未经过 preprocess_query，先保存 user_query chunk（原文），
            # 保证前端先收到用户消息、再收到拒答内容，chunk 顺序与正常回答一致。
            if not user_query_chunk_saved:
                await save_user_query_chunk_once(
                    display_user_query=original_user_query,
                    query_preprocessed=False,
                )

            # 拒答 content chunk：content 与 voice_content 同文案，不进入
            # 公式转换 / MathAgentService / RAG。chunk_type 与普通回答一致用 "token"，
            # 便于前端按既有逻辑渲染；finish chunk 与 [DONE] 交给收尾段统一发送。
            content_chunk_data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": SERVER_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": reject_message,
                            "voice_content": reject_message,
                        },
                        "finish_reason": None,
                    }
                ],
            }
            chunk_sequence += 1
            await save_stream_chunk(
                db,
                chat_id,
                chunk_sequence,
                session_id,
                request.user_id,
                request.employee_id,
                "token",
                content_chunk_data,
                current_state.get("conversation_id"),
            )
            yield json.dumps(content_chunk_data)
            logger.info(
                f"Sensitive word detected | reject response emitted | "
                f"chat_id={chat_id} | session_id={session_id} | "
                f"user_id={request.user_id} | employee_id={request.employee_id}"
            )
            # 跳出 workflow 消费循环，由收尾段统一发送 finish chunk + 保存对话。
            break

        # 当到达 generate_answer 节点时，开始流式输出
        if node_name == "generate_answer":
            # 兜底：异常路径导致 post_classification_preprocess 未触发 chunk 保存时，
            # 在进入流式输出前补存一次，保证前端不会丢 user_query。正常路径不应走到这里。
            if not user_query_chunk_saved:
                display_user_query = (
                    current_state.get("user_query") or original_user_query
                )
                query_preprocessed = bool(
                    current_state.get("query_preprocessed")
                    or display_user_query != original_user_query
                )
                logger.warning(
                    f"user_query chunk fallback save before generate_answer | "
                    f"query_preprocessed={query_preprocessed} | display_user_query={display_user_query[:200]!r}"
                )
                await save_user_query_chunk_once(
                    display_user_query=display_user_query,
                    query_preprocessed=query_preprocessed,
                )

            streaming_type = current_state.get("streaming_type")

            # 调试：打印当前状态中的关键字段
            logger.info(
                f"generate_answer state | streaming_type={streaming_type} | "
                f"has_streaming_llm={current_state.get('streaming_llm') is not None} | "
                f"intent={current_state.get('intent')}"
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
                ttfb_ms = int(
                    (time.time() - initial_state["workflow_start_time"]) * 1000
                )
                current_state["ttfb_ms"] = ttfb_ms
                logger.info(
                    f"Using preset answer | existing_answer={existing_answer} | ttfb_ms={ttfb_ms}"
                )

                # 流式返回预设答案（通过 sentence_buffer 分段处理）
                enable_math_sentence_conversion = bool(
                    current_state.get("is_math_problem", False)
                )
                for char in existing_answer:
                    segment = sentence_buffer.add(char)
                    if segment:
                        (
                            chunk_sequence,
                            chunk_data,
                        ) = await _stream_segment_with_formula_conversion(
                            segment,
                            chat_id,
                            created,
                            SERVER_MODEL,
                            db,
                            chunk_sequence,
                            session_id,
                            request.user_id,
                            request.employee_id,
                            current_state.get("conversation_id"),
                            prefer_zh_output=prefer_zh_output,
                            enable_math_sentence_conversion=enable_math_sentence_conversion,
                            log_prefix="PresetText",
                        )
                        yield json.dumps(chunk_data)

                # 刷新 buffer 中剩余内容
                final_segment = await sentence_buffer.flush(is_final=True)
                if final_segment:
                    (
                        chunk_sequence,
                        chunk_data,
                    ) = await _stream_segment_with_formula_conversion(
                        final_segment.content,
                        chat_id,
                        created,
                        SERVER_MODEL,
                        db,
                        chunk_sequence,
                        session_id,
                        request.user_id,
                        request.employee_id,
                        current_state.get("conversation_id"),
                        prefer_zh_output=prefer_zh_output,
                        enable_math_sentence_conversion=enable_math_sentence_conversion,
                        log_prefix="PresetText-FinalSegment",
                    )
                    yield json.dumps(chunk_data)

                # 发送结束标记
                chunk_sequence += 1
                finish_chunk_data["metadata"]["conversation_id"] = current_state.get(
                    "conversation_id", ""
                )
                normalized_sources = _attach_citations_to_lightrag_sources(
                    current_state.get("sources", [])
                )
                finish_chunk_data["metadata"]["sources"] = normalized_sources
                finish_chunk_data["metadata"]["citations"] = (
                    _build_chunk_citations_from_sources(normalized_sources)
                )
                finish_chunk_data["metadata"]["intent"] = current_state.get(
                    "intent", ""
                )
                finish_chunk_data["metadata"]["is_realtime_query"] = current_state.get(
                    "is_realtime_query", False
                )
                finish_chunk_data["metadata"]["realtime_category"] = current_state.get(
                    "realtime_category", ""
                )
                finish_chunk_data["metadata"]["confidence"] = 0.99
                await save_stream_chunk(
                    db,
                    chat_id,
                    chunk_sequence,
                    session_id,
                    request.user_id,
                    request.employee_id,
                    "done",
                    finish_chunk_data,
                    current_state.get("conversation_id", ""),
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
            enable_math_sentence_conversion = current_state.get(
                "is_math_problem", False
            )

            chunk_sequence, chunk_data = await _stream_segment_with_formula_conversion(
                preface,
                chat_id,
                created,
                SERVER_MODEL,
                db,
                chunk_sequence,
                session_id,
                request.user_id,
                request.employee_id,
                current_state.get("conversation_id"),
                prefer_zh_output=prefer_zh_output,
                enable_math_sentence_conversion=enable_math_sentence_conversion,
                log_prefix="Preface",
            )
            yield json.dumps(chunk_data)

            if not first_token_received:
                first_token_received = True
                ttfb_ms = int(
                    (time.time() - initial_state["workflow_start_time"]) * 1000
                )
                current_state["ttfb_ms"] = ttfb_ms
                logger.info(f"First token received | ttfb_ms={ttfb_ms}")
            model_name = streaming_type or model_name
            # 根据 streaming_type 选择不同的流式输出方式
            if streaming_type in ("rag_stream", "raganything_stream"):
                base_query = (
                    current_state.get("rag_query")
                    or current_state.get("effective_query")
                    or current_state.get("rewritten_query")
                    or current_state.get("user_query")
                    or ""
                )
                legacy_raw_query = (
                    current_state.get("raganything_query")
                    or current_state.get("rewritten_query")
                    or current_state.get("user_query")
                    or ""
                )

                mode = (
                    current_state.get("rag_mode")
                    or current_state.get("raganything_mode")
                    or os.getenv("TRAINING_RAG_QUERY_MODE", "hybrid")
                )
                backend = current_state.get("rag_backend") or os.getenv(
                    "TRAINING_RAG_BACKEND", "lightrag_file"
                )
                include_history = not (
                    backend == "lightrag_file" and not _training_rag_include_history()
                )
                if include_history:
                    raw_query = (
                        legacy_raw_query if backend == "raganything" else base_query
                    )
                    context_messages = (current_state.get("context") or {}).get(
                        "messages"
                    ) or []
                    history_prefix = _build_history_prefix_for_query(
                        context_messages=context_messages,
                        prefer_zh_output=prefer_zh_output,
                        context_dependence=current_state.get("context_dependence"),
                    )
                    if prefer_zh_output:
                        lang_lead = "本轮请使用简体中文回答以下问题。\n\n"
                        lang_tail = (
                            "\n\n[语言约束] 上文“对话历史”仅作上下文参考；"
                            "本次回复必须完整使用简体中文，不要输出英文段落或中英混合句子。"
                        )
                    else:
                        lang_lead = (
                            "Please answer the following question in English only.\n\n"
                        )
                        lang_tail = (
                            "\n\n[LANGUAGE CONSTRAINT] The conversation history above is only "
                            "for context reference. Your reply MUST be written entirely in English. "
                            "Do not output any Chinese characters or mixed Chinese-English sentences."
                        )
                    query = (history_prefix or "") + lang_lead + raw_query + lang_tail
                else:
                    query = base_query
                logger.info(
                    f"Using RAG stream | backend={backend} | include_history={str(include_history).lower()} | query={query[:80]} | mode={mode}"
                )
                rag_stream_error = False
                debug_meta = {
                    "chat_id": chat_id,
                    "session_id": session_id,
                    "user_id": request.user_id,
                    "employee_id": request.employee_id,
                    "backend": backend,
                    "mode": mode,
                    "streaming_type": streaming_type,
                }

                async for chunk in get_rag_stream(
                    query=query,
                    mode=mode,
                    prefer_zh_output=prefer_zh_output,
                    debug_meta=debug_meta,
                ):
                    chunk_type = chunk.get("type")
                    content = chunk.get("content")
                    if chunk_type == "chunk":
                        if content is None:
                            continue
                        full_answer += content
                        raw_token_index += 1
                        await save_raw_token(
                            db,
                            chat_id,
                            session_id,
                            request.user_id,
                            request.employee_id,
                            content,
                            raw_token_index,
                            "rag_stream",
                            current_state.get("conversation_id"),
                        )
                        segment = sentence_buffer.add(content)
                        if segment:
                            (
                                chunk_sequence,
                                chunk_data,
                            ) = await _stream_segment_with_formula_conversion(
                                segment,
                                chat_id,
                                created,
                                SERVER_MODEL,
                                db,
                                chunk_sequence,
                                session_id,
                                request.user_id,
                                request.employee_id,
                                current_state.get("conversation_id"),
                                prefer_zh_output=prefer_zh_output,
                                enable_math_sentence_conversion=enable_math_sentence_conversion,
                                log_prefix="RAGStream",
                            )
                            yield json.dumps(chunk_data)
                    elif chunk_type == "sources_info":
                        content: dict
                        logger.info(
                            f"📊 RAG召回: backend={backend}, "
                            f"entities={content.get('entities_count', 0)}, "
                            f"relationships={content.get('relationships_count', 0)}, "
                            f"chunks={content.get('chunks_count', 0)}, "
                            f"references={content.get('references_count', 0)}, "
                            f"candidate_chunks={content.get('candidate_chunks_count')}, "
                            f"final_chunks={content.get('final_chunks_count')}"
                        )
                        if (
                            int(content.get("entities_count", 0) or 0) == 0
                            and int(content.get("chunks_count", 0) or 0) == 0
                            and int(content.get("relationships_count", 0) or 0) == 0
                        ):
                            logger.warning(
                                f"RAG sources_info is empty | backend={backend} | "
                                f"mode={content.get('mode') or mode}"
                            )
                    elif chunk_type == "sources":
                        if content.get("chunks"):
                            current_state["sources"].append(
                                {
                                    "type": "chunk",
                                    "from": "【文档块】",
                                    "backend": backend,
                                    "chunks": content["chunks"],
                                }
                            )
                    elif chunk_type == "error":
                        logger.error(
                            f"RAG stream error | backend={backend} | {chunk['content']}"
                        )
                        rag_stream_error = True

                if await stop_if_disconnected("before_rag_final_segment"):
                    return

                final_segment = await sentence_buffer.flush(is_final=True)
                if final_segment:
                    (
                        chunk_sequence,
                        chunk_data,
                    ) = await _stream_segment_with_formula_conversion(
                        final_segment.content,
                        chat_id,
                        created,
                        SERVER_MODEL,
                        db,
                        chunk_sequence,
                        session_id,
                        request.user_id,
                        request.employee_id,
                        current_state.get("conversation_id"),
                        prefer_zh_output=prefer_zh_output,
                        enable_math_sentence_conversion=enable_math_sentence_conversion,
                        log_prefix="RAGStream FinalSegment",
                    )
                    yield json.dumps(chunk_data)

                stripped_answer = (full_answer or "").strip()
                need_rag_fallback = rag_stream_error or not stripped_answer
                if need_rag_fallback and _training_rag_general_fallback_enabled():
                    logger.info(
                        "RAG → general_llm fallback triggered | "
                        f"rag_stream_error={rag_stream_error}, "
                        f"answer_length={len(stripped_answer)}"
                    )
                    try:
                        from app.services.conversation.conversation_helpers import (
                            build_generation_messages as _build_general_messages,
                        )

                        fallback_messages = _build_general_messages(current_state)
                        fallback_llm, fallback_model = (
                            conversation_workflow.get_streaming_llm(current_state)
                        )
                        async for fb_chunk in fallback_llm.astream(fallback_messages):
                            fb_token = (
                                fb_chunk.content
                                if hasattr(fb_chunk, "content")
                                else str(fb_chunk)
                            )
                            if not fb_token:
                                continue
                            full_answer += fb_token
                            raw_token_index += 1
                            await save_raw_token(
                                db,
                                chat_id,
                                session_id,
                                request.user_id,
                                request.employee_id,
                                fb_token,
                                raw_token_index,
                                "rag_fallback",
                                current_state.get("conversation_id"),
                            )
                            fb_segment = sentence_buffer.add(fb_token)
                            if fb_segment:
                                (
                                    chunk_sequence,
                                    chunk_data,
                                ) = await _stream_segment_with_formula_conversion(
                                    fb_segment,
                                    chat_id,
                                    created,
                                    SERVER_MODEL,
                                    db,
                                    chunk_sequence,
                                    session_id,
                                    request.user_id,
                                    request.employee_id,
                                    current_state.get("conversation_id"),
                                    prefer_zh_output=prefer_zh_output,
                                    enable_math_sentence_conversion=enable_math_sentence_conversion,
                                    log_prefix="RAG-Fallback-LLM",
                                )
                                yield json.dumps(chunk_data)
                        fb_final = await sentence_buffer.flush(is_final=True)
                        if fb_final:
                            (
                                chunk_sequence,
                                chunk_data,
                            ) = await _stream_segment_with_formula_conversion(
                                fb_final.content,
                                chat_id,
                                created,
                                SERVER_MODEL,
                                db,
                                chunk_sequence,
                                session_id,
                                request.user_id,
                                request.employee_id,
                                current_state.get("conversation_id"),
                                prefer_zh_output=prefer_zh_output,
                                enable_math_sentence_conversion=enable_math_sentence_conversion,
                                log_prefix="RAG-Fallback-FinalSegment",
                            )
                            yield json.dumps(chunk_data)
                        # 标记当次实际生成模型，便于 metadata 与日志统计
                        model_name = fallback_model
                        current_state["streaming_type"] = "langchain_llm_fallback"
                        logger.info(
                            f"RAG fallback completed | model={fallback_model} | "
                            f"output_chars={len(full_answer)}"
                        )
                    except Exception as fallback_exc:
                        logger.error(
                            f"RAG → general_llm fallback failed | error={fallback_exc}",
                            exc_info=True,
                        )
                elif need_rag_fallback:
                    fixed_message = _training_rag_empty_message(prefer_zh_output)
                    full_answer = fixed_message
                    raw_token_index += 1
                    await save_raw_token(
                        db,
                        chat_id,
                        session_id,
                        request.user_id,
                        request.employee_id,
                        fixed_message,
                        raw_token_index,
                        "rag_stream",
                        current_state.get("conversation_id"),
                    )
                    (
                        chunk_sequence,
                        chunk_data,
                    ) = await _stream_segment_with_formula_conversion(
                        fixed_message,
                        chat_id,
                        created,
                        SERVER_MODEL,
                        db,
                        chunk_sequence,
                        session_id,
                        request.user_id,
                        request.employee_id,
                        current_state.get("conversation_id"),
                        prefer_zh_output=prefer_zh_output,
                        enable_math_sentence_conversion=enable_math_sentence_conversion,
                        log_prefix="RAGStream EmptyFallback",
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
                    (
                        chunk_sequence,
                        chunk_data,
                    ) = await _stream_segment_with_formula_conversion(
                        direct_text,
                        chat_id,
                        created,
                        SERVER_MODEL,
                        db,
                        chunk_sequence,
                        session_id,
                        request.user_id,
                        request.employee_id,
                        current_state.get("conversation_id"),
                        enable_math_sentence_conversion=enable_math_sentence_conversion,
                        log_prefix="DirectText",
                    )
                    yield json.dumps(chunk_data)

                # 更新最终状态，保存完整答案
                final_state = current_state
                final_state["final_answer"] = full_answer

            elif streaming_type == "math_llm":
                # 数学模型推理流式输出
                streaming_llm = current_state.get("streaming_llm")
                messages = current_state.get("streaming_messages")
                # math_runtime_mode 已由 generate_answer 节点写入 state（值为
                # direct/cot/tir）；不再从 ChatOpenAI 对象 getattr 一个不存在的 mode
                # 属性，旧 ``llm`` 模式已彻底删除。
                math_runtime_mode = current_state.get("math_runtime_mode") or "direct"
                if not streaming_llm or not messages:
                    logger.error(
                        "streaming_llm or messages not configured for math_llm type"
                    )
                    continue
                _llm_base_url = getattr(
                    streaming_llm, "openai_api_base", None
                ) or getattr(streaming_llm, "base_url", "unknown")
                logger.info(
                    f"Using math LLM stream | mode={math_runtime_mode} | "
                    f"model={model_name} | base_url={_llm_base_url}"
                )

                # 使用真正的流式输出
                logger.info(
                    f"Starting streaming response with astream | type=math_llm | mode={math_runtime_mode}"
                )
                first_token_received = False
                full_answer = ""
                # ── 数学模型调试 dump：保存本次数学模型输入/输出为 JSON，便于排查 ──
                math_debug_path = None
                math_debug_payload = None
                math_output_parts: list[str] = []

                start_time = time.perf_counter()
                try:
                    if await stop_if_disconnected("before_math_llm_stream"):
                        return

                    # 创建调试 JSON（状态 running）；初始化失败不影响数学调用
                    if is_math_debug_dump_enabled():
                        try:
                            math_debug_path = create_math_debug_file(
                                user_id=request.user_id,
                                employee_id=request.employee_id,
                                session_id=session_id,
                                chat_id=chat_id,
                                runtime_mode=math_runtime_mode,
                            )
                            math_model_name = (
                                getattr(streaming_llm, "model_name", None)
                                or getattr(streaming_llm, "model", None)
                                or model_name
                            )
                            math_debug_payload = build_base_debug_payload(
                                status="running",
                                user_id=request.user_id,
                                employee_id=request.employee_id,
                                session_id=session_id,
                                chat_id=chat_id,
                                runtime_mode=math_runtime_mode,
                                model_name=math_model_name,
                                base_url=_llm_base_url,
                                user_query=current_state.get("user_query") or "",
                                messages=messages,
                                state=current_state,
                            )
                            safe_write_math_debug(math_debug_path, math_debug_payload)
                            logger.info(
                                f"[MathDebugDump] Created math debug json | path={math_debug_path}"
                            )
                        except Exception:
                            logger.exception(
                                "[MathDebugDump] Failed to initialize math debug dump"
                            )
                            math_debug_path = None
                            math_debug_payload = None

                    async for chunk in streaming_llm.astream(messages):
                        if await stop_if_disconnected("math_llm_chunk"):
                            return
                        token = (
                            chunk.content if hasattr(chunk, "content") else str(chunk)
                        )
                        if token:
                            # 保存模型原始增量（未经 think 标签过滤）用于调试
                            math_output_parts.append(token)
                            raw_token_index += 1
                            await save_raw_token(
                                db,
                                chat_id,
                                session_id,
                                request.user_id,
                                request.employee_id,
                                token,
                                raw_token_index,
                                "math_llm",
                                current_state.get("conversation_id"),
                            )
                            # 过滤 think 标签
                            # print(f'token={token!r}|', end='') # 本行日志疯狂打印，不要随意开启
                            filtered_token = think_tag_buffer.add(token)
                            if filtered_token:
                                full_answer += filtered_token
                                segment = sentence_buffer.add(filtered_token)
                                if segment:
                                    (
                                        chunk_sequence,
                                        chunk_data,
                                    ) = await _stream_segment_with_formula_conversion(
                                        segment,
                                        chat_id,
                                        created,
                                        SERVER_MODEL,
                                        db,
                                        chunk_sequence,
                                        session_id,
                                        request.user_id,
                                        request.employee_id,
                                        current_state.get("conversation_id"),
                                        prefer_zh_output=prefer_zh_output,
                                        enable_math_sentence_conversion=True,
                                        log_prefix="Math-LLM-Stream",
                                    )
                                    yield json.dumps(chunk_data)

                                    if not first_token_received:
                                        first_token_received = True
                                        ttfb_ms = int(
                                            (
                                                time.time()
                                                - initial_state["workflow_start_time"]
                                            )
                                            * 1000
                                        )
                                        current_state["ttfb_ms"] = ttfb_ms
                                        logger.info(
                                            f"First token received | ttfb_ms={ttfb_ms}"
                                        )
                except asyncio.CancelledError:
                    # 客户端断开 / 任务取消：记录 cancelled 后必须继续 raise
                    if math_debug_payload is not None:
                        math_debug_payload = mark_cancelled(
                            math_debug_payload,
                            output_content="".join(math_output_parts),
                            duration_ms=int((time.perf_counter() - start_time) * 1000),
                        )
                        safe_write_math_debug(math_debug_path, math_debug_payload)
                    raise
                except Exception as e:
                    logger.error(
                        f"Math LLM astream failed | base_url={_llm_base_url} | model={model_name} | "
                        f"error_type={type(e).__name__} | tokens_sent_so_far={raw_token_index}",
                        exc_info=True,
                    )
                    # 记录 error 调试 JSON（仍在 except 内，traceback 可取）；不吞异常
                    if math_debug_payload is not None:
                        math_debug_payload = mark_error(
                            math_debug_payload,
                            exc=e,
                            output_content="".join(math_output_parts),
                            duration_ms=int((time.perf_counter() - start_time) * 1000),
                        )
                        safe_write_math_debug(math_debug_path, math_debug_payload)
                    raise

                duration = int((time.perf_counter() - start_time) * 1000)
                logger.info(
                    f"Math-LLM done | duration={duration}ms | output_chars={len(full_answer)}"
                )

                # 数学模型正常结束：更新调试 JSON 为 completed
                if math_debug_payload is not None:
                    _math_output_content = "".join(math_output_parts)
                    math_debug_payload = mark_completed(
                        math_debug_payload,
                        output_content=_math_output_content,
                        duration_ms=duration,
                    )
                    safe_write_math_debug(math_debug_path, math_debug_payload)
                    logger.info(
                        f"[MathDebugDump] Completed math debug json | path={math_debug_path} | "
                        f"output_chars={len(_math_output_content)} | duration_ms={duration}"
                    )

                # 客户端断开后不再做最终 segment 的 boxed→语音转换（LLM 调用，开销大）
                if await stop_if_disconnected("before_math_final_segment"):
                    return

                # 刷新 buffer 中剩余内容
                final_segment = await sentence_buffer.flush(is_final=True)
                if final_segment:
                    (
                        chunk_sequence,
                        chunk_data,
                    ) = await _stream_segment_with_formula_conversion(
                        final_segment.content,
                        chat_id,
                        created,
                        SERVER_MODEL,
                        db,
                        chunk_sequence,
                        session_id,
                        request.user_id,
                        request.employee_id,
                        current_state.get("conversation_id"),
                        prefer_zh_output=prefer_zh_output,
                        enable_math_sentence_conversion=True,
                        log_prefix="Math-LLM FinalSegment",
                    )
                    yield json.dumps(chunk_data)

                final_state = current_state
                final_state["final_answer"] = full_answer

            elif streaming_type == "langchain_llm":
                # LangChain LLM 流式输出（原有逻辑）, 可以不使用话术
                streaming_llm = current_state.get("streaming_llm")
                messages = current_state.get("streaming_messages")
                if not streaming_llm or not messages:
                    logger.error(
                        "streaming_llm or messages not configured for langchain_llm type"
                    )
                    continue
                _llm_base_url = getattr(
                    streaming_llm, "openai_api_base", None
                ) or getattr(streaming_llm, "base_url", "unknown")
                logger.info(
                    f"Using LangChain LLM stream | model={model_name} | base_url={_llm_base_url}"
                )

                # 实时查询+年份锚点：先生成再清洗，避免回答出现冲突年份
                target_year = current_state.get("target_year")
                if current_state.get("is_realtime_query") and target_year is not None:
                    # 非流式生成完整回答
                    try:
                        resp = await streaming_llm.ainvoke(messages)
                    except Exception as e:
                        logger.error(
                            f"LLM ainvoke failed | base_url={_llm_base_url} | model={model_name} | error_type={type(e).__name__}",
                            exc_info=True,
                        )
                        raise
                    text = resp.content if hasattr(resp, "content") else str(resp)
                    # 年份一致性清洗
                    text = _enforce_target_year_consistency(text, target_year)
                    if text:
                        full_answer += text
                        # 分段流式输出
                        (
                            chunk_sequence,
                            chunk_data,
                        ) = await _stream_segment_with_formula_conversion(
                            text,
                            chat_id,
                            created,
                            SERVER_MODEL,
                            db,
                            chunk_sequence,
                            session_id,
                            request.user_id,
                            request.employee_id,
                            current_state.get("conversation_id"),
                            prefer_zh_output=prefer_zh_output,
                            enable_math_sentence_conversion=enable_math_sentence_conversion,
                            log_prefix="Realtime-YearAnchor",
                        )
                        yield json.dumps(chunk_data)
                    # 更新最终会话状态
                    final_state = current_state
                    final_state["final_answer"] = full_answer
                    continue

                try:
                    async for chunk in streaming_llm.astream(messages):
                        token = chunk.content
                        if token:
                            full_answer += token
                            raw_token_index += 1
                            await save_raw_token(
                                db,
                                chat_id,
                                session_id,
                                request.user_id,
                                request.employee_id,
                                token,
                                raw_token_index,
                                "langchain_llm",
                                current_state.get("conversation_id"),
                            )
                            segment = sentence_buffer.add(token)
                            token_len = len(token)
                            buffer_len = sentence_buffer.get_buffer_length()
                            if segment or ("$$" in token[:10]):
                                logger.info(
                                    f"[STREAMING] token_len={token_len}, buffer_len={buffer_len}, "
                                    f"has_segment={bool(segment)}, token_preview={repr(token[:50])}, "
                                    f"buffer_start={repr(sentence_buffer.buffer[:30])}, buffer_end={repr(sentence_buffer.buffer[-30:])}"
                                )
                            if segment:
                                (
                                    chunk_sequence,
                                    chunk_data,
                                ) = await _stream_segment_with_formula_conversion(
                                    segment,
                                    chat_id,
                                    created,
                                    SERVER_MODEL,
                                    db,
                                    chunk_sequence,
                                    session_id,
                                    request.user_id,
                                    request.employee_id,
                                    current_state.get("conversation_id"),
                                    prefer_zh_output=prefer_zh_output,
                                    enable_math_sentence_conversion=enable_math_sentence_conversion,
                                    log_prefix="",
                                )
                                yield json.dumps(chunk_data)
                except Exception as e:
                    logger.error(
                        f"LLM astream failed | base_url={_llm_base_url} | model={model_name} | "
                        f"error_type={type(e).__name__} | tokens_sent_so_far={raw_token_index}",
                        exc_info=True,
                    )
                    raise
                if await stop_if_disconnected("before_langchain_final_segment"):
                    return
                final_segment = await sentence_buffer.flush(is_final=True)
                if final_segment:
                    (
                        chunk_sequence,
                        chunk_data,
                    ) = await _stream_segment_with_formula_conversion(
                        final_segment.content,
                        chat_id,
                        created,
                        SERVER_MODEL,
                        db,
                        chunk_sequence,
                        session_id,
                        request.user_id,
                        request.employee_id,
                        current_state.get("conversation_id"),
                        prefer_zh_output=prefer_zh_output,
                        enable_math_sentence_conversion=enable_math_sentence_conversion,
                        log_prefix="FinalSegment",
                    )
                    yield json.dumps(chunk_data)

                final_state = current_state
                final_state["final_answer"] = full_answer

    # 收尾段统一在 try 内执行：客户端断开 / 任务取消时抛出 CancelledError，
    # 这里单独捕获并记录后重新抛出——客户端断开是正常路径，不应记为错误，
    # 也不应继续向已关闭的连接推送 done chunk。
    try:
        if final_state is None:
            final_state = current_state

        # 兜底：workflow 在 preprocess_query 之前退出（敏感词 break / 异常）时，
        # 前端仍能收到 user_query chunk，保存原始输入。
        if not user_query_chunk_saved:
            await save_user_query_chunk_once(
                display_user_query=original_user_query,
                query_preprocessed=False,
            )

        sources = final_state.get("sources", [])

        final_user_query = current_state.get("user_query") or original_user_query
        _update_finish_chunk_metadata(
            finish_chunk_data, final_state, final_user_query, model_name, sources
        )

        finish_chunk_data.setdefault("metadata", {})["think_timing"] = (
            think_tag_buffer.get_timing_metadata()
        )

        chunk_sequence += 1
        await save_stream_chunk(
            db,
            chat_id,
            chunk_sequence,
            session_id,
            request.user_id,
            request.employee_id,
            "done",
            finish_chunk_data,
            final_state.get("conversation_id", ""),
        )

        yield json.dumps(finish_chunk_data)

        # 保存 final_state 用于调试
        save_dir = "finish_chunk_data"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = sanitize_filename(final_user_query)
        filename = f"{timestamp}_{session_id}_{safe_query}_finish_chunk_data.json"
        filepath = os.path.join(save_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(finish_chunk_data, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"[调试] 保存 conversation_state 到 {filepath}")

        # 流式结束保存会话，保证上下文记忆
        try:
            await conversation_workflow.save_conversation(final_state)
        except Exception as e:
            logger.error(
                f"Failed to persist streaming conversation at end: {e}", exc_info=True
            )

        yield "[DONE]"
    except asyncio.CancelledError:
        logger.info(
            f"[Stream] Streaming task cancelled | chat_id={chat_id} | session_id={session_id}"
        )
        raise
