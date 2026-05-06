"""
Helper utilities for conversation workflow.

This module provides:
- Node timing utilities
- LLM selection logic for hybrid routing
- Message building helpers
- Personality description helpers
"""

import time
from datetime import datetime
import re
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Tuple

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.core.config import settings
from app.core.logging import get_logger
from app.services.conversation.conversation_state import ConversationState
from app.utils.common import detect_dominant_language
from prompts.prompts import GEOMETRY_FORMULA_BOOK, MATH_SYSTEM_PROMPT, PHI4_SIMPLE_SYSTEM_PROMPT
logger = get_logger(__name__)


def _build_realtime_temporal_guardrail(prefer_zh_output: bool) -> str:
    """通用时间一致性约束：统一时间锚点，避免混入冲突年份/时间线。"""
    now = datetime.now()
    if prefer_zh_output:
        return (
            f"时间一致性要求：当前系统日期为 {now.year}年{now.month}月{now.day}日。\n"
            "先将用户问题中的时间指代映射到同一时间框架；最终结论只能对应一个目标时间，"
            "不得混入其它年份或其它时间线。若检索证据时间冲突，优先采用与目标时间一致的证据；"
            "若仍无法唯一确认，请明确说明不确定。"
        )
    return (
        f"Temporal consistency: current system date is {now.strftime('%Y-%m-%d')}.\n"
        "Resolve time references in one coherent timeline and provide one final answer aligned to a single target time."
        " Do not mix other years/timelines. If evidence conflicts by time, prioritize target-time-consistent evidence;"
        " if still ambiguous, state uncertainty clearly."
    )


def _build_system_time_context(prefer_zh_output: bool) -> str:
    """注入当前系统时间，用于正确解析「今年/今天」等相对时间"""
    now = datetime.now()
    if prefer_zh_output:
        return (
            "系统时间上下文：\n"
            f"- 当前本地日期时间：{now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- 当前年份：{now.year}\n"
            "当用户使用“今年/明年/去年/现在/今天”等相对时间表达时，默认基于上述系统时间解释，"
            "除非用户明确指定了其他时间。"
        )
    return (
        "System time context:\n"
        f"- Current local datetime: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- Current year: {now.year}\n"
        "When the user uses relative time expressions (e.g. this year/next year/today/now), "
        "interpret them using the system time above unless the user explicitly specifies otherwise."
    )


def resolve_target_year_from_query(query: str, now: datetime | None = None) -> int | None:
    """从问题中解析目标年份：今年/明年/去年 → 对应数字年份"""
    if now is None:
        now = datetime.now()
    q = (query or "").strip().lower()
    if not q:
        return None
    if re.search(r"(今年|本年|this\s+year)", q, re.IGNORECASE):
        return now.year
    if re.search(r"(明年|下一年|next\s+year)", q, re.IGNORECASE):
        return now.year + 1
    if re.search(r"(去年|上一年|last\s+year)", q, re.IGNORECASE):
        return now.year - 1
    return None


# =============================================================================
# Node Timing Utility
# =============================================================================
@asynccontextmanager
async def time_node(node_name: str, state: ConversationState, llm_instance=None):
    """
    Async context manager for timing node execution.

    Tracks execution time for each workflow node and stores in state for analysis.
    Timing data is logged and included in final conversation record.

    Args:
        node_name: Name of the workflow node
        state: Conversation state object
        llm_instance: The workflow instance (for accessing as a method if needed)

    Example:
        async with time_node("knowledge_retrieval", state):
            # retrieval logic here
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration_ms = int((time.time() - start_time) * 1000)

        if "node_timings" not in state:
            state["node_timings"] = {}
        state["node_timings"][node_name] = duration_ms

        logger.info(f"[TIMING] {node_name} - {duration_ms}ms")


# =============================================================================
# LLM Selection Utility
# =============================================================================
def select_llm(state: ConversationState, local_llm, remote_llm) -> Tuple[Any, str]:
    """
    Select appropriate LLM based on query context (hybrid mode only).

    Selection logic (hybrid mode):
    1. Primary: complexity_score (0-10)
       - 0-6分: 使用本地 Ollama (简单到中等复杂)
       - 7-10分: 使用外部 API (高复杂度)
    2. Special cases:
       - greeting, FAQ matched: 强制使用本地模型

    Args:
        state: Current conversation state
        local_llm: Local Ollama LLM instance
        remote_llm: Remote OpenAI-style LLM instance

    Returns:
        Tuple of (llm, model_name)
    """
    routing_mode = getattr(settings, "llm_routing_mode", "local_only")

    # Non-hybrid modes: return pre-configured LLM
    if routing_mode == "local_only":
        return local_llm, settings.ollama_model
    elif routing_mode == "remote_only":
        return remote_llm, settings.openai_model

    # Hybrid mode: dynamic selection based on complexity score
    complexity_score = state.get("complexity_score", 3.0)
    complexity_reason = state.get("complexity_reason", "unknown")
    is_greeting = state.get("intent") == "greeting"
    faq_matched = state.get("faq_matched")

    # 复杂度阈值：超过此分数使用外部模型
    complexity_threshold = getattr(settings, "complexity_threshold", 7.0)

    # 决策逻辑
    use_remote = False
    reason = []

    # 特殊情况：问候语和FAQ命中强制使用本地模型
    if is_greeting:
        reason = ["greeting"]
    elif faq_matched:
        reason = ["faq_matched"]
    # 主要决策：基于复杂度分数
    elif complexity_score >= complexity_threshold:
        use_remote = True
        reason = [f"complexity_{complexity_score:.1f}", complexity_reason]
    else:
        reason = [f"complexity_{complexity_score:.1f}", complexity_reason]

    model_name = settings.openai_model if use_remote else settings.ollama_model
    logger.info(
        "LLM selection",
        routing_mode="hybrid",
        selected=model_name,
        complexity_score=complexity_score,
        complexity_reason=complexity_reason,
        reason=reason,
        threshold=complexity_threshold,
    )

    if use_remote:
        return remote_llm, settings.openai_model
    return local_llm, settings.ollama_model


# =============================================================================
# Personality Helpers
# =============================================================================
#: 人格描述：以 "key" 为枚举主键，按语言提供本地化文案。
#: 新增语种只需在每个 map 的 key 下加一个语言代码 → 文案；不需要改任何业务代码。
_PERSONALITY_TONE_I18N: Dict[str, Dict[str, str]] = {
    "professional": {"zh": "专业严谨", "en": "professional and rigorous"},
    "friendly": {"zh": "友好亲切", "en": "friendly and warm"},
    "formal": {"zh": "正式庄重", "en": "formal and dignified"},
    "casual": {"zh": "轻松随意", "en": "casual and relaxed"},
}

_PERSONALITY_STYLE_I18N: Dict[str, Dict[str, str]] = {
    "concise": {"zh": "简明扼要", "en": "concise and to the point"},
    "detailed": {"zh": "详细周到", "en": "detailed and thorough"},
    "conversational": {"zh": "对话式", "en": "conversational"},
    "instructional": {"zh": "指导式", "en": "instructional"},
}

_PERSONALITY_FORMALITY_I18N: Dict[str, Dict[str, str]] = {
    "high": {"zh": "高度正式（使用敬语）", "en": "highly formal (use honorifics)"},
    "moderate": {"zh": "适度正式", "en": "moderately formal"},
    "low": {"zh": "轻松口语化", "en": "casual and colloquial"},
}

_PERSONALITY_DEFAULT_I18N: Dict[str, Dict[str, str]] = {
    "tone": {"zh": "专业", "en": "professional"},
    "style": {"zh": "友好", "en": "friendly"},
    "formality": {"zh": "适度", "en": "moderate"},
}


def _i18n_lookup(
    table: Dict[str, Dict[str, str]],
    key: str,
    lang: str,
    short_default: str,
) -> str:
    """
    在 i18n 映射表中按 ``key + lang`` 查询文案。

    严格保留旧实现的语义：
    - ``key`` 命中：返回该项的 ``lang`` 文案；
    - ``key`` 未命中：返回简短默认词 ``short_default``（不再二次查表，避免把
      "unknown" 静默升级为 "professional and rigorous" 这类长描述）。

    若映射表里某项缺失目标语言，再退回到中文 / 任意可用文案，保证不抛异常。
    """
    entry = table.get(key)
    if entry is None:
        return short_default
    return entry.get(lang) or entry.get("zh") or next(iter(entry.values()), short_default)


def get_personality_description(
    personality: dict, prefer_zh_output: bool = True
) -> Tuple[str, str, str]:
    """
    Get personality description for system prompt.

    Args:
        personality: Personality dict from employee config
        prefer_zh_output: 输出语言偏好。``True`` 返回中文文案、``False`` 返回英文文案。
            缺省为 ``True`` 以保持对老调用方的向后兼容（默认中文）。

    Returns:
        Tuple of (tone_desc, style_desc, formality_desc)，文案语言与 ``prefer_zh_output`` 一致。

    设计说明：
        - 为何要本地化？人格描述会被拼进 system prompt。若英文 system prompt 中混入
          "Tone: 友好亲切" 这类中文片段，部分中文偏好的模型（如 qwen 系列）会被
          诱导用中文回答英文问题。本地化后整个 system prompt 语言统一，从源头消除
          这类"语言污染"。
        - 为何用 i18n 映射表？保持原始 key 仍是英文枚举（"professional" 等），文案
          按需扩展任意语种，业务代码不需要任何 if/else。
        - 行为对齐旧版：未知 personality 枚举值时回退到简短默认词，与旧实现等价。
    """
    lang = "zh" if prefer_zh_output else "en"
    tone_key = personality.get("tone", "professional")
    style_key = personality.get("style", "friendly")
    formality_key = personality.get("formality", "moderate")

    tone_desc = _i18n_lookup(
        _PERSONALITY_TONE_I18N, tone_key, lang, _PERSONALITY_DEFAULT_I18N["tone"][lang]
    )
    style_desc = _i18n_lookup(
        _PERSONALITY_STYLE_I18N, style_key, lang, _PERSONALITY_DEFAULT_I18N["style"][lang]
    )
    formality_desc = _i18n_lookup(
        _PERSONALITY_FORMALITY_I18N, formality_key, lang, _PERSONALITY_DEFAULT_I18N["formality"][lang]
    )

    return tone_desc, style_desc, formality_desc


# =============================================================================
# Context Building Helpers
# =============================================================================
#: 上下文段落标签的多语言文案（仅用于 LLM 提示，不外暴露给用户）。
_CONTEXT_LABELS_I18N: Dict[str, Dict[str, str]] = {
    "compressed_header": {
        "zh": "[压缩后的参考信息]",
        "en": "[Compressed reference]",
    },
    "kb_header": {"zh": "[知识库参考{i}]", "en": "[KB reference {i}]"},
    "web_header": {"zh": "[网络资料{i}]", "en": "[Web result {i}]"},
    "web_title": {"zh": "标题:", "en": "Title:"},
    "web_content": {"zh": "内容:", "en": "Content:"},
    "web_source": {"zh": "来源:", "en": "Source:"},
    "empty": {"zh": "（暂无相关参考资料）", "en": "(No relevant reference available.)"},
}


def build_context_text(state: ConversationState) -> str:
    """
    Build context text from retrieved docs and web search results.

    Uses compressed context if available, otherwise builds from sources.

    Labels (e.g. ``[KB reference 1]`` / ``Title:``) are localized to
    ``prefer_zh_output``，避免英文 system prompt 中混入中文标签污染输出语言。

    Args:
        state: Current conversation state

    Returns:
        Formatted context string for LLM prompt
    """
    lang = "zh" if state.get("prefer_zh_output", True) else "en"
    L = {k: v[lang] for k, v in _CONTEXT_LABELS_I18N.items()}

    if state.get("compressed_context"):
        return f"{L['compressed_header']}\n{state['compressed_context']}"

    context_parts = []
    for i, doc in enumerate(state.get("retrieved_docs", [])[:3], 1):
        context_parts.append(f"{L['kb_header'].format(i=i)}\n{doc.get('content', '')[:500]}")

    web_results = state.get("web_search_results", [])
    if web_results and state.get("web_search_used", False):
        for i, web_result in enumerate(web_results[:3], 1):
            content = str(web_result.get("content", "") or "")
            # 网络文本截断：超过阈值取头+尾，否则取头部
            if len(content) > 2400:
                content_excerpt = (
                    content[:1200] + "\n...\n" + content[-800:]
                )
            else:
                content_excerpt = content[:1200]
            context_parts.append(
                f"{L['web_header'].format(i=i)}\n{L['web_title']} {web_result.get('title', '')}\n"
                f"{L['web_content']} {content_excerpt}\n"
                f"{L['web_source']} {web_result.get('url', '')}"
            )

    return "\n\n".join(context_parts) if context_parts else L["empty"]


#: 数据来源指示文案（拼接进 system prompt 的"上下文信息"标题旁）。
_SOURCE_INDICATOR_I18N: Dict[str, Dict[str, str]] = {
    "web": {"zh": "（包含最新网络信息）", "en": "(includes the latest web results)"},
    "kb": {"zh": "（基于知识库）", "en": "(based on the knowledge base)"},
}


def get_source_indicator(state: ConversationState) -> str:
    """
    Get source indicator based on data sources used.

    指示词随 ``prefer_zh_output`` 本地化，避免英文 system prompt 中混入中文。

    Args:
        state: Current conversation state

    Returns:
        Source indicator string（带括号）；无来源时返回空串。
    """
    lang = "zh" if state.get("prefer_zh_output", True) else "en"
    if state.get("web_search_used", False):
        return _SOURCE_INDICATOR_I18N["web"][lang]
    if state.get("retrieved_docs"):
        return _SOURCE_INDICATOR_I18N["kb"][lang]
    return ""


def _build_conversation_history(state: ConversationState, max_messages: int = 12) -> List:
    """
    Build conversation history messages from state.

    Args:
        state: Current conversation state
        max_messages: Maximum number of chat messages (user+assistant) to include.
            默认 12 条 = 最近 6 轮（user+assistant 对），与产品要求"固定保留最近 6 轮上下文"对齐，
            既保障多轮追问的连贯性（不丢最近用户句），又限制窗口大小避免无关旧主题干扰。

    Returns:
        List of Message objects from conversation history
    """
    messages = []
    for msg in state.get("context", {}).get("messages", [])[-max_messages:]:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg.get("content", "")))
    return messages


# =============================================================================
# Message Building Helpers
# =============================================================================
def build_greeting_messages(
    state: ConversationState, employee_config: Dict[str, Any]
) -> List:
    """
    Build LLM messages for greeting scenario.

    Characteristics:
    - No RAG or web search needed
    - Natural, friendly response based on personality
    - Encourages further interaction

    Args:
        state: Current conversation state
        employee_config: Employee configuration dict

    Returns:
        List of Message objects
    """
    personality = employee_config.get("personality", {})
    role = employee_config.get("role", "AI助手")
    greeting = employee_config.get("greeting", "您好")
    name = employee_config.get("name", "AI助手")
    description = employee_config.get("description", "专业的AI助手")

    entities = state.get("entities", {})
    greeting_type = entities.get("greeting_type", "basic")
    matched_keyword = entities.get("matched_keyword", "")

    tone_desc, _, _ = get_personality_description(personality)
    formality_desc = (
        "高度正式" if personality.get("formality") == "high" else "适度正式"
    )

    style_hints = {
        "time": f"根据时间（{matched_keyword}）给予相应的热情问候，并自然地询问用户今天需要什么帮助",
        "casual": "用轻松活泼的方式回应，表现出随时准备提供帮助的状态",
        "polite": "以礼貌、耐心的方式回应，让用户感受到专业和尊重",
        "basic": "用简洁友好的方式回应，自然地引导用户说明需求",
    }
    style_hint = style_hints.get(greeting_type, style_hints["basic"])

    system_prompt = f"""你是 {name}，{role}。

角色定位：
{description}

个性特征：
- 语气风格：{tone_desc}
- 正式程度：{formality_desc}

开场白：
{greeting}

**当前场景**：用户向你发起问候（"{matched_keyword}"）。

回答要求：
1. {style_hint}
2. 回复要简洁（不超过 50 字）
3. 保持{tone_desc}的语气风格
4. 不要提及"我是AI"或"我是机器人"
5. 不要提供任何具体信息（除非用户主动询问）
6. 如果用户用英文你也用英文

用户原话：
{state["user_query"]}
"""

    prefer_zh_output = state.get("prefer_zh_output", True)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(_build_conversation_history(state, max_messages=6))
    # 按语言追加提问：英文提问强制要求英文回复
    if prefer_zh_output:
        messages.append(HumanMessage(content=state["user_query"]))
    else:
        messages.append(HumanMessage(content=f"Please reply in English.\n\n{state['user_query']}"))

    logger.debug(
        "Greeting messages built",
        greeting_type=greeting_type,
        matched_keyword=matched_keyword,
    )

    return messages


def _select_llm_facing_query(state: ConversationState) -> str:
    """
    选择送给 LLM 的"用户问题"文本。

    设计：优先使用 ``rewritten_query``（消歧后通常更利于检索 / 推理），
    但当其主导语言与 ``prefer_zh_output`` 期望的输出语言不一致时，
    回退到原始 ``user_query``，避免"英文问、中文答"或反向的混语回复。

    这是 LLM 路径上的"语言一致性最后防线"。即使上游的 ``sanitize_resolved_query``
    漏检（或其它路径绕过 sanitize），此处仍能兜住。
    """
    user_query = (state.get("user_query") or "").strip()
    rewritten = (state.get("rewritten_query") or "").strip()
    if not rewritten or rewritten == user_query:
        return user_query or rewritten

    expected_lang = "zh" if state.get("prefer_zh_output", True) else "en"
    rewritten_lang = detect_dominant_language(rewritten)
    if rewritten_lang and rewritten_lang != expected_lang:
        logger.info(
            "Rewritten query language mismatch with prefer_zh_output; "
            f"falling back to original. expected={expected_lang}, got={rewritten_lang}, "
            f"user_query={user_query[:80]!r}, rewritten={rewritten[:80]!r}"
        )
        return user_query or rewritten
    return rewritten


def build_generation_messages(state: ConversationState) -> List:
    """
    Build LLM messages for answer generation.

    Handles different scenarios:
    - Interruption: Short acknowledgment response
    - Greeting: Simple, friendly response
    - Realtime + Web search: Emphasize network sources
    - Regular RAG: Knowledge-based response

    Args:
        state: Current conversation state

    Returns:
        List of Message objects for LLM
    """
    employee_config = state.get("employee_config", {})
    effective_query = _select_llm_facing_query(state)

    intent = state.get("intent")
    if intent == "greeting":
        return build_greeting_messages(state, employee_config)

    personality = employee_config.get("personality", {})
    role = employee_config.get("role", "AI助手")
    greeting = employee_config.get("greeting", "您好")
    name = employee_config.get("name", "AI助手")
    description = employee_config.get("description", "专业的AI助手")

    context_text = build_context_text(state)
    source_indicator = get_source_indicator(state)
    prefer_zh_output = state.get("prefer_zh_output", True)
    # 人格描述按语言本地化，避免英文 system prompt 中混入中文短语诱导模型用中文回答。
    tone_desc, style_desc, formality_desc = get_personality_description(
        personality, prefer_zh_output=prefer_zh_output
    )
    system_time_context = _build_system_time_context(prefer_zh_output)
    now = datetime.now()
    target_year = resolve_target_year_from_query(effective_query, now)

    # 构建基础系统提示（中文/英文）
    if prefer_zh_output:
        base_prompt = f"""你是 {name}，{role}。

角色定位：
{description}

个性特征：
- 语气风格：{tone_desc}
- 沟通方式：{style_desc}
- 正式程度：{formality_desc}

开场白：
{greeting}
"""
    else:
        base_prompt = f"""You are {name}, a helpful assistant.

Role:
{description}

Style:
- Tone: {tone_desc}
- Communication: {style_desc}
- Formality: {formality_desc}
"""

    # Add scenario-specific instructions
    if state.get("web_search_used", False) and state.get("is_realtime_query", False):
        temporal_guardrail = _build_realtime_temporal_guardrail(prefer_zh_output)
        realtime_category = state.get("realtime_category", "")
        if realtime_category == "weather":
            requirements = f"""**重要提示**：用户询问的是天气信息，系统已通过网络搜索获取了最新数据。

回答要求：
1. **必须基于下方提供的网络资料回答**
2. 字数严格限制在 60 字以内
3. 格式：一句话，不用列表、减号、复杂格式
4. 直接提取天气数据（温度、风力等）
5. 保持{tone_desc}的语气风格
6. **禁止：信息来源、网站链接、"信息来源"字样**
7. 简单直接地提供天气信息。
上下文信息{source_indicator}：
{context_text}

用户问题：
{effective_query}
"""
            if not prefer_zh_output:
                requirements = f"""IMPORTANT: The user asks about weather. The system has retrieved up-to-date web results.

Requirements:
1. Answer strictly based on the web context below
2. Keep it very short (about one sentence)
3. One sentence only (no lists / bullets)
4. Extract concrete data (temperature, wind, etc.)
5. Keep a {tone_desc} tone
6. Do NOT include sources, links, or the words "source" / "references"

Context {source_indicator}:
{context_text}

User question:
{effective_query}
"""
        elif realtime_category == "news":
            requirements = f"""**重要提示**：用户询问的是新闻信息，系统已通过网络搜索获取了最新数据。

回答要求：
1. **必须基于下方提供的网络资料回答**
2. 字数严格限制在 100 字以内
3. 格式：直接说要点，不用列表、减号、复杂格式
4. 保持{tone_desc}的语气风格
5. **禁止：信息来源、网站链接、"信息来源"字样**
6. 若问题属于“某职务/某角色目前是谁/哪位”的人物归属查询：必须输出**网络资料中明确出现的人名**；禁止凭记忆猜测或输出不在资料中的人名；如资料未出现该人名，必须说明“未从当前网络资料中确认到答案”并建议重试检索。
7. 简单直接地提供新闻要点。
上下文信息{source_indicator}：
{context_text}

用户问题：
{effective_query}
"""
            if not prefer_zh_output:
                requirements = f"""IMPORTANT: The user asks about news. The system has retrieved up-to-date web results.

Requirements:
1. Answer strictly based on the web context below
2. Keep it under ~100 words
3. Plain text, key points only (no lists / bullets)
4. Keep a {tone_desc} tone
5. Do NOT include sources, links, or the words "source" / "references"

Context {source_indicator}:
{context_text}

User question:
{effective_query}
"""
        elif realtime_category == "market":
            requirements = f"""**重要提示**：用户询问的是价格/市场信息，系统已通过网络搜索获取了最新数据。

回答要求：
1. **必须基于下方提供的网络资料回答**
2. 字数严格限制在 50 字以内
3. 格式：直接报数字，不用列表、减号、复杂格式
4. 保持{tone_desc}的语气风格
5. **禁止：信息来源、网站链接、"信息来源"字样**
6. 简单直接地提供价格信息。
上下文信息{source_indicator}：
{context_text}

用户问题：
{effective_query}

"""
            if not prefer_zh_output:
                requirements = f"""IMPORTANT: The user asks about prices/market. The system has retrieved up-to-date web results.

Requirements:
1. Answer strictly based on the web context below
2. Keep it under ~50 words
3. Prefer directly stating numbers
4. Keep a {tone_desc} tone
5. Do NOT include sources, links, or the words "source" / "references"

Context {source_indicator}:
{context_text}

User question:
{effective_query}
"""
        elif realtime_category == "traffic":
            # 先验信息（由 generate_answer 节点注入）：基于当前时段/星期的拥堵估算 + 时间锚点。
            # web 资料命中"今日具体路况"时优先用真实数据；否则使用先验给出合理估算，避免"无法回答"。
            _est = state.get("traffic_estimate") or {}
            _est_level = _est.get("level", "")
            _est_reason = _est.get("reason", "")
            _est_now = _est.get("now_iso", "")
            _est_weekday_zh = _est.get("weekday_zh", "")
            _est_weekday_en = _est.get("weekday_en", "")
            _prior_zh = (
                f"\n时段先验（系统本地推算，供资料缺失时兜底使用）：\n"
                f"- 当前时间：{_est_now}（{_est_weekday_zh}）\n"
                f"- 时段拥堵估算：{_est_level}（{_est_reason}）\n"
                if _est_level else ""
            )
            _prior_en = (
                f"\nTime-of-day prior (locally derived; use when web context is insufficient):\n"
                f"- Current time: {_est_now} ({_est_weekday_en})\n"
                f"- Estimated congestion level: {_est_level} ({_est_reason})\n"
                if _est_level else ""
            )
            requirements = f"""**重要提示**：用户询问的是路况/拥堵信息，系统已通过网络搜索获取了相关资料。
{_prior_zh}
回答要求：
1. **优先采用网络资料中明确指向"今日 + 用户询问城市"的具体路况数据**（如出现具体路段、时间戳、官方平台数据等）；
2. **若网络资料没有覆盖到用户询问城市的今日具体路况**：禁止说"网络资料中没有相关信息"或"无法判断"等回避语，改为：
   (a) 基于"时段先验"给出整体拥堵等级判断（轻度/中等/较拥堵）及简要依据（早晚高峰/平峰/节假日/周末等）；
   (b) 给出 1-2 条可执行的通用出行建议（如错峰、避开高峰路段、选择公共交通）；
   (c) 末尾推荐用户使用高德地图 / 百度地图等专业实时路况服务获取精确数据；
3. **禁止把与"今日 + 询问城市"无关的网页内容（如汽车广告快讯、历史文章）当作今日路况依据**；只能作为"通常拥堵规律"的弱背景；
4. **关于"假期/节日/放假"之类的判断**：除非网络资料中明确给出了具体日期且能对应到今天，并明确说明"假期/放假/节日"，否则禁止把"假期第一天/节假日"等当作今天事实写进结论；
5. **禁止在回答中出现"假期/节假日/放假/长假/假期第一天"等字样**，除非满足第 4 条的"同一天日期+明确假期"条件；
6. 保持{tone_desc}的语气风格；
7. **禁止：信息来源、网站链接、"信息来源"字样**。

上下文信息{source_indicator}：
{context_text}

用户问题：
{effective_query}
"""
            if not prefer_zh_output:
                requirements = f"""IMPORTANT: The user asks about traffic/congestion. The system has retrieved web results.
{_prior_en}
Requirements:
1. Prefer concrete \"today + the user's city\" traffic data from the web context (specific roads, timestamps, official platforms);
2. If the web context does NOT cover today's concrete traffic for the asked city, DO NOT say \"the context has no info\" or \"cannot determine\"; instead:
   (a) Give an overall congestion level using the time-of-day prior above (light / moderate / heavy) with a brief reason (rush hour / off-peak / weekend / holiday etc.);
   (b) Provide 1-2 practical, generic travel tips (avoid peak windows, alternative roads, public transit, etc.);
   (c) End by recommending the user check professional real-time map services (Amap / Baidu Maps) for precise data;
3. Do NOT treat unrelated content (auto-industry news, historical articles) as today's traffic; only use as weak background;
4. Do NOT mention \"holiday / day-off / public holiday\" unless the web context explicitly states the date matches today AND the holiday;
5. Keep a {tone_desc} tone;
6. Do NOT include sources, links, or the words \"source\" / \"references\".

Context {source_indicator}:
{context_text}

User question:
{effective_query}
"""
        else:
            requirements = f"""**重要提示**：用户询问的是实时信息，系统已通过网络搜索获取了最新数据。

回答要求：
1. **必须基于下方提供的网络资料回答**
2. 直接提取网络资料中的关键信息
3. 保持{tone_desc}的语气风格
4. 回答简洁明了，重点突出数据
5. **禁止：信息来源、网站链接、"信息来源"字样**

上下文信息{source_indicator}：
{context_text}

用户问题：
{effective_query}

请基于上述网络资料，提供准确的实时信息回答。"""
            if not prefer_zh_output:
                requirements = f"""IMPORTANT: The user asks about real-time information. The system has retrieved up-to-date web results.

Requirements:
1. Answer strictly based on the web context below
2. Extract key facts and numbers
3. Keep a {tone_desc} tone
4. Do NOT include sources, links, or the words "source" / "references"

Context {source_indicator}:
{context_text}

User question:
{effective_query}
"""
        requirements = requirements + "\n\n" + temporal_guardrail
        if target_year is not None:
            if prefer_zh_output:
                requirements += f"\n时间锚定：本问题目标年份为 {target_year} 年；若回答包含具体日期，年份必须与该目标年份一致。"
            else:
                requirements += (
                    f"\nTime anchor: target year for this query is {target_year}; "
                    "if you provide a concrete date, its year must match this target year."
                )
        if prefer_zh_output:
            requirements += (
                "\n输出风格要求：直接给出最终结论（可附一个简短依据），"
                "不要追加与结论无关的反问或重复说明。"
            )
        else:
            requirements += (
                "\nStyle requirement: provide the final conclusion directly (optionally one short rationale), "
                "without adding unrelated follow-up questions or repetitive caveats."
            )
    else:
        requirements = f"""回答要求：
1. 严格基于提供的上下文信息回答，不编造内容
2. 如果上下文不足，诚实告知并建议联系人工客服
3. 保持{tone_desc}的语气风格
4. 回答简洁明了，重点突出
5. 如有多个信息源，优先使用最相关的内容
6. **禁止：信息来源、网站链接、"信息来源"字样**

上下文信息{source_indicator}：
{context_text}

用户问题：
{effective_query}

"""
        if not prefer_zh_output:
            requirements = f"""Requirements:
1. Answer strictly based on the provided context; do not fabricate
2. If context is insufficient, say so and suggest contacting human support
3. Keep a {tone_desc} tone
4. Be concise and clear
5. Do NOT include sources, links, or the words "source" / "references"

Context {source_indicator}:
{context_text}

User question:
{effective_query}
"""
        # 实时查询但未获得联网结果：仍需保持时间一致性，避免模型自行混入错误年份。
        if state.get("is_realtime_query", False):
            requirements = requirements + "\n\n" + _build_realtime_temporal_guardrail(prefer_zh_output)
            if prefer_zh_output:
                requirements += "\n若当前上下文无法支撑唯一结论，请明确说明无法确认，不要补充未经证实的年份或日期。"
            else:
                requirements += (
                    "\nIf context is insufficient for a unique temporal answer, clearly state uncertainty "
                    "and do not introduce unverified years or dates."
                )

    system_prompt = base_prompt + "\n" + system_time_context + "\n\n" + requirements
    if not prefer_zh_output:
        system_prompt = "Answer in English only.\n\n" + system_prompt

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(_build_conversation_history(state, max_messages=12))
    if prefer_zh_output:
        messages.append(HumanMessage(content=effective_query))
    else:
        messages.append(HumanMessage(content=f"Please answer in English only.\n\n{effective_query}"))

    return messages


def build_math_generation_messages(state: ConversationState) -> List:
    """
    构建数学问题的 Phi-4 模型消息。

    Args:
        state: Current conversation state

    Returns:
        List of Message objects for Phi-4 LLM
    """
    query = (state.get("user_query") or "").strip()
    normalized_query = _normalize_math_query(query)
    if _is_simple_math_query(normalized_query):
        # 简单计算题：保持极简输出
        messages = [SystemMessage(content=PHI4_SIMPLE_SYSTEM_PROMPT)]
        messages.append(HumanMessage(content=normalized_query))
        return messages

    # 非简单题：统一使用“解题行为流程”+“公式库”，不再针对具体题目写死分支
    sys_prompt = MATH_SYSTEM_PROMPT + "\n\n" + GEOMETRY_FORMULA_BOOK
    messages = [SystemMessage(content=sys_prompt)]
    messages.append(HumanMessage(content=normalized_query))

    return messages


def _normalize_math_query(query: str) -> str:
    """轻量归一化：将“派/pi”统一为 π，减少符号漏写/误写。"""
    q = (query or "").strip()
    # 常见口语/拼写归一
    q = q.replace("派", "π")
    q = q.replace("pi", "π").replace("PI", "π").replace("Pi", "π")
    return q


def _is_simple_math_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    if len(q) > 28:
        return False
    complex_markers = ("证明", "推导", "分析", "为什么", "思路", "过程", "几何", "应用题", "函数", "方程组")
    if any(m in q for m in complex_markers):
        return False
    has_number = bool(re.search(r"[0-9一二三四五六七八九十百千万两零]", q))
    has_op = any(op in q for op in ("+", "-", "*", "/", "加", "减", "乘", "除", "×", "÷", "等于"))
    asks_value = any(k in q for k in ("等于几", "多少", "=?", "＝", "="))
    return has_number and has_op and asks_value


def heuristic_complexity(query: str) -> float:
    """
    启发式复杂度评估（当 LLM 评估失败时的备用方案）。

    Args:
        query: 用户查询

    Returns:
        复杂度分数 0-10
    """
    score = 3.0
    length = len(query)

    # Length factor
    if length > 100:
        score += 2
    elif length > 50:
        score += 1

    # Complex keywords
    complex_keywords = [
        "为什么",
        "为何",
        "如何",
        "怎样",
        "怎么",
        "比较",
        "对比",
        "区别",
        "差异",
        "分析",
        "评估",
        "评价",
        "总结",
        "影响",
        "后果",
        "原因",
        "导致",
        "关系",
        "关联",
        "相关性",
    ]
    if any(kw in query for kw in complex_keywords):
        score += 2

    # Multiple questions
    if "，" in query or "。" in query or "？" in query or "?" in query:
        score += 1

    # Simple patterns (reduce complexity)
    simple_patterns = [
        "是什么",
        "什么是",
        "多少",
        "几个",
        "天气",
        "价格",
        "多少钱",
        "怎么",
    ]
    if any(p in query for p in simple_patterns) and length < 30:
        score -= 1

    return max(0.0, min(10.0, score))
