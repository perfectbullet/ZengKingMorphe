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
from prompts.prompts import PHI4_SYSTEM_PROMPT
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
def get_personality_description(personality: dict) -> Tuple[str, str, str]:
    """
    Get personality description for system prompt.

    Args:
        personality: Personality dict from employee config

    Returns:
        Tuple of (tone_desc, style_desc, formality_desc)
    """
    tone_map = {
        "professional": "专业严谨",
        "friendly": "友好亲切",
        "formal": "正式庄重",
        "casual": "轻松随意",
    }

    style_map = {
        "concise": "简明扼要",
        "detailed": "详细周到",
        "conversational": "对话式",
        "instructional": "指导式",
    }

    formality_map = {
        "high": "高度正式（使用敬语）",
        "moderate": "适度正式",
        "low": "轻松口语化",
    }

    tone_desc = tone_map.get(personality.get("tone", "professional"), "专业")
    style_desc = style_map.get(personality.get("style", "friendly"), "友好")
    formality_desc = formality_map.get(personality.get("formality", "moderate"), "适度")

    return tone_desc, style_desc, formality_desc


# =============================================================================
# Context Building Helpers
# =============================================================================
def build_context_text(state: ConversationState) -> str:
    """
    Build context text from retrieved docs and web search results.

    Uses compressed context if available, otherwise builds from sources.

    Args:
        state: Current conversation state

    Returns:
        Formatted context string for LLM prompt
    """
    if state.get("compressed_context"):
        return f"[压缩后的参考信息]\n{state['compressed_context']}"

    context_parts = []
    for i, doc in enumerate(state.get("retrieved_docs", [])[:3], 1):
        context_parts.append(f"[知识库参考{i}]\n{doc.get('content', '')[:500]}")

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
                f"[网络资料{i}]\n标题: {web_result.get('title', '')}\n"
                f"内容: {content_excerpt}\n"
                f"来源: {web_result.get('url', '')}"
            )

    return "\n\n".join(context_parts) if context_parts else "（暂无相关参考资料）"


def get_source_indicator(state: ConversationState) -> str:
    """
    Get source indicator based on data sources used.

    Args:
        state: Current conversation state

    Returns:
        Source indicator string
    """
    if state.get("web_search_used", False):
        return "（包含最新网络信息）"
    elif state.get("retrieved_docs"):
        return "（基于知识库）"
    return ""


def _build_conversation_history(state: ConversationState, max_messages: int = 10) -> List:
    """
    Build conversation history messages from state.

    Args:
        state: Current conversation state
        max_messages: Maximum number of chat messages (user+assistant) to include.
            默认 10 条≈5 轮，与 session 侧最近上下文窗口对齐，避免多轮节日追问时丢最近用户句。

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
    effective_query = state.get("rewritten_query") or state["user_query"]

    intent = state.get("intent")
    if intent == "greeting":
        return build_greeting_messages(state, employee_config)

    personality = employee_config.get("personality", {})
    role = employee_config.get("role", "AI助手")
    greeting = employee_config.get("greeting", "您好")
    name = employee_config.get("name", "AI助手")
    description = employee_config.get("description", "专业的AI助手")
    tone_desc, style_desc, formality_desc = get_personality_description(personality)

    context_text = build_context_text(state)
    source_indicator = get_source_indicator(state)
    prefer_zh_output = state.get("prefer_zh_output", True)
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
            requirements = f"""**重要提示**：用户询问的是路况/拥堵信息，系统已通过网络搜索获取了相关资料。

回答要求：
1. **必须基于下方提供的网络资料回答**
2. **优先判断资料是否包含“今天/当前/更新时间/日期”线索**：若包含，则你必须给出结论性判断（例如“整体偏堵/较通畅/高峰更明显”）并简要说明依据（如车流量增加、交警提示、高峰提前等）
3. **禁止把与今天无关的历史文章当成“今天路况”**：只能把它作为“通常/容易拥堵时段或路段”的补充背景
4. **关于“假期/节日/放假”之类的判断**：除非网络资料中明确给出了具体日期且能对应到今天，并明确说明“假期/放假/节日”，否则禁止把“假期第一天/节假日”等当作今天事实写进结论
5. **禁止在回答中出现“假期/节假日/放假/长假/假期第一天”等字样**，除非满足第4条的“同一天日期+明确假期”条件
6. 若资料完全缺少任何时间线索或与用户问题不相关，才可以说明无法确定；否则不要用“无法确认”来回避结论
7. 保持{tone_desc}的语气风格
8. **禁止：信息来源、网站链接、"信息来源"字样**

上下文信息{source_indicator}：
{context_text}

用户问题：
{effective_query}
"""
            if not prefer_zh_output:
                requirements = f"""IMPORTANT: The user asks about traffic/congestion. The system has retrieved relevant web results.

Requirements:
1. Answer strictly based on the web context below
2. First check whether the context contains clear \"today/now/update time/date\" signals; if yes, you MUST provide a conclusion (e.g., \"likely congested\" / \"smooth\" / \"peak hours worse\") and briefly justify it (traffic volume increase, police advisory, peak starts earlier, etc.)
3. Do NOT treat unrelated historical articles as today's traffic; they can only be used as general background
4. You may say \"cannot determine\" ONLY if the context has no time signals or is irrelevant; otherwise do not evade giving a conclusion
5. Keep a {tone_desc} tone
6. Do NOT include sources, links, or the words \"source\" / \"references\"

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
    messages.extend(_build_conversation_history(state, max_messages=10))
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
    # 使用 Phi-4 模型进行数学推理
    messages = [SystemMessage(content=PHI4_SYSTEM_PROMPT)]

    # 添加用户问题
    messages.append(HumanMessage(content=state["user_query"]))

    return messages


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
