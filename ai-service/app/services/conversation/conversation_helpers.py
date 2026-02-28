"""
Helper utilities for conversation workflow.

This module provides:
- Node timing utilities
- LLM selection logic for hybrid routing
- Message building helpers
- Personality description helpers
"""
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Tuple

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.core.config import settings
from app.core.logging import get_logger
from app.services.conversation.conversation_state import ConversationState, GREETING_KEYWORDS, INTERRUPTION_KEYWORDS

logger = get_logger(__name__)


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
def select_llm(
    state: ConversationState,
    local_llm,
    local_grader_llm,
    remote_llm,
    remote_grader_llm
) -> Tuple[Any, Any, str]:
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
        local_grader_llm: Local grader LLM instance
        remote_llm: Remote OpenAI-style LLM instance
        remote_grader_llm: Remote grader LLM instance

    Returns:
        Tuple of (llm, grader_llm, model_name)
    """
    routing_mode = getattr(settings, 'llm_routing_mode', 'local_only')

    # Non-hybrid modes: return pre-configured LLM
    if routing_mode == 'local_only':
        return local_llm, local_grader_llm, settings.ollama_model
    elif routing_mode == 'remote_only':
        return remote_llm, remote_grader_llm, settings.openai_model

    # Hybrid mode: dynamic selection based on complexity score
    complexity_score = state.get("complexity_score", 3.0)
    complexity_reason = state.get("complexity_reason", "unknown")
    is_greeting = state.get("intent") == "greeting"
    faq_matched = state.get("faq_matched")

    # 复杂度阈值：超过此分数使用外部模型
    complexity_threshold = getattr(settings, 'complexity_threshold', 7.0)

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
        threshold=complexity_threshold
    )

    if use_remote:
        return remote_llm, remote_grader_llm, settings.openai_model
    return local_llm, local_grader_llm, settings.ollama_model


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
        "casual": "轻松随意"
    }

    style_map = {
        "concise": "简明扼要",
        "detailed": "详细周到",
        "conversational": "对话式",
        "instructional": "指导式"
    }

    formality_map = {
        "high": "高度正式（使用敬语）",
        "moderate": "适度正式",
        "low": "轻松口语化"
    }

    tone_desc = tone_map.get(personality.get("tone", "professional"), "专业")
    style_desc = style_map.get(personality.get("style", "concise"), "简明")
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
            context_parts.append(
                f"[网络资料{i}]\n标题: {web_result.get('title', '')}\n"
                f"内容: {web_result.get('content', '')[:400]}\n"
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


def _build_conversation_history(state: ConversationState, max_turns: int = 5) -> List:
    """
    Build conversation history messages from state.

    Args:
        state: Current conversation state
        max_turns: Maximum number of conversation turns to include

    Returns:
        List of Message objects from conversation history
    """
    messages = []
    for msg in state.get("context", {}).get("messages", [])[-max_turns:]:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg.get("content", "")))
    return messages


# =============================================================================
# Message Building Helpers
# =============================================================================
def build_greeting_messages(
    state: ConversationState,
    employee_config: Dict[str, Any]
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
    name = employee_config.get('name', 'AI助手')
    description = employee_config.get('description', '专业的AI助手')

    entities = state.get("entities", {})
    greeting_type = entities.get("greeting_type", "basic")
    matched_keyword = entities.get("matched_keyword", "")

    tone_desc, _, _ = get_personality_description(personality)
    formality_desc = "高度正式" if personality.get('formality') == 'high' else "适度正式"

    style_hints = {
        "time": f"根据时间（{matched_keyword}）给予相应的热情问候，并自然地询问用户今天需要什么帮助",
        "casual": "用轻松活泼的方式回应，表现出随时准备提供帮助的状态",
        "polite": "以礼貌、耐心的方式回应，让用户感受到专业和尊重",
        "basic": "用简洁友好的方式回应，自然地引导用户说明需求"
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

用户原话：
{state['user_query']}

请生成自然、友好的问候回应。"""

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(_build_conversation_history(state, max_turns=3))
    messages.append(HumanMessage(content=state["user_query"]))

    logger.debug(
        "Greeting messages built",
        greeting_type=greeting_type,
        matched_keyword=matched_keyword
    )

    return messages


def build_interruption_messages(
    state: ConversationState,
    employee_config: Dict[str, Any]
) -> List:
    """
    Build LLM messages for interruption scenario.

    Characteristics:
    - No RAG or web search needed
    - Short, acknowledging response
    - User wants to stop or pause current conversation

    Args:
        state: Current conversation state
        employee_config: Employee configuration dict

    Returns:
        List of Message objects
    """
    personality = employee_config.get("personality", {})
    name = employee_config.get('name', 'AI助手')

    entities = state.get("entities", {})
    interruption_type = entities.get("interruption_type", "stop")

    tone_desc, _, _ = get_personality_description(personality)

    response_hints = {
        "stop": "简短回应，表示已收到用户停止的指令",
        "done": "简短回应，表示确认",
        "interrupt": "简短回应，表示等待用户继续",
        "dismiss": "简短回应，表示已明白用户意思"
    }
    response_hint = response_hints.get(interruption_type, "简短回应，表示收到")

    system_prompt = f"""你是 {name}。

**当前场景**：用户发起了打断/停止的信号。

回答要求：
1. {response_hint}
2. 回复极其简洁（不超过 10 字）
3. 保持{tone_desc}的语气风格
4. 不要说任何多余的话

用户原话：
{state['user_query']}

请生成最简短的确认回应。"""

    messages = [SystemMessage(content=system_prompt)]
    messages.append(HumanMessage(content=state["user_query"]))

    logger.debug(
        "Interruption messages built",
        interruption_type=interruption_type
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

    intent = state.get("intent")
    if intent == "interruption":
        return build_interruption_messages(state, employee_config)
    if intent == "greeting":
        return build_greeting_messages(state, employee_config)

    personality = employee_config.get("personality", {})
    role = employee_config.get("role", "AI助手")
    greeting = employee_config.get("greeting", "您好")
    name = employee_config.get('name', 'AI助手')
    description = employee_config.get('description', '专业的AI助手')
    tone_desc, style_desc, formality_desc = get_personality_description(personality)

    context_text = build_context_text(state)
    source_indicator = get_source_indicator(state)

    # Build base system prompt
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

    # Add scenario-specific instructions
    if state.get("web_search_used", False) and state.get("is_realtime_query", False):
        requirements = f"""**重要提示**：用户询问的是实时信息（如{state.get('realtime_category', '最新动态')}），系统已通过网络搜索获取了最新数据。

回答要求：
1. **必须基于下方提供的网络资料回答**
2. 直接提取网络资料中的关键信息
3. 保持{tone_desc}的语气风格
4. 回答简洁明了，重点突出具体数据
5. 可在回答末尾简要注明信息来源
6. **不要说"无法提供实时数据"**

上下文信息{source_indicator}：
{context_text}

用户问题：
{state['user_query']}

请基于上述网络资料，提供准确的实时信息回答。"""
    else:
        requirements = f"""回答要求：
1. 严格基于提供的上下文信息回答，不编造内容
2. 如果上下文不足，诚实告知并建议联系人工客服
3. 保持{tone_desc}的语气风格
4. 回答简洁明了，重点突出
5. 如有多个信息源，优先使用最相关的内容

上下文信息{source_indicator}：
{context_text}

用户问题：
{state['user_query']}

请提供专业、准确的回答。"""

    system_prompt = base_prompt + "\n" + requirements

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(_build_conversation_history(state, max_turns=5))
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
        "为什么", "为何", "如何", "怎样", "怎么",
        "比较", "对比", "区别", "差异",
        "分析", "评估", "评价", "总结",
        "影响", "后果", "原因", "导致",
        "关系", "关联", "相关性"
    ]
    if any(kw in query for kw in complex_keywords):
        score += 2

    # Multiple questions
    if "，" in query or "。" in query or "？" in query or "?" in query:
        score += 1

    # Simple patterns (reduce complexity)
    simple_patterns = ["是什么", "什么是", "多少", "几个", "天气", "价格", "多少钱", "怎么"]
    if any(p in query for p in simple_patterns) and length < 30:
        score -= 1

    return max(0.0, min(10.0, score))
