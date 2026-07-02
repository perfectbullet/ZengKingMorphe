"""
对话工作流辅助工具集。

本模块提供：
- 节点计时工具
- 混合路由 LLM 选择逻辑
- 消息构建辅助函数
- 人格描述辅助函数
"""

import os
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
from app.services.math_agent_service import MathAgentService

logger = get_logger(__name__)


def clean_user_query(text: str) -> str:
    """
    清理用户查询，移除前导标点符号。

    Args:
        text: 用户查询文本

    Returns:
        移除前导标点后的文本
    """
    text = re.sub(r"^[，。！？、；：,.?!;:\s]+", "", text)
    return text.lstrip()


def prefer_zh_output(user_query: str) -> bool:
    """
    判断输出语言偏好：含中文→中文，含英文→英文，其余默认中文。

    Args:
        user_query: 用户查询文本

    Returns:
        True 表示偏好中文输出，False 表示偏好英文输出
    """
    if not user_query:
        return True
    if re.search(r"[一-鿿]", user_query):
        return True
    if re.search(r"[A-Za-z]", user_query):
        return False
    return True


def resolve_prefer_zh_output(state: ConversationState) -> bool:
    """
    解析本轮输出语言偏好（True=中文，False=英文）。

    设计要点：
    - **优先**取 ``state["prefer_zh_output"]``：调用方（chat_stream_v1）已根据
      用户问句语言显式设置，是最权威的来源；
    - **兜底**当 state 中缺失或为 None 时（典型场景：LangGraph 1.x 按 TypedDict
      schema 严格过滤掉未声明字段时；或老调用方未塞入此字段时），按
      ``user_query`` 的主导语言现场推断，**不再无脑默认中文**——这是过去
      “英文问、中英文混合答”的根因之一；
    - **最后**仍兜底为 True（中文）以兼容空 query / 纯数字符号问句的旧行为。

    任何工作流节点在拼 system prompt / 选模板前都应通过本函数取值，
    而不是直接 ``state.get("prefer_zh_output", True)`` 静默吞掉异常状态。
    """
    explicit = state.get("prefer_zh_output")
    if explicit is not None:
        return bool(explicit)
    user_query = (state.get("user_query") or "").strip()
    if user_query:
        lang = detect_dominant_language(user_query)
        if lang == "en":
            return False
        if lang == "zh":
            return True
    return True


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


def resolve_target_year_from_query(
    query: str, now: datetime | None = None
) -> int | None:
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
    异步上下文管理器，用于节点执行计时。

    跟踪每个工作流节点的执行时间并存入状态中供分析。
    计时数据会写入日志并包含在最终对话记录中。

    Args:
        node_name: 工作流节点名称
        state: 对话状态对象
        llm_instance: 工作流实例（按需访问方法）

    Example:
        async with time_node("knowledge_retrieval", state):
            # 检索逻辑
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
    根据查询上下文选择合适的 LLM（仅混合模式）。

    选择逻辑（混合模式）：
    1. 主要依据：complexity_score（0-10）
       - 0-6分：使用本地 Ollama（简单到中等复杂）
       - 7-10分：使用外部 API（高复杂度）
    2. 特殊情况：
       - greeting、FAQ 命中：强制使用本地模型
       - 「现任 X 是谁」类需要广博世界知识的问题：强制使用远端 LLM
         （本地小模型常常回避或答错）

    Args:
        state: 当前对话状态
        local_llm: 本地 Ollama LLM 实例
        remote_llm: 远端 OpenAI 风格 LLM 实例

    Returns:
        (llm, model_name) 元组
    """
    routing_mode = getattr(settings, "llm_routing_mode", "local_only")

    # 统一模型名解析
    model_name = os.getenv("LLM_MODEL") or settings.ollama_model

    # Non-hybrid modes: return pre-configured LLM
    if routing_mode == "local_only":
        return local_llm, model_name
    elif routing_mode == "remote_only":
        return remote_llm, model_name

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

    # 「需要广博/最新世界知识」的事实性人事查询：复杂度评估节点对它们看起来"短而
    # 简单"会给出 3.0 分这种低分，但本地小模型（qwen3:14b 等）面对这类问题常
    # 表现为"我无法提供具体姓名"或答出过时人名（如"李克强"→"李强"）。
    # 直接基于查询本身重新判定：命中「现任/当前 + 公共职务 + 谁」三元模式时强制
    # 走远端大模型——它们的预训练语料对国际、国家级公共职务覆盖远比本地 7B 完整。
    # 不依赖 evaluate_complexity 节点：GENERAL_LLM 分支并不经过它。
    user_query = (state.get("rewritten_query") or state.get("user_query") or "").strip()
    if not is_greeting and not faq_matched and _needs_big_world_knowledge(user_query):
        use_remote = True
        reason = ["needs_big_world_knowledge", *reason]

    model_name = os.getenv("LLM_MODEL") or settings.ollama_model
    # 使用 f-string 输出选择详情，避免 Loguru 静默吞掉 keyword arg。
    logger.info(
        f"LLM selection: routing_mode=hybrid, selected={model_name}, "
        f"complexity_score={complexity_score}, complexity_reason={complexity_reason}, "
        f"reason={reason}, threshold={complexity_threshold}, "
        f"query={(state.get('user_query') or '')[:60]!r}"
    )

    if use_remote:
        return remote_llm, model_name
    return local_llm, model_name


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
    return (
        entry.get(lang) or entry.get("zh") or next(iter(entry.values()), short_default)
    )


def get_personality_description(
    personality: dict, prefer_zh_output: bool = True
) -> Tuple[str, str, str]:
    """
    获取用于 system prompt 的人格描述。

    Args:
        personality: 员工配置中的人格字典
        prefer_zh_output: 输出语言偏好。``True`` 返回中文文案、``False`` 返回英文文案。
            缺省为 ``True`` 以保持对老调用方的向后兼容（默认中文）。

    Returns:
        (tone_desc, style_desc, formality_desc) 元组，文案语言与 ``prefer_zh_output`` 一致。

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
        _PERSONALITY_STYLE_I18N,
        style_key,
        lang,
        _PERSONALITY_DEFAULT_I18N["style"][lang],
    )
    formality_desc = _i18n_lookup(
        _PERSONALITY_FORMALITY_I18N,
        formality_key,
        lang,
        _PERSONALITY_DEFAULT_I18N["formality"][lang],
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
    从检索到的文档和网络搜索结果构建上下文文本。

    有压缩上下文时优先使用，否则从数据源构建。

    标签（如 ``[知识库参考1]`` / ``标题:``）根据 ``prefer_zh_output`` 本地化，
    避免英文 system prompt 中混入中文标签污染输出语言。

    Args:
        state: 当前对话状态

    Returns:
        格式化后的上下文字符串，用于 LLM 提示
    """
    prefer_zh_output = resolve_prefer_zh_output(state)
    lang = "zh" if prefer_zh_output else "en"
    L = {k: v[lang] for k, v in _CONTEXT_LABELS_I18N.items()}

    if state.get("compressed_context"):
        return f"{L['compressed_header']}\n{state['compressed_context']}"

    context_parts = []
    for i, doc in enumerate(state.get("retrieved_docs", [])[:3], 1):
        context_parts.append(
            f"{L['kb_header'].format(i=i)}\n{doc.get('content', '')[:500]}"
        )

    web_results = state.get("web_search_results", [])
    if web_results and state.get("web_search_used", False):
        # 语言一致性过滤（仅对英文输出生效）：开了海外代理后 Tavily 易返回大段
        # 日文/中文资料（按出口 IP 推断地理偏好），这些 CJK 段会污染英文输出。
        # 中文方向暂不过滤——避免误伤英文权威源（论文/Reddit 等）。
        # 不依赖关键字 / 词典，复用基于 Unicode 块的 detect_dominant_language。
        kept = 0
        for web_result in web_results:
            if kept >= 3:
                break
            title = str(web_result.get("title", "") or "")
            content = str(web_result.get("content", "") or "")
            if not prefer_zh_output:
                combined_lang = detect_dominant_language((title + " " + content)[:1500])
                if combined_lang == "zh":
                    logger.info(
                        f"Web result filtered out for English output (CJK dominant): "
                        f"title={title[:60]!r}, url={web_result.get('url', '')!r}"
                    )
                    continue
            # 网络文本截断：超过阈值取头+尾，否则取头部
            if len(content) > 2400:
                content_excerpt = content[:1200] + "\n...\n" + content[-800:]
            else:
                content_excerpt = content[:1200]
            kept += 1
            context_parts.append(
                f"{L['web_header'].format(i=kept)}\n{L['web_title']} {title}\n"
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
    lang = "zh" if resolve_prefer_zh_output(state) else "en"
    if state.get("web_search_used", False):
        return _SOURCE_INDICATOR_I18N["web"][lang]
    if state.get("retrieved_docs"):
        return _SOURCE_INDICATOR_I18N["kb"][lang]
    return ""


def _build_conversation_history(
    state: ConversationState, max_messages: int = 12
) -> List:
    """
    从状态中构建对话历史消息列表。

    Args:
        state: 当前对话状态
        max_messages: 最大消息数量（user + assistant 总数）。
            默认 12 条 = 最近 6 轮（user+assistant 对），与产品要求"固定保留最近 6 轮上下文"对齐，
            既保障多轮追问的连贯性（不丢最近用户句），又限制窗口大小避免无关旧主题干扰。

    语言一致性过滤（双向对称）：
        “本轮输出语言只跟当前问句的语言相关”。反复使用同一 session 来回切换中英文
        测试时，历史会积累异种语言消息；把这些消息原样注入给 LLM，会和 web context、
        员工角色描述等其他语料叠加，让小模型（如 qwen3:14b）在本轮问句下飘移到错
        误语言（典型表现：“英文问、中英文混合答”）。
        这里对两个方向都启用过滤——主导语言与本轮目标语言不一致的历史消息一律
        丢弃；纯数字/标点等无法判定语言的消息保留。

    动态上下文记忆（与上面 classify_query_type 节点联动）：
        当 ``state["context_dependence"] == "unrelated"`` 时，本轮问句被判定与历史
        无关（指代/承接/话题延续都不成立），此时返回空列表——LLM 接收到的就只是
        本轮问句 + system prompt，等价于"全新对话"。这是"动态上下文记忆"规则的
        核心落地点：与历史相关 → 注入完整历史；无关 → 完全不注入。

    Returns:
        对话历史的 Message 对象列表
    """
    messages = []
    # 动态上下文记忆短路：本轮与历史无关 → 一律不注入历史。
    # 用 explicit "unrelated" 比较而非 truthy 判断，向后兼容老调用方（state 中无该字段时退化为旧行为）。
    if state.get("context_dependence") == "unrelated":
        logger.info(
            "Conversation history skipped due to dynamic context memory: "
            f"reason={state.get('context_dependence_reason')!r}, "
            f"query={(state.get('user_query') or '')[:60]!r}"
        )
        return messages
    prefer_zh = resolve_prefer_zh_output(state)
    target_lang = "zh" if prefer_zh else "en"
    skipped = 0
    for msg in state.get("context", {}).get("messages", [])[-max_messages:]:
        content = msg.get("content", "") or ""
        msg_lang = detect_dominant_language(content)
        # 主导语言可识别且与本轮目标语言不一致 → 丢弃；不可识别（空/纯数字/符号）→ 保留。
        if msg_lang and msg_lang != target_lang:
            skipped += 1
            continue
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=content))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=content))
    if skipped > 0:
        logger.info(
            f"Conversation history filtered for {target_lang.upper()} output: "
            f"skipped {skipped} cross-lang msg(s), kept {len(messages)}"
        )
    return messages


# =============================================================================
# Message Building Helpers
# =============================================================================
def build_greeting_messages(
    state: ConversationState, employee_config: Dict[str, Any]
) -> List:
    """
    构建问候场景下的 LLM 消息列表。

    特点：
    - 不需要 RAG 或联网搜索
    - 基于人格特征的自然、友好回复
    - 鼓励用户继续互动

    Args:
        state: 当前对话状态
        employee_config: 员工配置字典

    Returns:
        Message 对象列表
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

    prefer_zh_output = resolve_prefer_zh_output(state)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(_build_conversation_history(state, max_messages=6))
    # 按语言追加提问：英文提问强制要求英文回复
    if prefer_zh_output:
        messages.append(HumanMessage(content=state["user_query"]))
    else:
        messages.append(
            HumanMessage(content=f"Please reply in English.\n\n{state['user_query']}")
        )

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

    expected_lang = "zh" if resolve_prefer_zh_output(state) else "en"
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
    构建用于答案生成的 LLM 消息列表。

    处理不同场景：
    - 打断：简短确认回复
    - 问候：简单、友好的回复
    - 实时 + 联网搜索：强调网络资料来源
    - 常规 RAG：基于知识库的回答

    Args:
        state: 当前对话状态

    Returns:
        LLM 消息对象列表
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
    # 提前一次性解析输出语言偏好，避免下游各处 state.get(..., True) 静默走中文兜底。
    prefer_zh_output = resolve_prefer_zh_output(state)
    # base_prompt 字段语言守门：当目标输出是英文，但员工配置 (name/description)
    # 是中文（典型客服场景），直接填入英文模板会让 system prompt 混入大段 CJK，
    # 进而拉低 LLM 对"输出英文"的服从度。这里用通用英文占位符替代，保留"员工
    # 角色"概念但去除语言污染。中文输出方向不动（中文配置→中文模板天然一致）。
    if not prefer_zh_output:
        if detect_dominant_language(name) == "zh":
            name = "the assistant"
        if detect_dominant_language(description) == "zh":
            description = "A helpful AI assistant."

    context_text = build_context_text(state)
    source_indicator = get_source_indicator(state)
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

    # 人工概念上下文处理：命中人工概念库后严格依据概念内容回答，
    # 禁止引入资料正文之外的知识，禁止混合其他上下文来源。
    if state.get("concept_retrieval_hit") and state.get("concept_context"):
        concept_context = state.get("concept_context") or {}
        concept_name = concept_context.get("concept_name", "")
        concept_domain = concept_context.get("domain", "")
        concept_content = concept_context.get("content", "")

        if prefer_zh_output:
            concept_requirements = f"""【人工概念资料】
概念名：{concept_name}
领域：{concept_domain}
资料正文：
{concept_content}

回答要求：
1. 只能依据【人工概念资料】回答用户问题；
2. 禁止引入资料正文之外的知识、例子、历史背景、推广形式或应用场景；
3. 如果资料正文没有提供相关内容，明确说明"该概念文档中未提供"；
4. 尽量贴近原文表达；
5. 如果原文已经完整，可以直接转述原文；
6. 不要提及 RAG、知识库、检索过程、数据来源路径；
7. 不要追加与问题无关的反问。

用户问题：
{effective_query}
"""
        else:
            concept_requirements = f"""【Manual Concept Material】
Concept Name: {concept_name}
Domain: {concept_domain}
Content:
{concept_content}

Answer Requirements:
1. Answer ONLY based on the 【Manual Concept Material】 above;
2. Do NOT introduce knowledge, examples, background, or applications beyond the provided content;
3. If the content doesn't provide relevant information, explicitly state "not provided in this concept document";
4. Stay close to the original text expression;
5. If the original text is complete, you may paraphrase it directly;
6. Do NOT mention RAG, knowledge bases, retrieval processes, or data sources;
7. Do NOT add unrelated follow-up questions.

User Question:
{effective_query}
"""

        messages = [
            SystemMessage(content=base_prompt + concept_requirements),
            *_build_conversation_history(state),
            HumanMessage(content=effective_query),
        ]
        return messages

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
1. **摘录可核对的数字**：上证指数、深证成指、创业板指等主要指数的**收盘点位**，仅当下方摘录对**同一指数**有**明确文字**写出（如出现「上证综指收盘××××点」「收报××××点」且与指数名称对应）才可引用该数字。**禁止**从表格碎片、无关列（如百分比、市盈率、排名）、或摘录中未写明的数字推理、拼凑、编造收盘点位。
2. **强约束**：若你无法在摘录中为「某个指数」划出**同一短短语**内同时出现的**指数指称**与**「收报/收盘/收于」+ 具体数字 + `点`」**，则该轮回答**整段不得出现**任何「××××.xx点」类收盘数字（含整数或小数）；不得用记忆或站外常识补数。
3. 若摘录**没有**清楚给出用户所指指数的收盘点数：不得输出具体点位；用一两句话概括摘录中的**涨跌态势、成交额/量能描述、领涨领跌板块**（摘录有写才写）；可一句提示用户在本机行情软件查看最新收盘。
4. 字数控制在约 90 字以内。
5. 保持{tone_desc}的语气风格。
6. **禁止**：外链、「依据」「来源于」及任一网站名称；**禁止**出现「实时行情数据」式来源套话；摘录未出现时不得写出看似精确的指数收盘数字。

上下文信息{source_indicator}：
{context_text}

用户问题：
{effective_query}

"""
            if not prefer_zh_output:
                requirements = f"""IMPORTANT: The user asks about prices/market. The system has retrieved up-to-date web results.

Requirements:
1. **Quotable numbers only**: For major index closing levels (Shanghai SZSE Chinext etc.), cite a numeric close **only** if the excerpt explicitly states it for **that same index** (e.g. clear \"closed at #### points\" paired with the index name). Do **not** invent or stitch digits from unrelated table columns or ratios.
2. **Hard rule**: If you cannot point to a single short phrase in the excerpts where the **index name** and **closing level** (with the word for \"points\" in the target language) clearly co-occur, you must **not** output any fabricated \"####.## points\" figures.
3. If excerpts lack an explicit closing level for the asked index: do **not** output a fabricated point level; summarize direction/moves and sectors **only from excerpts**, and optionally suggest checking a live quotes app for the exact close.
4. Keep it roughly under ~90 words.
5. Keep a {tone_desc} tone.
6. Do NOT name sources/sites, use \"according to …\", or stock phrases like \"real-time market data\"; never output a precise close not present verbatim in context.

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
                if _est_level
                else ""
            )
            _prior_en = (
                f"\nTime-of-day prior (locally derived; use when web context is insufficient):\n"
                f"- Current time: {_est_now} ({_est_weekday_en})\n"
                f"- Estimated congestion level: {_est_level} ({_est_reason})\n"
                if _est_level
                else ""
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
        elif realtime_category == "time":
            requirements = f"""**重要提示**：用户询问日期/时刻类实时信息。

回答要求：
1. **必须以资料中标题含「权威时钟」的一条为准**：其中时刻由服务器按时区 Asia/Shanghai（北京时间/东八区）计算；
2. **禁止**采用其它资料片段里与用户问题矛盾的「时分秒」（常见于网页缓存或未标注时区的 UTC）；
3. **直接回答**用户问的几点、星期几、日期等，一两句话即可；
4. 保持{tone_desc}的语气风格；
5. **禁止**：信息来源说明、网站链接、「信息来源」字样。

上下文信息{source_indicator}：
{context_text}

用户问题：
{effective_query}
"""
            if not prefer_zh_output:
                requirements = f"""IMPORTANT: The user asks for date/time information.

Requirements:
1. If the context includes an authoritative wall-clock entry for Beijing/Asia/Shanghai, treat it as **the only source** for civil time-of-day (HH:MM:SS).
2. **Do NOT** pick conflicting timestamps from other snippets (common cache/timezone mistakes).
3. Answer the asked date/weekday/time directly in plain language.
4. Keep a {tone_desc} tone.
5. Do NOT include sources, links, or the words "source" / "references".

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
        # 当 RAG / web 完全没有提供上下文时，强制 LLM「严格基于上下文回答」会让
        # 它对所有事实题（如「现任 X 是谁」）一律回避或猜测，反而拉低准确率。
        # 这里检测「无上下文」状态，切到「允许 LLM 用自身知识作答」的提示文案：
        #   - 有上下文（RAG 召回 / web 搜到 / FAQ 命中）→ 沿用旧的「严格按上下文」；
        #   - 无上下文 → 允许使用 LLM 训练知识，但仍要求"不知道就说不知道"，
        #     避免出现"硬要求严格按上下文 → LLM 拒答 / 编造"的冲突。
        # 通用判定：retrieved_docs 非空 / web_search_used 为真 / 显式压缩上下文，
        # 任一为真即认为有上下文；否则按"无上下文"分支构建提示。
        has_context = bool(
            state.get("retrieved_docs")
            or state.get("web_search_used")
            or state.get("compressed_context")
        )
        if has_context:
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
        else:
            requirements = f"""回答要求：
1. 没有外部上下文资料，可基于你自身的训练知识直接作答
2. 保持{tone_desc}的语气风格，回答简洁明了、重点突出
3. 如果你**确实不掌握**某项信息，请直接说明无法确认，而不是编造或提供过时信息
4. **禁止：信息来源、网站链接、"信息来源"字样**

用户问题：
{effective_query}

"""
            if not prefer_zh_output:
                requirements = f"""Requirements:
1. There is no external retrieval context for this question; you may rely on your own pretraining knowledge to answer directly.
2. Keep a {tone_desc} tone; be concise and clear.
3. If you genuinely do not know a specific fact, say so explicitly rather than fabricate or guess.
4. Do NOT include sources, links, or the words "source" / "references".

User question:
{effective_query}

"""
            # 「现任 X 职务」类事实查询的"知识时效"加固：
            # DeepSeek-V3 等大模型自身权重里其实记得「李强 2023-03 起任国务院总理」，
            # 但默认 prompt 中保留了「如果不掌握就说不知道」的弱化指令，叠加
            # temperature=0.7（请求默认值，已在 get_streaming_llm 中改为 0），
            # 模型偶尔会主动套用「截至我的知识更新（2021/2023）……」这类训练截止
            # 套话，把上一任「李克强」当作"已知最准答案"复述出来。
            # 这里对命中三元启发式（时间敏感词 + 谁 + 公共职务）的查询追加一段
            # 强约束，明确：(a) 必须给出最新已知任命人物（不是上一任）；
            # (b) 禁止以"训练截止/2023年/2021年"为由复述旧人物；(c) 没有把握时
            # 应当回答"无法确认"而非沿用上一任名单。
            # 这条修饰**仅在没有 web/RAG 上下文**时生效，与原"严格按上下文"分支隔离。
            if _needs_big_world_knowledge(effective_query):
                # 「现任 X 职务是谁」类事实查询的"知识时效"加固。
                #
                # 经验教训（重要）：
                #   早期版本里同时写了「无法确认就说不知道」与「禁止使用旧人物」两条
                #   约束，模型（DeepSeek-V3, temp=0）会优先取「无法确认」这条最安全的
                #   出口，导致所有此类查询都返回「无法确认当前在任者……」，把本来能
                #   正确回忆的事实也压抑掉了——这是把「安全兜底」写得太突出的副作用。
                #
                # 现在采用「明确鼓励 + 弱化兜底」的写法：
                #   - 强调「你的训练语料覆盖到 2024 年之后的主要人事任命，请直接给出
                #     最新一任的姓名」；
                #   - 不再在 prompt 里把「我不知道」列为一个并列选项；
                #   - 仍然禁止以「训练截止 2021/2023」为由复述上一任，避免之前出现的
                #     「现任国务院总理是李克强」这类前任倒灌。
                #
                # 与 get_streaming_llm 中的 temperature=0 配合，使 DeepSeek-V3 在
                # 「现任国务院总理是谁」这类问题上稳定输出「李强」，消除随机性。
                if prefer_zh_output:
                    requirements += (
                        "\n[人事归属事实加固]\n"
                        "本问询询问的是「某公共职务现任由谁担任」（典型如"
                        "现任国务院总理、现任国家主席、现任联合国秘书长 等）。"
                        "请按以下方式作答：\n"
                        "- 你的预训练语料已覆盖到 2024 年及之后的主要公共职务任命事实，"
                        "请**直接给出当前在任者的姓名**（最新一任），并简要补一句任命/就任时间；\n"
                        "- **严禁**以「截至我的训练数据/知识更新（如 2021、2023 等）」为由，"
                        "把已卸任的前任姓名当作「现任」复述出来——即便你对最新人事不完全确定，"
                        "也不要回退到一个你明确知道已经卸任的人；\n"
                        "- 仅在你完全没有任何相关候选人的训练记忆时，才回答"
                        "「无法确认当前在任者，建议查阅官方权威发布」；\n"
                        "- 答案只给出**一个明确人物姓名**，不要并列多个候选。"
                    )
                else:
                    requirements += (
                        "\n[Officeholder fact reinforcement]\n"
                        "This query asks who currently holds a specific public office "
                        "(e.g. current Premier, current President, current UN Secretary-General). "
                        "Please answer as follows:\n"
                        "- Your pretraining corpus already covers major public-office "
                        "appointments through 2024 and beyond. **Output the name of the "
                        "current incumbent directly**, optionally with a short note on when "
                        "they were appointed.\n"
                        "- It is **strictly forbidden** to fall back to a former officeholder "
                        'by citing your "training-data cutoff" (e.g. 2021/2023). Even if '
                        "you are not fully certain about the very latest changes, do NOT "
                        "name someone you know has already left office.\n"
                        "- Only if you have no relevant candidate at all in your training "
                        'memory, reply: "I cannot confirm the current officeholder; please '
                        'check official sources."\n'
                        "- Output exactly one definitive name; do not list multiple candidates."
                    )
        # 实时查询但未获得联网结果：仍需保持时间一致性，避免模型自行混入错误年份。
        if state.get("is_realtime_query", False):
            requirements = (
                requirements
                + "\n\n"
                + _build_realtime_temporal_guardrail(prefer_zh_output)
            )
            if prefer_zh_output:
                requirements += "\n若当前上下文无法支撑唯一结论，请明确说明无法确认，不要补充未经证实的年份或日期。"
            else:
                requirements += (
                    "\nIf context is insufficient for a unique temporal answer, clearly state uncertainty "
                    "and do not introduce unverified years or dates."
                )

    system_prompt = base_prompt + "\n" + system_time_context + "\n\n" + requirements
    # 语言隔离声明（紧贴 user 消息的最后一段 system 文本）：
    # 当 web context / 历史对话 / 角色描述 中出现非目标语言（如开了海外代理后
    # Tavily 返回大段日文资料），仅靠 user message 末尾的语言锁权重不足以
    # 抵消 system 里的语料密度，需要在 system 末尾再补一道明确指令。
    # 改动 prompt-only，不动数据流。
    if not prefer_zh_output:
        system_prompt = (
            "Answer in English only.\n\n"
            + system_prompt
            + "\n\n[FINAL LANGUAGE CONSTRAINT] Regardless of the language used in the "
            "web context, chat history, or role description above, your final answer "
            "MUST be written entirely in English. Do not output any Chinese / Japanese / "
            "other CJK characters. Treat non-English text in the context as reference "
            "information only — translate any key facts you cite into English."
        )
    else:
        system_prompt = (
            system_prompt
            + "\n\n[最终语言约束] 无论上文资料、历史对话或角色描述中出现何种语言，"
            "你的最终回答必须完整使用简体中文。如需引用上文资料中的非中文要点，"
            "请将其翻译为简体中文后再使用，不要直接照搬原文语言。"
        )

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(_build_conversation_history(state, max_messages=12))
    # 语言锁定（最后一道墙）：放在 user 问句之后，利用 LLM 的 recency bias，
    # 抵消"base_prompt 中的员工配置中文字段 + 历史对话语言 + web context 异种语言"
    # 等多重稀释源。问句前置的"Please answer in English only."权重不够，
    # 需要在最贴近模型生成位置再加一遍。
    if prefer_zh_output:
        zh_lock = (
            "\n\n请使用简体中文回答。"
            "若上文资料/历史对话中含有其它语言，仅作为信息参考，"
            "你的最终回答必须是简体中文。"
        )
        messages.append(HumanMessage(content=effective_query + zh_lock))
    else:
        en_lock = (
            "\n\nIMPORTANT: Reply strictly in English. "
            "Do not output any Chinese characters at all. "
            "If the context or chat history contains non-English content, "
            "treat it only as reference; your final answer must be in English."
        )
        messages.append(HumanMessage(content=effective_query + en_lock))

    # 汇总日志：方便诊断"语言锁是否真的把 CJK 清出 LLM 输入"。
    # 当 prefer_zh_output=False（英文输出）但 system 仍含较多 CJK 字符，
    # 说明还有未识别的污染源。生产环境可通过日志级别关闭。
    try:
        sys_text = messages[0].content
        cjk_in_sys = sum(1 for ch in sys_text if "\u4e00" <= ch <= "\u9fff")
        logger.info(
            f"build_generation_messages: prefer_zh={prefer_zh_output}, "
            f"history_msgs={len(messages) - 2}, system_chars={len(sys_text)}, "
            f"system_cjk_chars={cjk_in_sys}, "
            f"effective_query={effective_query[:60]!r}"
        )
    except Exception:
        pass

    return messages


def build_math_generation_messages(
    state: ConversationState,
    include_system_prompt: bool = True,
    math_runtime_mode: str = "direct",
    math_runtime_lang: str = "zh",
) -> List:
    """
    构建数学问题的模型消息。

    系统提示词统一来自 ``MathAgentService.get_system_prompt(mode, lang)``：
    - ``direct`` 模式：``include_system_prompt=True``，注入对应语言的 direct system prompt；
    - ``cot`` / ``tir`` 模式：``include_system_prompt=False``，模式提示词由 Qwen-Agent
      自己的 Assistant / TIRMathAgent system_message 决定，这里不再额外注入，避免两套
      模式提示互相叠加。

    数学历史一律禁用（``history_msgs=[]``）。

    Args:
        state: Current conversation state
        include_system_prompt: 是否注入数学 system prompt
        math_runtime_mode: 数学运行模式（direct / cot / tir），决定使用哪套提示词
        math_runtime_lang: 提示词语言（zh / en）

    Returns:
        List of Message objects for math LLM
    """
    messages = []
    if include_system_prompt:
        system_prompt = MathAgentService.get_system_prompt(
            mode=math_runtime_mode,
            lang=math_runtime_lang,
        )
        messages.append(SystemMessage(content=system_prompt))

    # 数学模型一律不携带历史对话：历史上下文会污染推理（如敏感词命中后的拒答
    # 话术、空 assistant 消息等被一并送入），导致数学模型 prompt 串入噪声。
    history_msgs: List = []

    # 数学题统一以 state["user_query"] 为「当前问题」：post_classification_preprocess 会
    # 把数学题 LaTeX 化后写回 state["user_query"]，所以这里读 user_query 才能拿到转换后题干。
    # （数学题在 resolve_context_query 中均设置 rewritten_query == user_query，不会改写题干。）
    effective_query = (state.get("user_query") or "").strip()

    # 数学追问：把历史上下文作为「对话上下文」提供给数学模型，但不拼进当前问题、
    # 也不参与 word_to_latex。完整数学题 math_context_used 为 False，不会注入。
    math_context_text = state.get("math_context_text")
    math_context_used = bool(state.get("math_context_used"))

    if math_context_used and math_context_text:
        user_content = (
            f"[对话上下文]\n{math_context_text}\n\n"
            f"[当前问题]\n{effective_query}"
        )
    else:
        user_content = effective_query

    messages.append(HumanMessage(content=user_content))

    logger.info(
        "build_math_generation_messages [qwen_math]: "
        f"include_system_prompt={include_system_prompt}, "
        f"math_runtime_mode={math_runtime_mode}, "
        f"math_runtime_lang={math_runtime_lang}, "
        f"history_msgs={len(history_msgs)}, "
        f"math_history_disabled=True, "
        f"math_context_used={math_context_used}, "
        f"effective_query={effective_query!r}"
    )
    return messages

    # # 以下在 qwen数学模型的情况下都不生效
    # # --- 通用数学路径 ---
    # if _is_simple_math_query(raw_query):
    #     # 简单计算题：保持极简输出（不注入历史，避免噪声拉高首字延迟）
    #     messages = [SystemMessage(content=MATH_SIMPLE_SYSTEM_PROMPT)]
    #     messages.append(HumanMessage(content=raw_query))
    #     return messages

    # # 非简单题：统一使用"解题行为流程"+"公式库"，不再针对具体题目写死分支
    # sys_prompt = MATH_SYSTEM_PROMPT + "\n\n" + GEOMETRY_FORMULA_BOOK
    # messages: List = [SystemMessage(content=sys_prompt)]

    # # 注入最近对话历史，与 build_generation_messages 行为对齐：
    # # - 动态上下文记忆判定为 "unrelated" 时 _build_conversation_history 自动返回 []
    # # - 跨语言历史会被语言一致性过滤掉，避免污染输出语言
    # history_msgs = _build_conversation_history(state, max_messages=12)
    # messages.extend(history_msgs)

    # # 优先用消歧改写后的问句（_select_llm_facing_query 内部已做语言一致性兜底）；
    # # 再做一次 π / pi / 派 归一化，避免历史 / 改写过程引入的符号写法差异。
    # effective_query = _select_llm_facing_query(state)

    # messages.append(HumanMessage(content=effective_query))

    # logger.info(
    #     "build_math_generation_messages [math_llm]: "
    #     f"history_msgs={len(history_msgs)}, "
    #     f"context_dependence={state.get('context_dependence')!r}, "
    #     f"query_rewritten={state.get('query_rewritten', False)}, "
    #     f"effective_query={effective_query[:80]!r}"
    #     f"sys_prompt={sys_prompt}"
    # )

    # return messages


def _is_simple_math_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    if len(q) > 28:
        return False
    complex_markers = (
        "证明",
        "推导",
        "分析",
        "为什么",
        "思路",
        "过程",
        "几何",
        "应用题",
        "函数",
        "方程组",
    )
    if any(m in q for m in complex_markers):
        return False
    has_number = bool(re.search(r"[0-9一二三四五六七八九十百千万两零]", q))
    has_op = any(
        op in q for op in ("+", "-", "*", "/", "加", "减", "乘", "除", "×", "÷", "等于")
    )
    asks_value = any(k in q for k in ("等于几", "多少", "=?", "＝", "="))
    return has_number and has_op and asks_value


# 「需要广博/最新世界知识」的关键词组合：典型代表是「现任 / 当前 + 公共职务 + 谁」。
# 本地小模型（如 qwen3:14b）这类问题的命中率很低——要么没训练、要么回避不答；
# 远端 DeepSeek 等大模型对这类知识覆盖完整得多。在「混合路由」模式里，把命中
# 此模式的问题手动抬高复杂度分数，让它们越过 ``complexity_threshold`` 走远端 LLM。
# 词表与正则集中在这里维护，避免把"该走哪个模型"的判断散落到各处。
import re as _re_complexity  # 避免与外部 re 命名冲突，仅在本函数局部使用。

_BIG_KNOWLEDGE_TIME_RE = _re_complexity.compile(
    r"(目前|当前|现任|现阶段|本届|现今|现在在任)"
)
_BIG_KNOWLEDGE_WHO_RE = _re_complexity.compile(
    r"(谁|哪位|哪一位|哪个|哪一个|是谁|何人|哪几位)"
)
_BIG_KNOWLEDGE_OFFICE_RE = _re_complexity.compile(
    r"(总理|主席|总统|首相|省长|市长|县长|区长|州长|"
    r"部长|司长|厅长|局长|处长|科长|主任|书记|总书记|阁员|内阁|"
    r"领导人|领导|元首|大使|代表|议员|議員|"
    r"秘书长|秘书|理事长|会长|主任|议长|主席团)"
)
_BIG_KNOWLEDGE_TIME_RE_EN = _re_complexity.compile(
    r"\b(current|present|incumbent|now)\b", _re_complexity.IGNORECASE
)
_BIG_KNOWLEDGE_WHO_RE_EN = _re_complexity.compile(
    r"\bwho(?:'s|\s+is|\s+are|\s+was|\s+were)\b", _re_complexity.IGNORECASE
)
_BIG_KNOWLEDGE_OFFICE_RE_EN = _re_complexity.compile(
    r"\b(president|premier|prime\s+minister|mayor|governor|secretary|chancellor|minister)\b",
    _re_complexity.IGNORECASE,
)


def _needs_big_world_knowledge(query: str) -> bool:
    """是否「需要广博世界知识」类查询（现任/当前 + 职务 + 谁）。"""
    if not query:
        return False
    if (
        _BIG_KNOWLEDGE_TIME_RE.search(query)
        and _BIG_KNOWLEDGE_WHO_RE.search(query)
        and _BIG_KNOWLEDGE_OFFICE_RE.search(query)
    ):
        return True
    if (
        _BIG_KNOWLEDGE_TIME_RE_EN.search(query)
        and _BIG_KNOWLEDGE_WHO_RE_EN.search(query)
        and _BIG_KNOWLEDGE_OFFICE_RE_EN.search(query)
    ):
        return True
    return False


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

    # 「现任 X 是谁」类需要最新/广博世界知识：本地小模型回答常常回避或错答，
    # 强制把复杂度抬到阈值以上让 hybrid 路由切到远端大模型（DeepSeek 等）。
    # 这里只对命中三元模式（时间敏感词 + 谁 + 公共职务）的问题加分，避免把
    # 「他是哪位科学家」这类历史题也牵连过去——它们的「时间敏感词」一般缺失。
    if _needs_big_world_knowledge(query):
        score = max(score, 8.0)

    return max(0.0, min(10.0, score))
