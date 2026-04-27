"""
Node implementations for LangGraph conversation workflow.

This module contains all node functions that process the conversation state:
- Configuration loading nodes
- Input validation nodes
- Query classification nodes
- Complexity evaluation nodes
- Web search nodes
- Answer generation nodes
- Conversation saving nodes

Note: Simplified workflow using RAGAnything for RAG retrieval.
Removed nodes: intent_recognition, knowledge_retrieval, grade_documents,
               compress_context, match_faq, rewrite_query
"""
import hashlib
import time
from datetime import datetime
import re
from datetime import timedelta

from langchain_community.tools.tavily_search import TavilySearchResults

from app.core.config import settings
from app.core.database import get_database
from app.core.logging import get_logger
from app.models.database import ConversationModel, SessionModel
from app.services.conversation.conversation_state import (
    ConversationState,
    GREETING_KEYWORDS,
    DEFAULT_SENSITIVE_WORDS,
    DEFAULT_SENSITIVE_WORDS_LOWER,
)
from app.services.conversation.conversation_helpers import (
    time_node,
    heuristic_complexity,
    build_generation_messages,
    build_math_generation_messages
)

logger = get_logger(__name__)

# 相对日历日的偏移（天）：列表顺序很重要——长的、更具体的词必须排在前面，
# 例如「大前天」若排在「前天」之后会错判。
_RELATIVE_DAY_OFFSET_ZH: tuple[tuple[str, int], ...] = (
    ("大前天", -3),
    ("前天", -2),
    ("昨天", -1),
    ("今天", 0),
    ("明天", 1),
    ("后天", 2),
    ("大后天", 3),
)
_RELATIVE_DAY_OFFSET_EN: tuple[tuple[str, int], ...] = (
    ("three days ago", -3),
    ("day before yesterday", -2),
    ("yesterday", -1),
    ("today", 0),
    ("tomorrow", 1),
    ("day after tomorrow", 2),
    ("three days later", 3),
)


def _relative_calendar_day_offset(q_lower: str) -> int | None:
    """命中「今天/昨天/前天/大前天…」等表述时返回 timedelta 天数偏移，否则 None。"""
    for phrase, off in _RELATIVE_DAY_OFFSET_ZH:
        if phrase in q_lower:
            return off
    for phrase, off in _RELATIVE_DAY_OFFSET_EN:
        if phrase in q_lower:
            return off
    return None


def _split_user_questions(text: str) -> list[str]:
    """按标点切分用户问题，返回清理后的问题列表"""
    if not text:
        return []
    # 按中英文句末标点分割
    parts = re.split(r"[?？!！。；;\n]+", text)
    return [p.strip() for p in parts if p and p.strip()]


_WEEKDAY_CN_INDEX: dict[str, int] = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_WEEK_PREFIX_OFFSET: dict[str, int] = {
    "上上上周": -3,
    "上上周": -2,
    "上周": -1,
    "本周": 0,
    "这周": 0,
    "本星期": 0,
    "这星期": 0,
    "下周": 1,
    "下下周": 2,
    "下下下周": 3,
}


def _parse_weekday_date_cn(q_lower: str, now: datetime) -> tuple[str, datetime] | None:
    """解析中文星期日期（上周六/下周一），返回标签+日期"""
    # 必须含周/星期关键词，避免误判
    if ("周" not in q_lower) and ("星期" not in q_lower) and not any(
        p in q_lower for p in _WEEK_PREFIX_OFFSET.keys()
    ):
        return None

    # 正则提取前缀+星期
    m = re.search(
        r"(?P<prefix>上上上周|上上周|上周|本周|这周|本星期|这星期|下周|下下周|下下下周|上星期|本星期|这星期|下星期|上上星期|下下星期)?(?:(?:周|星期)?)(?P<wd>[一二三四五六日天])",
        q_lower,
    )
    if not m:
        return None
    prefix = m.group("prefix") or "本周"
    wd = m.group("wd")
    prefix_norm = prefix.replace("星期", "周")
    if prefix_norm not in _WEEK_PREFIX_OFFSET:
        prefix_norm = prefix_norm.replace("上上周", "上上周").replace("下下周", "下下周")
    # 获取偏移量与星期索引
    week_off = _WEEK_PREFIX_OFFSET.get(prefix_norm)
    wd_idx = _WEEKDAY_CN_INDEX.get(wd)
    if week_off is None or wd_idx is None:
        return None

    # 计算目标日期（周一为一周起点）
    monday = now.date() - timedelta(days=now.weekday())
    target_date = monday + timedelta(days=week_off * 7 + wd_idx)
    # 生成标签
    if "星期" in q_lower or "星期" in (prefix or ""):
        label = prefix_norm.replace("周", "星期") + wd
    else:
        label = prefix_norm + wd
    target_dt = datetime.combine(target_date, now.time())
    return label, target_dt


_DAY_LABEL_ZH: dict[int, str] = {
    -3: "大前天",
    -2: "前天",
    -1: "昨天",
    0: "今天",
    1: "明天",
    2: "后天",
    3: "大后天",
}
_DAY_LABEL_EN: dict[int, str] = {
    -3: "Three days ago",
    -2: "The day before yesterday",
    -1: "Yesterday",
    0: "Today",
    1: "Tomorrow",
    2: "The day after tomorrow",
    3: "Three days from now",
}


# =============================================================================
# Base class for node implementations
# =============================================================================
class ConversationNodes:
    """
    Container class for all workflow nodes.

    Each node is an async method that takes a ConversationState
    and returns an updated ConversationState.
    """

    def __init__(self, workflow_instance):
        """
        Initialize nodes with reference to parent workflow.

        Args:
            workflow_instance: The ConversationWorkflow instance
        """
        self.workflow = workflow_instance

    # -------------------------------------------------------------------------
    # Workflow Nodes - Configuration Loading
    # -------------------------------------------------------------------------
    async def load_employee_config(self, state: ConversationState) -> ConversationState:
        """
        Load digital employee configuration from MongoDB.

        Retrieves employee-specific settings including:
        - Basic info (name, role, description)
        - Personality (tone, style, formality)
        - Capabilities (web_search_enabled, kb_ids)
        - FAQ settings (faq_sim_threshold, faq_top_k)

        Args:
            state: Current conversation state

        Returns:
            Updated state with employee_config populated

        Raises:
            ValueError: If employee_id not found in database
        """
        async with time_node("load_employee_config", state):
            db = await get_database()
            employee = await db.digital_employee_configs.find_one({"employee_id": state["employee_id"]})

            if not employee:
                error_msg = f"Employee config not found: employee_id={state['employee_id']}"
                logger.error(f"{error_msg}", exc_info=False)
                raise ValueError(error_msg)

            employee.pop("_id", None)
            state["employee_config"] = employee

            # 兼容两种 kb_ids 存储路径: knowledge.kb_ids 或根级别 kb_ids
            kb_ids = None
            knowledge = employee.get("knowledge")
            if knowledge and isinstance(knowledge, dict):
                kb_ids = knowledge.get("kb_ids")
            if not kb_ids:
                kb_ids = employee.get("kb_ids", [])
            if kb_ids is None:
                kb_ids = []

            logger.info(
                f"Employee config loaded: employee_id={state['employee_id']}, "
                f"name={employee.get('name')}, kb_ids={kb_ids}, kb_count={len(kb_ids)}"
            )

        return state

    async def load_session_context(self, state: ConversationState) -> ConversationState:
        """
        Load or create session context from MongoDB.

        Handles:
        - Loading existing session with message history
        - Creating new session for first-time users
        - Updating last_activity timestamp

        Args:
            state: Current conversation state

        Returns:
            Updated state with context populated
        """
        async with time_node("load_session_context", state):
            try:
                db = await get_database()
                session = await db.sessions.find_one({"session_id": state["session_id"]})

                if session:
                    # Load recent messages (last 10 for context)
                    state["context"] = {
                        "messages": session.get("context_messages", [])[-10:],
                        "message_count": session.get("message_count", 0)
                    }
                    # Update activity timestamp
                    await db.sessions.update_one(
                        {"session_id": state["session_id"]},
                        {"$set": {"last_activity": datetime.now()}}
                    )
                else:
                    # Create new session
                    session_model = SessionModel(
                        session_id=state["session_id"],
                        user_id=state["user_id"],
                        employee_id=state["employee_id"],
                        status="active",
                        message_count=0
                    )
                    await db.sessions.insert_one(session_model.model_dump())
                    state["context"] = {"messages": [], "message_count": 0}
                    state["sources"] = []  # 初始化 sources 列表

                logger.debug(
                    f"Session context loaded: session_id={state['session_id']}, "
                    f"message_count={state['context']['message_count']}"
                )

            except Exception as e:
                logger.error(f"Failed to load session context: {str(e)}", exc_info=True)
                state["context"] = {"messages": [], "message_count": 0}

        return state

    # -------------------------------------------------------------------------
    # Workflow Nodes - Input Processing
    # -------------------------------------------------------------------------
    async def validate_input(self, state: ConversationState) -> ConversationState:
        """
        Validate and sanitize user input.

        Performs basic validation:
        - Check for empty or malicious input
        - Detect sensitive content using default + employee-specific words

        Args:
            state: Current conversation state

        Returns:
            Updated state with validation results
        """
        async with time_node("validate_input", state):
            query = state["user_query"].strip().lower()

            employee_config = state.get("employee_config") or {}

            # Build sensitive words list (default + employee-specific)
            sensitive_words = set(DEFAULT_SENSITIVE_WORDS)

            # Add employee-specific sensitive words from safe_rule.sensitive_ids
            safe_rule = employee_config.get("safe_rule") or {}
            sensitive_ids = safe_rule.get("sensitive_ids") or []
            employee_words_lower: set = set()
            if sensitive_ids:
                # Load sensitive words from database
                db = get_database()
                if sensitive_ids:
                    cursor = db.thesaurus_sensitive.find(
                        {"thesaurus_id": {"$in": sensitive_ids}},
                        {"word": 1, "_id": 0}
                    )
                    sensitive_docs = await cursor.to_list(length=None)
                    employee_words = [doc.get("word", "") for doc in sensitive_docs if doc.get("word")]
                    sensitive_words.update(employee_words)
                    employee_words_lower = {w.lower() for w in employee_words if w}

            # Check if query contains any sensitive word (using pre-computed lowercase set)
            check_words_lower = DEFAULT_SENSITIVE_WORDS_LOWER | employee_words_lower
            query_lower = query.lower()
            has_sensitive = any(word in query_lower for word in check_words_lower)

            state["has_sensitive"] = has_sensitive

            # DEBUG: 输出敏感词检测结果
            matched_words = [w for w in sensitive_words if w.lower() in query_lower]
            logger.info(
                f"[DEBUG] Sensitive check | query={query[:50]} | "
                f"words_count={len(sensitive_words)} | has_sensitive={has_sensitive} | "
                f"matched={matched_words[:5]}"
            )

            if has_sensitive:
                logger.info(
                    f"Sensitive word detected in query, user_id={state.get('user_id')}, "
                    f"employee_id={state.get('employee_id')}, query={query[:100]}"
                )
        return state

    # -------------------------------------------------------------------------
    # Workflow Nodes - Query Classification (Early Exit)
    # -------------------------------------------------------------------------
    async def classify_query_type(self, state: ConversationState) -> ConversationState:
        """
        Query Classification - 快速识别查询类型并路由。

        检测顺序 (从快到慢):
        1. 问候语检测 (关键词匹配)
        2. 实时查询检测 (关键词匹配)
        3. 其他 (继续正常流程)

        Args:
            state: Current conversation state

        Returns:
            Updated state with query_type classification
        """
        async with time_node("classify_query_type", state):
            query = state["user_query"].strip().lower()

            # 预判是否含业务/实时信号（避免“请问xx天气”被误判为 greeting）
            realtime_signal = False
            if settings.realtime_query_enabled:
                # 实时关键词（与 realtime_keywords_ordered 保持一致）
                _realtime_keywords_for_guard = [
                    # weather
                    "天气", "气温", "降雨", "降水", "温度",
                    # 新闻/政务
                    "新闻", "热点", "最新", "资讯", "动态", "头条",
                    "目前", "现任", "现在", "当前", "总理", "国务院总理", "国家总理", "国家领导人",
                    # 行情
                    "股价", "汇率", "行情", "股市", "价格", "金价", "银价", "油价", "多少钱",
                    # 时间
                    "今天", "明天", "昨天", "前天", "大前天", "后天", "大后天", "几月几号", "几号", "几点", "日期",
                    # 英文
                    "current", "premier", "prime minister", "state council", "president",
                    "today", "date", "time", "weather", "news", "market",
                ]
                realtime_signal = any(k in query for k in _realtime_keywords_for_guard)

            # 1. 检测问候语
            for category, keywords in GREETING_KEYWORDS.items():
                if any(kw in query for kw in keywords):
                    # 若同时包含明显的业务/实时信号，则不要走 greeting，继续后续 realtime 检测
                    if realtime_signal:
                        break
                    state["intent"] = "greeting"
                    state["complexity_score"] = 0.0
                    state["complexity_reason"] = "greeting"
                    state["is_realtime_query"] = False
                    # 添加 greeting source
                    state["sources"].append({
                        "type": "text",
                        "from": "greeting",
                        "text": query,
                        "citations": []
                    })
                    logger.info(f"Query classified: greeting: category={category}")
                    return state

            # 2. 检测实时查询
            if settings.realtime_query_enabled:
                # 天气追问兜底：识别“明天呢？”等短问，避免误判为时间
                follow_up_weather_queries = {
                    "明天呢", "明天呢?", "明天呢？",
                    "那明天呢", "那明天呢?", "那明天呢？",
                    "明天怎么样", "明天怎么样?", "明天怎么样？",
                }
                if query in follow_up_weather_queries:
                    prev_user_msg = ""
                    for msg in reversed(state.get("context", {}).get("messages", [])):
                        if msg.get("role") == "user":
                            prev_user_msg = (msg.get("content") or "").lower()
                            break
                    if prev_user_msg and any(kw in prev_user_msg for kw in ["天气", "气温", "降雨", "降水", "温度"]):
                        state["is_realtime_query"] = True
                        state["realtime_category"] = "weather"
                        state["realtime_detect_reason"] = "follow_up:weather"
                        state["intent"] = "general_query"
                        logger.info("Query classified: realtime follow-up: category=weather")
                        return state

                # 按优先级匹配：天气/新闻/行情 > 时间，避免“今天天气”误判时间
                realtime_keywords_ordered = [
                    ("weather", ["天气", "气温", "降雨", "降水", "温度"]),
                    # news 也包含英文“当前领导/职位”类问法，确保走联网拿最新信息
                    ("news", [
                        "新闻", "热点", "最新", "资讯", "动态", "头条",
                        "目前", "现任", "现在", "当前", "最新",
                        "总理", "国务院总理", "国家总理", "国家领导人", "领导是谁", "是哪位领导", "是谁",
                        "current", "who is the current", "who is current",
                        "premier", "prime minister", "state council", "president",
                        "who is the", "incumbent",
                    ]),
                    ("market", ["股价", "汇率", "行情", "股市", "价格", "金价", "银价", "油价", "多少钱"]),
                    # 中英实时触发词，避免LLM返回过期/幻觉内容
                    ("time", [
                        "今天", "明天", "昨天", "前天", "大前天", "后天", "大后天",
                        "最近", "现在", "本周", "本月", "当前", "几月几号", "几号", "几点", "日期",
                        "today", "date", "what's the date", "what is the date", "what day is it", "current date", "today's date",
                        "time", "what time", "current time", "now"
                    ]),
                ]
                for category, keywords in realtime_keywords_ordered:
                    matched_kw = next((kw for kw in keywords if kw in query), None)
                    if matched_kw:
                        state["is_realtime_query"] = True
                        state["realtime_category"] = category
                        state["realtime_detect_reason"] = f"keyword:{matched_kw}"
                        state["intent"] = "general_query"
                        logger.info(
                            f"Query classified: realtime: category={category}, matched={matched_kw}"
                        )
                        return state
            
            # 4. 默认为一般查询
            state["is_realtime_query"] = False
            state["intent"] = "general_query"
            logger.debug("Query classified as general")
        return state

    # -------------------------------------------------------------------------
    # Math Problem Detection
    # -------------------------------------------------------------------------
    # 概念性问题排除关键词（不算数学题）
    CONCEPT_KEYWORDS = ["是什么", "什么是", "介绍", "解释", "定义", "概念"]
    # 数学操作关键词
    MATH_OPERATION_KEYWORDS = ["求", "计算", "解", "证明", "推导", "化简"]
    # 数学对象关键词
    MATH_OBJECT_KEYWORDS = ["函数", "方程", "不等式", "集合", "数列", "三角函数", "导数", "积分", "极限", "椭圆", "双曲线", "抛物线"]

    async def check_math_problem(self, state: ConversationState) -> ConversationState:
        """
        数学问题检测节点 - 检测用户查询是否为数学问题。

        使用基于规则的模式匹配：
        - 排除概念性问题（"什么是函数"、"介绍一下三角函数"等）
        - 检测数学操作+数学对象（"求函数值域"、"解方程"等）

        Args:
            state: Current conversation state

        Returns:
            Updated state with is_math_problem populated
        """
        async with time_node("check_math_problem", state):
            query = state["user_query"].strip()

            # 执行数学问题检测
            is_math, reason = self._detect_math_problem(query)
            state["is_math_problem"] = is_math

            logger.info(
                f"Math detection result: is_math={is_math}, reason={reason}, "
                f"query={query[:50]}"
            )

            # 如果是数学问题，添加来源信息
            if is_math:
                state["sources"].append({
                    "type": "text",
                    "from": "phi4_math",
                    "text": query,
                    "citations": []
                })
                logger.info(f"Math problem source added | query={query[:50]}")

        return state

    @staticmethod
    def _detect_math_problem(query: str) -> tuple[bool, str]:
        """
        基于规则的模式匹配检测数学问题。

        规则:
        1. 概念性问题排除（"是什么"、"什么是"等 + 短问题）
        2. 同时包含数学操作关键词 + 数学对象关键词 → 数学题
        3. 其他 → 普通查询

        Args:
            query: 用户查询

        Returns:
            (is_math, reason) - 是否为数学问题及原因
        """
        query_lower = query.strip().lower()

        # 1. 概念性问题排除
        for concept_kw in ConversationNodes.CONCEPT_KEYWORDS:
            if concept_kw in query_lower and len(query) < 50:
                return (False, "concept_question")

        # 2. 同时包含数学操作+数学对象 → 数学题
        has_operation = any(kw in query_lower for kw in ConversationNodes.MATH_OPERATION_KEYWORDS)
        has_object = any(kw in query_lower for kw in ConversationNodes.MATH_OBJECT_KEYWORDS)

        if has_operation and has_object:
            return (True, "math_problem")

        return (False, "general_query")

    @staticmethod
    def route_after_math_check(state: ConversationState) -> str:
        """
        路由决策: 数学检测后的下一步。

        Args:
            state: Current conversation state

        Returns:
            目标节点名称 (math/normal)
        """
        if state.get("is_math_problem", False):
            return "math"
        return "normal"

    @staticmethod
    def route_after_classification(state: ConversationState) -> str:
        """
        路由决策: 查询分类后的下一步。

        Args:
            state: Current conversation state

        Returns:
            目标节点名称 (greeting/realtime/normal)
        """
        intent = state.get("intent")
        # 问候语直接跳到生成答案
        if intent == "greeting":
            return "greeting"
        if state.get("is_realtime_query"):
            return "realtime"
        return "normal"

    # -------------------------------------------------------------------------
    # Workflow Nodes - Complexity Evaluation
    # -------------------------------------------------------------------------
    async def evaluate_complexity(self, state: ConversationState) -> ConversationState:
        """
        Query Complexity Evaluation - 评估问题复杂度，决定使用本地还是外部模型。

        使用本地 Ollama 模型快速评估问题复杂度（0-10分）：
        - 0-3分：简单问题 - 本地 Ollama 足够
        - 4-6分：中等复杂 - 可用本地，必要时用外部
        - 7-10分：复杂问题 - 使用外部 API 模型

        复杂度评估维度：
        1. 问题长度（越长越复杂）
        2. 问题类型（简单问答 vs 复杂推理）
        3. 是否需要多步推理
        4. 是否需要综合多个信息源
        5. 意图是否清晰

        Args:
            state: Current conversation state

        Returns:
            Updated state with complexity_score and complexity_reason populated
        """
        async with time_node("evaluate_complexity", state):
            query = state["user_query"].strip()

            # 默认复杂度
            state["complexity_score"] = 3.0
            state["complexity_reason"] = "default"

            # 非混合模式：跳过复杂度评估
            routing_mode = getattr(settings, 'llm_routing_mode', 'local_only')
            if routing_mode != 'hybrid':
                logger.debug(
                    f"Complexity evaluation skipped (routing_mode={routing_mode})"
                )
                return state

            # 问候语检测已在 classify_query_type 节点中处理，此处无需重复

            # 快速启发式评估（跳过 LLM 调用以提高响应速度）
            # LLM 评估虽然更准确，但会增加 15+ 秒延迟，影响用户体验
            score = heuristic_complexity(query)

            # 根据查询特征优化 reason 分类
            reason = "heuristic"
            if any(kw in query for kw in ["集合", "函数", "定理", "公式", "定义", "什么是"]):
                reason = "数学概念"
            elif any(kw in query for kw in ["证明", "推导", "为什么"]):
                reason = "数学推理"
            elif any(kw in query for kw in ["分析", "比较", "总结"]):
                reason = "综合分析"

            state["complexity_score"] = score
            state["complexity_reason"] = reason

            logger.info(
                f"Complexity evaluated (heuristic): {score}/10 - {reason}"
            )

        return state

    # -------------------------------------------------------------------------
    # Workflow Nodes - Web Search
    # -------------------------------------------------------------------------
    async def web_search(self, state: ConversationState) -> ConversationState:
        """
        Web Search - Fetch realtime information from the internet.

        Uses Tavily Search API to get current data for:
        - Realtime queries (weather, news, prices)
        - Low relevance fallback (when knowledge base doesn't have answer)

        Checks:
        1. Web search enabled in settings
        2. Tavily API key configured
        3. Employee has web_search capability

        Args:
            state: Current conversation state

        Returns:
            Updated state with web_search_results populated
        """
        async with time_node("web_search", state):
            try:
                # Check if web search is enabled
                if not settings.web_search_enabled:
                    logger.info("Web search is disabled in settings")
                    state["web_search_results"] = []
                    state["web_search_used"] = False
                    state["web_search_error"] = None
                    return state

                if not settings.tavily_api_key:
                    logger.warning("Tavily API key is not configured")
                    state["web_search_results"] = []
                    state["web_search_used"] = False
                    state["web_search_error"] = "Tavily API key not configured"
                    return state

                # Check employee config for web search permission
                employee_config = state.get("employee_config", {})
                capabilities = employee_config.get("capabilities", {})
                if not capabilities.get("web_search_enabled", True):
                    logger.info(f"Web search disabled for employee: {state.get('employee_id')}")
                    state["web_search_results"] = []
                    state["web_search_used"] = False
                    state["web_search_error"] = None
                    return state

                query = state["user_query"]
                logger.info(
                    f"Web search started: query={query[:100]}, "
                    f"is_realtime={state.get('is_realtime_query')}"
                )

                # Perform search
                web_search_tool = TavilySearchResults(
                    max_results=5,  # 返回结果数量，默认 5
                    search_depth="basic" # 搜索深度："basic" (免费) 或 "advanced" (付费)
                )
                search_results = await web_search_tool.ainvoke({"query": query})

                # Format results
                formatted_results = []
                api_error_message = None

                if search_results:
                    # 先判断 search_results 类型
                    if not isinstance(search_results, list):
                        # 处理非列表返回值（可能是错误字符串）
                        results_str = str(search_results)
                        if "401" in results_str or "Unauthorized" in results_str or "authentication" in results_str.lower():
                            api_error_message = "当前无法进行网络检索（API Key 可能过期或无效）"
                            logger.warning(
                                f"Web search API error: error_type={type(search_results).__name__}, "
                                f"error={results_str[:200]}"
                            )
                    else:
                        # 是列表，检查每个元素
                        for result in search_results:
                            if not isinstance(result, dict):
                                result_str = str(result)
                                if "401" in result_str or "Unauthorized" in result_str or "authentication" in result_str.lower():
                                    api_error_message = "当前无法进行网络检索（API Key 可能过期或无效）"
                                    logger.warning(
                                        f"Web search API error in result: error={result_str[:200]}"
                                    )
                                    break

                        # 只有没有 API 错误时才格式化结果
                        if not api_error_message:
                            for i, result in enumerate(search_results[:settings.web_search_max_results], 1):
                                if not isinstance(result, dict):
                                    logger.warning(
                                        f"Invalid search result type: result_type={type(result).__name__}, "
                                        f"result={str(result)[:200]}"
                                    )
                                    continue

                                formatted_results.append({
                                    "rank": i,
                                    "title": result.get("title", ""),
                                    "url": result.get("url", ""),
                                    "content": result.get("content", "")[:500],
                                    "score": result.get("score", 0.0)
                                })

                state["web_search_results"] = formatted_results
                state["web_search_used"] = len(formatted_results) > 0
                state["web_search_error"] = api_error_message

                # 添加 web_search source
                if formatted_results:
                    citations = []
                    for result in formatted_results[:3]:  # 最多3个
                        citations.append({
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "score": result.get("score", 0.0),
                            "snippet": result.get("content", "")[:100]
                        })

                    state["sources"].append({
                        "type": "text",
                        "from": "web_search",
                        "text": state.get("rewritten_query", state["user_query"]),
                        "citations": citations
                    })

                logger.info(
                    f"Web search completed: results_count={len(formatted_results)}, "
                    f"has_error={api_error_message is not None}"
                )

            except Exception as e:
                logger.error(f"Web search failed: {str(e)}", exc_info=True)
                state["web_search_results"] = []
                state["web_search_used"] = False
                state["web_search_error"] = f"Web search failed: {str(e)}"

        return state

    # -------------------------------------------------------------------------
    # Workflow Nodes - Answer Generation
    # -------------------------------------------------------------------------
    async def generate_answer(self, state: ConversationState) -> ConversationState:
        """
        配置流式输出对象 - 根据意图和数据源选择不同的流式输出方式

        设置的状态字段：
        - state["streaming_llm"]     : 流式输出对象（LLM 或 None）
        - state["streaming_type"]   : "langchain_llm" 或 "raganything_stream"
        - state["streaming_messages"]: messages（LangChain LLM 使用）
        - state["raganything_query"] : 查询文本（RAGAnything 使用）
        - state["raganything_mode"]  : 检索模式（RAGAnything 使用）
        - state["confidence"]        : 置信度分数

        实际的流式输出在 chat_stream_v1.py 中根据这些配置执行。
        """
        async with time_node("generate_answer", state):
            # If final_answer is already set (direct match), skip placeholder
            if state.get("final_answer"):
                logger.info(
                    f"Direct match answer already set, skipping LLM generation: answer_length={len(state['final_answer'])}"
                )
                return state

            intent = state.get("intent")
            web_search_used = state.get("web_search_used", False)

            if state.get("is_realtime_query") and state.get("realtime_category") == "time":
                # 尝试获取已生成的时间文本，若无则重新生成
                direct_text = state.get("direct_text_answer")
                if not direct_text:
                    now = datetime.now()
                    user_query = state.get("user_query", "")
                    q_lower = (user_query or "").strip().lower()

                    # 一句多问：按标点切分为多个子问题，按顺序逐条回答
                    sub_questions = _split_user_questions(user_query)
                    if not sub_questions:
                        sub_questions = [user_query]

                    prefer_zh_output = re.search(r"[\u4e00-\u9fff]", user_query) is not None
                    answers: list[str] = []
                    for sq in sub_questions:
                        sq_lower = sq.strip().lower()
                        # 优先解析星期日期，再解析相对日期
                        label = None
                        week_parsed = _parse_weekday_date_cn(sq_lower, now)
                        if week_parsed:
                            label, target_dt = week_parsed
                            day_offset = (target_dt.date() - now.date()).days
                        else:
                            # 解析昨天/前天等偏移量
                            resolved_offset = _relative_calendar_day_offset(sq_lower)
                            day_offset = resolved_offset if resolved_offset is not None else 0
                            target_dt = now + timedelta(days=day_offset)

                        # 判断是否询问具体时间
                        asks_clock_only = any(
                            k in sq_lower
                            for k in ("几点", "什么时间", "现在几点", "当前时间", "时辰")
                        ) or any(k in sq_lower for k in ("what time", "current time"))
                        asks_date_focus = ("日期" in sq_lower) or ("几号" in sq_lower)

                         # 生成中英文回答
                        if prefer_zh_output:
                            weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][target_dt.weekday()]
                            day_word = label or _DAY_LABEL_ZH.get(day_offset, "今天")
                            line = f"{day_word}是{target_dt.year}年{target_dt.month}月{target_dt.day}日（{weekday_cn}）。"
                            # 只有明确问“几点/什么时间”才附带当前时刻
                            if asks_clock_only:
                                line = line[:-1] + f"，当前时间{now.strftime('%H:%M:%S')}。"
                            answers.append(line)
                        else:
                            weekday_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][target_dt.weekday()]
                            day_word = _DAY_LABEL_EN.get(day_offset, "Today")
                            en_verb = "is" if day_offset == 0 else ("was" if day_offset < 0 else "will be")
                            line = f"{day_word} {en_verb} {target_dt.strftime('%B')} {target_dt.day}, {target_dt.year} ({weekday_en})."
                            if asks_clock_only:
                                line += f" The current time is {now.strftime('%H:%M:%S')}."
                            answers.append(line)

                    direct_text = "\n".join(answers) if answers else ""
                    state["direct_text_answer"] = direct_text

                # 直接文本输出，不调用模型
                state["streaming_llm"] = None
                state["streaming_messages"] = None
                state["streaming_type"] = "direct_text" # 设置流式类型为【直接文本输出】
                state["confidence"] = 0.99
                state["final_answer"] = ""  # 占位符，流式输出无需预生成
                logger.info("Streaming configured: type=direct_text, realtime_category=time")
                return state


            # 计算置信度
            confidence = 0.5  # Base confidence
            if intent == "greeting":
                confidence = 0.98
            elif web_search_used:
                web_results = state.get("web_search_results", [])
                if web_results:
                    avg_web_score = sum(r.get("score", 0.5) for r in web_results) / len(web_results)
                    confidence = max(0.75, avg_web_score)
            else:  # normal query with RAGAnything
                confidence = 0.8

            # 根据意图和数据源配置流式输出
            # 数学问题特殊处理
            if state.get("is_math_problem", False):
                # 使用 Phi-4 LLM 进行数学推理
                messages = build_math_generation_messages(state)
                streaming_llm, model_name = self.workflow.get_phi4_streaming_llm(state)
                state["streaming_llm"] = streaming_llm
                state["streaming_messages"] = messages
                state["streaming_type"] = "phi4_math"
                logger.info(
                    f"Streaming configured: type=phi4_math, model={model_name}, "
                    f"query={state['user_query'][:50]}..."
                )
            elif intent == "greeting" or web_search_used:
                # 使用 LangChain LLM（原有逻辑）
                messages = build_generation_messages(state)
                streaming_llm, model_name = self.workflow.get_streaming_llm(state)
                state["streaming_llm"] = streaming_llm
                state["streaming_messages"] = messages
                state["streaming_type"] = "langchain_llm"
                logger.info(
                    f"Streaming configured: type=langchain_llm, intent={intent}, "
                    f"web_search_used={web_search_used}, model={model_name}"
                )
            else:  # normal - 需要召回文档，使用 RAGAnything
                employee_config = state.get("employee_config", {})
                kb_ids = employee_config.get("kb_ids", []) or employee_config.get("capabilities", {}).get("kb_ids", [])
                raganything_enabled = getattr(settings, "raganything_enabled", True)

                # 如果 RAG 未启用 或 无知识库 → 降级为普通 LLM 生成
                if (not raganything_enabled) or (not kb_ids):
                    messages = build_generation_messages(state)
                    streaming_llm, model_name = self.workflow.get_streaming_llm(state)
                    state["streaming_llm"] = streaming_llm
                    state["streaming_messages"] = messages
                    state["streaming_type"] = "langchain_llm"
                    logger.info(
                        f"Streaming configured: type=langchain_llm (fallback), "
                        f"reason={'raganything_disabled' if not raganything_enabled else 'no_kb_ids'}, "
                        f"model={model_name}"
                    )
                # 启用 RAG → 使用 RAGAnything 混合检索
                else:
                    # 需要召回文档，使用 RAGAnything
                    state["streaming_llm"] = None
                    state["streaming_messages"] = None
                    state["streaming_type"] = "raganything_stream"
                    state["raganything_query"] = state["user_query"]
                    state["raganything_mode"] = "hybrid"

                    logger.info(
                        f"Streaming configured: type=raganything_stream, intent={intent}, "
                        f"mode=hybrid, query={state['user_query'][:50]}..."
                    )

            state["confidence"] = confidence
            state["final_answer"] = ""  # Placeholder for streaming

        return state

    # -------------------------------------------------------------------------
    # Workflow Nodes - Answer Verification
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Workflow Nodes - Save Conversation
    # -------------------------------------------------------------------------
    async def save_conversation(self, state: ConversationState) -> ConversationState:
        """
        Save Conversation Record - Persist conversation to MongoDB.

        Saves:
        - User query and AI response
        - Performance metrics (timing, confidence)
        - Data sources used (KB, web search)
        - Intent and verification results

        Also updates session with new messages.

        Args:
            state: Current conversation state

        Returns:
            Updated state with conversation_id populated
        """
        async with time_node("save_conversation", state):
            try:
                db = await get_database()

                # Generate conversation ID
                session_id = state["session_id"]
                timestamp = datetime.now().timestamp()
                conv_id = f"conv_{hashlib.md5(f'{session_id}_{timestamp}'.encode()).hexdigest()[:12]}"
                state["conversation_id"] = conv_id

                # Calculate total response time
                workflow_start_time = state.get("workflow_start_time", time.time())
                total_time_ms = int((time.time() - workflow_start_time) * 1000)
                state["response_time_ms"] = total_time_ms

                # Create conversation record
                employee_config = state.get("employee_config", {})
                conversation = ConversationModel(
                    conversation_id=conv_id,
                    session_id=state["session_id"],
                    user_id=state["user_id"],
                    user_name=state.get('user_name', ''),
                    head_url=state.get('head_url', ''),
                    employee_id=state["employee_id"],
                    employee_name=employee_config.get("name", ""),
                    user_query=state["user_query"],
                    ai_response=state["final_answer"],
                    is_realtime_query=state.get("is_realtime_query", False),
                    realtime_category=state.get("realtime_category"),
                    intent=state.get("intent"),
                    kb_used=state.get("kb_used", []),
                    web_search_used=state.get("web_search_used", False),
                    web_search_results=[
                        {
                            "rank": result.get("rank"),
                            "title": result.get("title"),
                            "url": result.get("url"),
                            "score": result.get("score", 0.0)
                        }
                        for result in state.get("web_search_results", [])[:5]
                    ],
                    retrieved_docs=[
                        {
                            "doc_id": doc.get("doc_id"),
                            "kb_id": doc.get("kb_id"),
                            "score": doc.get("rrf_score", 0.0)
                        }
                        for doc in state.get("retrieved_docs", [])[:3]
                    ],
                    relevance_score=state.get("relevance_score", 0.0),
                    confidence=state.get("confidence", 0.0),
                    response_time_ms=total_time_ms
                )

                await db.conversations.insert_one(conversation.model_dump())

                # Update session context messages
                await db.sessions.update_one(
                    {"session_id": state["session_id"]},
                    {
                        "$push": {
                            "context_messages": {
                                "$each": [
                                    {"role": "user", "content": state["user_query"]},
                                    {"role": "assistant", "content": state["final_answer"]}
                                ],
                                "$slice": -20  # Keep last 20 messages
                            }
                        },
                        "$inc": {"message_count": 1}
                    }
                )

                # Log timing summary
                node_timings = state.get("node_timings", {})
                ttfb_ms = state.get("ttfb_ms")

                logger.info(
                    f"[TIMING_SUMMARY] Conversation completed - "
                    f"total: {total_time_ms}ms, ttfb: {ttfb_ms}ms, "
                    f"nodes: {node_timings}"
                )

            except Exception as e:
                logger.error(f"Failed to save conversation: {str(e)}", exc_info=True)

        return state
