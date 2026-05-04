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
from datetime import timedelta
import re

import httpx
from bs4 import BeautifulSoup
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
from app.services.query_classifier import (
    ClassificationResult,
    augment_dialog_with_persisted_turns,
    format_dialog_for_resolver,
    get_query_classifier,
)
from app.services.realtime_intent_heuristic import heuristic_realtime_category
from app.services.conversation.conversation_helpers import (
    time_node,
    heuristic_complexity,
    build_generation_messages,
    build_math_generation_messages,
    resolve_target_year_from_query,
)
from app.services.calendar_time_resolver import calendar_direct_text_answer
from app.services.web_search_recency import (
    DEFAULT_POLICIES,
    build_time_anchored_query,
    filter_and_sort_by_recency,
)

logger = get_logger(__name__)

# 启发式联网补位：不覆盖问候、噪声、数学题；也不重复覆盖已是实时的分支
_REALTIME_HEURISTIC_SKIP_LABELS = frozenset({"greeting", "noise", "math_problem", "realtime_query"})


def _extract_weather_location(query: str) -> str | None:
    """尽量从自然语言天气问句中提取地点，失败时返回 None。"""
    q = (query or "").strip()
    if not q:
        return None
    # 先尝试“X天气”结构
    m = re.search(r"([\u4e00-\u9fffA-Za-z]{2,20})天气", q)
    if m:
        loc = m.group(1).strip()
        loc = re.sub(r"^(请问|麻烦问下|问下|请帮我查一下|帮我查一下)", "", loc).strip()
        loc = re.sub(r"(今天|今日|明天|后天|本周|这周|当前|现在)$", "", loc).strip()
        if loc:
            return loc
    # 回退：保留中文/字母，截取前部作为 geocoding 查询
    simple = re.sub(r"[^\u4e00-\u9fffA-Za-z]", "", q)
    if 2 <= len(simple) <= 20:
        return simple
    return None


def _extract_traffic_location(query: str) -> str | None:
    """从路况问题提取城市，如“成都堵不堵”→“成都”"""
    q = (query or "").strip()
    if not q:
        return None
    m = re.search(r"([\u4e00-\u9fffA-Za-z]{2,20})(?:堵不堵|拥堵|路况|交通)", q)
    if m:
        loc = m.group(1).strip()
        loc = re.sub(r"^(请问|我想了解一下|想了解一下|帮我查一下|查一下|今天|今日|现在)", "", loc).strip()
        loc = re.sub(r"(今天|今日|现在)$", "", loc).strip()
        if loc:
            return loc
    simple = re.sub(r"[^\u4e00-\u9fffA-Za-z]", "", q)
    simple = re.sub(r"(堵不堵|拥堵|路况|交通|今天|今日|现在)", "", simple)
    return simple if 2 <= len(simple) <= 20 else None


async def _open_meteo_weather_fallback(query: str) -> dict | None:
    """免费天气接口兜底查询，返回结构化天气结果"""
    loc = _extract_weather_location(query)
    if not loc:
        return None
    timeout = httpx.Timeout(8.0, connect=4.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # 地理编码获取经纬度
        geo_resp = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": loc, "count": 1, "language": "zh", "format": "json"},
        )
        if geo_resp.status_code != 200:
            return None
        geo_data = geo_resp.json() or {}
        results = geo_data.get("results") or []
        if not results:
            return None
        top = results[0]
        lat = top.get("latitude")
        lon = top.get("longitude")
        if lat is None or lon is None:
            return None
        city_name = top.get("name") or loc
        country = top.get("country") or ""

        # 查询天气
        weather_resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                "timezone": "Asia/Shanghai",
            },
        )
        if weather_resp.status_code != 200:
            return None
        weather_data = weather_resp.json() or {}
        current = weather_data.get("current") or {}
        temp = current.get("temperature_2m")
        feels = current.get("apparent_temperature")
        wind = current.get("wind_speed_10m")
        code = current.get("weather_code")
        obs_time = current.get("time")
        if temp is None and feels is None and wind is None and code is None:
            return None

        content = (
            f"{city_name}{('·' + country) if country else ''} 当前天气："
            f"气温 {temp}°C，体感 {feels}°C，风速 {wind}km/h，天气代码 {code}，观测时间 {obs_time}。"
        )
        return {
            "rank": 1,
            "title": f"{city_name} 实时天气（Open-Meteo）",
            "url": f"https://open-meteo.com/en/docs?latitude={lat}&longitude={lon}",
            "content": content,
            "score": 0.75,
        }


async def _duckduckgo_traffic_fallback(query: str) -> list[dict]:
    """交通检索兜底：使用 DuckDuckGo 公共搜索抓取摘要。"""
    try:
        loc = _extract_traffic_location(query) or "当地"
        search_q = f"{loc} 今天 路况 拥堵"
        timeout = httpx.Timeout(10.0, connect=4.0)
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
        async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
            resp = await client.get("https://duckduckgo.com/html/", params={"q": search_q})
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, "html.parser")
            items: list[dict] = []
            for i, node in enumerate(soup.select(".result"), 1):
                a = node.select_one("a.result__a")
                snip = node.select_one(".result__snippet")
                if not a:
                    continue
                title = (a.get_text(" ", strip=True) or "").strip()
                href = (a.get("href") or "").strip()
                content = (snip.get_text(" ", strip=True) if snip else "").strip()
                if not (title and href):
                    continue
                items.append(
                    {
                        "rank": i,
                        "title": title[:200],
                        "url": href,
                        "content": content[:2000],
                        "score": max(0.5, 0.9 - i * 0.1),
                    }
                )
                if len(items) >= 5:
                    break
            return items
    except Exception as e:
        logger.warning(f"DuckDuckGo traffic fallback failed: {e}")
        return []


def _traffic_congestion_estimate(now: datetime) -> tuple[str, str]:
    """无实时数据时，按时段估算拥堵等级"""
    h = now.hour
    weekday = now.weekday()  # Mon=0
    is_workday = weekday < 5
    # 早晚高峰
    if is_workday and (7 <= h <= 9 or 17 <= h <= 19):
        return "较拥堵", "工作日通勤高峰"
     # 工作日白天
    if is_workday and (10 <= h <= 16):
        return "中等拥堵", "工作日白天车流较大"
    # 工作日其他时间
    if is_workday:
        return "基本通畅", "非通勤高峰时段"
    # 周末白天
    if 10 <= h <= 20:
        return "中等拥堵", "周末商圈/景点出行时段"
    return "基本通畅", "周末非高峰时段"


def _normalize_realtime_category(reason: str | None) -> str:
    """将QueryClassifier的reason映射为内部统一枚举：weather/time/news/market/traffic/general"""
    if not reason:
        return "general"
    r = reason.strip().lower()
    # 别名映射表（集中维护）
    aliases = {
        # 英文 / 规范值
        "weather": "weather",
        "time": "time",
        "news": "news",
        "market": "market",
        "traffic": "traffic",
        "general": "general",
        # 常见中文别名（可按需追加）
        "天气": "weather",
        "气象": "weather",
        "时间": "time",
        "日期": "time",
        "时刻": "time",
        "时钟": "time",
        "新闻": "news",
        "时事": "news",
        "资讯": "news",
        "股市": "market",
        "股票": "market",
        "行情": "market",
        "汇率": "market",
        "金价": "market",
        "交通": "traffic",
        "路况": "traffic",
        "堵车": "traffic",
        "拥堵": "traffic",
    }
    return aliases.get(r, r if r in {"weather", "time", "news", "market", "traffic", "general"} else "general")


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
        Query Classification - 使用 LLM 进行细粒度查询分类并路由。

        使用 QueryClassifier 进行分类，支持 9 种类别：
        - math_problem, concept_explain, greeting, english_query
        - realtime_query, general_knowledge, chit_chat, noise, other

        分类结果映射到 workflow state：
        - greeting → intent="greeting", complexity_score=0.0
        - realtime_query → is_realtime_query=True
        - math_problem → is_math_problem=True
        - noise → 返回友好提示后结束
        - 其他 → 继续正常流程

        Args:
            state: Current conversation state

        Returns:
            Updated state with query_type classification
        """
        async with time_node("classify_query_type", state):
            # 1.基础变量初始化
            query = state["user_query"].strip()
            context_messages = (state.get("context") or {}).get("messages") or []
            # 获取上一轮用户问题
            last_user_query = next(
                (
                    (m.get("content") or "").strip()
                    for m in reversed(context_messages)
                    if m.get("role") == "user" and (m.get("content") or "").strip()
                ),
                "",
            )
            # 2.格式化对话上下文
            dialog_text = format_dialog_for_resolver(context_messages)
            # 统计上下文用户消息数量
            session_user_count = sum(
                1 for m in context_messages if m.get("role") == "user" and (m.get("content") or "").strip()
            )
            # 上下文不足2轮时，从DB补全历史对话
            if state.get("session_id") and session_user_count < 2:
                try:
                    db = await get_database()
                    recent_turns = await db.conversations.find(
                        {"session_id": state.get("session_id", "")},
                        {"_id": 0, "user_query": 1, "ai_response": 1},
                    ).sort("created_at", 1).limit(30).to_list(length=30)
                    dialog_text = augment_dialog_with_persisted_turns(
                        dialog_text, list(recent_turns), query
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to merge persisted dialog for resolver: session_id={state.get('session_id')}, error={e}",
                        exc_info=True,
                    )
            # 3.意图分类器初始化与原始意图预判
            classifier = get_query_classifier()
            original_ctx_result = await classifier.aclassify(query, context_query=last_user_query or None)
            # 高置信度非实时查询，保留原始意图
            preserve_original_intent = (
                original_ctx_result.label != "realtime_query"
                and original_ctx_result.confidence == "high"
                and len(query) >= 8
            )
            # 4.上下文消歧，改写问句
            resolved = query
            if dialog_text.strip():
                resolved = await classifier.aresolve_standalone_query(query, dialog_text)
                if not (resolved or "").strip():
                    resolved = query
            resolved = resolved.strip()
            state["rewritten_query"] = resolved
            state["query_rewritten"] = resolved != query

            # 5.对改写后的问句分类
            result_llm = await classifier.aclassify(resolved, context_query=None)
            # 短问句场景：恢复原始实时查询意图
            if (
                result_llm.label != "realtime_query"
                and original_ctx_result.label == "realtime_query"
                and original_ctx_result.confidence == "high"
                and len(query) < 8
            ):
                logger.info(
                    "Restore realtime intent for short contextual follow-up: "
                    f"query={query[:80]}, resolved={resolved[:80]}"
                )
                result_llm = original_ctx_result
            # 高置信度原始意图：防止改写偏移
            if (
                result_llm.label == "realtime_query"
                and original_ctx_result.label != "realtime_query"
                and original_ctx_result.confidence == "high"
            ):
                logger.info(
                    "Keep original intent classification to avoid rewrite drift: "
                    f"original={original_ctx_result.label}, rewritten={result_llm.label}, "
                    f"query={query[:80]}, resolved={resolved[:80]}"
                )
                result_llm = original_ctx_result
            # 6.启发式规则：升级为实时查询
            boost = heuristic_realtime_category(resolved)
            if (
                boost is not None
                and result_llm.label not in _REALTIME_HEURISTIC_SKIP_LABELS
                and not preserve_original_intent
            ):
                logger.info(
                    f"Realtime heuristic upgrade: boost={boost}, llm_label={result_llm.label}, "
                    f"query={resolved[:80]}"
                )
                result = ClassificationResult(
                    label="realtime_query",
                    confidence=result_llm.confidence,
                    reason=boost,
                )
            else:
                result = result_llm

            logger.info(
                f"LLM classification: label={result.label}, confidence={result.confidence}, "
                f"reason={result.reason}, query={resolved[:50]}"
            )

            # 7.解析目标年份，存入状态：用于后续生成阶段保持“今年/明年/去年”一致。
            target_year = resolve_target_year_from_query(resolved)
            if target_year is not None:
                state["target_year"] = target_year

            # 8.保存分类结果到状态
            state["classification_label"] = result.label
            state["classification_confidence"] = result.confidence
            state["classification_reason"] = result.reason

            # 9.根据分类标签设置状态
            match result.label:
                # 问候语
                case "greeting":
                    state["intent"] = "greeting"
                    state["complexity_score"] = 0.0
                    state["complexity_reason"] = "greeting"
                    state["is_realtime_query"] = False
                    # 10. 添加问候语来源
                    state["sources"].append({
                        "type": "text",
                        "from": "greeting",
                        "text": query,
                        "citations": []
                    })
                # 实时查询
                case "realtime_query":
                    state["is_realtime_query"] = True
                    state["realtime_category"] = _normalize_realtime_category(result.reason)
                    state["realtime_detect_reason"] = f"llm:{result.confidence}"
                    state["intent"] = "general_query"
                # 数学题
                case "math_problem":
                    state["is_math_problem"] = True
                    state["intent"] = "general_query"
                # 无效噪声
                case "noise":
                    state["intent"] = "noise"
                    state["sources"].append({
                        "type": "text",
                        "from": "noise_response",
                        "text": "抱歉，我没有听清您的问题，请再重复一次。",
                        "citations": []
                    })
                # 默认通用查询
                case _:
                    state["is_realtime_query"] = False
                    state["intent"] = "general_query"

            # 10.日历日期直出答案（优先原问句，再用改写后问句）避免把“习俗/由来”等非日期问题误转为日期回答。
            # `language_hint_query` 始终传入用户原始问句，避免改写后的查询语言污染输出语言判定
            # （例如：英文问句被消歧/改写为中文，造成英文问、中文答的混语回复）。
            direct = calendar_direct_text_answer(
                query,
                bool(state.get("prefer_zh_output", True)),
                anchor_year=state.get("target_year"),
                language_hint_query=state.get("user_query") or query,
            )
            if not direct and result.label == "realtime_query":
                direct = calendar_direct_text_answer(
                    resolved,
                    bool(state.get("prefer_zh_output", True)),
                    anchor_year=state.get("target_year"),
                    language_hint_query=state.get("user_query") or query,
                )
            # 命中日历答案，设置本地计算状态
            if direct:
                state["direct_text_answer"] = direct
                state["is_realtime_query"] = True
                state["realtime_category"] = "time"
                state["realtime_detect_reason"] = "local_calendar_resolver"
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
            目标节点名称 (greeting/realtime/math/normal)
        """
        intent = state.get("intent")
        # 问候语和噪声输入直接跳到生成答案
        if intent in ("greeting", "noise"):
            return "greeting"
        # 实时查询 → web search
        if state.get("is_realtime_query"):
            return "realtime"
        # 数学问题 → 直接生成答案（使用 Phi-4）
        if state.get("is_math_problem"):
            return "math"
        # 其他 → 复杂度评估
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

                # 日历问题：优先本地推算，避免分类器把 reason 标成 general 仍去联网抄错误示例
                if state.get("is_realtime_query"):
                    q_cal = (state.get("rewritten_query") or state.get("user_query") or "").strip()
                    direct_cal = calendar_direct_text_answer(
                        q_cal,
                        bool(state.get("prefer_zh_output", True)),
                        anchor_year=state.get("target_year"),
                        language_hint_query=state.get("user_query") or q_cal,
                    )
                    if direct_cal:
                        state["web_search_results"] = []
                        state["web_search_used"] = False
                        state["web_search_error"] = None
                        state["direct_text_answer"] = direct_cal
                        state["realtime_category"] = "time"
                        logger.info(
                            "Web search skipped: local_calendar_resolver, "
                            f"query={state.get('user_query', '')[:80]}"
                        )
                        return state

                query = state["user_query"]
                logger.info(
                    f"Web search started: query={query[:100]}, "
                    f"is_realtime={state.get('is_realtime_query')}"
                )

                now = datetime.now()
                realtime_category = state.get("realtime_category", "") or "general"
                policy = DEFAULT_POLICIES.get(realtime_category, DEFAULT_POLICIES["general"])
                anchored_query = query
                used_duck_fallback = False

                # 实时类查询统一追加时间锚点（通用）：避免“今年/当前/最近”等相对时间漂移。
                if state.get("is_realtime_query"):
                    anchored_query = build_time_anchored_query(query, now, policy)

                # 执行联网搜索
                web_search_tool = TavilySearchResults(
                    max_results=10,  # 返回结果数量，默认 5
                    search_depth="basic", # 搜索深度："basic" (免费) 或 "advanced" (付费)
                    tavily_api_key=settings.tavily_api_key,
                )
                search_results = await web_search_tool.ainvoke({"query": anchored_query})
                # 锚定查询可能过窄：实时查询无结果时，用原始查询重试一次（通用回退）。
                if (
                    state.get("is_realtime_query")
                    and anchored_query != query
                    and (not search_results)
                ):
                    logger.info("Web search retry with raw query after anchored miss")
                    search_results = await web_search_tool.ainvoke({"query": query})
                # 交通类查询：首轮无结果时，使用更精准的关键词（地点+路况+拥堵）重试搜索
                if (
                    state.get("is_realtime_query")
                    and state.get("realtime_category") == "traffic"
                    and (not search_results)
                ):
                    loc = _extract_traffic_location(query) or "当地"
                    focused_query = f"{loc} 实时路况 拥堵 情况"
                    logger.info(f"Web search retry with traffic-focused query: {focused_query}")
                    search_results = await web_search_tool.ainvoke({"query": focused_query})
                # 重试仍无结果：启用 DuckDuckGo 兜底搜索
                if (
                    state.get("is_realtime_query")
                    and state.get("realtime_category") == "traffic"
                    and (not search_results)
                ):
                    logger.info("Web search fallback retry with DuckDuckGo for traffic")
                    search_results = await _duckduckgo_traffic_fallback(query)
                    used_duck_fallback = bool(search_results)

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
                            selected_results = (
                                search_results[:settings.web_search_max_results]
                                if used_duck_fallback
                                else filter_and_sort_by_recency(search_results, now, policy)[:settings.web_search_max_results]
                            )
                            for i, result in enumerate(selected_results, 1):
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
                                    "content": result.get("content", "")[:2000],
                                    "score": result.get("score", 0.0)
                                })

                state["web_search_results"] = formatted_results
                state["web_search_used"] = len(formatted_results) > 0
                state["web_search_error"] = api_error_message

                # 天气兜底：无搜索结果且无API错误时，调用免费天气接口
                if (
                    not state["web_search_used"]
                    and not api_error_message
                    and state.get("realtime_category") == "weather"
                ):
                    fallback = await _open_meteo_weather_fallback(query)
                    if fallback:
                        formatted_results = [fallback]
                        state["web_search_results"] = formatted_results
                        state["web_search_used"] = True
                        state["web_search_error"] = None
                        logger.info(
                            f"Web search fallback used: open_meteo, query={query[:80]}"
                        )

                # 添加 web_search source
                if formatted_results:
                    citations = []
                    for result in formatted_results[:3]:  # 最多3个
                        citations.append({
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "score": result.get("score", 0.0),
                            "snippet": result.get("content", "")[:300]
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

            # 噪声输入：使用预设的友好响应，不需要调用 LLM
            if intent == "noise":
                # 从 sources 中获取预设的响应
                noise_source = next((s for s in state.get("sources", []) if s.get("from") == "noise_response"), None)
                if noise_source:
                    state["final_answer"] = noise_source.get("text", "抱歉，我没有听清您的问题，请再重复一次。")
                    state["confidence"] = 0.99
                    state["streaming_llm"] = None
                    state["streaming_type"] = "text"  # 标记为纯文本输出
                    logger.info("Noise input detected, using preset response")
                else:
                    # 回退到 greeting 逻辑
                    state["final_answer"] = "抱歉，我没有听清您的问题，请再重复一次。"
                    state["confidence"] = 0.99
                    state["streaming_llm"] = None
                    state["streaming_type"] = "text"
                return state

            # 纯日期/星期类：本地确定性推算，直接输出（不依赖分类 reason=time，也不依赖联网）
            if state.get("is_realtime_query"):
                direct = state.get("direct_text_answer")
                if not direct:
                    q_cal = (state.get("rewritten_query") or state.get("user_query") or "").strip()
                    direct = calendar_direct_text_answer(
                        q_cal,
                        bool(state.get("prefer_zh_output", True)),
                        anchor_year=state.get("target_year"),
                        language_hint_query=state.get("user_query") or q_cal,
                    )
                if direct:
                    state["direct_text_answer"] = direct
                    state["realtime_category"] = "time"
                    state["streaming_llm"] = None
                    state["streaming_messages"] = None
                    state["streaming_type"] = "direct_text"
                    state["confidence"] = 0.99
                    state["final_answer"] = ""
                    logger.info(
                        "Streaming configured: type=direct_text, realtime_category=time (local_calendar_resolver)"
                    )
                    return state

            # 实时查询在“联网不可用/失败”时：返回统一提示语，避免模型臆测。
            # 若仅是无足够结果（但无错误），继续走 LLM 流程，让模型基于已有上下文尽力作答。
            if state.get("is_realtime_query") and not state.get("web_search_used", False):
                err = state.get("web_search_error")
                # 网络检索失败：返回固定提示
                if err:
                    direct = "当前网络检索不可用，暂时无法确认实时信息，请稍后重试。"
                    state["direct_text_answer"] = direct
                    state["streaming_llm"] = None
                    state["streaming_messages"] = None
                    state["streaming_type"] = "direct_text"
                    state["confidence"] = 0.7
                    state["final_answer"] = ""
                    logger.info(
                        "Streaming configured: type=direct_text, realtime_web_error"
                    )
                    return state
                # 交通类：按时段给出拥堵预估
                if state.get("realtime_category") == "traffic":
                    lvl, reason = _traffic_congestion_estimate(datetime.now())
                    direct = f"当前路况判断：{lvl}。依据：{reason}。"
                    state["direct_text_answer"] = direct
                    state["streaming_llm"] = None
                    state["streaming_messages"] = None
                    state["streaming_type"] = "direct_text"
                    state["confidence"] = 0.68
                    state["final_answer"] = ""
                    logger.info(
                        "Streaming configured: type=direct_text, realtime_traffic_estimate"
                    )
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
