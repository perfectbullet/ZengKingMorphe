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

from langchain_community.tools.tavily_search import TavilySearchResults

from app.core.config import settings
from app.core.database import get_database
from app.core.logging import get_logger
from app.models.database import ConversationModel, SessionModel
from app.services.conversation.conversation_state import ConversationState, GREETING_KEYWORDS
from app.services.conversation.conversation_helpers import (
    time_node,
    heuristic_complexity
)
from app.services.conversation.conversation_helpers import build_generation_messages

logger = get_logger(__name__)


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
        - Detect sensitive content (placeholder)

        Args:
            state: Current conversation state

        Returns:
            Updated state with validation results
        """
        async with time_node("validate_input", state):
            # Basic validation - can be extended
            state["has_sensitive"] = False
        return state

    # -------------------------------------------------------------------------
    # Workflow Nodes - Query Classification (Early Exit)
    # -------------------------------------------------------------------------
    async def classify_query_type(self, state: ConversationState) -> ConversationState:
        """
        Query Classification - 快速识别查询类型并路由。

        检测顺序 (从快到慢):
        1. 打断检测 (关键词匹配)
        2. 问候语检测 (关键词匹配)
        3. 实时查询检测 (关键词匹配)
        4. 其他 (继续正常流程)

        Args:
            state: Current conversation state

        Returns:
            Updated state with query_type classification
        """
        async with time_node("classify_query_type", state):
            query = state["user_query"].strip().lower()

            # 1. 检测问候语
            for category, keywords in GREETING_KEYWORDS.items():
                if any(kw in query for kw in keywords):
                    state["intent"] = "greeting"
                    state["complexity_score"] = 0.0
                    state["complexity_reason"] = "greeting"
                    state["is_realtime_query"] = False
                    logger.info(f"Query classified: greeting: category={category}")
                    return state

            # 3. 检测实时查询
            if settings.realtime_query_enabled:
                realtime_keywords = {
                    "time": ["今天", "明天", "昨天", "最近", "现在", "本周", "本月", "当前"],
                    "weather": ["天气", "气温", "降雨", "降水", "温度"],
                    "news": ["新闻", "热点", "最新", "资讯", "动态", "头条"],
                    "market": ["股价", "汇率", "行情", "股市", "价格", "金价", "银价", "油价", "多少钱"],
                }
                for category, keywords in realtime_keywords.items():
                    if any(kw in query for kw in keywords):
                        state["is_realtime_query"] = True
                        state["realtime_category"] = category
                        state["realtime_detect_reason"] = f"keyword:{keywords[0] if keywords else category}"
                        state["intent"] = "general_query"
                        logger.info(f"Query classified: realtime: category={category}")
                        return state

            # 4. 默认为一般查询
            state["is_realtime_query"] = False
            state["intent"] = "general_query"
            logger.debug("Query classified as general")
        return state

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
    # Workflow Nodes - Query Rewriting - 已删除，由 RAGAnything 处理
    # -------------------------------------------------------------------------
    # Note: rewrite_query 节点已删除，查询重写功能由 RAGAnything 内部处理

    # -------------------------------------------------------------------------
    # Workflow Nodes - FAQ Matching (Fast Path) - 已删除，由 RAGAnything 处理
    # -------------------------------------------------------------------------
    # Note: match_faq 节点已删除，FAQ 匹配功能由 RAGAnything 内部处理

    # -------------------------------------------------------------------------
    # Workflow Nodes - Intent Recognition - 已删除，功能与 classify_query_type 重复
    # -------------------------------------------------------------------------
    # Note: recognize_intent 节点已删除，意图识别由 classify_query_type 处理

    # -------------------------------------------------------------------------
    # Workflow Nodes - Knowledge Retrieval (RAG) - 已删除，由 RAGAnything 处理
    # -------------------------------------------------------------------------
    # Note: knowledge_retrieval 节点已删除，RAG 检索由 RAGAnything 内部处理

    # -------------------------------------------------------------------------
    # Workflow Nodes - Document Grading - 已删除，由 RAGAnything 处理
    # -------------------------------------------------------------------------
    # Note: grade_documents 节点已删除，文档评分由 RAGAnything 内部处理

    # -------------------------------------------------------------------------
    # Workflow Nodes - Context Compression - 已删除，由 RAGAnything 处理
    # -------------------------------------------------------------------------
    # Note: compress_context 节点已删除，上下文压缩由 RAGAnything 内部处理

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
                web_search_tool = TavilySearchResults(k=3)
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
            if intent == "greeting" or web_search_used:
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
