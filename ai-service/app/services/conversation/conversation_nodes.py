"""
Node implementations for LangGraph conversation workflow.

This module contains all node functions that process the conversation state:
- Configuration loading nodes
- Input validation nodes
- Query classification nodes
- Complexity evaluation nodes
- Query rewriting nodes
- FAQ matching nodes
- Intent recognition nodes
- Knowledge retrieval nodes
- Document grading/reranking nodes
- Context compression nodes
- Web search nodes
- Answer generation nodes
- Answer verification nodes
- Conversation saving nodes
"""
import hashlib
import json
import random
import time
from datetime import datetime

from langchain_community.tools.tavily_search import TavilySearchResults

from app.core.config import settings
from app.core.database import get_database
from app.core.logging import get_logger
from app.models.database import ConversationModel, SessionModel
from app.services.conversation.conversation_state import ConversationState, GREETING_KEYWORDS
from app.services.conversation.conversation_helpers import (
    time_node, select_llm, build_generation_messages,
    heuristic_complexity
)
from app.services.rag_service import rag_retrieval

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
                logger.error(error_msg)
                raise ValueError(error_msg)

            employee.pop("_id", None)
            state["employee_config"] = employee
            logger.info(
                "Employee config loaded",
                employee_id=state["employee_id"],
                name=employee.get("name")
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
                    "Session context loaded",
                    session_id=state["session_id"],
                    message_count=state["context"]["message_count"]
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

            # 1. 检测问候语
            for category, keywords in GREETING_KEYWORDS.items():
                if any(kw in query for kw in keywords):
                    state["intent"] = "greeting"
                    state["complexity_score"] = 0.0
                    state["complexity_reason"] = "greeting"
                    state["is_realtime_query"] = False
                    logger.info("Query classified: greeting", category=category)
                    return state

            # 2. 检测实时查询
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
                        logger.info("Query classified: realtime", category=category)
                        return state

            # 3. 默认为一般查询
            state["is_realtime_query"] = False
            state["intent"] = "general_query"
            logger.debug("Query classified: general")
        return state

    @staticmethod
    def route_after_classification(state: ConversationState) -> str:
        """
        路由决策: 查询分类后的下一步。

        Args:
            state: Current conversation state

        Returns:
            目标节点名称
        """
        if state.get("intent") == "greeting":
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

            try:
                # 使用本地 Ollama 模型评估复杂度（快速调用）
                complexity_prompt = f"""请评估以下用户查询的复杂度（0-10分）。

用户查询: {query}

评分标准：
- 0-3分（简单）：直接的事实问答、天气/价格查询、简单知识查询
- 4-6分（中等）：需要一定推理、涉及多个方面、需要上下文理解
- 7-10分（复杂）：需要多步推理、模糊问题需要意图澄清、多文档综合分析

请以JSON格式返回：
{{"score": 分数0-10, "reason": "简短原因说明"}}

只返回JSON，不要有其他内容："""

                response = await self.workflow.local_llm.ainvoke(complexity_prompt)
                response_text = response.content.strip()

                try:
                    result = json.loads(response_text)
                    score = float(result.get("score", 3.0))
                    reason = result.get("reason", "unknown")

                    # 确保分数在合理范围内
                    score = max(0.0, min(10.0, score))
                    state["complexity_score"] = score
                    state["complexity_reason"] = reason

                    logger.info(
                        f"Complexity evaluated: {score}/10 - {reason}",
                        score=score,
                        reason=reason
                    )

                except json.JSONDecodeError:
                    # JSON 解析失败，使用启发式规则
                    state["complexity_score"] = heuristic_complexity(query)
                    state["complexity_reason"] = "heuristic_fallback"

            except Exception as e:
                logger.warning(f"LLM complexity evaluation failed: {e}, using heuristic")
                state["complexity_score"] = heuristic_complexity(query)
                state["complexity_reason"] = "heuristic_fallback"

        return state

    async def rewrite_query(self, state: ConversationState) -> ConversationState:
        """
        Query Rewriting - Optimize short queries for better retrieval.

        Strategy:
        - Only rewrite queries shorter than 20 characters
        - Expand with relevant keywords while preserving original intent
        - Skip for greetings and FAQ-ready queries

        Configuration: QUERY_REWRITE_ENABLED (default: False)

        Args:
            state: Current conversation state

        Returns:
            Updated state with rewritten_query populated
        """
        async with time_node("rewrite_query", state):
            state["query_rewritten"] = False
            state["rewritten_query"] = state["user_query"]

            # Check if query rewriting is enabled
            rewrite_enabled = getattr(settings, 'query_rewrite_enabled', False)
            if not rewrite_enabled:
                logger.debug("Query rewriting disabled")
                return state

            query = state["user_query"].strip()

            # Only rewrite short queries
            if len(query) >= 20:
                logger.debug(f"Query too long for rewriting: {len(query)} chars")
                return state

            # Skip greetings
            query_lower = query.lower()
            for keywords in GREETING_KEYWORDS.values():
                if any(kw in query_lower for kw in keywords):
                    logger.debug("Greeting detected, skipping query rewrite")
                    return state

            try:
                rewrite_prompt = f"""你是一个查询优化助手。请将用户查询重写为更具体的搜索语句,用于知识库检索。

原查询: {query}

要求:
1. 保持原意不变
2. 增加相关关键词和同义词
3. 使查询更具体、更完整
4. 只返回重写后的查询,不要解释
5. 长度控制在50字以内

重写后的查询:"""

                # Get appropriate LLM for current state (hybrid routing)
                llm, _, model_name = select_llm(
                    state,
                    self.workflow.local_llm,
                    self.workflow.local_grader_llm,
                    self.workflow.remote_llm,
                    self.workflow.remote_grader_llm
                )
                response = await llm.ainvoke(rewrite_prompt)
                rewritten = response.content.strip()

                # Validate rewrite result
                if rewritten and len(rewritten) > len(query) and len(rewritten) < 100:
                    state["rewritten_query"] = rewritten
                    state["query_rewritten"] = True
                    logger.info(
                        "Query rewritten successfully",
                        original=query[:50],
                        rewritten=rewritten[:50]
                    )
                else:
                    logger.warning(
                        "Query rewrite result invalid, using original",
                        rewritten=rewritten[:50] if rewritten else "empty"
                    )

            except Exception as e:
                logger.error(f"Query rewriting failed: {str(e)}", exc_info=True)
                state["rewritten_query"] = state["user_query"]

        return state

    # -------------------------------------------------------------------------
    # Workflow Nodes - FAQ Matching (Fast Path)
    # -------------------------------------------------------------------------
    async def match_faq(self, state: ConversationState) -> ConversationState:
        """
        FAQ Fast-Path Matching - Direct answer for common questions.

        Process:
        1. Hybrid search (vector + keyword) with RRF fusion
        2. Filter by faq_sim_threshold
        3. Randomly select from multiple answers
        4. Skip if RRF score < 0.02 (avoid false positives)

        If matched: Skip RAG pipeline, use FAQ answer directly
        If not matched: Continue to intent recognition

        Args:
            state: Current conversation state

        Returns:
            Updated state with faq_matched populated (or None)
        """
        async with time_node("match_faq", state):
            try:
                query = state["user_query"]
                employee_id = state["employee_id"]

                # Get FAQ config from employee settings
                db = await get_database()
                digital_config = await db.digital_employee_configs.find_one({"employee_id": employee_id})

                if not digital_config:
                    logger.info(f"No employee config found for {employee_id}, skipping FAQ")
                    state["faq_matched"] = None
                    return state

                faq_sim_threshold = digital_config.get("faq_sim_threshold", 0.0)
                faq_top_k = digital_config.get("faq_top_k", 3)

                logger.info(
                    "FAQ matching started",
                    employee_id=employee_id,
                    threshold=faq_sim_threshold,
                    top_k=faq_top_k
                )

                # Perform FAQ hybrid search
                faq_results = await rag_retrieval.faq_hybrid_search(
                    query=query,
                    employee_id=employee_id,
                    faq_sim_threshold=faq_sim_threshold,
                    faq_top_k=faq_top_k
                )

                if not faq_results:
                    logger.info("No FAQ matched above threshold")
                    state["faq_matched"] = None
                    return state

                # Get best FAQ result
                best_faq_result = faq_results[0]
                faq_id = best_faq_result["faq_id"]
                rrf_score = best_faq_result["rrf_score"]

                # Quality check: skip if RRF score too low
                if rrf_score < 0.02:
                    logger.warning(
                        "FAQ RRF score too low, skipping",
                        faq_id=faq_id,
                        rrf_score=rrf_score
                    )
                    state["faq_matched"] = None
                    return state

                # Retrieve full FAQ from MongoDB
                faq_doc = await db.faqs.find_one({"faq_id": faq_id})

                if not faq_doc or faq_doc.get("is_enable", 0) != 1:
                    logger.info(f"FAQ {faq_id} not found or disabled")
                    state["faq_matched"] = None
                    return state

                # Select random answer from answers array
                answers = faq_doc.get("answers", [])
                if not answers:
                    logger.warning(f"FAQ {faq_id} has no answers")
                    state["faq_matched"] = None
                    return state

                selected_answer = random.choice(answers)

                # Set FAQ match state
                state["final_answer"] = selected_answer
                state["confidence"] = min(0.95, rrf_score)
                state["intent"] = "faq_match"
                state["faq_matched"] = {
                    "faq_id": faq_id,
                    "question_name": faq_doc.get("question_name"),
                    "rrf_score": rrf_score,
                    "vector_score": best_faq_result.get("vector_score"),
                    "keyword_score": best_faq_result.get("keyword_score"),
                    "selected_answer": selected_answer,
                    "total_answers": len(answers)
                }

                logger.info(
                    "FAQ answer selected",
                    faq_id=faq_id,
                    rrf_score=rrf_score
                )

            except Exception as e:
                logger.error(f"FAQ matching failed: {str(e)}", exc_info=True)
                state["faq_matched"] = None

        return state

    # -------------------------------------------------------------------------
    # Workflow Nodes - Intent Recognition
    # -------------------------------------------------------------------------
    async def recognize_intent(self, state: ConversationState) -> ConversationState:
        """
        Intent Recognition - Categorize user query for appropriate handling.

        Supported intents:
        - greeting: Simple greetings (fast response, no RAG needed)
        - general_query: Default for knowledge base lookup

        Args:
            state: Current conversation state

        Returns:
            Updated state with intent and entities populated
        """
        async with time_node("recognize_intent", state):
            query = state["user_query"].strip().lower()

            # Check for greeting intent
            for category, keywords in GREETING_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in query:
                        state["intent"] = "greeting"
                        state["entities"] = {
                            "greeting_type": category,
                            "matched_keyword": keyword
                        }
                        logger.info(
                            "Greeting detected",
                            category=category,
                            keyword=keyword
                        )
                        return state

            # Default: general query requiring knowledge retrieval
            state["intent"] = "general_query"
            state["entities"] = {}

            logger.debug("Intent recognized as general_query")

        return state

    # -------------------------------------------------------------------------
    # Workflow Nodes - Knowledge Retrieval (RAG)
    # -------------------------------------------------------------------------
    async def knowledge_retrieval(self, state: ConversationState) -> ConversationState:
        """
        Knowledge Base Retrieval - Hybrid search for relevant documents.

        Process:
        1. Get kb_ids from employee config
        2. Use rewritten query if available
        3. Perform hybrid search (vector + BM25 + RRF fusion)

        Args:
            state: Current conversation state

        Returns:
            Updated state with retrieved_docs and kb_used populated
        """
        async with time_node("knowledge_retrieval", state):
            try:
                kb_ids = state["employee_config"].get("kb_ids", [])
                search_query = state.get("rewritten_query", state["user_query"])

                logger.info(
                    "Knowledge retrieval started",
                    employee_id=state["employee_id"],
                    kb_count=len(kb_ids),
                    query_rewritten=state.get("query_rewritten", False)
                )

                # Perform RAG search
                results = await rag_retrieval.search(
                    query=search_query,
                    kb_ids=kb_ids if kb_ids else None,
                    top_k=5,
                    use_hybrid=True
                )

                state["retrieved_docs"] = results
                state["kb_used"] = list(set([
                    doc.get("kb_id") for doc in results if doc.get("kb_id")
                ]))

                logger.info(
                    "Knowledge retrieval completed",
                    results_count=len(results),
                    kb_used=state["kb_used"]
                )

            except Exception as e:
                logger.error(f"Knowledge retrieval failed: {str(e)}", exc_info=True)
                state["retrieved_docs"] = []
                state["kb_used"] = []

        return state

    async def grade_documents(self, state: ConversationState) -> ConversationState:
        """
        Document Grading - Calculate relevance score for retrieved documents.

        Uses the top document's RRF score as the overall relevance score.
        This score determines if we should fallback to web search.

        Args:
            state: Current conversation state

        Returns:
            Updated state with relevance_score populated
        """
        async with time_node("grade_documents", state):
            docs = state.get("retrieved_docs", [])

            if not docs:
                state["relevance_score"] = 0.0
            else:
                # Use top document's RRF score as overall relevance
                state["relevance_score"] = docs[0].get("rrf_score", 0.0)

            logger.info(f"Document grading: relevance_score={state['relevance_score']:.4f}")

        return state

    async def rerank_documents(self, state: ConversationState) -> ConversationState:
        """
        Document Reranking - Reorder retrieved docs by semantic relevance.

        Strategy:
        - Only rerank when 3-10 documents retrieved
        - Use Grader LLM with JSON output for structured scoring
        - Sort by new scores for better context ordering

        Configuration: RERANK_ENABLED (default: False)

        Args:
            state: Current conversation state

        Returns:
            Updated state with retrieved_docs reordered
        """
        async with time_node("rerank_documents", state):
            docs = state.get("retrieved_docs", [])

            rerank_enabled = getattr(settings, 'rerank_enabled', False)
            if not rerank_enabled:
                logger.debug("Reranking disabled")
                return state

            # Skip if too few or too many docs
            if len(docs) <= 2 or len(docs) > 10:
                logger.debug(f"Document count不适合rerank: {len(docs)}")
                return state

            try:
                docs_text = "\n\n".join([
                    f"[文档{i+1}]\n{doc.get('content', '')[:300]}"
                    for i, doc in enumerate(docs[:5])
                ])

                rerank_prompt = f"""请根据用户问题对以下文档进行相关性评分。

用户问题: {state["user_query"]}

{docs_text}

评分标准:
- 1.0: 完全相关,直接回答了问题
- 0.7-0.9: 高度相关,包含答案的关键信息
- 0.4-0.6: 部分相关,需要推理才能回答
- 0.1-0.3: 低相关,仅提及相关主题
- 0.0: 不相关

请以JSON格式返回评分,格式如下:
{{"scores": [0.9, 0.7, 0.5, 0.2, 0.0]}}

只返回JSON,不要有其他内容:"""

                # Get appropriate Grader LLM for current state (hybrid routing)
                _, grader_llm, model_name = select_llm(
                    state,
                    self.workflow.local_llm,
                    self.workflow.local_grader_llm,
                    self.workflow.remote_llm,
                    self.workflow.remote_grader_llm
                )
                response = await grader_llm.ainvoke(rerank_prompt)
                response_text = response.content.strip()

                try:
                    result = json.loads(response_text)
                    scores = result.get("scores", [])

                    if len(scores) == len(docs[:5]):
                        # Reorder by new scores
                        indexed_docs = list(enumerate(docs[:5]))
                        indexed_docs.sort(key=lambda x: scores[x[0]], reverse=True)
                        reranked_docs = [doc for _, doc in indexed_docs]
                        if len(docs) > 5:
                            reranked_docs.extend(docs[5:])

                        state["retrieved_docs"] = reranked_docs

                        logger.info(
                            "Documents reranked successfully",
                            original_scores=[f"{d.get('rrf_score', 0):.3f}" for d in docs[:3]],
                            new_scores=[f"{s:.3f}" for s in scores[:3]]
                        )

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse rerank JSON: {e}")

            except Exception as e:
                logger.error(f"Document reranking failed: {str(e)}", exc_info=True)

        return state

    async def compress_context(self, state: ConversationState) -> ConversationState:
        """
        Context Compression - Reduce token usage by compressing retrieved content.

        Strategy:
        - Compress when total context > 1500 characters
        - Preserve information relevant to user query
        - Remove redundant and irrelevant content

        Configuration: CONTEXT_COMPRESSION_ENABLED (default: False)

        Args:
            state: Current conversation state

        Returns:
            Updated state with compressed_context populated
        """
        async with time_node("compress_context", state):
            state["compressed_context"] = None

            compression_enabled = getattr(settings, 'context_compression_enabled', False)
            if not compression_enabled:
                logger.debug("Context compression disabled")
                return state

            # Calculate total context length
            docs = state.get("retrieved_docs", [])
            web_results = state.get("web_search_results", [])
            total_length = (
                sum(len(doc.get('content', '')) for doc in docs) +
                sum(len(r.get('content', '')) for r in web_results)
            )

            if total_length < 1500:
                logger.debug(f"Context too short for compression: {total_length} chars")
                return state

            try:
                docs_text = "\n\n".join([
                    f"[文档{i+1}] {doc.get('content', '')[:400]}"
                    for i, doc in enumerate(docs[:3])
                ])

                compress_prompt = f"""请将以下文档内容压缩成最精炼的关键信息。

用户问题: {state["user_query"]}

{docs_text}

压缩要求:
1. 只保留与用户问题相关的信息
2. 去除重复和冗余内容
3. 使用简洁的语言
4. 压缩后的内容不超过500字
5. 保留关键数据和事实

压缩后的内容:"""

                # Get appropriate LLM for current state (hybrid routing)
                llm, _, model_name = select_llm(
                    state,
                    self.workflow.local_llm,
                    self.workflow.local_grader_llm,
                    self.workflow.remote_llm,
                    self.workflow.remote_grader_llm
                )
                response = await llm.ainvoke(compress_prompt)
                compressed = response.content.strip()

                if 100 < len(compressed) < total_length:
                    state["compressed_context"] = compressed
                    logger.info(
                        "Context compressed successfully",
                        original_length=total_length,
                        compressed_length=len(compressed),
                        compression_ratio=f"{(1 - len(compressed) / total_length) * 100:.1f}%"
                    )

            except Exception as e:
                logger.error(f"Context compression failed: {str(e)}", exc_info=True)

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
                    logger.info("Web search disabled in settings")
                    state["web_search_results"] = []
                    state["web_search_used"] = False
                    return state

                if not settings.tavily_api_key:
                    logger.warning("Tavily API key not configured")
                    state["web_search_results"] = []
                    state["web_search_used"] = False
                    return state

                # Check employee config for web search permission
                employee_config = state.get("employee_config", {})
                capabilities = employee_config.get("capabilities", {})
                if not capabilities.get("web_search_enabled", True):
                    logger.info(f"Web search disabled for employee: {state.get('employee_id')}")
                    state["web_search_results"] = []
                    state["web_search_used"] = False
                    return state

                query = state["user_query"]
                logger.info(
                    "Web search started",
                    query=query[:100],
                    is_realtime=state.get('is_realtime_query')
                )

                # Perform search
                web_search_tool = TavilySearchResults(k=3)
                search_results = await web_search_tool.ainvoke({"query": query})

                # Format results
                formatted_results = []
                if search_results:
                    for i, result in enumerate(search_results[:settings.web_search_max_results], 1):
                        formatted_results.append({
                            "rank": i,
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "content": result.get("content", "")[:500],
                            "score": result.get("score", 0.0)
                        })

                state["web_search_results"] = formatted_results
                state["web_search_used"] = len(formatted_results) > 0

                logger.info(
                    "Web search completed",
                    results_count=len(formatted_results)
                )

            except Exception as e:
                logger.error(f"Web search failed: {str(e)}", exc_info=True)
                state["web_search_results"] = []
                state["web_search_used"] = False

        return state

    # -------------------------------------------------------------------------
    # Workflow Nodes - Answer Generation
    # -------------------------------------------------------------------------
    async def generate_answer(self, state: ConversationState) -> ConversationState:
        """
        Answer Generation Preparation - Calculate confidence and prepare metadata.

        Note: Actual LLM streaming happens in the API endpoint.
        This node only prepares the state with confidence scores.

        Args:
            state: Current conversation state

        Returns:
            Updated state with confidence calculated
        """
        async with time_node("generate_answer", state):
            # Calculate confidence based on data sources
            confidence = 0.5  # Base confidence

            if state.get("faq_matched"):
                confidence = 0.95
            elif state.get("web_search_used", False):
                web_results = state.get("web_search_results", [])
                if web_results:
                    avg_web_score = sum(r.get("score", 0.5) for r in web_results) / len(web_results)
                    confidence = max(0.75, avg_web_score)
            elif state.get("retrieved_docs"):
                confidence = max(0.6, state.get("relevance_score", 0.7))

            state["confidence"] = confidence
            state["final_answer"] = ""  # Placeholder for streaming

            logger.info(
                "Ready for answer generation",
                confidence=confidence,
                web_search_used=state.get('web_search_used'),
                kb_docs_count=len(state.get('retrieved_docs', []))
            )

        return state

    # -------------------------------------------------------------------------
    # Workflow Nodes - Answer Verification
    # -------------------------------------------------------------------------
    async def verify_answer(self, state: ConversationState) -> ConversationState:
        """
        Answer Verification - Check consistency with source documents.

        Uses Grader LLM to verify:
        1. Key information is from source documents
        2. No hallucination or fabricated content
        3. No contradictions with source material

        Configuration: ANSWER_VERIFICATION_ENABLED (default: False)

        Args:
            state: Current conversation state

        Returns:
            Updated state with verification_result populated
        """
        async with time_node("verify_answer", state):
            state["answer_verified"] = False
            state["verification_result"] = None

            verification_enabled = getattr(settings, 'answer_verification_enabled', False)
            if not verification_enabled:
                logger.debug("Answer verification disabled")
                return state

            # Skip for FAQ and greeting (already validated)
            if state.get("faq_matched") or state.get("intent") == "greeting":
                logger.debug("Skipping verification for FAQ/greeting")
                return state

            answer = state.get("final_answer", "")
            docs = state.get("retrieved_docs", [])

            if not answer or len(answer) < 20 or not docs:
                logger.debug("Insufficient data for verification")
                return state

            try:
                docs_text = "\n\n".join([
                    f"[源文档{i+1}] {doc.get('content', '')[:400]}"
                    for i, doc in enumerate(docs[:3])
                ])

                verify_prompt = f"""请检查以下生成的答案是否与源文档一致。

用户问题: {state["user_query"]}

源文档:
{docs_text}

生成的答案:
{answer}

验证要求:
1. 检查答案中的关键信息是否在源文档中
2. 检查是否有幻觉或编造的内容
3. 检查是否有与源文档矛盾的陈述

请以JSON格式返回验证结果:
{{
    "is_consistent": true/false,
    "confidence": 0.0-1.0,
    "issues": ["不一致点1", "不一致点2"],
    "summary": "验证总结"
}}

只返回JSON,不要有其他内容:"""

                # Get appropriate Grader LLM for current state (hybrid routing)
                _, grader_llm, model_name = select_llm(
                    state,
                    self.workflow.local_llm,
                    self.workflow.local_grader_llm,
                    self.workflow.remote_llm,
                    self.workflow.remote_grader_llm
                )
                response = await grader_llm.ainvoke(verify_prompt)
                response_text = response.content.strip()

                try:
                    result = json.loads(response_text)
                    state["answer_verified"] = True
                    state["verification_result"] = result

                    if not result.get("is_consistent", True):
                        logger.warning(
                            "Answer inconsistency detected",
                            confidence=result.get("confidence", 0.8),
                            issues=result.get("issues", [])[:3]
                        )
                    else:
                        logger.info(
                            "Answer verification passed",
                            confidence=result.get("confidence", 0.8)
                        )

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse verification JSON: {e}")

            except Exception as e:
                logger.error(f"Answer verification failed: {str(e)}", exc_info=True)

        return state

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
