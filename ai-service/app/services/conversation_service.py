"""
LangGraph-based Conversation Workflow for Digital Employee.

This module implements a state-based conversation workflow using LangGraph, supporting:
- Multi-source knowledge retrieval (RAG + FAQ + Web Search)
- Query optimization (rewriting, compression)
- Dual-LLM architecture (local Ollama + remote OpenAI-style API)
- Streaming responses with performance monitoring

Workflow Graph:
    load_employee_config → load_session_context → input_validation
        → rewrite_query → check_realtime_query
        → [conditional: realtime?] → web_search OR match_faq
        → [conditional: FAQ matched?] → generate_answer OR recognize_intent
        → [conditional: greeting?] → generate_answer OR knowledge_retrieval
        → grade_documents → rerank_documents → compress_context
        → [conditional: low relevance?] → web_search OR generate_answer
        → generate_answer → verify_answer → save_conversation → END
"""
import os
import time
from pathlib import Path
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from operator import add
from datetime import datetime
from contextlib import asynccontextmanager

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langchain_community.tools.tavily_search import TavilySearchResults

from app.core.config import settings
from app.core.logging import get_logger
from app.core.database import get_database
from app.services.rag_service import rag_retrieval
from app.models.database import ConversationModel, SessionModel

logger = get_logger(__name__)

# =============================================================================
# Greeting Detection Keywords
# Used for quick pattern matching to identify simple greeting queries
# =============================================================================
GREETING_KEYWORDS = {
    # Basic greetings (你好, hi, hello, etc.)
    "basic": ["你好", "您好", "hi", "hello", "嗨"],
    # Time-based greetings (早上好, 晚上好, etc.)
    "time": [
        "早上好", "早", "上午好", "中午好", "下午好",
        "晚上好", "晚安"
    ],
    # Casual greetings (哈喽, 在吗, etc.)
    "casual": ["哈喽", "在吗", "在不在", "有人吗"],
    # Polite opening phrases (打扰一下, 请问, etc.)
    "polite": ["打扰一下", "请问", "不好意思", "劳驾"]
}


# =============================================================================
# Conversation State Definition
# =============================================================================
class ConversationState(TypedDict):
    """
    State object that flows through the LangGraph workflow nodes.

    Contains all information needed for conversation processing:
    - User input (user_query, user_id, session_id, employee_id)
    - Configuration (employee_config)
    - Detection results (is_realtime_query, intent, faq_matched)
    - RAG data (retrieved_docs, relevance_score, kb_used)
    - Web search (web_search_results, web_search_used)
    - Output (final_answer, confidence, conversation_id)
    - Performance metrics (response_time_ms, node_timings, ttfb_ms)
    - Optimization flags (query_rewritten, compressed_context, answer_verified)
    """
    messages: Annotated[List, add]
    user_query: str
    user_id: str
    session_id: str
    employee_id: str
    employee_config: Dict[str, Any]
    is_realtime_query: bool
    realtime_category: str
    realtime_detect_reason: str
    intent: str
    entities: Dict[str, Any]
    retrieved_docs: List[Dict[str, Any]]
    relevance_score: float
    web_search_results: List[Dict[str, Any]]
    final_answer: str
    confidence: float
    context: Dict[str, Any]
    has_sensitive: bool
    error: Optional[Dict[str, Any]]
    faq_matched: Optional[Dict[str, Any]]
    kb_used: List[str]
    web_search_used: bool
    conversation_id: str
    response_time_ms: int
    workflow_start_time: float
    node_timings: Dict[str, float]
    ttfb_ms: Optional[int]
    rewritten_query: str
    query_rewritten: bool
    compressed_context: Optional[str]
    answer_verified: bool
    verification_result: Optional[Dict[str, Any]]


# =============================================================================
# Conversation Workflow Class
# =============================================================================
class ConversationWorkflow:
    """
    LangGraph-based conversation workflow for digital employee interactions.

    Features:
    - Dual-LLM support: Local Ollama for fast responses, remote API for complex tasks
    - Hybrid retrieval: Vector search + keyword search + RRF fusion
    - FAQ fast-path: Direct answer matching with configurable threshold
    - Realtime query detection: Auto-route to web search for time-sensitive queries
    - Query optimization: Rewriting, context compression, document reranking
    - Answer verification: Consistency checking against source documents

    Workflow consists of 15 nodes connected by conditional edges.
    """

    def __init__(self):
        """Initialize workflow with configured LLM instances."""
        # Initialize LLM based on configuration
        if settings.use_ollama:
            logger.info(
                "Initializing Ollama LLMs",
                model=settings.ollama_model,
                base_url=settings.ollama_base_url
            )
            # Main LLM for answer generation (streaming enabled)
            self.llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0,
                streaming=True,
                keep_alive=-1  # Keep model loaded indefinitely
            )
            # Grader LLM for document scoring (JSON output mode)
            self.grader_llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_grader_model,
                temperature=0,
                format="json",
                keep_alive=-1
            )
            logger.info(f"Ollama LLMs initialized: model={settings.ollama_model}")
        else:
            # Use OpenAI-style API (e.g., SiliconFlow, DeepSeek)
            logger.info(
                "Initializing OpenAI-style LLMs",
                model=settings.openai_model,
                base_url=settings.openai_api_base
            )
            self.llm = ChatOpenAI(
                base_url=settings.openai_api_base,
                api_key=settings.siliconflow_api_key,
                model=settings.openai_model,
                temperature=settings.openai_temperature,
                streaming=True,
            )
            self.grader_llm = ChatOpenAI(
                base_url=settings.openai_api_base,
                api_key=settings.siliconflow_api_key,
                model=settings.openai_grader_model,
                temperature=0,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            logger.info(f"OpenAI LLMs initialized: model={settings.openai_model}")

        # Initialize web search tool
        self.web_search_tool = TavilySearchResults(k=3)

        # Build the workflow graph
        self.workflow = self._build_workflow()

    # -------------------------------------------------------------------------
    # Node Timing Utility
    # -------------------------------------------------------------------------
    @asynccontextmanager
    async def _time_node(self, node_name: str, state: ConversationState):
        """
        Async context manager for timing node execution.

        Tracks execution time for each workflow node and stores in state for analysis.
        Timing data is logged and included in final conversation record.

        Args:
            node_name: Name of the workflow node
            state: Conversation state object

        Example:
            async with self._time_node("knowledge_retrieval", state):
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

    # -------------------------------------------------------------------------
    # Workflow Graph Building
    # -------------------------------------------------------------------------
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow with all nodes and conditional edges.

        Graph structure:
        - Entry: load_employee_config
        - Middle: 13 processing nodes with conditional routing
        - Exit: save_conversation → END

        Returns:
            Compiled StateGraph ready for execution
        """
        graph = StateGraph(ConversationState)

        # Add all 15 workflow nodes
        graph.add_node("load_employee_config", self.load_employee_config)
        graph.add_node("load_session_context", self.load_session_context)
        graph.add_node("input_validation", self.validate_input)
        graph.add_node("rewrite_query", self.rewrite_query)
        graph.add_node("match_faq", self.match_faq)
        graph.add_node("check_realtime_query", self.check_realtime_query)
        graph.add_node("intent_recognition", self.recognize_intent)
        graph.add_node("knowledge_retrieval", self.knowledge_retrieval)
        graph.add_node("grade_documents", self.grade_documents)
        graph.add_node("rerank_documents", self.rerank_documents)
        graph.add_node("compress_context", self.compress_context)
        graph.add_node("web_search", self.web_search)
        graph.add_node("generate_answer", self.generate_answer)
        graph.add_node("verify_answer", self.verify_answer)
        graph.add_node("save_conversation", self.save_conversation)

        # Set entry point
        graph.set_entry_point("load_employee_config")

        # Define sequential edges
        graph.add_edge("load_employee_config", "load_session_context")
        graph.add_edge("load_session_context", "input_validation")
        graph.add_edge("input_validation", "rewrite_query")
        graph.add_edge("rewrite_query", "check_realtime_query")

        # Conditional routing after realtime check
        graph.add_conditional_edges(
            "check_realtime_query",
            lambda state: "web_search" if state.get("is_realtime_query") else "match_faq",
            {
                "web_search": "web_search",
                "match_faq": "match_faq"
            }
        )

        # Conditional routing after FAQ matching
        graph.add_conditional_edges(
            "match_faq",
            lambda state: "generate_answer" if state.get("faq_matched") else "intent_recognition",
            {
                "generate_answer": "generate_answer",
                "intent_recognition": "intent_recognition"
            }
        )

        # Conditional routing after intent recognition
        graph.add_conditional_edges(
            "intent_recognition",
            lambda state: "generate_answer" if state.get("intent") == "greeting" else "knowledge_retrieval",
            {
                "generate_answer": "generate_answer",
                "knowledge_retrieval": "knowledge_retrieval"
            }
        )

        # RAG pipeline edges
        graph.add_edge("knowledge_retrieval", "grade_documents")
        graph.add_edge("grade_documents", "rerank_documents")
        graph.add_edge("rerank_documents", "compress_context")

        # Conditional routing after context compression
        graph.add_conditional_edges(
            "compress_context",
            lambda state: "web_search" if state.get("relevance_score", 0) < settings.relevance_threshold else "generate_answer",
            {
                "web_search": "web_search",
                "generate_answer": "generate_answer"
            }
        )

        # Final sequence
        graph.add_edge("web_search", "generate_answer")
        graph.add_edge("generate_answer", "verify_answer")
        graph.add_edge("verify_answer", "save_conversation")
        graph.add_edge("save_conversation", END)

        # Compile and export graph for debugging
        compiled_graph = graph.compile()
        self._dump_graph_debug(compiled_graph)
        return compiled_graph

    def _dump_graph_debug(self, compiled_graph) -> None:
        """
        Export graph structure to Mermaid format for visualization.

        Output file: graph_debug/crag_graph.mmd
        Controlled by: CRAG_DUMP_GRAPH env variable (default: enabled)
        """
        dump_flag = os.getenv("CRAG_DUMP_GRAPH", "1").lower()
        if dump_flag in {"0", "false", "no"}:
            return

        try:
            graph_view = compiled_graph.get_graph(xray=True)
            output_dir = Path(os.getenv("CRAG_GRAPH_DIR", "./graph_debug"))
            output_dir.mkdir(parents=True, exist_ok=True)

            # Extract Mermaid source
            mermaid_src = None
            try:
                mermaid_src = graph_view.draw_mermaid()
                logger.info("Mermaid source extracted successfully")
            except Exception as src_exc:
                logger.warning(f"Failed to extract mermaid source: {src_exc}")
                (output_dir / "crag_graph_mermaid_extract_error.txt").write_text(
                    str(src_exc), encoding="utf-8"
                )

            # Save Mermaid file
            mermaid_path = output_dir / "crag_graph.mmd"
            if mermaid_src:
                mermaid_path.write_text(mermaid_src, encoding="utf-8")
                print(f"[OK] Mermaid source saved at {mermaid_path}")
            else:
                repr_text = repr(graph_view)
                (output_dir / "crag_graph_view_repr.txt").write_text(repr_text, encoding="utf-8")
                print("[WARN] Mermaid source unavailable, saved repr to crag_graph_view_repr.txt")

        except Exception as exc:
            logger.error(f"Graph debug dump failed: {exc}", exc_info=True)
            try:
                output_dir = Path(os.getenv("CRAG_GRAPH_DIR", "./graph_debug"))
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "crag_graph_dump_exception.txt").write_text(
                    str(exc), encoding="utf-8"
                )
            except Exception:
                pass
            print(f"[WARN] Unable to dump graph: {exc}")

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
        async with self._time_node("load_employee_config", state):
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
        async with self._time_node("load_session_context", state):
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
        async with self._time_node("validate_input", state):
            # Basic validation - can be extended
            state["has_sensitive"] = False
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
        async with self._time_node("rewrite_query", state):
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

                response = await self.llm.ainvoke(rewrite_prompt)
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
        async with self._time_node("match_faq", state):
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

                import random
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
    # Workflow Nodes - Realtime Query Detection
    # -------------------------------------------------------------------------
    async def check_realtime_query(self, state: ConversationState) -> ConversationState:
        """
        Realtime Query Detection - Identify time-sensitive queries.

        Detects queries requiring current information:
        - Time expressions (今天, 明天, 最近)
        - Weather (天气, 气温)
        - News (新闻, 热点, 最新)
        - Market data (股价, 价格, 汇率)

        If detected: Skip FAQ/RAG, route directly to web search

        Args:
            state: Current conversation state

        Returns:
            Updated state with is_realtime_query flag
        """
        async with self._time_node("check_realtime_query", state):
            if not settings.realtime_query_enabled:
                state["is_realtime_query"] = False
                return state

            query = state["user_query"].lower()

            # Realtime keyword categories
            realtime_keywords = {
                "time": ["今天", "明天", "昨天", "最近", "现在", "本周", "本月", "当前"],
                "weather": ["天气", "气温", "降雨", "降水", "温度"],
                "news": ["新闻", "热点", "最新", "资讯", "动态", "头条"],
                "market": ["股价", "汇率", "行情", "股市", "价格", "金价", "银价", "油价", "多少钱"],
            }

            for category, keywords in realtime_keywords.items():
                for keyword in keywords:
                    if keyword in query:
                        state["is_realtime_query"] = True
                        state["realtime_category"] = category
                        state['realtime_detect_reason'] = f"keyword:{keyword}"
                        logger.info(
                            "Realtime query detected",
                            category=category,
                            keyword=keyword
                        )
                        return state

            state["is_realtime_query"] = False
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
        async with self._time_node("recognize_intent", state):
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
        async with self._time_node("knowledge_retrieval", state):
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
        async with self._time_node("grade_documents", state):
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
        async with self._time_node("rerank_documents", state):
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

                response = await self.grader_llm.ainvoke(rerank_prompt)
                response_text = response.content.strip()

                import json
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
        async with self._time_node("compress_context", state):
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

                response = await self.llm.ainvoke(compress_prompt)
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
        async with self._time_node("web_search", state):
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
                search_results = await self.web_search_tool.ainvoke({"query": query})

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
        async with self._time_node("generate_answer", state):
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
        async with self._time_node("verify_answer", state):
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

                response = await self.grader_llm.ainvoke(verify_prompt)
                response_text = response.content.strip()

                import json
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
        async with self._time_node("save_conversation", state):
            try:
                db = await get_database()

                # Generate conversation ID
                import hashlib
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

    # -------------------------------------------------------------------------
    # Helper Methods for Message Building
    # -------------------------------------------------------------------------
    def _get_personality_description(self, personality: dict) -> tuple[str, str, str]:
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
        style_desc = style_map.get(personality.get("style", "friendly"), "友好")
        formality_desc = formality_map.get(personality.get("formality", "moderate"), "适度")

        return tone_desc, style_desc, formality_desc

    def _build_context_text(self, state: ConversationState) -> str:
        """
        Build context text from retrieved docs and web search results.

        Uses compressed context if available, otherwise builds from sources.

        Args:
            state: Current conversation state

        Returns:
            Formatted context string for LLM prompt
        """
        # Use compressed context if available
        if state.get("compressed_context"):
            return f"[压缩后的参考信息]\n{state['compressed_context']}"

        # Build from retrieved docs
        context_parts = []
        for i, doc in enumerate(state.get("retrieved_docs", [])[:3], 1):
            context_parts.append(f"[知识库参考{i}]\n{doc.get('content', '')[:500]}")

        # Add web search results
        web_results = state.get("web_search_results", [])
        if web_results and state.get("web_search_used", False):
            for i, web_result in enumerate(web_results[:3], 1):
                context_parts.append(
                    f"[网络资料{i}]\n标题: {web_result.get('title', '')}\n"
                    f"内容: {web_result.get('content', '')[:400]}\n"
                    f"来源: {web_result.get('url', '')}"
                )

        return "\n\n".join(context_parts) if context_parts else "（暂无相关参考资料）"

    def _get_source_indicator(self, state: ConversationState) -> str:
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

    def build_generation_messages(self, state: ConversationState) -> List:
        """
        Build LLM messages for answer generation.

        Handles different scenarios:
        - Greeting: Simple, friendly response
        - Realtime + Web search: Emphasize network sources
        - Regular RAG: Knowledge-based response

        Args:
            state: Current conversation state

        Returns:
            List of Message objects for LLM
        """
        employee_config = state.get("employee_config", {})

        # Handle greeting separately
        if state.get("intent") == "greeting":
            return self._build_greeting_messages(state, employee_config)

        # Get personality and basic config
        personality = employee_config.get("personality", {})
        role = employee_config.get("role", "AI助手")
        greeting = employee_config.get("greeting", "您好")
        tone_desc, style_desc, formality_desc = self._get_personality_description(personality)

        context_text = self._build_context_text(state)
        source_indicator = self._get_source_indicator(state)

        # System prompt for realtime queries with web search
        if state.get("web_search_used", False) and state.get("is_realtime_query", False):
            system_prompt = f"""你是 {employee_config.get('name', 'AI助手')}，{role}。

角色定位：
{employee_config.get('description', '专业的AI助手')}

个性特征：
- 语气风格：{tone_desc}
- 沟通方式：{style_desc}
- 正式程度：{formality_desc}

开场白：
{greeting}

**重要提示**：用户询问的是实时信息（如{state.get('realtime_category', '最新动态')}），系统已通过网络搜索获取了最新数据。

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
            # Regular RAG-based system prompt
            system_prompt = f"""你是 {employee_config.get('name', 'AI助手')}，{role}。

角色定位：
{employee_config.get('description', '专业的AI助手')}

个性特征：
- 语气风格：{tone_desc}
- 沟通方式：{style_desc}
- 正式程度：{formality_desc}

开场白：
{greeting}

回答要求：
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

        # Build messages list
        messages = [SystemMessage(content=system_prompt)]

        # Add conversation history (last 5 turns)
        for msg in state.get("context", {}).get("messages", [])[-5:]:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg.get("content", "")))

        # Current query
        messages.append(HumanMessage(content=state["user_query"]))

        return messages

    def _build_greeting_messages(
        self,
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

        # Get greeting type from entities
        entities = state.get("entities", {})
        greeting_type = entities.get("greeting_type", "basic")
        matched_keyword = entities.get("matched_keyword", "")

        tone_desc, _, _ = self._get_personality_description(personality)
        formality_desc = "高度正式" if personality.get('formality') == 'high' else "适度正式"

        # Style hints based on greeting type
        style_hints = {
            "time": f"根据时间（{matched_keyword}）给予相应的热情问候，并自然地询问用户今天需要什么帮助",
            "casual": "用轻松活泼的方式回应，表现出随时准备提供帮助的状态",
            "polite": "以礼貌、耐心的方式回应，让用户感受到专业和尊重",
            "basic": "用简洁友好的方式回应，自然地引导用户说明需求"
        }
        style_hint = style_hints.get(greeting_type, style_hints["basic"])

        system_prompt = f"""你是 {employee_config.get('name', 'AI助手')}，{role}。

角色定位：
{employee_config.get('description', '专业的AI助手')}

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

        # Add conversation history (last 3 turns)
        for msg in state.get("context", {}).get("messages", [])[-3:]:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg.get("content", "")))

        # Current greeting
        messages.append(HumanMessage(content=state["user_query"]))

        logger.debug(
            "Greeting messages built",
            greeting_type=greeting_type,
            matched_keyword=matched_keyword
        )

        return messages


# =============================================================================
# Global Workflow Instance
# =============================================================================
conversation_workflow = ConversationWorkflow()
