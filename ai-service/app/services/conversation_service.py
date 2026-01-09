"""
LangGraph conversation workflow.
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
# from langchain_community.tools.bing_search import BingSearchResults

from app.core.config import settings
from app.core.logging import get_logger
from app.core.database import get_database
from app.services.rag_service import rag_retrieval
from app.models.database import ConversationModel, SessionModel

logger = get_logger(__name__)

# 问候语关键词定义（用于快速检测）
GREETING_KEYWORDS = {
    # 基础问候
    "basic": ["你好", "您好", "hi", "hello", "嗨"],
    # 时间问候
    "time": [
        "早上好", "早", "上午好", "中午好", "下午好",
        "晚上好", "晚安"
    ],
    # 简单打招呼
    "casual": ["哈喽", "在吗", "在不在", "有人吗"],
    # 礼貌用语
    "polite": ["打扰一下", "请问", "不好意思", "劳驾"]
}


# Define conversation state
class ConversationState(TypedDict):
    """State for conversation workflow."""
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
    # FAQ matching
    faq_matched: Optional[Dict[str, Any]]
    # Tracking
    kb_used: List[str]
    web_search_used: bool
    conversation_id: str
    response_time_ms: int
    # Performance monitoring
    workflow_start_time: float
    node_timings: Dict[str, float]
    ttfb_ms: Optional[int]
    # Query optimization
    rewritten_query: str
    query_rewritten: bool
    # Context compression
    compressed_context: Optional[str]
    # Answer verification
    answer_verified: bool
    verification_result: Optional[Dict[str, Any]]


class ConversationWorkflow:
    """LangGraph-based conversation workflow."""
    
    def __init__(self):
        # Initialize LLM based on configuration
        if settings.use_ollama:
            logger.info(f"Ollama configuration detected, initializing Ollama LLMs,model= {settings.ollama_model}")
            # Use Ollama with keep_alive to prevent model unloading
            self.llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0,
                streaming=True,
                keep_alive=-1  # Keep model loaded indefinitely
            )
            self.grader_llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_grader_model,
                temperature=0,
                format="json",  # 强制 JSON
                keep_alive=-1  # Keep model loaded indefinitely
            )
            logger.info(f"Using Ollama LLM: model={settings.ollama_model}")
        else:
            # Use OpenAI-style API (e.g., SiliconFlow)
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
            logger.info(f"Using OpenAI-style LLM: model={settings.openai_model}")
        
        # 初始化 Web 搜索工具
        self.web_search_tool = TavilySearchResults(k=3)
        # self.web_search_tool = BingSearchResults(k=3)

        self.workflow = self._build_workflow()

    @asynccontextmanager
    async def _time_node(self, node_name: str, state: ConversationState):
        """
        Async context manager for timing node execution.

        Usage:
            async with self._time_node("node_name", state):
                # node logic here
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = int((time.time() - start_time) * 1000)

            # Store timing in state
            if "node_timings" not in state:
                state["node_timings"] = {}
            state["node_timings"][node_name] = duration_ms

            # Log timing
            logger.info(f"[TIMING] {node_name} - {duration_ms}ms")

    def _build_workflow(self) -> StateGraph:
        """Build the conversation workflow graph."""
        graph = StateGraph(ConversationState)

        # Add nodes
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

        # Add edges
        graph.add_edge("load_employee_config", "load_session_context")
        graph.add_edge("load_session_context", "input_validation")
        graph.add_edge("input_validation", "rewrite_query")
        graph.add_edge("rewrite_query", "check_realtime_query")

        # Conditional: realtime query -> web search, else -> match FAQ
        graph.add_conditional_edges(
            "check_realtime_query",
            lambda state: "web_search" if state.get("is_realtime_query") else "match_faq",
            {
                "web_search": "web_search",
                "match_faq": "match_faq"
            }
        )

        # Conditional: FAQ matched -> generate answer, else -> intent recognition
        graph.add_conditional_edges(
            "match_faq",
            lambda state: "generate" if state.get("faq_matched") else "intent_recognition",
            {
                "generate": "generate_answer",
                "intent_recognition": "intent_recognition"
            }
        )

        # Conditional: greeting -> generate answer directly, else -> knowledge retrieval
        graph.add_conditional_edges(
            "intent_recognition",
            lambda state: "generate_answer" if state.get("intent") == "greeting" else "knowledge_retrieval",
            {
                "generate_answer": "generate_answer",
                "knowledge_retrieval": "knowledge_retrieval"
            }
        )
        graph.add_edge("knowledge_retrieval", "grade_documents")
        graph.add_edge("grade_documents", "rerank_documents")
        graph.add_edge("rerank_documents", "compress_context")

        # Conditional: low relevance -> web search, else -> compress_context
        graph.add_conditional_edges(
            "compress_context",
            lambda state: "web_search" if state.get("relevance_score", 0) < settings.relevance_threshold else "generate_answer",
            {
                "web_search": "web_search",
                "generate_answer": "generate_answer"
            }
        )

        graph.add_edge("web_search", "generate_answer")
        graph.add_edge("generate_answer", "verify_answer")
        graph.add_edge("verify_answer", "save_conversation")
        graph.add_edge("save_conversation", END)
        compiled_stateGraph = graph.compile()
        self._dump_graph_debug(compiled_stateGraph)
        return compiled_stateGraph
        
    def _dump_graph_debug(self, compiled_stateGraph) -> None:
        """保存图结构用于调试"""
        dump_flag = os.getenv("CRAG_DUMP_GRAPH", "1").lower()
        if dump_flag in {"0", "false", "no"}:
            return

        try:
            graph_view = compiled_stateGraph.get_graph(xray=True)
            output_dir = Path(os.getenv("CRAG_GRAPH_DIR", "./graph_debug"))
            output_dir.mkdir(parents=True, exist_ok=True)

            # 正确的 mermaid 提取方式
            mermaid_src = None
            try:
                mermaid_src = graph_view.draw_mermaid()  # 返回 mermaid 字符串
                logger.info("Mermaid source extracted successfully")
            except Exception as src_exc:
                logger.warning(f"Failed to extract mermaid source: {src_exc}")
                (output_dir / "crag_graph_mermaid_extract_error.txt").write_text(
                    str(src_exc), encoding="utf-8"
                )

            # 保存 mermaid 源文件
            mermaid_path = output_dir / "crag_graph.mmd"
            if mermaid_src:
                mermaid_path.write_text(mermaid_src, encoding="utf-8")
                print(f"[OK] Mermaid source saved at {mermaid_path}")
            else:
                # 回退：保存 repr 以便调试
                repr_text = repr(graph_view)
                (output_dir / "crag_graph_view_repr.txt").write_text(repr_text, encoding="utf-8")
                print("[WARN] Mermaid source unavailable, saved repr to crag_graph_view_repr.txt")

            # 尝试远程渲染（如果启用）
            # 渲染在windows开发机上进行
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

    async def load_employee_config(self, state: ConversationState) -> ConversationState:
        """Load employee configuration (enhanced with full config)."""
        async with self._time_node("load_employee_config", state):
            db = await get_database()
            employee = await db.digital_employee_configs.find_one({"employee_id": state["employee_id"]})
            logger.info(f"Loading employee config: employee_id={state['employee_id']}")
            if not employee:
                error_msg = f"Employee config not found: employee_id={state['employee_id']}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            employee.pop("_id", None)
            state["employee_config"] = employee

            logger.info(f"Employee config loaded: {employee}")
        return state
    
    async def load_session_context(self, state: ConversationState) -> ConversationState:
        """Load session context."""
        async with self._time_node("load_session_context", state):
            try:
                db = await get_database()
                session = await db.sessions.find_one({"session_id": state["session_id"]})

                if session:
                    # Load recent messages
                    state["context"] = {
                        "messages": session.get("context_messages", [])[-10:],  # Last 10 messages
                        "message_count": session.get("message_count", 0)
                    }

                    # Update last activity
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

                logger.info(f"Loaded session context: session_id={state['session_id']}")
            except Exception as e:
                logger.error(f"Failed to load session context: error={str(e)}", exc_info=True)
                state["context"] = {"messages": [], "message_count": 0}
        return state
    
    async def validate_input(self, state: ConversationState) -> ConversationState:
        """Validate input."""
        async with self._time_node("validate_input", state):
            # Simple validation - already done at API level
            state["has_sensitive"] = False
        return state

    async def rewrite_query(self, state: ConversationState) -> ConversationState:
        """
        Query Rewriting - 使用LLM重写用户查询,提升检索准确度。

        策略:
        - 对短查询(<20字)进行扩展
        - 保持原意,增加关键词
        - 使用快速模型减少延迟

        配置: 通过环境变量 QUERY_REWRITE_ENABLED 控制
        """
        async with self._time_node("rewrite_query", state):
            state["query_rewritten"] = False
            state["rewritten_query"] = state["user_query"]

            # 检查是否启用查询重写
            rewrite_enabled = getattr(settings, 'query_rewrite_enabled', False)
            if not rewrite_enabled:
                logger.debug("Query rewriting disabled")
                return state

            query = state["user_query"].strip()

            # 只重写短查询 (少于20字)
            if len(query) >= 20:
                logger.debug(f"Query too long for rewriting, using original: {len(query)} chars")
                return state

            # 问候语不需要重写
            query_lower = query.lower()
            for keywords in GREETING_KEYWORDS.values():
                if any(kw in query_lower for kw in keywords):
                    logger.debug("Greeting detected, skipping query rewrite")
                    return state

            try:
                # 构造重写prompt
                rewrite_prompt = f"""你是一个查询优化助手。请将用户查询重写为更具体的搜索语句,用于知识库检索。

原查询: {query}

要求:
1. 保持原意不变
2. 增加相关关键词和同义词
3. 使查询更具体、更完整
4. 只返回重写后的查询,不要解释
5. 长度控制在50字以内

重写后的查询:"""

                # 使用LLM重写 (非流式,速度快)
                response = await self.llm.ainvoke(rewrite_prompt)
                rewritten = response.content.strip()

                # 验证重写结果
                if rewritten and len(rewritten) > len(query) and len(rewritten) < 100:
                    state["rewritten_query"] = rewritten
                    state["query_rewritten"] = True
                    logger.info(
                        f"Query rewritten successfully",
                        original=query[:50],
                        rewritten=rewritten[:50],
                        original_len=len(query),
                        rewritten_len=len(rewritten)
                    )
                else:
                    logger.warning(
                        f"Query rewrite result invalid, using original",
                        rewritten=rewritten[:50] if rewritten else "empty"
                    )

            except Exception as e:
                logger.error(f"Query rewriting failed, using original query: {str(e)}", exc_info=True)
                # 失败时使用原始查询
                state["rewritten_query"] = state["user_query"]

        return state

    async def match_faq(self, state: ConversationState) -> ConversationState:
        """
        Match FAQ using hybrid search (vector + keyword + RRF fusion).
        如果FAQ的RRF分数 >= faq_sim_threshold，直接返回FAQ答案（随机选择），
        跳过后续的RAG检索和LLM生成。
        """
        async with self._time_node("match_faq", state):
            try:
                query = state["user_query"]
                employee_id = state["employee_id"]

                # Get FAQ configuration from digital_employee_configs
                db = await get_database()
                digital_config = await db.digital_employee_configs.find_one({"employee_id": employee_id})

                if not digital_config:
                    logger.info(f"No digital employee config found for {employee_id}, skipping FAQ matching")
                    state["faq_matched"] = None
                    return state

                faq_sim_threshold = digital_config.get("faq_sim_threshold", 0.0)
                faq_top_k = digital_config.get("faq_top_k", 3)

                logger.info(f"FAQ matching: employee_id={employee_id}, threshold={faq_sim_threshold}, top_k={faq_top_k}")

                # Perform FAQ hybrid search (vector + keyword + RRF)
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

                # Get the best FAQ (highest RRF score)
                best_faq_result = faq_results[0]
                faq_id = best_faq_result["faq_id"]
                rrf_score = best_faq_result["rrf_score"]

                logger.info(f"FAQ matched: faq_id={faq_id}, rrf_score={rrf_score:.4f}, vector_score={best_faq_result.get('vector_score', 0):.4f}, keyword_score={best_faq_result.get('keyword_score', 0):.4f}")

                # Retrieve full FAQ from MongoDB
                faq_doc = await db.faqs.find_one({"faq_id": faq_id})

                if not faq_doc:
                    logger.warning(f"FAQ {faq_id} not found in MongoDB, skipping")
                    state["faq_matched"] = None
                    return state

                # Check if FAQ is enabled and within time range
                if faq_doc.get("is_enable", 0) != 1:
                    logger.info(f"FAQ {faq_id} is disabled, skipping")
                    state["faq_matched"] = None
                    return state

                # Check time range (optional: add time validation logic here)
                # start_time = faq_doc.get("start_time")
                # end_time = faq_doc.get("end_time")
                # now = datetime.utcnow().isoformat()
                # if start_time and now < start_time: ...
                # if end_time and now > end_time: ...

                # Select answer randomly from answers array
                answers = faq_doc.get("answers", [])
                if not answers:
                    logger.warning(f"FAQ {faq_id} has no answers, skipping")
                    state["faq_matched"] = None
                    return state

                import random
                selected_answer = random.choice(answers)

                # FAQ 质量检查：如果 RRF 分数过低（< 0.02），跳过以避免误匹配
                # RRF 分数范围通常在 0.01-0.05 之间，低于 0.02 表示相关性很低
                if rrf_score < 0.02:
                    logger.warning(
                        f"FAQ RRF score too low ({rrf_score:.4f} < 0.02), skipping FAQ match to avoid false positive",
                        faq_id=faq_id,
                        question=faq_doc.get('question_name')[:50],
                        vector_score=best_faq_result.get('vector_score', 0),
                        keyword_score=best_faq_result.get('keyword_score', 0)
                    )
                    state["faq_matched"] = None
                    return state

                # Set state for direct FAQ answer return
                state["final_answer"] = selected_answer
                state["confidence"] = min(0.95, rrf_score)  # Cap at 0.95
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
                    f"FAQ answer selected: faq_id={faq_id}, question={faq_doc.get('question_name')[:50]}, "
                    f"rrf_score={rrf_score:.4f}, vector_score={best_faq_result.get('vector_score', 0):.4f}, "
                    f"keyword_score={best_faq_result.get('keyword_score', 0):.4f}, answer_length={len(selected_answer)}"
                )

            except Exception as e:
                logger.error(f"FAQ matching failed: error={str(e)}", exc_info=True)
                state["faq_matched"] = None

        return state
    
    async def check_realtime_query(self, state: ConversationState) -> ConversationState:
        """Check if query needs realtime information."""
        async with self._time_node("check_realtime_query", state):
            if not settings.realtime_query_enabled:
                state["is_realtime_query"] = False
                return state

            query = state["user_query"].lower()

            # Realtime keywords（包含金融价格查询）
            realtime_keywords = {
                "time": ["今天", "明天", "昨天", "最近", "现在", "本周", "本月", "当前"],
                "weather": ["天气", "气温", "降雨", "降水", "温度"],
                "news": ["新闻", "热点", "最新", "资讯", "动态", "头条"],
                "market": ["股价", "汇率", "行情", "股市", "价格", "金价", "银价", "油价", "多少钱", "最新价格", "实时价格"],
            }

            for category, keywords in realtime_keywords.items():
                for keyword in keywords:
                    if keyword in query:
                        state["is_realtime_query"] = True
                        state["realtime_category"] = category
                        state['realtime_detect_reason'] = f"keyword:{keyword}"
                        logger.info(f"Realtime query detected: category={category}, keyword={keyword}")
                        return state

            state["is_realtime_query"] = False
        return state

    async def recognize_intent(self, state: ConversationState) -> ConversationState:
        """
        识别用户意图（增强版：支持问候检测）。

        检测优先级：
        1. 问候语检测（关键词匹配）
        2. 其他意图（保留扩展空间）
        3. 默认：general_query
        """
        async with self._time_node("recognize_intent", state):
            query = state["user_query"].strip().lower()

            # 1. 问候语检测
            for category, keywords in GREETING_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in query:
                        state["intent"] = "greeting"
                        state["entities"] = {"greeting_type": category, "matched_keyword": keyword}

                        logger.info(
                            "Greeting detected",
                            employee_id=state["employee_id"],
                            query=query[:50],
                            greeting_type=category,
                            matched_keyword=keyword
                        )
                        return state

            # 2. 其他意图识别（保留扩展空间）
            # 未来可添加：投诉、咨询、预约等

            # 3. 默认：一般查询
            state["intent"] = "general_query"
            state["entities"] = {}

            logger.debug(
                "Intent recognized",
                employee_id=state["employee_id"],
                query=query[:50],
                intent="general_query"
            )
        return state
    
    async def knowledge_retrieval(self, state: ConversationState) -> ConversationState:
        """Retrieve relevant knowledge."""
        async with self._time_node("knowledge_retrieval", state):
            try:
                # Get KB IDs from employee config (at root level, not in capabilities)
                kb_ids = state["employee_config"].get("kb_ids", [])

                # 使用重写后的查询进行检索 (如果有的话)
                search_query = state.get("rewritten_query", state["user_query"])

                logger.info(
                    "Retrieving knowledge from KB",
                    employee_id=state["employee_id"],
                    kb_ids=kb_ids,
                    kb_count=len(kb_ids),
                    query_rewritten=state.get("query_rewritten", False),
                    original_query=state["user_query"][:50],
                    search_query=search_query[:50]
                )

                # Search using RAG
                results = await rag_retrieval.search(
                    query=search_query,
                    kb_ids=kb_ids if kb_ids else None,
                    top_k=5,
                    use_hybrid=True
                )

                state["retrieved_docs"] = results
                state["kb_used"] = list(set([doc.get("kb_id") for doc in results if doc.get("kb_id")]))

                logger.info(f"Knowledge retrieval completed: results_count={len(results)}, kb_used={state['kb_used']}")

            except Exception as e:
                logger.error(f"Knowledge retrieval failed: error={str(e)}", exc_info=True)
                state["retrieved_docs"] = []
                state["kb_used"] = []
        return state
    
    async def grade_documents(self, state: ConversationState) -> ConversationState:
        """Grade document relevance."""
        async with self._time_node("grade_documents", state):
            docs = state.get("retrieved_docs", [])

            if not docs:
                state["relevance_score"] = 0.0
                return state

            # Use the top document's score as relevance score
            state["relevance_score"] = docs[0].get("rrf_score", 0.0) if docs else 0.0

            logger.info(f"Document grading completed: relevance_score={state['relevance_score']}")
        return state

    async def rerank_documents(self, state: ConversationState) -> ConversationState:
        """
        Reranking - 使用Grader LLM对检索到的文档重新评分和排序。

        策略:
        - 只对检索到3-10个文档时进行rerank
        - 使用Grader LLM的JSON模式强制输出结构化评分
        - 根据用户查询的语义相关性评分

        配置: 通过环境变量 RERANK_ENABLED 控制
        """
        async with self._time_node("rerank_documents", state):
            docs = state.get("retrieved_docs", [])

            # 检查是否启用rerank
            rerank_enabled = getattr(settings, 'rerank_enabled', False)
            if not rerank_enabled:
                logger.debug("Reranking disabled")
                return state

            # 文档太少或太多时不rerank
            if len(docs) <= 2:
                logger.debug(f"Too few documents for reranking: {len(docs)}")
                return state
            if len(docs) > 10:
                logger.debug(f"Too many documents for reranking: {len(docs)}, skipping")
                return state

            try:
                # 构造rerank prompt
                docs_text = "\n\n".join([
                    f"[文档{i+1}]\n{doc.get('content', '')[:300]}"
                    for i, doc in enumerate(docs[:5])  # 最多rerank前5个
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

                # 使用Grader LLM评分
                response = await self.grader_llm.ainvoke(rerank_prompt)
                response_text = response.content.strip()

                # 解析JSON响应
                import json
                try:
                    result = json.loads(response_text)
                    scores = result.get("scores", [])

                    if len(scores) == len(docs[:5]):
                        # 根据新分数重新排序
                        indexed_docs = list(enumerate(docs[:5]))
                        indexed_docs.sort(key=lambda x: scores[x[0]], reverse=True)

                        # 更新docs顺序
                        reranked_docs = [doc for _, doc in indexed_docs]
                        if len(docs) > 5:
                            reranked_docs.extend(docs[5:])

                        state["retrieved_docs"] = reranked_docs

                        logger.info(
                            f"Documents reranked successfully",
                            original_scores=[f"{d.get('rrf_score', 0):.3f}" for d in docs[:3]],
                            new_scores=[f"{s:.3f}" for s in scores[:3]],
                            doc_order_changed=True
                        )
                    else:
                        logger.warning(
                            f"Rerank scores count mismatch: expected {len(docs[:5])}, got {len(scores)}"
                        )

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse rerank JSON response: {e}, using original order")

            except Exception as e:
                logger.error(f"Document reranking failed, using original order: {str(e)}", exc_info=True)

        return state

    async def web_search(self, state: ConversationState) -> ConversationState:
        """Perform web search using Tavily."""
        async with self._time_node("web_search", state):
            try:
                # Check if web search is enabled
                if not settings.web_search_enabled:
                    logger.info("Web search disabled in settings")
                    state["web_search_results"] = []
                    state["web_search_used"] = False
                    return state

                # Check if Tavily API key is configured
                if not settings.tavily_api_key:
                    logger.warning("Tavily API key not configured, skipping web search")
                    state["web_search_results"] = []
                    state["web_search_used"] = False
                    return state

                # Check employee config for web search permission
                employee_config = state.get("employee_config", {})
                capabilities = employee_config.get("capabilities", {})
                web_search_enabled = capabilities.get("web_search_enabled", True)

                if not web_search_enabled:
                    logger.info(f"Web search disabled for employee: employee_id={state.get('employee_id')}")
                    state["web_search_results"] = []
                    state["web_search_used"] = False
                    return state

                query = state["user_query"]

                # Perform web search
                logger.info(f"Performing web search: query={query[:100]}, is_realtime={state.get('is_realtime_query')}, realtime_category={state.get('realtime_category')}")

                # Call Tavily search tool
                search_results = await self.web_search_tool.ainvoke({"query": query})

                # Format results
                formatted_results = []
                if search_results:
                    for i, result in enumerate(search_results[:settings.web_search_max_results], 1):
                        formatted_result = {
                            "rank": i,
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "content": result.get("content", "")[:500],  # Truncate to 500 chars
                            "score": result.get("score", 0.0)
                        }
                        formatted_results.append(formatted_result)

                state["web_search_results"] = formatted_results
                state["web_search_used"] = len(formatted_results) > 0

                logger.info(f"Web search completed: results_count={len(formatted_results)}, has_results={state['web_search_used']}")

            except Exception as e:
                logger.error(f"Web search failed: error={str(e)}, query={state.get('user_query', '')[:100]}", exc_info=True)
                # Don't fail the entire workflow, just continue without web results
                state["web_search_results"] = []
                state["web_search_used"] = False
        return state

    async def compress_context(self, state: ConversationState) -> ConversationState:
        """
        Context Compression - 智能压缩检索到的上下文,减少token使用。

        策略:
        - 当上下文超过1500字时进行压缩
        - 保留与用户问题最相关的信息
        - 去除冗余和无关内容

        配置: 通过环境变量 CONTEXT_COMPRESSION_ENABLED 控制
        """
        async with self._time_node("compress_context", state):
            state["compressed_context"] = None

            # 检查是否启用上下文压缩
            compression_enabled = getattr(settings, 'context_compression_enabled', False)
            if not compression_enabled:
                logger.debug("Context compression disabled")
                return state

            # 获取所有上下文内容
            docs = state.get("retrieved_docs", [])
            web_results = state.get("web_search_results", [])

            # 计算总上下文长度
            total_context_length = sum(len(doc.get('content', '')) for doc in docs)
            total_context_length += sum(len(r.get('content', '')) for r in web_results)

            # 只有上下文超过1500字时才压缩
            if total_context_length < 1500:
                logger.debug(f"Context too short for compression: {total_context_length} chars")
                return state

            try:
                # 构造压缩prompt
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

                # 使用LLM压缩
                response = await self.llm.ainvoke(compress_prompt)
                compressed = response.content.strip()

                if len(compressed) > 100 and len(compressed) < total_context_length:
                    state["compressed_context"] = compressed
                    logger.info(
                        f"Context compressed successfully",
                        original_length=total_context_length,
                        compressed_length=len(compressed),
                        compression_ratio=f"{(1 - len(compressed) / total_context_length) * 100:.1f}%"
                    )
                else:
                    logger.warning(
                        f"Compression result invalid, using original context",
                        compressed_len=len(compressed),
                        original_len=total_context_length
                    )

            except Exception as e:
                logger.error(f"Context compression failed, using original: {str(e)}", exc_info=True)

        return state

    async def generate_answer(self, state: ConversationState) -> ConversationState:
        """Prepare for answer generation (placeholder for streaming)."""
        async with self._time_node("generate_answer", state):
            # This node only prepares metadata, no actual LLM call
            # Real generation happens in streaming methods for token-level streaming

            # Calculate confidence based on available information sources
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
            state["final_answer"] = ""  # Placeholder

            logger.info(f"Ready for answer generation: confidence={confidence}, web_search_used={state.get('web_search_used')}, kb_docs_count={len(state.get('retrieved_docs', []))}")
        return state
    
    def build_generation_messages(self, state: ConversationState) -> List:
        """Build messages for LLM generation (used by streaming methods)."""
        employee_config = state.get("employee_config", {})

        # 针对问候场景的专门处理
        if state.get("intent") == "greeting":
            return self._build_greeting_messages(state, employee_config)

        # 原有逻辑：FAQ、实时查询、常规场景
        personality = employee_config.get("personality", {})
        role = employee_config.get("role", "AI助手")
        greeting = employee_config.get("greeting", "您好")

        # 优先使用压缩后的上下文
        if state.get("compressed_context"):
            context_text = f"[压缩后的参考信息]\n{state['compressed_context']}"
            logger.debug("Using compressed context for generation")
        else:
            # Build context from retrieved docs
            context_parts = []
            for i, doc in enumerate(state.get("retrieved_docs", [])[:3], 1):
                context_parts.append(f"[知识库参考{i}]\n{doc.get('content', '')[:500]}")

            # Add web search results if available
            web_results = state.get("web_search_results", [])
            if web_results and state.get("web_search_used", False):
                for i, web_result in enumerate(web_results[:3], 1):
                    web_context = f"[网络资料{i}]\n标题: {web_result.get('title', '')}\n内容: {web_result.get('content', '')[:400]}\n来源: {web_result.get('url', '')}"
                    context_parts.append(web_context)

            context_text = "\n\n".join(context_parts) if context_parts else "（暂无相关参考资料）"

        # Add information source indicator
        source_indicator = ""
        if state.get("web_search_used", False):
            source_indicator = "（包含最新网络信息）"
        elif state.get("retrieved_docs"):
            source_indicator = "（基于知识库）"
        
        # Build personality-based system prompt
        tone_desc = {
            "professional": "专业严谨",
            "friendly": "友好亲切",
            "formal": "正式庄重",
            "casual": "轻松随意"
        }.get(personality.get("tone", "professional"), "专业")
        
        style_desc = {
            "concise": "简明扼要",
            "detailed": "详细周到",
            "conversational": "对话式",
            "instructional": "指导式"
        }.get(personality.get("style", "friendly"), "友好")
        
        formality_desc = {
            "high": "高度正式（使用敬语）",
            "moderate": "适度正式",
            "low": "轻松口语化"
        }.get(personality.get("formality", "moderate"), "适度")
        
        # 针对实时查询和网络搜索场景，使用不同的系统提示
        if state.get("web_search_used", False) and state.get("is_realtime_query", False):
            # 实时查询场景：强调使用网络搜索结果
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
1. **必须基于下方提供的网络资料回答**，这些是通过实时搜索获得的最新信息
2. 直接提取网络资料中的关键信息，如价格、数据、时间等
3. 保持{tone_desc}的语气风格
4. 回答简洁明了，重点突出具体数据
5. 如果网络资料中有多个相关信息源，整合后给出完整回答
6. 可在回答末尾简要注明信息来源（如"以上信息来自[来源名称]"）
7. **不要说"无法提供实时数据"或"知识库不包含"这类话**，因为网络搜索结果就是实时数据

上下文信息{source_indicator}：
{context_text}

用户问题：
{state['user_query']}

请基于上述网络资料，提供准确的实时信息回答。"""
        else:
            # 常规场景：知识库检索或一般问答
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
6. 如果使用了网络资料，可在回答末尾注明信息来源

上下文信息{source_indicator}：
{context_text}

用户问题：
{state['user_query']}

请提供专业、准确的回答。"""
        
        # Build messages
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
        构建问候场景的 LLM 消息。

        特点：
        - 不依赖任何上下文（无 RAG、无网络搜索）
        - 基于数字员工个性生成友好回应
        - 简洁自然，鼓励用户进一步交流
        """
        personality = employee_config.get("personality", {})
        role = employee_config.get("role", "AI助手")
        greeting = employee_config.get("greeting", "您好")

        # 获取问候类型（从 entities）
        entities = state.get("entities", {})
        greeting_type = entities.get("greeting_type", "basic")
        matched_keyword = entities.get("matched_keyword", "")

        # 构建个性描述
        tone_desc = {
            "professional": "专业严谨",
            "friendly": "友好亲切",
            "formal": "正式庄重",
            "casual": "轻松随意"
        }.get(personality.get("tone", "professional"), "专业")

        # 根据问候类型调整回复风格
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
- 正式程度：{'高度正式' if personality.get('formality') == 'high' else '适度正式'}

开场白：
{greeting}

**当前场景**：用户向你发起问候（"{matched_keyword}"）。

回答要求：
1. {style_hint}
2. 回复要简洁（不超过 50 字），不要过度冗长
3. 保持{tone_desc}的语气风格，符合你的角色设定
4. 不要提及"我是AI"或"我是机器人"，自然地像真人一样回应
5. 不要提供任何具体信息或产品介绍（除非用户主动询问）
6. 可以根据用户的历史对话上下文（如果有）调整问候的个性化程度

用户原话：
{state['user_query']}

请生成自然、友好的问候回应。"""

        messages = [SystemMessage(content=system_prompt)]

        # 添加历史对话上下文（最近 3 轮）
        for msg in state.get("context", {}).get("messages", [])[-3:]:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg.get("content", "")))

        # 当前问候
        messages.append(HumanMessage(content=state["user_query"]))

        logger.debug(
            "Greeting messages built",
            employee_id=state["employee_id"],
            greeting_type=greeting_type,
            personality_tone=personality.get("tone", "professional")
        )

        return messages

    async def verify_answer(self, state: ConversationState) -> ConversationState:
        """
        Answer Consistency Check - 验证生成的答案是否与源文档一致。

        策略:
        - 使用Grader LLM检查答案与源文档的一致性
        - 如果检测到不一致,记录警告但不修改答案(避免延迟)
        - 保存验证结果到conversation记录

        配置: 通过环境变量 ANSWER_VERIFICATION_ENABLED 控制
        """
        async with self._time_node("verify_answer", state):
            state["answer_verified"] = False
            state["verification_result"] = None

            # 检查是否启用答案验证
            verification_enabled = getattr(settings, 'answer_verification_enabled', False)
            if not verification_enabled:
                logger.debug("Answer verification disabled")
                return state

            # FAQ匹配和问候语不需要验证
            if state.get("faq_matched") or state.get("intent") == "greeting":
                logger.debug("Skipping answer verification for FAQ/greeting")
                return state

            # 获取答案和源文档
            answer = state.get("final_answer", "")
            docs = state.get("retrieved_docs", [])

            if not answer or len(answer) < 20:
                logger.debug("Answer too short for verification")
                return state

            if not docs:
                logger.debug("No source documents for verification")
                return state

            try:
                # 构造验证prompt
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

                # 使用Grader LLM验证
                response = await self.grader_llm.ainvoke(verify_prompt)
                response_text = response.content.strip()

                # 解析JSON响应
                import json
                try:
                    result = json.loads(response_text)

                    state["answer_verified"] = True
                    state["verification_result"] = result

                    is_consistent = result.get("is_consistent", True)
                    confidence = result.get("confidence", 0.8)

                    if not is_consistent:
                        logger.warning(
                            f"Answer inconsistency detected",
                            confidence=confidence,
                            issues=result.get("issues", [])[:3],
                            summary=result.get("summary", "")[:100]
                        )
                    else:
                        logger.info(
                            f"Answer verification passed",
                            confidence=confidence,
                            summary=result.get("summary", "")[:100]
                        )

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse verification JSON: {e}")

            except Exception as e:
                logger.error(f"Answer verification failed: {str(e)}", exc_info=True)

        return state

    async def save_conversation(self, state: ConversationState) -> ConversationState:
        """Save conversation to database."""
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

                # Update session context
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

                logger.info(f"Conversation saved: conversation_id={conv_id}")

                # Log timing summary
                node_timings = state.get("node_timings", {})
                ttfb_ms = state.get("ttfb_ms")

                logger.info(
                    f"[TIMING_SUMMARY] Conversation completed - "
                    f"total: {total_time_ms}ms, "
                    f"ttfb: {ttfb_ms}ms, "
                    f"nodes: {node_timings}"
                )

            except Exception as e:
                logger.error(f"Failed to save conversation: error={str(e)}", exc_info=True)

        return state

# Global workflow instance
conversation_workflow = ConversationWorkflow()
