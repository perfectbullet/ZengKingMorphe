"""
LangGraph conversation workflow.
"""
import os
from pathlib import Path
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from operator import add
from datetime import datetime

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


class ConversationWorkflow:
    """LangGraph-based conversation workflow."""
    
    def __init__(self):
        # Initialize LLM based on configuration
        if settings.use_ollama:
            logger.info(f"Ollama configuration detected, initializing Ollama LLMs,model= {settings.ollama_model}")
            # Use Ollama
            self.llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0,
                streaming=True,
            )
            self.grader_llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_grader_model,
                temperature=0,
                format="json",  # 强制 JSON
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
    
    def _build_workflow(self) -> StateGraph:
        """Build the conversation workflow graph."""
        graph = StateGraph(ConversationState)
        
        # Add nodes
        graph.add_node("load_employee_config", self.load_employee_config)
        graph.add_node("load_session_context", self.load_session_context)
        graph.add_node("input_validation", self.validate_input)
        graph.add_node("match_faq", self.match_faq)
        graph.add_node("check_realtime_query", self.check_realtime_query)
        graph.add_node("intent_recognition", self.recognize_intent)
        graph.add_node("knowledge_retrieval", self.knowledge_retrieval)
        graph.add_node("grade_documents", self.grade_documents)
        graph.add_node("web_search", self.web_search)
        graph.add_node("generate_answer", self.generate_answer)
        graph.add_node("save_conversation", self.save_conversation)
        
        # Set entry point
        graph.set_entry_point("load_employee_config")
        
        # Add edges
        graph.add_edge("load_employee_config", "load_session_context")
        graph.add_edge("load_session_context", "input_validation")
        graph.add_edge("input_validation", "match_faq")
        
        # Conditional: FAQ matched -> generate answer, else -> check realtime
        graph.add_conditional_edges(
            "match_faq",
            lambda state: "generate" if state.get("faq_matched") else "check_realtime",
            {
                "generate": "generate_answer",
                "check_realtime": "check_realtime_query"
            }
        )
        
        # Conditional: realtime query -> web search, else -> intent recognition
        graph.add_conditional_edges(
            "check_realtime_query",
            lambda state: "web_search" if state.get("is_realtime_query") else "intent_recognition",
            {
                "web_search": "web_search",
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
        
        # Conditional: low relevance -> web search, else -> generate answer
        graph.add_conditional_edges(
            "grade_documents",
            lambda state: "web_search" if state.get("relevance_score", 0) < settings.relevance_threshold else "generate_answer",
            {
                "web_search": "web_search",
                "generate_answer": "generate_answer"
            }
        )
        
        graph.add_edge("web_search", "generate_answer")
        graph.add_edge("generate_answer", "save_conversation")
        graph.add_edge("save_conversation", END)
        compiled_stateGraph = graph.compile()
        self._dump_graph_debug(compiled_stateGraph)
        return compiled_stateGraph
        
    def _dump_graph_debug(self, compiled_stateGraph) -> None:
        """保存图结构用于调试"""
        dump_flag = os.getenv("CRAG_DUMP_GRAPH", "1").lower()
        if dump_flag in {"0", "false", "no"}:
            return

        use_remote = os.getenv("CRAG_RENDER_REMOTE", "0").lower() not in {"0", "false", "no"}

        try:
            graph_view = compiled_stateGraph.get_graph(xray=True)
            output_dir = Path(os.getenv("CRAG_GRAPH_DIR", "./graph_debug"))
            output_dir.mkdir(parents=True, exist_ok=True)

            # ✅ 正确的 mermaid 提取方式
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
                print(f"✓ Mermaid source saved at {mermaid_path}")
            else:
                # 回退：保存 repr 以便调试
                repr_text = repr(graph_view)
                (output_dir / "crag_graph_view_repr.txt").write_text(repr_text, encoding="utf-8")
                print("⚠️ Mermaid source unavailable, saved repr to crag_graph_view_repr.txt")

            # 尝试远程渲染（如果启用）
            if use_remote and mermaid_src:
                try:
                    from langgraph.graph.graph import MermaidDrawMethod
                    png_bytes = graph_view.draw_mermaid_png(
                        draw_method=MermaidDrawMethod.API,  # 使用远程 API
                        max_retries=3,
                        retry_delay=1.0
                    )
                    (output_dir / "crag_graph.png").write_bytes(png_bytes)
                    print(f"✓ Graph PNG rendered at {output_dir / 'crag_graph.png'}")
                except Exception as remote_exc:
                    logger.warning(f"Remote PNG rendering failed: {remote_exc}")
                    (output_dir / "crag_graph_render_error.txt").write_text(
                        str(remote_exc), encoding="utf-8"
                    )
                    print("⚠️ Remote rendering failed (see crag_graph_render_error.txt)")
                    print("💡 Use local rendering: Set CRAG_RENDER_REMOTE=0 or install pyppeteer")

            # 本地渲染建议（如果远程失败）
            if not use_remote and mermaid_src:
                print("ℹ️ Mermaid source available at {mermaid_path}")
                print("💡 To render locally:")
                print("   1. Install mermaid-cli: npm install -g @mermaid-js/mermaid-cli")
                print("   2. Run: mmdc -i {mermaid_path} -o {output_dir / 'crag_graph.png'}")
                print("   OR set CRAG_RENDER_REMOTE=1 to use remote API")

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
            print(f"⚠️ Unable to dump graph: {exc}")

    async def load_employee_config(self, state: ConversationState) -> ConversationState:
        """Load employee configuration (enhanced with full config)."""
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
            return state
            
        except Exception as e:
            logger.error(f"Failed to load session context: error={str(e)}", exc_info=True)
            state["context"] = {"messages": [], "message_count": 0}
            return state
    
    async def validate_input(self, state: ConversationState) -> ConversationState:
        """Validate input."""
        # Simple validation - already done at API level
        state["has_sensitive"] = False
        return state
    
    async def match_faq(self, state: ConversationState) -> ConversationState:
        """
        Match FAQ using hybrid search (vector + keyword + RRF fusion).
        
        如果FAQ的RRF分数 >= faq_sim_threshold，直接返回FAQ答案（随机选择），
        跳过后续的RAG检索和LLM生成。
        """
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
            
            logger.info(f"FAQ answer selected: faq_id={faq_id}, question={faq_doc.get('question_name')[:50]}, answer_length={len(selected_answer)}")
            
        except Exception as e:
            logger.error(f"FAQ matching failed: error={str(e)}", exc_info=True)
            state["faq_matched"] = None
        
        return state
    
    async def check_realtime_query(self, state: ConversationState) -> ConversationState:
        """Check if query needs realtime information."""
        if not settings.realtime_query_enabled:
            state["is_realtime_query"] = False
            return state
        
        query = state["user_query"].lower()
        
        # Realtime keywords
        realtime_keywords = {
            "time": ["今天", "明天", "昨天", "最近", "现在", "本周", "本月", "当前"],
            "weather": ["天气", "气温", "降雨", "降水", "温度"],
            "news": ["新闻", "热点", "最新", "资讯", "动态", "头条"],
            "market": ["股价", "汇率", "行情", "股市"],
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
        try:
            # Get KB IDs from employee config
            kb_ids = state["employee_config"].get("capabilities", {}).get("kb_ids", [])
            
            # Search using RAG
            results = await rag_retrieval.search(
                query=state["user_query"],
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
        docs = state.get("retrieved_docs", [])
        
        if not docs:
            state["relevance_score"] = 0.0
            return state
        
        # Use the top document's score as relevance score
        state["relevance_score"] = docs[0].get("rrf_score", 0.0) if docs else 0.0
        
        logger.info(f"Document grading completed: relevance_score={state['relevance_score']}")
        return state
    
    async def web_search(self, state: ConversationState) -> ConversationState:
        """Perform web search using Tavily."""
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
    
    async def generate_answer(self, state: ConversationState) -> ConversationState:
        """Prepare for answer generation (placeholder for streaming)."""
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

    async def save_conversation(self, state: ConversationState) -> ConversationState:
        """Save conversation to database."""
        try:
            db = await get_database()
            
            # Generate conversation ID
            import hashlib
            session_id = state["session_id"]
            timestamp = datetime.now().timestamp()
            conv_id = f"conv_{hashlib.md5(f'{session_id}_{timestamp}'.encode()).hexdigest()[:12]}"
            state["conversation_id"] = conv_id
            
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
                response_time_ms=state.get("response_time_ms", 0)
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
            
        except Exception as e:
            logger.error(f"Failed to save conversation: error={str(e)}", exc_info=True)
        
        return state

# Global workflow instance
conversation_workflow = ConversationWorkflow()
