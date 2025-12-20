"""
LangGraph conversation workflow.
"""
import os
from pathlib import Path
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from operator import add
from datetime import datetime

# Load environment variables before importing settings
from dotenv import load_dotenv
load_dotenv()

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
            logger.info("Using Ollama LLM", model=settings.ollama_model)
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
            logger.info("Using OpenAI-style LLM", model=settings.openai_model)
        
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
        
        graph.add_edge("intent_recognition", "knowledge_retrieval")
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
                print(f"⚠️ Mermaid source unavailable, saved repr to crag_graph_view_repr.txt")

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
                    print(f"⚠️ Remote rendering failed (see crag_graph_render_error.txt)")
                    print(f"💡 Use local rendering: Set CRAG_RENDER_REMOTE=0 or install pyppeteer")

            # 本地渲染建议（如果远程失败）
            if not use_remote and mermaid_src:
                print(f"ℹ️ Mermaid source available at {mermaid_path}")
                print(f"💡 To render locally:")
                print(f"   1. Install mermaid-cli: npm install -g @mermaid-js/mermaid-cli")
                print(f"   2. Run: mmdc -i {mermaid_path} -o {output_dir / 'crag_graph.png'}")
                print(f"   OR set CRAG_RENDER_REMOTE=1 to use remote API")

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
        try:
            db = await get_database()
            employee = await db.employee_configs.find_one({"employee_id": state["employee_id"]})
            
            if not employee:
                logger.warning("Employee config not found, using defaults", employee_id=state["employee_id"])
                # Default configuration
                state["employee_config"] = {
                    "name": "AI助手",
                    "role": "通用助理",
                    "description": "专业的AI助手",
                    "personality": {
                        "tone": "professional",
                        "style": "friendly",
                        "language": "zh-CN",
                        "formality": "moderate"
                    },
                    "capabilities": {
                        "kb_ids": [],
                        "web_search_enabled": True,
                        "max_context_turns": 10
                    },
                    "greeting": "您好，我是AI助手，很高兴为您服务。",
                    "faqs": []
                }
            else:
                employee.pop("_id", None)
                state["employee_config"] = employee
            
            logger.info(
                "Employee config loaded",
                employee_id=state["employee_id"],
                kb_count=len(state["employee_config"].get("capabilities", {}).get("kb_ids", [])),
                faq_count=len(state["employee_config"].get("faqs", []))
            )
            return state
            
        except Exception as e:
            logger.error("Failed to load employee config", error=str(e), exc_info=True)
            state["employee_config"] = {}
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
            
            logger.info("Loaded session context", session_id=state["session_id"])
            return state
            
        except Exception as e:
            logger.error("Failed to load session context", error=str(e), exc_info=True)
            state["context"] = {"messages": [], "message_count": 0}
            return state
    
    async def validate_input(self, state: ConversationState) -> ConversationState:
        """Validate input."""
        # Simple validation - already done at API level
        state["has_sensitive"] = False
        return state
    
    async def match_faq(self, state: ConversationState) -> ConversationState:
        """Match FAQ using keyword and similarity matching."""
        try:
            query = state["user_query"]
            faqs = state["employee_config"].get("faqs", [])
            
            if not faqs:
                logger.info("No FAQs configured, skipping FAQ matching")
                state["faq_matched"] = None
                return state
            
            # Strategy 1: Keyword matching (fast filter)
            keyword_matches = []
            for faq in faqs:
                keywords = faq.get("keywords", [])
                question = faq.get("question", "")
                
                # Check if query contains FAQ keywords
                if any(kw in query for kw in keywords):
                    keyword_matches.append(faq)
                # Or query is very similar to FAQ question
                elif query in question or question in query:
                    keyword_matches.append(faq)
            
            # If matched, use the first match
            if keyword_matches:
                best_faq = keyword_matches[0]
                state["final_answer"] = best_faq["answer"]
                state["confidence"] = 0.95
                state["intent"] = "faq_match"
                state["faq_matched"] = {
                    "faq_id": best_faq.get("faq_id"),
                    "question": best_faq.get("question"),
                    "category": best_faq.get("category")
                }
                
                logger.info(
                    "FAQ matched",
                    faq_id=best_faq.get("faq_id"),
                    question=best_faq.get("question")
                )
            else:
                state["faq_matched"] = None
            
        except Exception as e:
            logger.error("FAQ matching failed", error=str(e), exc_info=True)
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
                    state["realtime_detect_reason"] = f"keyword:{keyword}"
                    logger.info(
                        "Realtime query detected",
                        category=category,
                        keyword=keyword
                    )
                    return state
        
        state["is_realtime_query"] = False
        return state
    
    async def recognize_intent(self, state: ConversationState) -> ConversationState:
        """Recognize user intent (simplified)."""
        # Simplified intent recognition - just mark as general_query
        state["intent"] = "general_query"
        state["entities"] = {}
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
            
            logger.info(
                "Knowledge retrieval completed",
                results_count=len(results),
                kb_used=state["kb_used"]
            )
            
        except Exception as e:
            logger.error("Knowledge retrieval failed", error=str(e), exc_info=True)
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
        
        logger.info("Document grading completed", relevance_score=state["relevance_score"])
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
                logger.info(
                    "Web search disabled for employee",
                    employee_id=state.get("employee_id")
                )
                state["web_search_results"] = []
                state["web_search_used"] = False
                return state
            
            query = state["user_query"]
            
            # Perform web search
            logger.info(
                "Performing web search",
                query=query[:100],
                is_realtime=state.get("is_realtime_query", False),
                realtime_category=state.get("realtime_category")
            )
            
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
            
            logger.info(
                "Web search completed",
                results_count=len(formatted_results),
                has_results=state["web_search_used"]
            )
            
        except Exception as e:
            logger.error(
                "Web search failed",
                error=str(e),
                query=state.get("user_query", "")[:100],
                exc_info=True
            )
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
        
        logger.info(
            "Ready for answer generation",
            confidence=confidence,
            web_search_used=state.get("web_search_used", False),
            kb_docs_count=len(state.get("retrieved_docs", []))
        )
        
        return state
    
    def build_generation_messages(self, state: ConversationState) -> List:
        """Build messages for LLM generation (used by streaming methods)."""
        employee_config = state.get("employee_config", {})
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
7. 对于实时性问题（天气、新闻等），优先使用网络资料

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
            
            logger.info("Conversation saved", conversation_id=conv_id)
            
        except Exception as e:
            logger.error("Failed to save conversation", error=str(e), exc_info=True)
        
        return state
    
    async def run(self, state: ConversationState) -> ConversationState:
        """Run the workflow."""
        start_time = datetime.now()
        
        try:
            # Execute the workflow， get the final state
            result = await self.workflow.ainvoke(state)
            
            # For non-streaming mode, actually generate the answer
            if not result.get("final_answer") and not result.get("error"):
                messages = self.build_generation_messages(result)
                response = await self.llm.ainvoke(messages)
                result["final_answer"] = response.content
                
                logger.info(
                    "Answer generated (non-streaming)",
                    answer_length=len(result["final_answer"]),
                    confidence=result.get("confidence", 0.0)
                )
            
            # Calculate response time
            end_time = datetime.now()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)
            result["response_time_ms"] = response_time_ms
            
            return result
            
        except Exception as e:
            logger.error("Workflow execution failed", error=str(e), exc_info=True)
            raise


# Global workflow instance
conversation_workflow = ConversationWorkflow()
