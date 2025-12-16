"""
LangGraph conversation workflow.
"""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from operator import add
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
                api_key=settings.openai_api_key or settings.siliconflow_api_key or "",
                model=settings.openai_model,
                temperature=settings.openai_temperature,
                streaming=True,
            )
            self.grader_llm = ChatOpenAI(
                base_url=settings.openai_api_base,
                api_key=settings.openai_api_key or settings.siliconflow_api_key or "",
                model=settings.openai_grader_model,
                temperature=0,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            logger.info("Using OpenAI-style LLM", model=settings.openai_model)
        
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the conversation workflow graph."""
        graph = StateGraph(ConversationState)
        
        # Add nodes
        graph.add_node("load_employee_config", self.load_employee_config)
        graph.add_node("load_session_context", self.load_session_context)
        graph.add_node("input_validation", self.validate_input)
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
        graph.add_edge("input_validation", "check_realtime_query")
        
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
        
        return graph.compile()
    
    async def load_employee_config(self, state: ConversationState) -> ConversationState:
        """Load employee configuration."""
        try:
            db = await get_database()
            employee = await db.employee_configs.find_one({"employee_id": state["employee_id"]})
            
            if not employee:
                logger.warning("Employee config not found", employee_id=state["employee_id"])
                state["employee_config"] = {}
            else:
                employee.pop("_id", None)
                state["employee_config"] = employee
            
            logger.info("Loaded employee config", employee_id=state["employee_id"])
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
                    {"$set": {"last_activity": datetime.utcnow()}}
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
        """Perform web search (placeholder)."""
        # TODO: Implement Tavily integration in Phase 3
        state["web_search_results"] = []
        state["web_search_used"] = False
        
        logger.info("Web search skipped (not implemented yet)")
        return state
    
    async def generate_answer(self, state: ConversationState) -> ConversationState:
        """Generate answer using LLM."""
        try:
            # Build context from retrieved docs
            context_parts = []
            for i, doc in enumerate(state.get("retrieved_docs", [])[:3], 1):
                context_parts.append(f"参考{i}: {doc.get('content', '')[:500]}")
            context_text = "\n\n".join(context_parts) if context_parts else "无相关参考资料"
            
            # Build messages
            messages = []
            
            # System message with employee persona
            employee_config = state.get("employee_config", {})
            personality = employee_config.get("personality", {})
            
            system_msg = f"""你是{employee_config.get('name', '助手')}，{employee_config.get('role', '智能助手')}。
{employee_config.get('description', '')}
请用{personality.get('tone', 'professional')}的语气，{personality.get('style', 'friendly')}的风格回答用户问题。"""
            
            messages.append(SystemMessage(content=system_msg))
            
            # Add conversation history
            for msg in state.get("context", {}).get("messages", [])[-5:]:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))
            
            # Current query with context
            user_msg = f"""用户问题：{state['user_query']}

参考知识：
{context_text}

请基于上述参考知识回答用户问题。如果参考知识不足以回答问题，请诚实说明。"""
            
            messages.append(HumanMessage(content=user_msg))
            
            # Generate response
            response = await self.llm.ainvoke(messages)
            state["final_answer"] = response.content
            state["confidence"] = 0.8  # Placeholder
            
            logger.info("Answer generated", answer_length=len(state["final_answer"]))
            
        except Exception as e:
            logger.error("Answer generation failed", error=str(e), exc_info=True)
            state["final_answer"] = "抱歉，我暂时无法回答这个问题。请稍后再试。"
            state["confidence"] = 0.0
        
        return state
    
    async def save_conversation(self, state: ConversationState) -> ConversationState:
        """Save conversation to database."""
        try:
            db = await get_database()
            
            # Generate conversation ID
            import hashlib
            conv_id = f"conv_{hashlib.md5(f'{state['session_id']}_{datetime.utcnow().timestamp()}'.encode()).hexdigest()[:12]}"
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
        start_time = datetime.utcnow()
        
        try:
            result = await self.workflow.ainvoke(state)
            
            # Calculate response time
            end_time = datetime.utcnow()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)
            result["response_time_ms"] = response_time_ms
            
            return result
            
        except Exception as e:
            logger.error("Workflow execution failed", error=str(e), exc_info=True)
            raise


# Global workflow instance
conversation_workflow = ConversationWorkflow()
