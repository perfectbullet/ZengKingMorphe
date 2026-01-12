"""
LangGraph-based Conversation Workflow for Digital Employee.

This module implements a state-based conversation workflow using LangGraph, supporting:
- Multi-source knowledge retrieval (RAG + FAQ + Web Search)
- Query optimization (rewriting, compression)
- Dual-LLM architecture (local Ollama + remote OpenAI-style API)
- Streaming responses with performance monitoring

Workflow Graph (17 nodes):
    load_employee_config → load_session_context → input_validation
        → classify_query_type
        → [conditional: greeting?] → generate_answer
        → [conditional: realtime?] → web_search
        → [conditional: normal?] → evaluate_complexity → rewrite_query → match_faq
        → [conditional: FAQ matched?] → generate_answer OR intent_recognition
        → knowledge_retrieval → grade_documents → rerank_documents → compress_context
        → [conditional: low relevance?] → web_search OR generate_answer
        → generate_answer → verify_answer → save_conversation → END
"""
import os
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from app.core.config import settings
from app.core.logging import get_logger
from app.services.conversation.conversation_state import ConversationState
from app.services.conversation.conversation_nodes import ConversationNodes

logger = get_logger(__name__)


# =============================================================================
# Conversation Workflow Class
# =============================================================================
class ConversationWorkflow:
    """
    LangGraph-based conversation workflow for digital employee interactions.

    Features:
    - Dual-LLM support: Local Ollama for fast responses, remote API for complex tasks
    - Hybrid routing: Automatically select LLM based on query complexity
    - Hybrid retrieval: Vector search + keyword search + RRF fusion
    - FAQ fast-path: Direct answer matching with configurable threshold
    - Realtime query detection: Auto-route to web search for time-sensitive queries
    - Query optimization: Rewriting, context compression, document reranking
    - Answer verification: Consistency checking against source documents

    Workflow consists of 17 nodes connected by conditional edges.

    LLM Routing Strategy (hybrid mode):
    - Use local Ollama for: greetings, FAQs, simple queries (<30 chars), early turns
    - Use remote API for: RAG retrieval, web search, long context, complex queries
    """

    def __init__(self):
        """Initialize workflow with dual LLM instances for hybrid routing."""
        # Initialize local LLM (Ollama) - for fast, simple responses
        logger.info(
            "Initializing local Ollama LLM",
            model=settings.ollama_model,
            base_url=settings.ollama_base_url
        )
        self.local_llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0,
            streaming=True,
            keep_alive=-1
        )
        self.local_grader_llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_grader_model,
            temperature=0,
            format="json",
            keep_alive=-1
        )

        # Initialize remote LLM (OpenAI-style API) - for complex, accurate responses
        logger.info(
            "Initializing remote OpenAI-style LLM",
            model=settings.openai_model,
            base_url=settings.openai_api_base
        )
        self.remote_llm = ChatOpenAI(
            base_url=settings.openai_api_base,
            api_key=settings.siliconflow_api_key,
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            streaming=True,
        )
        self.remote_grader_llm = ChatOpenAI(
            base_url=settings.openai_api_base,
            api_key=settings.siliconflow_api_key,
            model=settings.openai_grader_model,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        # Set default LLM based on routing mode
        routing_mode = getattr(settings, 'llm_routing_mode', 'local_only')
        if routing_mode == 'local_only':
            self.llm = self.local_llm
            self.grader_llm = self.local_grader_llm
            logger.info("LLM routing mode: local_only - using Ollama only")
        elif routing_mode == 'remote_only':
            self.llm = self.remote_llm
            self.grader_llm = self.remote_grader_llm
            logger.info("LLM routing mode: remote_only - using remote API only")
        else:  # hybrid mode - will select dynamically per request
            self.llm = self.local_llm  # default to local
            self.grader_llm = self.local_grader_llm
            logger.info("LLM routing mode: hybrid - will select dynamically")

        # Initialize nodes container
        self.nodes = ConversationNodes(self)

        # Build the workflow graph
        self.workflow = self._build_workflow()

    # -------------------------------------------------------------------------
    # LLM Selection Methods (delegated to helpers, but exposed here)
    # -------------------------------------------------------------------------
    def get_active_llm(self, state: ConversationState):
        """
        Get the appropriate LLM for the current state.

        This method should be called before LLM invocations to ensure
        the correct model is used based on routing strategy.

        Args:
            state: Current conversation state

        Returns:
            Tuple of (llm, grader_llm, model_name)
        """
        from app.services.conversation.conversation_helpers import select_llm
        return select_llm(
            state,
            self.local_llm,
            self.local_grader_llm,
            self.remote_llm,
            self.remote_grader_llm
        )

    def get_streaming_llm(self, state: ConversationState):
        """
        Get the appropriate streaming LLM for answer generation.

        This is used by the chat endpoint for token-level streaming.
        Returns the LLM instance and model name based on routing strategy.

        Args:
            state: Current conversation state

        Returns:
            Tuple of (llm, model_name) for streaming
        """
        llm, _, model_name = self.get_active_llm(state)
        return llm, model_name

    # -------------------------------------------------------------------------
    # Message Building Methods (delegated to helpers)
    # -------------------------------------------------------------------------
    def build_generation_messages(self, state: ConversationState):
        """
        Build LLM messages for answer generation.

        Args:
            state: Current conversation state

        Returns:
            List of Message objects for LLM
        """
        from app.services.conversation.conversation_helpers import build_generation_messages
        return build_generation_messages(state)

    async def save_conversation(self, state: ConversationState):
        """
        Save conversation record (delegates to nodes container).

        Args:
            state: Current conversation state

        Returns:
            Updated state with conversation_id populated
        """
        return await self.nodes.save_conversation(state)

    # -------------------------------------------------------------------------
    # Workflow Graph Building
    # -------------------------------------------------------------------------
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow with all nodes and conditional edges.

        Graph structure:
        - Entry: load_employee_config
        - Middle: 17 processing nodes with conditional routing
        - Exit: save_conversation → END

        Returns:
            Compiled StateGraph ready for execution
        """
        graph = StateGraph(ConversationState)

        # Add all 17 workflow nodes (delegated to nodes container)
        graph.add_node("load_employee_config", self.nodes.load_employee_config)
        graph.add_node("load_session_context", self.nodes.load_session_context)
        graph.add_node("input_validation", self.nodes.validate_input)
        graph.add_node("classify_query_type", self.nodes.classify_query_type)
        graph.add_node("evaluate_complexity", self.nodes.evaluate_complexity)
        graph.add_node("rewrite_query", self.nodes.rewrite_query)
        graph.add_node("match_faq", self.nodes.match_faq)
        graph.add_node("intent_recognition", self.nodes.recognize_intent)
        graph.add_node("knowledge_retrieval", self.nodes.knowledge_retrieval)
        graph.add_node("grade_documents", self.nodes.grade_documents)
        graph.add_node("rerank_documents", self.nodes.rerank_documents)
        graph.add_node("compress_context", self.nodes.compress_context)
        graph.add_node("web_search", self.nodes.web_search)
        graph.add_node("generate_answer", self.nodes.generate_answer)
        graph.add_node("verify_answer", self.nodes.verify_answer)
        graph.add_node("save_conversation", self.nodes.save_conversation)

        # Set entry point
        graph.set_entry_point("load_employee_config")

        # Define sequential edges
        graph.add_edge("load_employee_config", "load_session_context")
        graph.add_edge("load_session_context", "input_validation")
        graph.add_edge("input_validation", "classify_query_type")

        # Conditional routing after query classification
        graph.add_conditional_edges(
            "classify_query_type",
            self.nodes.route_after_classification,
            {
                "greeting": "generate_answer",      # Greeting → direct to answer
                "realtime": "web_search",           # Realtime query → web search
                "normal": "evaluate_complexity"     # Normal query → complexity eval
            }
        )

        # Normal flow: complexity → rewrite → FAQ
        graph.add_edge("evaluate_complexity", "rewrite_query")
        graph.add_edge("rewrite_query", "match_faq")

        # Conditional routing after FAQ matching
        graph.add_conditional_edges(
            "match_faq",
            lambda state: "generate_answer" if state.get("faq_matched") else "intent_recognition",
            {
                "generate_answer": "generate_answer",
                "intent_recognition": "intent_recognition"
            }
        )

        # Intent recognition now mainly handles general_query routing
        graph.add_edge("intent_recognition", "knowledge_retrieval")

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


# =============================================================================
# Global Workflow Instance
# =============================================================================
conversation_workflow = ConversationWorkflow()
