"""
LangGraph-based Conversation Workflow for Digital Employee.

This module implements a state-based conversation workflow using LangGraph, supporting:
- Multi-source knowledge retrieval (RAG + FAQ + Web Search)
- Query optimization (rewriting, compression)
- Dual-LLM architecture (local Ollama + remote OpenAI-style API)
- Streaming responses with performance monitoring

Workflow Graph (10 nodes):
    load_employee_config → load_session_context → input_validation
        → classify_query_type
        → [conditional: greeting?] → generate_answer
        → [conditional: realtime?] → web_search → generate_answer
        → [conditional: normal?] → evaluate_complexity → generate_answer
        → generate_answer → save_conversation → END

Note: Simplified workflow using RAGAnything for RAG retrieval.
Removed nodes: intent_recognition, knowledge_retrieval, grade_documents,
               compress_context, match_faq, rewrite_query
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
    - RAGAnything integration: Knowledge graph + vector retrieval with streaming
    - Realtime query detection: Auto-route to web search for time-sensitive queries
    - Simplified workflow: 10 nodes (down from 17)

    Workflow consists of 10 nodes connected by conditional edges.

    LLM Routing Strategy (hybrid mode):
    - Use local Ollama for: greetings, simple queries (<30 chars), early turns
    - Use remote API for: RAGAnything queries, web search, long context, complex queries
    """

    def __init__(self):
        """Initialize workflow with dual LLM instances for hybrid routing."""
        # Initialize local LLM (Ollama) - for fast, simple responses
        self.local_llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0,
            streaming=True,
            keep_alive=-1
        )
        # 打印 local_llm 配置
        logger.info(
            f"Local LLM configured | base_url={self.local_llm.base_url} | model={self.local_llm.model} | "
            f"temperature={self.local_llm.temperature} | keep_alive={self.local_llm.keep_alive}"
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

        # Set default LLM based on routing mode
        routing_mode = getattr(settings, 'llm_routing_mode', 'local_only')
        if routing_mode == 'local_only':
            self.llm = self.local_llm
            logger.info("LLM routing mode: local_only - using Ollama only")
        elif routing_mode == 'remote_only':
            self.llm = self.remote_llm
            logger.info("LLM routing mode: remote_only - using remote API only")
        else:  # hybrid mode - will select dynamically per request
            self.llm = self.local_llm  # default to local
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
            Tuple of (llm, model_name)
        """
        from app.services.conversation.conversation_helpers import select_llm
        return select_llm(
            state,
            self.local_llm,
            self.remote_llm
        )

    def get_streaming_llm(self, state: ConversationState):
        """
        Get the appropriate streaming LLM for answer generation.

        This is used by the chat endpoint for token-level streaming.
        Returns the LLM instance and model name based on routing strategy.

        Supports dynamic LLM parameters from the request state:
        - llm_temperature: Sampling temperature
        - llm_top_p: Nucleus sampling parameter
        - llm_max_tokens: Maximum tokens to generate
        - llm_presence_penalty: Presence penalty
        - llm_frequency_penalty: Frequency penalty

        Args:
            state: Current conversation state

        Returns:
            Tuple of (llm, model_name) for streaming
        """
        llm, model_name = self.get_active_llm(state)

        # Check if custom LLM parameters are provided in the state
        temperature = state.get("llm_temperature")
        top_p = state.get("llm_top_p")
        max_tokens = state.get("llm_max_tokens")

        # If custom parameters are provided, create a new LLM instance with them
        if temperature is not None or top_p is not None or max_tokens is not None:
            # Import LLM classes
            from langchain_community.chat_models import ChatOllama
            from langchain_openai import ChatOpenAI

            # Determine which LLM type to use based on the current llm instance
            if isinstance(llm, ChatOllama):
                # Create new Ollama LLM with custom parameters
                base_url = getattr(llm, 'base_url', 'http://localhost:11434')
                model_name = getattr(llm, 'model_name', None) or getattr(llm, 'model', '')
                llm = ChatOllama(
                    base_url=base_url,
                    model=model_name,
                    temperature=temperature if temperature is not None else getattr(llm, 'temperature', 0.7),
                    top_p=top_p if top_p is not None else getattr(llm, 'top_p', None),
                    num_predict=max_tokens if max_tokens is not None else getattr(llm, 'num_predict', None),
                )
                logger.info(
                    "Created custom Ollama LLM for streaming",
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )
            elif isinstance(llm, ChatOpenAI):
                # Create new OpenAI LLM with custom parameters
                # ChatOpenAI uses openai_api_base for base URL in some versions
                base_url = getattr(llm, 'openai_api_base', 'https://api.openai.com/v1') or getattr(llm, 'base_url', 'https://api.openai.com/v1')
                api_key = getattr(llm, 'openai_api_key', '') or getattr(llm, 'api_key', '')
                model_name = getattr(llm, 'model_name', None) or getattr(llm, 'model', '')
                llm = ChatOpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    model=model_name,
                    temperature=temperature if temperature is not None else getattr(llm, 'temperature', 0.7),
                    max_tokens=max_tokens if max_tokens is not None else getattr(llm, 'max_tokens', None),
                )
                logger.info(
                    "Created custom OpenAI LLM for streaming",
                    temperature=temperature,
                    max_tokens=max_tokens
                )

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
        - Middle: 8 processing nodes with conditional routing
        - Exit: save_conversation → END

        Simplified workflow using RAGAnything for RAG retrieval.

        Returns:
            Compiled StateGraph ready for execution
        """
        graph = StateGraph(ConversationState)

        # Add all workflow nodes (delegated to nodes container)
        graph.add_node("load_employee_config", self.nodes.load_employee_config)
        graph.add_node("load_session_context", self.nodes.load_session_context)
        graph.add_node("input_validation", self.nodes.validate_input)
        graph.add_node("classify_query_type", self.nodes.classify_query_type)
        graph.add_node("evaluate_complexity", self.nodes.evaluate_complexity)
        graph.add_node("web_search", self.nodes.web_search)
        graph.add_node("generate_answer", self.nodes.generate_answer)
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

        # Normal flow: complexity → generate_answer (RAGAnything handles RAG)
        graph.add_edge("evaluate_complexity", "generate_answer")

        # Final sequence
        graph.add_edge("web_search", "generate_answer")
        graph.add_edge("generate_answer", "save_conversation")
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
            logger.exception("Graph debug dump failed")
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
