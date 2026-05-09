"""
LangGraph-based Conversation Workflow for Digital Employee.

This module implements a state-based conversation workflow using LangGraph, supporting:
- Multi-source knowledge retrieval (RAG + FAQ + Web Search)
- LLM-based query classification (QueryClassifier)
- Dual-LLM architecture (local Ollama + remote OpenAI-style API)
- Streaming responses with performance monitoring

Workflow Graph (8 nodes):
    load_employee_config → load_session_context → input_validation
        → classify_query_type
        → [conditional: greeting/noise?]   → generate_answer
        → [conditional: realtime?]          → web_search → generate_answer
        → [conditional: math?]              → generate_answer (Phi-4)
        → [conditional: rag (concept)?]     → evaluate_complexity → generate_answer (RAGAnything)
        → [conditional: general?]           → generate_answer (通用 LLM，不走 RAG)
        → save_conversation → END

意图 → 回答路径的映射统一维护在
``app/services/conversation/intent_routing.py``（INTENT_TO_ANSWER_MODE）。
新增分类标签或调整路由策略只改那张表，不需要改工作流。

Note: Simplified workflow using RAGAnything for RAG retrieval.
Removed nodes: intent_recognition, knowledge_retrieval, grade_documents,
               compress_context, match_faq, rewrite_query, check_math_problem
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
from app.utils.get_vllm_first_model import get_vllm_first_model

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
    - LLM-based query classification: QueryClassifier with 9 categories
    - Simplified workflow: 8 nodes

    Workflow consists of 8 nodes connected by conditional edges.

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

        Determinism override（事实性人事查询）:
            「现任 X 职务是谁」等需要广博/最新世界知识的查询若沿用调用方传入的
            ``temperature=0.7`` 默认值，会让 DeepSeek-V3 在「李强」与「李克强」
            之间随机偏移（实测 10 次约 20% 错答）。这里检测到此类 query 时强制
            把采样参数压回 ``temperature=0 / top_p=1``，让答案完全由模型权重决定，
            消除随机性带来的"一会对一会错"。注：复杂度评估、人格、其它生成路径
            不受影响——仅当 ``_needs_big_world_knowledge`` 命中时才覆盖。

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

        # 「现任 X 职务是谁」类事实查询的"确定性兜底"。
        # 复用 select_llm 已有的 _needs_big_world_knowledge 启发式，避免在两处分别
        # 维护词表；命中后强制压低采样随机性，确保 DeepSeek 等大模型给出稳定答案。
        from app.services.conversation.conversation_helpers import _needs_big_world_knowledge
        big_world_query = (state.get("rewritten_query") or state.get("user_query") or "").strip()
        if _needs_big_world_knowledge(big_world_query):
            temperature = 0.0
            top_p = 1.0
            logger.info(
                f"Force deterministic LLM (temperature=0, top_p=1) for big-world-knowledge query: "
                f"query={big_world_query[:80]!r}"
            )

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

    def get_phi4_streaming_llm(self, state: ConversationState):
        """
        动态创建 Phi-4 流式 LLM（不存储为实例变量）。

        根据环境配置创建 Phi-4 LLM 实例用于数学问题解答。

        Args:
            state: Current conversation state

        Returns:
            Tuple of (llm, model_name) for Phi-4 streaming

        Notes:
            - 读取环境变量而非从 state
            - vLLM 不需要真实 API key
        """
        # 检查是否启用
        enabled = os.getenv("PHI4_ENABLED", "true").lower() == "true"
        if not enabled:
            logger.info("Phi-4 disabled, falling back to default LLM")
            return self.get_streaming_llm(state)

        # 读取配置
        base_url = os.getenv("PHI4_BASE_URL", "http://192.168.8.235:8000/v1")
        model_id = get_vllm_first_model(base_url)
        # temperature = float(os.getenv("PHI4_TEMPERATURE", "0.0"))
        # max_tokens = int(os.getenv("PHI4_MAX_TOKENS", "16384"))
        max_tokens = 1024 * 3
        temperature = 0.6

        # 动态创建 ChatOpenAI 实例
        phi4_llm = ChatOpenAI(
            base_url=base_url,
            api_key="dummy-key",  # vLLM 不需要真实 key
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            top_p=0.95,
        )

        logger.info(
            f"Phi-4 LLM created | model={model_id} | base_url={base_url} | "
            f"temperature={temperature} | max_tokens={max_tokens}"
        )

        return phi4_llm, model_id

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
        - Middle: classify_query_type with conditional routing
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

        # Conditional routing after query classification.
        # path_map 的 key 必须与 ``intent_routing.ROUTE_BRANCH_*`` 一一对应，
        # 任何新增的分支都需要在这里登记，否则 LangGraph 会抛 KeyError。
        from app.services.conversation.intent_routing import (
            ROUTE_BRANCH_GENERAL,
            ROUTE_BRANCH_GREETING,
            ROUTE_BRANCH_MATH,
            ROUTE_BRANCH_RAG,
            ROUTE_BRANCH_REALTIME,
        )
        graph.add_conditional_edges(
            "classify_query_type",
            self.nodes.route_after_classification,
            {
                ROUTE_BRANCH_GREETING: "generate_answer",   # Greeting / noise → 直接回答
                ROUTE_BRANCH_REALTIME: "web_search",        # 实时类 → 联网检索
                ROUTE_BRANCH_MATH: "generate_answer",       # 数学题 → Phi-4 直接回答
                ROUTE_BRANCH_RAG: "evaluate_complexity",    # 概念/教材类 → 复杂度评估 → RAG
                ROUTE_BRANCH_GENERAL: "generate_answer",    # 通用 LLM（英语/常识/闲聊）→ 直接回答
            }
        )

        # Concept / textbook flow: complexity → generate_answer (RAGAnything)
        graph.add_edge("evaluate_complexity", "generate_answer")

        # Final sequence
        graph.add_edge("web_search", "generate_answer")
        graph.add_edge("generate_answer", "save_conversation")
        graph.add_edge("save_conversation", END)

        # Compile and export graph for debugging
        compiled_graph = graph.compile()
        # self._dump_graph_debug(compiled_graph)
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
