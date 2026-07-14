"""
基于 LangGraph 的数字员工对话工作流。

本模块实现基于状态机的对话工作流，支持：
- 多源知识检索（RAG + FAQ + Web Search）
- 基于 LLM 的查询分类（QueryClassifier）
- 双 LLM 架构（本地 Ollama + 远程 OpenAI 兼容 API）
- 流式响应与性能监控

工作流图（9 节点）：
    load_employee_config → load_session_context → input_validation
        → preprocess_query → classify_query_type → resolve_context_query → finalize_classification → post_classification_preprocess
        → [条件分支: greeting/noise?]   → generate_answer
        → [条件分支: realtime?]          → web_search → generate_answer
        → [条件分支: math?]              → generate_answer（数学模型；ASR→LaTeX 已在上一个节点完成）
        → [条件分支: rag (concept)?]     → evaluate_complexity → generate_answer（RAGAnything）
        → [条件分支: general?]           → generate_answer（通用 LLM，不走 RAG）
        → save_conversation → END

注：post_classification_preprocess 负责“分类后”的 ASR→LaTeX 转换——先由
classify_query_type 的 LLM 分类器判定是否为数学题，再统一调用 word_to_latex，
避免原先 preprocess_query 中 is_math_problem 启发式漏判排列组合题。

意图 → 回答路径的映射统一维护在
``app/services/conversation/intent_routing.py``（INTENT_TO_ANSWER_MODE）。
新增分类标签或调整路由策略只改那张表，不需要改工作流。

注：已使用 RAGAnything 简化 RAG 检索流程。
已移除节点：intent_recognition, knowledge_retrieval, grade_documents,
            compress_context, match_faq, rewrite_query, check_math_problem
"""
import os
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logging import get_logger
from app.services.conversation.conversation_state import ConversationState
from app.services.conversation.conversation_nodes import ConversationNodes
from app.services.math_agent_service import MathAgentService, MathRuntimeConfig
from app.utils.get_vllm_first_model import get_vllm_first_model

logger = get_logger(__name__)


# =============================================================================
# 对话工作流类
# =============================================================================
class ConversationWorkflow:
    """
    基于 LangGraph 的数字员工对话工作流。

    特性：
    - 双 LLM 支持：本地 Ollama 负责快速响应，远程 API 负责复杂任务
    - 混合路由：根据查询复杂度自动选择 LLM
    - RAGAnything 集成：知识图谱 + 向量检索 + 流式输出
    - 基于 LLM 的查询分类：QueryClassifier 支持 9 种意图
    - 精简工作流：8 节点

    工作流由 8 个节点通过条件边连接。

    LLM 路由策略（hybrid 模式）：
    - 本地 Ollama：问候语、简单查询（<30 字）、前几轮对话
    - 远程 API：RAGAnything 查询、联网搜索、长上下文、复杂查询
    """

    def __init__(self):
        """初始化工作流，创建双 LLM 实例用于混合路由。"""
        # 统一 LLM 配置（单一数据源）
        llm_base_url = os.getenv("LLM_BASE_URL")
        llm_model = os.getenv("LLM_MODEL")
        llm_api_key = os.getenv("LLM_API_KEY", "no-key")

        # 初始化本地 LLM — 用于快速、简单的响应
        self.local_llm = ChatOpenAI(
            base_url=llm_base_url,
            api_key=llm_api_key,
            model=llm_model,
            streaming=True,
        )
        logger.info(
            f"Local LLM configured | base_url={self.local_llm.openai_api_base} | "
            f"model={llm_model} | temperature={self.local_llm.temperature}"
        )

        # 初始化远程 LLM — 用于复杂、精确的响应
        # 统一配置下与 local_llm 相同，保留双对象架构供 hybrid 路由使用
        logger.info(
            f"Initializing remote LLM (unified config) | model={llm_model} | "
            f"base_url={llm_base_url}"
        )
        self.remote_llm = ChatOpenAI(
            base_url=llm_base_url,
            api_key=llm_api_key,
            model=llm_model,
            temperature=settings.openai_temperature,
            streaming=True,
        )

        # 初始化节点容器
        self.nodes = ConversationNodes(self)

        # 构建工作流图
        self.workflow = self._build_workflow()

    # -------------------------------------------------------------------------
    # LLM 选择方法（委托给 helpers，在此暴露接口）
    # -------------------------------------------------------------------------
    def get_active_llm(self, state: ConversationState):
        """
        根据当前状态获取合适的 LLM。

        在 LLM 调用前应先调用此方法，确保根据路由策略使用正确的模型。

        Args:
            state: 当前对话状态

        Returns:
            (llm, model_name) 元组
        """
        from app.services.conversation.conversation_helpers import select_llm
        return select_llm(
            state,
            self.local_llm,
            self.remote_llm
        )

    def get_streaming_llm(self, state: ConversationState):
        """
        获取用于答案生成的流式 LLM。

        用于聊天端点的 token 级流式输出。
        根据路由策略返回 LLM 实例和模型名。

        确定性覆盖（事实性人事查询）：
            「现任 X 职务是谁」等需要广博/最新世界知识的查询若沿用默认
            ``temperature``，会让 DeepSeek-V3 在「李强」与「李克强」之间随机偏移
            （实测 10 次约 20% 错答）。这里检测到此类 query 时强制把采样参数压回
            ``temperature=0 / top_p=1``，让答案完全由模型权重决定。
            仅当 ``_needs_big_world_knowledge`` 命中时才覆盖。

        Args:
            state: 当前对话状态

        Returns:
            (llm, model_name) 元组，用于流式输出
        """
        llm, model_name = self.get_active_llm(state)

        # 「现任 X 职务是谁」类事实查询的"确定性兜底"。
        # 复用 select_llm 已有的 _needs_big_world_knowledge 启发式，避免在两处分别
        # 维护词表；命中后强制压低采样随机性，确保 DeepSeek 等大模型给出稳定答案。
        from app.services.conversation.conversation_helpers import _needs_big_world_knowledge
        big_world_query = (state.get("rewritten_query") or state.get("user_query") or "").strip()
        if _needs_big_world_knowledge(big_world_query):
            base_url = getattr(llm, 'openai_api_base', 'https://api.openai.com/v1') or getattr(llm, 'base_url', 'https://api.openai.com/v1')
            api_key = getattr(llm, 'openai_api_key', '') or getattr(llm, 'api_key', '')
            model_name = getattr(llm, 'model_name', None) or getattr(llm, 'model', '')
            llm = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=model_name,
                temperature=0.0,
                top_p=1.0,
            )
            logger.info(
                f"Force deterministic LLM (temperature=0, top_p=1) for big-world-knowledge query: "
                f"query={big_world_query[:80]!r}"
            )

        return llm, model_name

    # -------------------------------------------------------------------------
    # 消息构建方法（委托给 helpers）
    # -------------------------------------------------------------------------
    def build_generation_messages(self, state: ConversationState):
        """
        构建用于答案生成的 LLM 消息列表。

        Args:
            state: 当前对话状态

        Returns:
            LLM 消息对象列表
        """
        from app.services.conversation.conversation_helpers import build_generation_messages
        return build_generation_messages(state)

    def get_math_streaming_llm(self, state: ConversationState):
        """
        动态创建数学流式 LLM（不存储为实例变量）。

        通过环境变量配置：
        - MATH_LLM_ENABLED: 是否启用（默认 true）
        - MATH_MODEL_BASE_URL: API 地址
        - MATH_MODEL_NAME: 模型名（不设则通过 vLLM 自动发现）
        - MATH_RUNTIME_MODE: direct / cot / tir（旧 ``llm`` 已删除，配置 ``llm`` 会抛错）
        - MATH_RUNTIME_LANG: zh / en（不设时按 query 粗略推断）
        - MATH_TEMPERATURE / MATH_TOP_P: 非 Qwen3-32B 模型的采样参数；
          Qwen3-32B 始终使用服务端 generation_config.json

        Args:
            state: 当前对话状态

        Returns:
            (llm, model_name) 元组，用于数学流式输出
        """
        # 检查是否启用
        enabled = os.getenv("MATH_LLM_ENABLED", "true").lower() == "true"
        if not enabled:
            logger.info("Math LLM disabled (MATH_LLM_ENABLED=false), falling back to default LLM")
            return self.get_streaming_llm(state)

        base_url = os.getenv("MATH_MODEL_BASE_URL")
        model_id = os.getenv("MATH_MODEL_NAME")
        math_api_key = (
            os.getenv("MATH_MODEL_API_KEY", "").strip().strip('"').strip("'")
            or "dummy-key"
        )

        # 未指定模型名时通过 vLLM 自动发现
        if not model_id:
            model_id = get_vllm_first_model(base_url)

        math_temperature = float(os.getenv("MATH_TEMPERATURE", 0.6))
        math_max_token = int(os.getenv("MATH_MAX_TOKEN", 10240))
        math_top_p = float(os.getenv("MATH_TOP_P", 0.95))
        use_model_generation_defaults = (
            MathAgentService.is_model_generation_default(model_id)
        )
        if use_model_generation_defaults:
            math_temperature = None
            math_top_p = None
        raw_runtime_mode = os.getenv("MATH_RUNTIME_MODE", "direct")
        # 非法值（含已删除的 llm）直接抛 ValueError，让配置问题暴露，不静默 fallback。
        runtime_mode = MathAgentService.validate_runtime_mode(raw_runtime_mode)
        runtime_lang = (os.getenv("MATH_RUNTIME_LANG") or "").strip().lower()

        query_text = (
            state.get("rewritten_query")
            or state.get("user_query")
            or ""
        )
        if runtime_lang not in {"zh", "en"}:
            runtime_lang = MathAgentService.resolve_lang(query_text, fallback="zh")

        math_service = MathAgentService(MathRuntimeConfig(
            base_url=base_url,
            api_key=math_api_key,
            model=model_id,
            temperature=math_temperature,
            max_tokens=math_max_token,
            streaming=True,
            top_p=math_top_p,
        ))
        math_llm = math_service.create_streaming_interface(
            mode=runtime_mode,
            lang=runtime_lang,
        )

        logger.info(
            f"Math runtime created | mode={runtime_mode} | lang={runtime_lang} | "
            f"model={model_id} | base_url={base_url} | "
            f"sampling={'model_default' if use_model_generation_defaults else 'env'} | "
            f"math_temperature={math_temperature} | math_top_p={math_top_p} | "
            f"math_max_token={math_max_token}"
        )

        # 把运行模式/语言写入 state，供下游节点（conversation_nodes）构建数学消息时读取，
        # 避免再从 ChatOpenAI 对象上 getattr 一个不存在的 mode 属性。
        state["math_runtime_mode"] = runtime_mode
        state["math_runtime_lang"] = runtime_lang

        return math_llm, model_id

    async def save_conversation(self, state: ConversationState):
        """
        保存对话记录（委托给节点容器）。

        Args:
            state: 当前对话状态

        Returns:
            更新后的状态（已填充 conversation_id）
        """
        return await self.nodes.save_conversation(state)

    # -------------------------------------------------------------------------
    # 工作流图构建
    # -------------------------------------------------------------------------
    def _build_workflow(self) -> StateGraph:
        """
        构建 LangGraph 工作流，包含所有节点和条件边。

        图结构：
        - 入口：load_employee_config
        - 中间：classify_query_type 条件路由
        - 出口：save_conversation → END

        使用 RAGAnything 简化 RAG 检索流程。

        Returns:
            编译完成的 StateGraph，可执行
        """
        graph = StateGraph(ConversationState)

        # 添加所有工作流节点（委托给节点容器）
        graph.add_node("load_employee_config", self.nodes.load_employee_config)
        graph.add_node("load_session_context", self.nodes.load_session_context)
        graph.add_node("input_validation", self.nodes.validate_input)
        graph.add_node("preprocess_query", self.nodes.preprocess_query)
        graph.add_node("classify_query_type", self.nodes.classify_query_type)
        graph.add_node("resolve_context_query", self.nodes.resolve_context_query)
        graph.add_node("finalize_classification", self.nodes.finalize_classification)
        graph.add_node("post_classification_preprocess", self.nodes.post_classification_preprocess)
        graph.add_node("evaluate_complexity", self.nodes.evaluate_complexity)
        graph.add_node("web_search", self.nodes.web_search)
        graph.add_node("generate_answer", self.nodes.generate_answer)
        graph.add_node("save_conversation", self.nodes.save_conversation)

        # 设置入口节点
        graph.set_entry_point("load_employee_config")

        # 定义顺序边
        graph.add_edge("load_employee_config", "load_session_context")
        graph.add_edge("load_session_context", "input_validation")
        graph.add_edge("input_validation", "preprocess_query")
        graph.add_edge("preprocess_query", "classify_query_type")
        # 分类与上下文消歧完成后由 answer_mode 决定实时、工训 RAG 或通用 LLM 路径。
        graph.add_edge("classify_query_type", "resolve_context_query")
        graph.add_edge("resolve_context_query", "finalize_classification")
        graph.add_edge("finalize_classification", "post_classification_preprocess")

        # 查询分类后的条件路由（挂在 post_classification_preprocess 之后）。
        # path_map 的 key 必须与 ``intent_routing.ROUTE_BRANCH_*`` 一一对应，
        # 任何新增的分支都需要在这里登记，否则 LangGraph 会抛 KeyError。
        from app.services.conversation.intent_routing import (
            ROUTE_BRANCH_GENERAL,
            ROUTE_BRANCH_GREETING,
            ROUTE_BRANCH_RAG,
            ROUTE_BRANCH_REALTIME,
        )
        graph.add_conditional_edges(
            "post_classification_preprocess",
            self.nodes.route_after_classification,
            {
                ROUTE_BRANCH_GREETING: "generate_answer",   # 问候/噪声 → 直接回答
                ROUTE_BRANCH_REALTIME: "web_search",        # 实时类 → 联网检索
                ROUTE_BRANCH_RAG: "evaluate_complexity",   # 工训教材类 → LightRAG
                ROUTE_BRANCH_GENERAL: "generate_answer",    # 通用 LLM（英语/常识/闲聊）→ 直接回答
            }
        )

        # 工训教材流程：复杂度评估 → generate_answer，后者配置 rag_stream。
        graph.add_edge("evaluate_complexity", "generate_answer")

        # 最终顺序
        graph.add_edge("web_search", "generate_answer")
        graph.add_edge("generate_answer", "save_conversation")
        graph.add_edge("save_conversation", END)

        # 编译并导出图（用于调试）
        compiled_graph = graph.compile()
        # self._dump_graph_debug(compiled_graph)
        return compiled_graph

    def _dump_graph_debug(self, compiled_graph) -> None:
        """
        将图结构导出为 Mermaid 格式，用于可视化。

        输出文件：graph_debug/crag_graph.mmd
        控制开关：CRAG_DUMP_GRAPH 环境变量（默认启用）
        """
        dump_flag = os.getenv("CRAG_DUMP_GRAPH", "1").lower()
        if dump_flag in {"0", "false", "no"}:
            return

        try:
            graph_view = compiled_graph.get_graph(xray=True)
            output_dir = Path(os.getenv("CRAG_GRAPH_DIR", "./graph_debug"))
            output_dir.mkdir(parents=True, exist_ok=True)

            # 提取 Mermaid 源码
            mermaid_src = None
            try:
                mermaid_src = graph_view.draw_mermaid()
                logger.info("Mermaid source extracted successfully")
            except Exception as src_exc:
                logger.warning(f"Failed to extract mermaid source: {src_exc}")
                (output_dir / "crag_graph_mermaid_extract_error.txt").write_text(
                    str(src_exc), encoding="utf-8"
                )

            # 保存 Mermaid 文件
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
# 全局工作流实例
# =============================================================================
conversation_workflow = ConversationWorkflow()
