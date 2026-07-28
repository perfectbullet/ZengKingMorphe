"""
Conversation State and Constants for LangGraph Workflow.

This module defines:
- ConversationState: TypedDict that flows through workflow nodes
- GREETING_KEYWORDS: Keyword patterns for greeting detection
"""
import re
from pathlib import Path
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Set
from operator import add

from app.core.logging import get_logger

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
# Default Sensitive Words
# Basic sensitive word list for content safety filtering.
# Used when employee-specific sensitive word lists are not available.
# =============================================================================
def _load_default_sensitive_words() -> Set[str]:
    """
    从文件加载默认敏感词列表。

    文件格式：
    - 一行一个敏感词
    - # 开头为注释
    - ( 开头为注释
    - 自动去重
    - 去掉长度为1的敏感词
    Returns:
        Set[str]: 去重后的敏感词集合
    """
    words: Set[str] = set()
    # 从 conversation_state.py 向上四级到 ai-service 目录
    # ai-service/app/services/conversation/conversation_state.py -> ai-service/
    file_path = Path(__file__).parent.parent.parent.parent / "DEFAULT_SENSITIVE_WORDS.txt"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 跳过空行、注释行（#或(开头）
                if not line or line.startswith("#") or line.startswith("("):
                    continue
                if len(line) < 2:
                    continue
                words.add(line)
        logger.info(f"Loaded {len(words)} default sensitive words from {file_path}")
    except FileNotFoundError:
        logger.warning(f"Sensitive words file not found: {file_path}, using empty list")

    return words


DEFAULT_SENSITIVE_WORDS = frozenset(_load_default_sensitive_words())
DEFAULT_SENSITIVE_WORDS_LOWER = frozenset(w.lower() for w in DEFAULT_SENSITIVE_WORDS)


def sensitive_term_matches_query(query_lower: str, term_lower: str) -> bool:
    """
    敏感词匹配：对纯英文/数字敏感词使用整词匹配，避免误杀；
    对含中文/符号的敏感词使用子串匹配，保证拦截效果。
    """
    if not term_lower:
        return False
    if not query_lower:
        return False
    ascii_alnum = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")
    # 非纯英文数字 → 子串匹配
    if not term_lower.isascii() or not all(c in ascii_alnum for c in term_lower):
        return term_lower in query_lower
    # 纯英文数字 → 整词匹配
    return re.search(r"\b" + re.escape(term_lower) + r"\b", query_lower) is not None

# =============================================================================
# Noise Preset Response Text
# =============================================================================
# 噪声分类命中且通过 ``apply_noise_preset_gate`` 三道闸门后展示的兜底话术。
#
# 设计要点（务必保持的措辞约束）：
# - **不要**出现"听清 / 没听见 / 听不到"等措辞——这些字眼在数字人侧会让用户
#   误以为麦克风/ASR 故障，把服务端的有意拦截当成硬件问题，造成线下排错噪声。
# - 措辞偏向"理解层面"：表达"我没理解到您的提问要点"，引导用户换种说法或补充信息。
# - 一处文案、多处复用：``conversation_nodes`` 的 ``classify_query_type`` 与
#   ``generate_answer`` 都从这里读取，避免散落多份硬编码。
NOISE_PRESET_RESPONSE_TEXT: str = "您的问题我没太理解，能否换种说法或补充更多信息？"


# =============================================================================
# Interruption Detection Keywords (REMOVED)
# Previously used for detecting user interruption intent during conversation.
# This functionality has been removed to allow interruption keywords
# to flow through the normal processing pipeline.
# =============================================================================

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
    - LLM parameters (llm_temperature, llm_top_p, etc.)
    - Additional context (channel_name, team_id)
    """
    messages: Annotated[List, add]
    user_query: str
    user_id: str
    user_name: str
    head_url: str
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
    direct_match: Optional[Dict[str, Any]]  # 直接匹配信息（仅数学教材知识库）
    kb_used: List[str]
    web_search_used: bool
    web_search_error: Optional[str]  # Web search error message (e.g., API key expired)
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
    complexity_score: float  # 0-10: 问题复杂度评分
    complexity_reason: str  # 复杂度评分的原因说明
    # Additional context from client
    channel_name: Optional[str]
    team_id: Optional[str]
    # Streaming output configuration for unified RAG integration
    streaming_type: Optional[str]  # "langchain_llm" / "rag_stream" / "math_llm"
    streaming_llm: Optional[Any]  # LLM instance (for langchain_llm)
    streaming_messages: Optional[List]  # Messages (for langchain_llm)
    rag_query: Optional[str]  # Query for unified RAG stream
    rag_mode: Optional[str]  # Unified RAG mode (hybrid, local, global, naive)
    direct_text_answer: Optional[str]  # 直接文本答案，不走LLM（如：系统时间）
    sources: List[Dict[str, Any]]  # 累积的数据源列表
    is_math_problem: bool  # 是否为数学题目
    # LLM-based classification results (from QueryClassifier)
    classification_label: Optional[str]  # 分类标签 (math_problem, concept_explain, greeting, etc.)
    classification_confidence: Optional[str]  # 置信度 (high/medium/low)
    classification_reason: Optional[str]  # 分类理由
    # 回答模式（由 INTENT_TO_ANSWER_MODE 表驱动；下游统一读这一个字段决定生成路径）
    # - "rag_with_fallback" / "general_llm" / "math_llm" / "web_search" / "preset_response"
    # 设计要点：把"该走 RAG 还是 web 还是直接 LLM"的判定从 generate_answer 里抽出来，
    # 让分类节点一次定级，避免下游各处再依赖 intent / is_realtime_query / is_math_problem 复合条件。
    answer_mode: Optional[str]
    target_year: Optional[int]  # 目标年份：今年/明年/去年解析后的年份
    # 动态上下文记忆：本轮问题与历史对话是否相关
    # - "related"   → 与历史相关（指代承接/话题延续/前文关联），下游需注入完整上下文
    # - "unrelated" → 与历史无关（全新话题），下游一律不注入历史，按全新问题处理
    # - None        → 尚未判定（默认）；首轮无历史时也可保持 None 表示无上下文可用
    context_dependence: Optional[str]
    context_dependence_reason: Optional[str]  # 判定来源：no_history / llm_yes / llm_no / fallback / disabled
    # 输出语言偏好（True=中文，False=英文）。
    # 注意：必须在 TypedDict 中声明，否则 LangGraph 1.x 在执行节点前会按 schema
    # 过滤掉未声明字段，导致下游读到空值后 fallback 到默认中文，出现
    # “英文问、中英文混合答”这类语言飘移 Bug。设为 Optional 兼容初始未设置场景。
    prefer_zh_output: Optional[bool]
    # ASR→LaTeX 分类后转换的决策与结果（由 post_classification_preprocess 节点写入）。
    # 时机说明：原先在 preprocess_query 里靠 is_math_problem 启发式决定是否转换，
    # 会漏掉“次品/测试/方法数”这类排列组合题；改为先由 classify_query_type 的 LLM
    # 分类器判定为 math_problem 后，再在 post_classification_preprocess 调用
    # word_to_latex，确保数学模型拿到的是转换后的题干。同样必须在 TypedDict 声明，
    # 否则被 LangGraph schema 过滤后 chat_stream_v1 读不到 query_preprocessed。
    asr_latex_should_run: Optional[bool]   # 分类后是否应当执行 ASR→LaTeX 转换
    asr_latex_converted: Optional[bool]    # 是否实际完成了转换（query 被改写）
    query_preprocessed: Optional[bool]     # user_query 是否被本节点改写（供前端 chunk 判定）
    asr_latex_before: Optional[str]        # 转换前原文（审计/排查）
    asr_latex_after: Optional[str]         # 转换后文本（审计/排查）
    # ── 分类 / 上下文消歧拆分后的中间状态 ──
    # 上下文消歧现已从 classify_query_type 独立为 resolve_context_query 节点；
    # 完整数学题不做上下文消歧，数学追问只用上下文不改写题干，非数学追问走普通 resolver。
    # 这些字段记录拆分链路各阶段的中间结论，便于排查 / 测试断言。
    raw_classification_label: Optional[str]      # 原始 query 的初步分类标签（classify_query_type）
    raw_classification_confidence: Optional[str] # 原始 query 的初步分类置信度
    raw_classification_reason: Optional[str]     # 原始 query 的初步分类理由
    context_resolution_mode: Optional[str]       # none / skipped_complete_math / math_context_only / normal_resolver
    context_resolution_skipped_reason: Optional[str]  # complete_math_problem / no_history / disabled / unrelated / empty_resolver_result
    math_context_used: Optional[bool]            # 数学追问是否使用历史上下文
    math_context_text: Optional[str]             # 数学追问使用的上下文文本（仅给 math prompt，不用于改写题干）
    effective_query: Optional[str]               # 最终用于分类 / 生成的 query（数学题仍以 post_classification_preprocess 后的 user_query 为准）
    # ── 人工概念检索状态 ──
    # 概念检索节点（concept_retrieval）写入的字段，用于人工概念库的精确匹配和 LightRAG 召回
    concept_retrieval_enabled: Optional[bool]    # 概念检索功能是否启用
    concept_retrieval_hit: Optional[bool]        # 是否命中人工概念库
    concept_retrieval_reason: Optional[str]      # 命中原因或未命中原因
    concept_context: Optional[Dict[str, Any]]   # 命中的概念上下文（含 concept_name, content, domain 等）
    concept_context_source: Optional[str]       # 概念上下文来源（如 "manual_concept_lightrag"）
