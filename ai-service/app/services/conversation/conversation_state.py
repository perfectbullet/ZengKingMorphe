"""
Conversation State and Constants for LangGraph Workflow.

This module defines:
- ConversationState: TypedDict that flows through workflow nodes
- GREETING_KEYWORDS: Keyword patterns for greeting detection
"""
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


DEFAULT_SENSITIVE_WORDS = list(_load_default_sensitive_words())


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
    # LLM parameters from OpenAI-style request
    llm_temperature: Optional[float]
    llm_top_p: Optional[float]
    llm_max_tokens: Optional[int]
    llm_presence_penalty: Optional[float]
    llm_frequency_penalty: Optional[float]
    llm_seed: Optional[int]
    llm_n: Optional[int]
    llm_tools: Optional[List[Dict[str, Any]]]
    # Additional context from client
    channel_name: Optional[str]
    team_id: Optional[str]
    # Streaming output configuration for RAGAnything integration
    streaming_type: Optional[str]  # "langchain_llm" or "raganything_stream"
    streaming_llm: Optional[Any]  # LLM instance (for langchain_llm)
    streaming_messages: Optional[List]  # Messages (for langchain_llm)
    raganything_query: Optional[str]  # Query for RAGAnything
    raganything_mode: Optional[str]  # RAGAnything mode (hybrid, local, global, naive)
    sources: List[Dict[str, Any]]  # 累积的数据源列表
    is_math_problem: bool  # 是否为数学题目
