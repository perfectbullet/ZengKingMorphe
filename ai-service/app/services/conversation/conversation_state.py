"""State and safety helpers shared by the conversation workflow."""

import re
from operator import add
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Set, TypedDict

from app.core.logging import get_logger

logger = get_logger(__name__)

GREETING_KEYWORDS = {
    "basic": ["你好", "您好", "hi", "hello", "嗨"],
    "time": ["早上好", "早", "上午好", "中午好", "下午好", "晚上好", "晚安"],
    "casual": ["哈喽", "在吗", "在不在", "有人吗"],
    "polite": ["打扰一下", "请问", "不好意思", "劳驾"],
}


def _load_default_sensitive_words() -> Set[str]:
    words: Set[str] = set()
    file_path = Path(__file__).parent.parent.parent.parent / "DEFAULT_SENSITIVE_WORDS.txt"
    try:
        for line in file_path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith(("#", "(")) and len(value) >= 2:
                words.add(value)
    except FileNotFoundError:
        logger.warning("Sensitive words file not found: %s", file_path)
    return words


DEFAULT_SENSITIVE_WORDS = frozenset(_load_default_sensitive_words())
DEFAULT_SENSITIVE_WORDS_LOWER = frozenset(word.lower() for word in DEFAULT_SENSITIVE_WORDS)


def sensitive_term_matches_query(query_lower: str, term_lower: str) -> bool:
    """Match ASCII terms at word boundaries and all other terms by substring."""
    if not query_lower or not term_lower:
        return False
    if not term_lower.isascii() or not term_lower.isalnum():
        return term_lower in query_lower
    return re.search(r"\b" + re.escape(term_lower) + r"\b", query_lower) is not None


class ConversationState(TypedDict):
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
    web_search_results: List[Dict[str, Any]]
    web_search_used: bool
    web_search_error: Optional[str]
    final_answer: str
    confidence: float
    context: Dict[str, Any]
    has_sensitive: bool
    error: Optional[Dict[str, Any]]
    faq_matched: Optional[Dict[str, Any]]
    conversation_id: str
    response_time_ms: int
    workflow_start_time: float
    node_timings: Dict[str, float]
    ttfb_ms: Optional[int]
    rewritten_query: str
    query_rewritten: bool
    answer_verified: bool
    verification_result: Optional[Dict[str, Any]]
    channel_name: Optional[str]
    team_id: Optional[str]
    streaming_llm: Optional[Any]
    streaming_messages: Optional[List]
    sources: List[Dict[str, Any]]
    classification_label: Optional[str]
    classification_confidence: Optional[str]
    classification_reason: Optional[str]
    target_year: Optional[int]
    context_dependence: Optional[str]
    context_dependence_reason: Optional[str]
    prefer_zh_output: Optional[bool]
    raw_classification_label: Optional[str]
    raw_classification_confidence: Optional[str]
    raw_classification_reason: Optional[str]
    context_resolution_mode: Optional[str]
    context_resolution_skipped_reason: Optional[str]
    effective_query: Optional[str]
