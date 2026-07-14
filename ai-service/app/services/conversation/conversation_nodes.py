"""Nodes for the common LLM conversation workflow."""

import hashlib
import time
from datetime import datetime

from langchain_community.tools.tavily_search import TavilySearchResults

from app.core.config import settings
from app.core.database import get_database
from app.core.logging import get_logger
from app.models.database import ConversationModel, SessionModel
from app.services.conversation.conversation_helpers import (
    build_generation_messages,
    clean_user_query,
    prefer_zh_output,
    resolve_target_year_from_query,
    time_node,
)
from app.services.conversation.conversation_state import (
    ConversationState,
    DEFAULT_SENSITIVE_WORDS,
    DEFAULT_SENSITIVE_WORDS_LOWER,
    sensitive_term_matches_query,
)
from app.services.conversation.intent_routing import (
    ROUTE_BRANCH_GENERAL,
    ROUTE_BRANCH_REALTIME,
)
from app.services.query_classifier import (
    ClassificationResult,
    augment_dialog_with_persisted_turns,
    format_dialog_for_resolver,
    get_query_classifier,
)
from app.services.realtime_intent_heuristic import heuristic_realtime_category

logger = get_logger(__name__)

_REALTIME_HEURISTIC_SKIP_LABELS = frozenset({"greeting", "noise", "realtime_query"})


class ConversationNodes:
    def __init__(self, workflow_instance):
        self.workflow = workflow_instance

    async def load_employee_config(self, state: ConversationState) -> ConversationState:
        async with time_node("load_employee_config", state):
            db = await get_database()
            employee = await db.digital_employee_configs.find_one(
                {"employee_id": state["employee_id"]}
            )
            if not employee:
                raise ValueError(f"Employee config not found: employee_id={state['employee_id']}")
            employee.pop("_id", None)
            setting_doc = await db.digital_employee_settings.find_one(
                {"employee_id": state["employee_id"]}
            )
            if setting_doc:
                setting_doc.pop("_id", None)
                employee["setting"] = setting_doc
            state["employee_config"] = employee
            logger.info("Employee config loaded | employee_id=%s", state["employee_id"])
        return state

    async def load_session_context(self, state: ConversationState) -> ConversationState:
        async with time_node("load_session_context", state):
            try:
                db = await get_database()
                session = await db.sessions.find_one({"session_id": state["session_id"]})
                if session:
                    state["context"] = {
                        "messages": (session.get("context_messages") or [])[-10:],
                        "message_count": session.get("message_count", 0),
                    }
                    await db.sessions.update_one(
                        {"session_id": state["session_id"]},
                        {"$set": {"last_activity": datetime.now()}},
                    )
                else:
                    session_model = SessionModel(
                        session_id=state["session_id"],
                        user_id=state["user_id"],
                        employee_id=state["employee_id"],
                        status="active",
                        message_count=0,
                    )
                    await db.sessions.insert_one(session_model.model_dump())
                    state["context"] = {"messages": [], "message_count": 0}
                state.setdefault("sources", [])
            except Exception as exc:
                logger.error("Failed to load session context: %s", exc, exc_info=True)
                state["context"] = {"messages": [], "message_count": 0}
                state.setdefault("sources", [])
        return state

    async def validate_input(self, state: ConversationState) -> ConversationState:
        async with time_node("input_validation", state):
            query_lower = (state.get("user_query") or "").strip().lower()
            employee_words: set[str] = set()
            sensitive_ids = ((state.get("employee_config") or {}).get("safe_rule") or {}).get(
                "sensitive_ids"
            ) or []
            if sensitive_ids:
                try:
                    db = await get_database()
                    records = await db.thesaurus_sensitive.find(
                        {"thesaurus_id": {"$in": sensitive_ids}}, {"word": 1, "_id": 0}
                    ).to_list(length=None)
                    employee_words = {str(item.get("word") or "").lower() for item in records}
                except Exception as exc:
                    logger.warning("Failed to load employee sensitive words: %s", exc)
            terms = DEFAULT_SENSITIVE_WORDS_LOWER | {term for term in employee_words if term}
            state["has_sensitive"] = any(
                sensitive_term_matches_query(query_lower, term) for term in terms
            )
            logger.info("Sensitive check | matched=%s", state["has_sensitive"])
        return state

    async def preprocess_query(self, state: ConversationState) -> ConversationState:
        async with time_node("preprocess_query", state):
            cleaned = clean_user_query((state.get("user_query") or "").strip())
            state["user_query"] = cleaned
            state["prefer_zh_output"] = prefer_zh_output(cleaned)
        return state

    async def classify_query_type(self, state: ConversationState) -> ConversationState:
        async with time_node("classify_query_type", state):
            query = (state.get("user_query") or "").strip()
            messages = (state.get("context") or {}).get("messages") or []
            last_user_query = next(
                (
                    (item.get("content") or "").strip()
                    for item in reversed(messages)
                    if item.get("role") == "user" and (item.get("content") or "").strip()
                ),
                "",
            )
            result = await get_query_classifier().aclassify(query, context_query=last_user_query or None)
            state["raw_classification_label"] = result.label
            state["raw_classification_confidence"] = result.confidence
            state["raw_classification_reason"] = result.reason
            state["classification_label"] = result.label
            state["classification_confidence"] = result.confidence
            state["classification_reason"] = result.reason
            logger.info("Raw query classification | label=%s | query=%r", result.label, query[:80])
        return state

    async def resolve_context_query(self, state: ConversationState) -> ConversationState:
        """Apply the same history-aware standalone-query resolution to every query."""
        async with time_node("resolve_context_query", state):
            query = (state.get("user_query") or "").strip()
            state["rewritten_query"] = query
            state["effective_query"] = query
            state["query_rewritten"] = False
            context_messages = (state.get("context") or {}).get("messages") or []
            dialog_text = format_dialog_for_resolver(context_messages)
            if not dialog_text.strip():
                state["context_dependence"] = "unrelated"
                state["context_dependence_reason"] = "no_history"
                state["context_resolution_mode"] = "none"
                state["context_resolution_skipped_reason"] = "no_history"
                return state
            if not getattr(settings, "dynamic_context_memory_enabled", True):
                state["context_dependence"] = "unrelated"
                state["context_dependence_reason"] = "disabled"
                state["context_resolution_mode"] = "none"
                state["context_resolution_skipped_reason"] = "disabled"
                return state
            if state.get("session_id"):
                try:
                    db = await get_database()
                    records = await db.conversations.find(
                        {"session_id": state["session_id"]}, {"_id": 0, "user_query": 1, "ai_response": 1}
                    ).sort("created_at", 1).limit(30).to_list(length=30)
                    dialog_text = augment_dialog_with_persisted_turns(dialog_text, list(records), query)
                except Exception as exc:
                    logger.warning("Failed to supplement dialog context: %s", exc)
            classifier = get_query_classifier()
            related, reason = await classifier.aclassify_context_dependence(query, dialog_text)
            state["context_dependence"] = "related" if related else "unrelated"
            state["context_dependence_reason"] = reason
            if not related:
                state["context_resolution_mode"] = "none"
                state["context_resolution_skipped_reason"] = "unrelated"
                return state
            resolved = (await classifier.aresolve_standalone_query(query, dialog_text)).strip() or query
            state["rewritten_query"] = resolved
            state["effective_query"] = resolved
            state["query_rewritten"] = resolved != query
            state["context_resolution_mode"] = "normal_resolver"
            state["context_resolution_skipped_reason"] = None
        return state

    async def finalize_classification(self, state: ConversationState) -> ConversationState:
        async with time_node("finalize_classification", state):
            query = (state.get("user_query") or "").strip()
            resolved = (state.get("effective_query") or query).strip()
            original = ClassificationResult(
                label=state.get("raw_classification_label") or "other",
                confidence=state.get("raw_classification_confidence") or "medium",
                reason=state.get("raw_classification_reason") or "raw",
            )
            result = original if resolved == query else await get_query_classifier().aclassify(resolved)
            realtime_reason = heuristic_realtime_category(resolved)
            if result.label not in _REALTIME_HEURISTIC_SKIP_LABELS and realtime_reason:
                result = ClassificationResult(
                    label="realtime_query", confidence=result.confidence, reason=realtime_reason
                )
            elif (
                getattr(settings, "realtime_query_llm_fallback_enabled", False)
                and result.label not in _REALTIME_HEURISTIC_SKIP_LABELS
                and result.confidence != "high"
                and await get_query_classifier().aneed_realtime(resolved)
            ):
                result = ClassificationResult(
                    label="realtime_query", confidence=result.confidence, reason="general"
                )
            state["classification_label"] = result.label
            state["classification_confidence"] = result.confidence
            state["classification_reason"] = result.reason
            state["is_realtime_query"] = result.label == "realtime_query"
            state["realtime_category"] = result.reason if state["is_realtime_query"] else ""
            state["realtime_detect_reason"] = f"classifier:{result.confidence}" if state["is_realtime_query"] else ""
            state["intent"] = "greeting" if result.label == "greeting" else "general_query"
            state["target_year"] = resolve_target_year_from_query(resolved)
            logger.info("Final classification | label=%s | realtime=%s", result.label, state["is_realtime_query"])
        return state

    @staticmethod
    def route_after_classification(state: ConversationState) -> str:
        if state.get("is_realtime_query"):
            return ROUTE_BRANCH_REALTIME
        return ROUTE_BRANCH_GENERAL

    async def web_search(self, state: ConversationState) -> ConversationState:
        """Fetch web results for realtime queries; failure remains LLM-generated."""
        async with time_node("web_search", state):
            state["web_search_results"] = []
            state["web_search_used"] = False
            state["web_search_error"] = None
            if not getattr(settings, "web_search_enabled", False):
                state["web_search_error"] = "web search is disabled"
                return state
            if not getattr(settings, "tavily_api_key", None):
                state["web_search_error"] = "Tavily API key not configured"
                return state
            capabilities = (state.get("employee_config") or {}).get("capabilities") or {}
            if not capabilities.get("web_search_enabled", True):
                state["web_search_error"] = "web search disabled for employee"
                return state
            query = (state.get("rewritten_query") or state.get("user_query") or "").strip()
            try:
                search = TavilySearchResults(api_key=settings.tavily_api_key, max_results=5)
                raw_results = await search.ainvoke({"query": query})
                results = raw_results if isinstance(raw_results, list) else []
                state["web_search_results"] = results
                state["web_search_used"] = bool(results)
                if results:
                    state.setdefault("sources", []).append({
                        "type": "web_search", "from": "web_search", "text": query,
                        "citations": [
                            {"title": item.get("title", ""), "url": item.get("url", ""), "snippet": (item.get("content") or "")[:300]}
                            for item in results[:5] if isinstance(item, dict)
                        ],
                    })
            except Exception as exc:
                state["web_search_error"] = str(exc)
                logger.warning("Web search failed: %s", exc, exc_info=True)
        return state

    async def generate_answer(self, state: ConversationState) -> ConversationState:
        async with time_node("generate_answer", state):
            messages = build_generation_messages(state)
            streaming_llm, model_name = self.workflow.get_streaming_llm(state)
            state["streaming_llm"] = streaming_llm
            state["streaming_messages"] = messages
            state["confidence"] = 0.8 if not state.get("web_search_used") else 0.9
            logger.info(
                "Streaming configured: type=general_llm | model=%s | is_realtime=%s | web_search_used=%s",
                model_name, state.get("is_realtime_query"), state.get("web_search_used"),
            )
        return state

    async def save_conversation(self, state: ConversationState) -> ConversationState:
        async with time_node("save_conversation", state):
            try:
                db = await get_database()
                timestamp = datetime.now().timestamp()
                seed = f"{state['session_id']}_{timestamp}"
                conv_id = f"conv_{hashlib.md5(seed.encode()).hexdigest()[:12]}"
                state["conversation_id"] = conv_id
                elapsed_ms = int((time.time() - state.get("workflow_start_time", time.time())) * 1000)
                state["response_time_ms"] = elapsed_ms
                config = state.get("employee_config") or {}
                conversation = ConversationModel(
                    conversation_id=conv_id, session_id=state["session_id"], user_id=state["user_id"],
                    user_name=state.get("user_name", ""), head_url=state.get("head_url", ""),
                    employee_id=state["employee_id"], employee_name=config.get("name", ""),
                    user_query=state.get("user_query", ""), ai_response=state.get("final_answer", ""),
                    is_realtime_query=state.get("is_realtime_query", False),
                    realtime_category=state.get("realtime_category"), intent=state.get("intent"),
                    web_search_used=state.get("web_search_used", False),
                    web_search_results=[
                        {"rank": index + 1, "title": item.get("title"), "url": item.get("url"), "score": item.get("score", 0.0)}
                        for index, item in enumerate(state.get("web_search_results", [])[:5]) if isinstance(item, dict)
                    ], confidence=state.get("confidence", 0.0), response_time_ms=elapsed_ms,
                )
                await db.conversations.insert_one(conversation.model_dump())
                await db.sessions.update_one(
                    {"session_id": state["session_id"]},
                    {"$push": {"context_messages": {"$each": [
                        {"role": "user", "content": state.get("user_query", "")},
                        {"role": "assistant", "content": state.get("final_answer", "")},
                    ], "$slice": -20}}, "$inc": {"message_count": 1}},
                )
            except Exception as exc:
                logger.error("Failed to save conversation: %s", exc, exc_info=True)
        return state
