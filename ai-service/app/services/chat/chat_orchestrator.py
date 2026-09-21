"""Convert ConversationWorkflow output plans into protocol-neutral answer events."""

import random
import re
import time
from collections.abc import AsyncIterator
from typing import Any, Callable, Optional

from app.core.logging import get_logger
from app.services.chat.answer_events import AnswerEvent, ChatContext
from app.services.chat.math_reasoning_stream import (
    MathModelConfig,
    MathReasoningStreamAdapter,
    ReasoningBuffer,
)
from app.utils.latex import normalize_latex_formulas
from app.utils.sentence_buffer import SentenceBuffer

logger = get_logger(__name__)

STATUS_TOKENS = (
    "好的，我正在梳理您的问题要点…",
    "这个我知道······",
    "等我一小下下······",
)

REALTIME_KEYWORDS = (
    "天气", "气温", "温度", "下雨", "下雪", "刮风",
    "股价", "股票", "汇率", "金价", "银价", "新闻", "今日", "最新", "实时",
)


class ChatOrchestrator:
    """Run the workflow and expose one real-time ``AnswerEvent`` stream."""

    def __init__(
        self,
        workflow=None,
        *,
        math_stream_adapter_factory: Callable[[MathModelConfig], MathReasoningStreamAdapter] = MathReasoningStreamAdapter,
    ) -> None:
        if workflow is None:
            from app.services.conversation_service import conversation_workflow

            workflow = conversation_workflow
        self.workflow = workflow
        self.math_stream_adapter_factory = math_stream_adapter_factory

    async def stream(self, context: ChatContext) -> AsyncIterator[AnswerEvent]:
        state = self._initial_state(context)
        sequence = 0
        full_answer = ""
        model_name: Optional[str] = None

        sequence += 1
        yield AnswerEvent(event_type="start", sequence=sequence)

        if any(keyword in context.user_query.lower() for keyword in REALTIME_KEYWORDS):
            sequence += 1
            yield AnswerEvent(
                event_type="status",
                content=random.choice(STATUS_TOKENS),
                content_type="status",
                sequence=sequence,
                metadata={"model_name": "status"},
            )

        async for workflow_event in self.workflow.workflow.astream(
            state,
            stream_mode="updates",
        ):
            for state_update in workflow_event.values():
                if state_update:
                    state.update(state_update)

            if state.get("has_sensitive"):
                rejection = "抱歉，您的问题包含敏感内容，请规范用语后再试。"
                full_answer = rejection
                sequence += 1
                yield AnswerEvent(
                    event_type="content",
                    content=rejection,
                    content_type="status",
                    sequence=sequence,
                    metadata={"finish_reason": "sensitive"},
                )
                break

            plan = self._resolve_answer_plan(state)
            if plan is None:
                continue

            plan_type = plan[0]
            if plan_type == "prebuilt":
                display_text = normalize_latex_formulas(plan[1])
                direct_match = state.get("direct_match") or {}
                segments = self._split_prebuilt(display_text)
                for segment in segments:
                    sequence += 1
                    metadata = self._event_metadata(state, model_name)
                    if direct_match:
                        metadata["direct_match"] = direct_match
                    yield AnswerEvent(
                        event_type="content",
                        content=segment,
                        content_type="latex" if "\\" in segment or "$" in segment else "markdown",
                        sequence=sequence,
                        metadata=metadata,
                    )
                full_answer = display_text
            elif plan_type == "math_llm":
                math_config, messages = plan[1], plan[2]
                model_name = math_config.model_name
                reasoning_enabled = self._reasoning_display_enabled()
                reasoning_buffer = ReasoningBuffer()
                first_token_received = False

                async for chunk in self.math_stream_adapter_factory(math_config).stream(messages):
                    model_name = chunk.model_name or model_name
                    if reasoning_enabled and chunk.reasoning:
                        buffered_reasoning = reasoning_buffer.add(chunk.reasoning)
                        if buffered_reasoning:
                            sequence += 1
                            yield AnswerEvent(
                                event_type="reasoning",
                                content=buffered_reasoning,
                                content_type="reasoning",
                                sequence=sequence,
                                metadata=self._event_metadata(state, model_name),
                            )

                    if not chunk.content:
                        continue

                    # Preserve the vLLM ordering boundary: no buffered reasoning
                    # may appear after the first visible answer token.
                    if reasoning_enabled:
                        buffered_reasoning = reasoning_buffer.flush()
                        if buffered_reasoning:
                            sequence += 1
                            yield AnswerEvent(
                                event_type="reasoning",
                                content=buffered_reasoning,
                                content_type="reasoning",
                                sequence=sequence,
                                metadata=self._event_metadata(state, model_name),
                            )
                    if not first_token_received:
                        state["ttfb_ms"] = int(
                            (time.time() - state["workflow_start_time"]) * 1000
                        )
                        first_token_received = True
                    full_answer += chunk.content
                    sequence += 1
                    yield AnswerEvent(
                        event_type="content",
                        content=chunk.content,
                        content_type="math",
                        sequence=sequence,
                        metadata=self._event_metadata(state, model_name),
                    )

                if reasoning_enabled:
                    buffered_reasoning = reasoning_buffer.flush()
                    if buffered_reasoning:
                        sequence += 1
                        yield AnswerEvent(
                            event_type="reasoning",
                            content=buffered_reasoning,
                            content_type="reasoning",
                            sequence=sequence,
                            metadata=self._event_metadata(state, model_name),
                        )
            else:
                streaming_llm, messages = plan[1], plan[2]
                model_name = (
                    getattr(streaming_llm, "model_name", None)
                    or getattr(streaming_llm, "model", None)
                    or ""
                )
                first_token_received = False
                async for chunk in streaming_llm.astream(messages):
                    token = getattr(chunk, "content", "")
                    if not token:
                        continue
                    if not isinstance(token, str):
                        token = str(token)
                    if not first_token_received:
                        state["ttfb_ms"] = int(
                            (time.time() - state["workflow_start_time"]) * 1000
                        )
                        first_token_received = True
                    full_answer += token
                    sequence += 1
                    yield AnswerEvent(
                        event_type="content",
                        content=token,
                        content_type=(
                            "math" if state.get("streaming_type") == "math_llm" else "markdown"
                        ),
                        sequence=sequence,
                        metadata=self._event_metadata(state, model_name),
                    )

            state["final_answer"] = full_answer
            await self.workflow.save_conversation(state)
            break

        state["final_answer"] = full_answer or state.get("final_answer", "")
        sequence += 1
        yield AnswerEvent(
            event_type="done",
            sequence=sequence,
            metadata=self._done_metadata(state, model_name),
        )

    def _resolve_answer_plan(self, state: dict[str, Any]):
        streaming_type = state.get("streaming_type")
        direct_match = state.get("direct_match")
        final_answer = state.get("final_answer") or ""

        if final_answer and (direct_match or streaming_type == "text"):
            return "prebuilt", final_answer

        if streaming_type == "direct_text":
            direct_text = state.get("direct_text_answer") or final_answer
            if direct_text:
                return "prebuilt", direct_text

        if streaming_type == "math_llm":
            math_config = state.get("math_stream_config")
            messages = state.get("streaming_messages")
            if math_config is not None and messages is not None:
                return "math_llm", math_config, messages
            raise RuntimeError("Math streaming state is incomplete")

        if streaming_type == "langchain_llm":
            llm = state.get("streaming_llm")
            messages = state.get("streaming_messages")
            if llm is not None and messages is not None:
                return "llm", llm, messages
            raise RuntimeError(f"Streaming state is incomplete for type={streaming_type}")

        # Preserve the old v2 fallback for unhandled output modes such as rag_stream.
        if streaming_type:
            messages = self.workflow.build_generation_messages(state)
            llm, _ = self.workflow.get_streaming_llm(state)
            return "llm", llm, messages

        return None

    @staticmethod
    def _split_prebuilt(text: str) -> list[str]:
        parts = re.split(r"([。！？\n])", text)
        return [part for part in SentenceBuffer._merge_punctuation_segments(parts) if part]

    @staticmethod
    def _event_metadata(state: dict[str, Any], model_name: Optional[str]) -> dict[str, Any]:
        return {
            "conversation_id": state.get("conversation_id", ""),
            "answer_mode": state.get("answer_mode") or state.get("streaming_type") or "",
            "model_name": model_name,
        }

    @staticmethod
    def _done_metadata(state: dict[str, Any], model_name: Optional[str]) -> dict[str, Any]:
        sources = _format_sources(
            state.get("retrieved_docs", []),
            state.get("web_search_results", []),
        )
        return {
            "conversation_id": state.get("conversation_id", ""),
            "confidence": state.get("confidence", 0.0),
            "kb_used": state.get("kb_used", []),
            "web_search_used": state.get("web_search_used", False),
            "sources": sources,
            "intent": state.get("intent", ""),
            "is_realtime_query": state.get("is_realtime_query", False),
            "realtime_category": state.get("realtime_category", ""),
            "model_name": model_name,
            "answer_mode": state.get("answer_mode") or state.get("streaming_type") or "",
        }

    @staticmethod
    def _initial_state(context: ChatContext) -> dict[str, Any]:
        return {
            "messages": [],
            "user_query": context.user_query,
            "user_id": context.user_id,
            "user_name": context.user_name,
            "head_url": context.head_url,
            "session_id": context.session_id,
            "employee_id": context.employee_id,
            "employee_config": {},
            "is_realtime_query": False,
            "realtime_category": "",
            "realtime_detect_reason": "",
            "intent": "",
            "entities": {},
            "retrieved_docs": [],
            "relevance_score": 0.0,
            "web_search_results": [],
            "final_answer": "",
            "confidence": 0.0,
            "context": {},
            "has_sensitive": False,
            "error": None,
            "faq_matched": None,
            "kb_used": [],
            "web_search_used": False,
            "web_search_error": None,
            "conversation_id": "",
            "response_time_ms": 0,
            "sources": [],
            "workflow_start_time": time.time(),
            "node_timings": {},
            "ttfb_ms": None,
            "channel_name": context.channel_name,
            "team_id": context.team_id,
            "classification_label": None,
            "classification_confidence": None,
            "classification_reason": None,
            "answer_mode": None,
            "math_stream_config": None,
        }

    @staticmethod
    def _reasoning_display_enabled() -> bool:
        import os

        return os.getenv("MATH_REASONING_DISPLAY_ENABLED", "true").lower() == "true"


def _format_sources(
    retrieved_docs: list[dict[str, Any]],
    web_search_results: list[dict[str, Any]],
    max_content_length: int = 200,
) -> dict[str, list[dict[str, Any]]]:
    sources: dict[str, list[dict[str, Any]]] = {"rag_sources": [], "web_sources": []}
    for index, document in enumerate(retrieved_docs[:3], 1):
        content = document.get("content", "")
        raw_score = document.get("rrf_score", document.get("score", 0.0))
        normalized_score = max(0.0, min(1.0, raw_score / (2.0 / 60)))
        source = {
            "rank": index,
            "doc_id": document.get("doc_id", ""),
            "kb_id": document.get("kb_id", ""),
            "content_snippet": content[:max_content_length] + (
                "..." if len(content) > max_content_length else ""
            ),
            "score": round(normalized_score, 4),
        }
        if "chunk_index" in document:
            source["chunk_index"] = document["chunk_index"]
        sources["rag_sources"].append(source)

    for result in web_search_results[:5]:
        sources["web_sources"].append({
            "rank": result.get("rank", 0),
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "score": round(result.get("score", 0.0), 4),
        })
    return sources
