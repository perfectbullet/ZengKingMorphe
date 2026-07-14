"""LangGraph workflow for the common streaming conversation path."""

import os

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.core.config import settings
from app.core.logging import get_logger
from app.services.conversation.conversation_nodes import ConversationNodes
from app.services.conversation.conversation_state import ConversationState, GREETING_KEYWORDS
from app.services.conversation.intent_routing import (
    ROUTE_BRANCH_GENERAL,
    ROUTE_BRANCH_REALTIME,
)

logger = get_logger(__name__)


class ConversationWorkflow:
    """Classify, optionally search the web, then stream one general LLM."""

    def __init__(self):
        llm_base_url = os.getenv("LLM_BASE_URL")
        llm_model = os.getenv("LLM_MODEL")
        llm_api_key = os.getenv("LLM_API_KEY", "no-key")
        self.local_llm = ChatOpenAI(
            base_url=llm_base_url,
            api_key=llm_api_key,
            model=llm_model,
            streaming=True,
        )
        self.remote_llm = ChatOpenAI(
            base_url=llm_base_url,
            api_key=llm_api_key,
            model=llm_model,
            temperature=settings.openai_temperature,
            streaming=True,
        )
        logger.info(f"General LLM configured | model={llm_model} | base_url={llm_base_url}")
        self.nodes = ConversationNodes(self)
        self.workflow = self._build_workflow()

    def get_active_llm(self, state: ConversationState):
        from app.services.conversation.conversation_helpers import select_llm

        return select_llm(state, self.local_llm, self.remote_llm)

    def get_streaming_llm(self, state: ConversationState):
        """Return the normal general-purpose streaming model."""
        return self.get_active_llm(state)

    def build_generation_messages(self, state: ConversationState):
        from app.services.conversation.conversation_helpers import build_generation_messages

        return build_generation_messages(state)

    async def save_conversation(self, state: ConversationState):
        return await self.nodes.save_conversation(state)

    def _build_workflow(self):
        graph = StateGraph(ConversationState)
        graph.add_node("load_employee_config", self.nodes.load_employee_config)
        graph.add_node("load_session_context", self.nodes.load_session_context)
        graph.add_node("input_validation", self.nodes.validate_input)
        graph.add_node("preprocess_query", self.nodes.preprocess_query)
        graph.add_node("classify_query_type", self.nodes.classify_query_type)
        graph.add_node("resolve_context_query", self.nodes.resolve_context_query)
        graph.add_node("finalize_classification", self.nodes.finalize_classification)
        graph.add_node("web_search", self.nodes.web_search)
        graph.add_node("generate_answer", self.nodes.generate_answer)
        graph.add_node("save_conversation", self.nodes.save_conversation)

        graph.set_entry_point("load_employee_config")
        graph.add_edge("load_employee_config", "load_session_context")
        graph.add_edge("load_session_context", "input_validation")
        graph.add_edge("input_validation", "preprocess_query")
        graph.add_edge("preprocess_query", "classify_query_type")
        graph.add_edge("classify_query_type", "resolve_context_query")
        graph.add_edge("resolve_context_query", "finalize_classification")
        graph.add_conditional_edges(
            "finalize_classification",
            self.nodes.route_after_classification,
            {
                ROUTE_BRANCH_REALTIME: "web_search",
                ROUTE_BRANCH_GENERAL: "generate_answer",
            },
        )
        graph.add_edge("web_search", "generate_answer")
        graph.add_edge("generate_answer", "save_conversation")
        graph.add_edge("save_conversation", END)
        return graph.compile()


conversation_workflow = ConversationWorkflow()
