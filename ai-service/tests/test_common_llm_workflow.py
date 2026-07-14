"""Focused regression checks for the single common-LLM runtime path."""

import asyncio
import inspect

from app.api.endpoints import chat_stream_v1
from app.services.conversation.conversation_helpers import build_generation_messages
from app.services.conversation.conversation_nodes import ConversationNodes
from app.services.conversation.conversation_state import ConversationState
from app.services.query_classifier import ClassificationLabel, QueryClassifier


def _state(**updates):
    base = dict(chat_stream_v1._STATE_DEFAULTS)
    base.update({"user_query": "求解 x²-5x+6<0", "employee_config": {}, "context": {}})
    base.update(updates)
    return base


def test_only_realtime_changes_workflow_branch():
    assert ConversationNodes.route_after_classification(_state(is_realtime_query=True)) == "realtime"
    assert ConversationNodes.route_after_classification(_state(is_realtime_query=False)) == "general"


def test_math_and_training_queries_have_no_special_labels():
    labels = set(ClassificationLabel.__args__)
    assert {"greeting", "english_query", "realtime_query", "general_knowledge", "chit_chat", "noise", "other"} == labels


def test_legacy_classifier_output_falls_back_to_other():
    classifier = QueryClassifier.__new__(QueryClassifier)
    result = classifier._parse_llm_response('{"label":"obsolete", "confidence":"high"}')
    assert result.label == "other"


def test_general_prompt_handles_noise_with_llm_instruction():
    messages = build_generation_messages(_state(classification_label="noise", user_query="嗯"))
    assert "补充信息" in messages[0].content


def test_realtime_failure_is_a_prompt_instruction_not_fixed_answer():
    messages = build_generation_messages(_state(
        user_query="北京今天天气怎么样", is_realtime_query=True, web_search_used=False,
        web_search_error="failed",
    ))
    assert "暂时无法确认" in messages[0].content


def test_generate_answer_configures_only_general_streaming_llm():
    class Workflow:
        def get_streaming_llm(self, state):
            return "llm", "general-model"

    nodes = ConversationNodes(Workflow())
    state = _state()
    async def run():
        return await nodes.generate_answer(state)
    result = asyncio.run(run())
    assert result["streaming_llm"] == "llm"
    assert result["streaming_messages"]


def test_v1_has_one_normal_astream_loop_and_no_legacy_mode_tokens():
    source = inspect.getsource(chat_stream_v1)
    assert source.count("streaming_llm.astream(messages)") == 1
    for legacy in ("rag_stream", "direct_text", "preset_response", "streaming_type"):
        assert legacy not in source


def test_state_has_no_special_runtime_fields():
    fields = set(ConversationState.__annotations__)
    for field in ("answer_mode", "direct_text_answer", "streaming_type", "rag_query", "is_math_problem"):
        assert field not in fields
