"""工业实训分类到 LightRAG 路由的离线回归测试。"""

from app.services.conversation.conversation_nodes import ConversationNodes
from app.services.conversation.intent_routing import (
    AnswerMode,
    resolve_answer_mode,
)
from app.services.query_classifier import ClassificationLabel, QueryClassifier


def test_classifier_labels_keep_industrial_training_and_drop_math_concept_labels():
    labels = set(ClassificationLabel.__args__)
    assert "industrial_training_query" in labels
    assert "math_problem" not in labels
    assert "concept_explain" not in labels


def test_industrial_training_label_routes_to_rag():
    assert resolve_answer_mode("industrial_training_query") is AnswerMode.RAG_WITH_FALLBACK
    assert ConversationNodes.route_after_classification(
        {"intent": "general_query", "is_realtime_query": False, "answer_mode": "rag_with_fallback"}
    ) == "rag"


def test_classifier_prompt_describes_industrial_training_questions():
    assert "industrial_training_query" in QueryClassifier.SYSTEM_PROMPT
    assert "平铺珐琅工艺" in QueryClassifier.SYSTEM_PROMPT


def test_removed_classifier_label_falls_back_to_general_knowledge():
    classifier = QueryClassifier.__new__(QueryClassifier)
    result = classifier._parse_llm_response(
        '{"label": "legacy_label", "confidence": "high", "reason": "old model"}'
    )
    assert result.label == "general_knowledge"
