from app.services.conversation.intent_routing import AnswerMode, resolve_answer_mode


def test_math_concept_explain_routes_to_lightrag():
    assert resolve_answer_mode("math_concept_explain") is AnswerMode.RAG_WITH_FALLBACK


def test_removed_concept_labels_fall_back_to_general_llm():
    assert resolve_answer_mode("concept_explain") is AnswerMode.GENERAL_LLM
    assert resolve_answer_mode("industrial_training_query") is AnswerMode.GENERAL_LLM
