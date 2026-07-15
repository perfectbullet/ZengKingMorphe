"""post_classification_preprocess 的工训兼容钩子测试。"""

import pytest

from app.services.conversation.conversation_nodes import ConversationNodes


@pytest.fixture
def nodes():
    return ConversationNodes(workflow_instance=None)


@pytest.mark.asyncio
async def test_post_classification_preprocess_preserves_prior_normalization(nodes):
    state = {
        "user_query": "失蜡铸造的工序是什么？",
        "industrial_term_normalized": True,
        "query_preprocessed": True,
    }

    result = await nodes.post_classification_preprocess(state)

    assert result["user_query"] == "失蜡铸造的工序是什么？"
    assert result["industrial_term_normalized"] is True
    assert result["query_preprocessed"] is True
    assert result["asr_latex_should_run"] is False
    assert result["asr_latex_converted"] is False


@pytest.mark.asyncio
async def test_post_classification_preprocess_does_not_mark_unmodified_query(nodes):
    state = {"user_query": "今天天气怎么样？", "industrial_term_normalized": False}

    result = await nodes.post_classification_preprocess(state)

    assert result["user_query"] == "今天天气怎么样？"
    assert result["query_preprocessed"] is False
    assert result["asr_latex_should_run"] is False
    assert result["asr_latex_converted"] is False
