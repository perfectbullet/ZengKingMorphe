"""ConversationNodes.preprocess_query 的工训术语归一化接入测试。"""

import pytest

from app.services.conversation.conversation_nodes import ConversationNodes


@pytest.mark.asyncio
async def test_preprocess_query_normalizes_before_classification_fields():
    nodes = ConversationNodes(workflow_instance=None)
    state = {"user_query": "简述石膏灌浆在湿蜡铸造中的作用。"}

    result = await nodes.preprocess_query(state)

    assert result["user_query"] == "简述石膏灌浆在失蜡铸造中的作用。"
    assert result["industrial_term_normalized"] is True
    assert result["industrial_term_before"] == "简述石膏灌浆在湿蜡铸造中的作用。"
    assert result["industrial_term_after"] == "简述石膏灌浆在失蜡铸造中的作用。"
    assert result["industrial_term_matches"] == [
        {
            "rule_id": "lost_wax_casting",
            "source": "湿蜡铸造",
            "target": "失蜡铸造",
            "start": 7,
            "end": 11,
            "priority": 100,
        }
    ]
    assert result["query_preprocessed"] is True
    assert result["prefer_zh_output"] is True
