"""
post_classification_preprocess 节点单测。

验证 ASR→LaTeX 转换时机重构后的核心行为：
- 数学题（分类命中）会调用 word_to_latex 并改写 user_query；
- 非数学题不调用 word_to_latex；
- 转换失败 / 无变化时保留原 query，不中断流程；
- answer_mode == math_llm 这一信号也能独立触发转换（兼容路由表解析路径）。

节点真实实例化成本高（依赖整个 workflow），但 post_classification_preprocess
不访问 self.workflow，因此用 ConversationNodes(workflow_instance=None) 轻量构造，
并通过 monkeypatch 替换 word_to_latex，避免外部 LLM 调用。
"""
import pytest

from app.services.conversation.conversation_nodes import ConversationNodes
from app.services.conversation.intent_routing import AnswerMode


WORD_TO_LATEX_PATH = "app.services.conversation.conversation_nodes.word_to_latex"


@pytest.fixture
def nodes():
    """轻量构造 ConversationNodes：被测节点不访问 self.workflow，传 None 即可。"""
    return ConversationNodes(workflow_instance=None)


@pytest.mark.asyncio
async def test_post_classification_preprocess_converts_math_query(monkeypatch, nodes):
    """数学题分类命中后应调用 word_to_latex 并改写 user_query。"""
    converted_text = "已知10件不同产品中有4件是次品。(1) 若..."

    async def fake_word_to_latex(query):
        return converted_text

    monkeypatch.setattr(WORD_TO_LATEX_PATH, fake_word_to_latex)

    original_query = "已知十件不同产品中有四件是次品。第一问..."
    state = {
        "user_query": original_query,
        "classification_label": "math_problem",
        "is_math_problem": True,
        "answer_mode": AnswerMode.MATH_LLM.value,
    }

    new_state = await nodes.post_classification_preprocess(state)

    assert new_state["asr_latex_should_run"] is True
    assert new_state["asr_latex_converted"] is True
    assert new_state["query_preprocessed"] is True
    assert "10件" in new_state["user_query"]
    assert new_state["asr_latex_before"] == original_query
    assert new_state["asr_latex_after"] == converted_text


@pytest.mark.asyncio
async def test_post_classification_preprocess_skips_non_math_query(monkeypatch, nodes):
    """非数学题不应调用 word_to_latex，user_query 保持不变。"""
    called = {"value": False}

    async def fake_word_to_latex(query):
        called["value"] = True
        return query

    monkeypatch.setattr(WORD_TO_LATEX_PATH, fake_word_to_latex)

    state = {
        "user_query": "今天天气怎么样？",
        "classification_label": "general_knowledge",
        "is_math_problem": False,
        "answer_mode": AnswerMode.GENERAL_LLM.value,
    }

    new_state = await nodes.post_classification_preprocess(state)

    assert called["value"] is False
    assert new_state["asr_latex_should_run"] is False
    assert new_state["asr_latex_converted"] is False
    assert new_state["query_preprocessed"] is False
    assert new_state["user_query"] == "今天天气怎么样？"


@pytest.mark.asyncio
async def test_post_classification_preprocess_keeps_original_on_failure(monkeypatch, nodes):
    """word_to_latex 抛异常时应保留原 query，且不中断流程。"""

    async def fake_word_to_latex(query):
        raise RuntimeError("mock failure")

    monkeypatch.setattr(WORD_TO_LATEX_PATH, fake_word_to_latex)

    original_query = "第一问求概率。"
    state = {
        "user_query": original_query,
        "classification_label": "math_problem",
        "is_math_problem": True,
        "answer_mode": AnswerMode.MATH_LLM.value,
    }

    new_state = await nodes.post_classification_preprocess(state)

    assert new_state["user_query"] == original_query
    assert new_state["asr_latex_converted"] is False
    assert new_state["query_preprocessed"] is False
    # 异常路径下 before/after 不应被写入
    assert "asr_latex_before" not in new_state


@pytest.mark.asyncio
async def test_post_classification_preprocess_converts_on_answer_mode_only(monkeypatch, nodes):
    """label 未判为 math_problem，但 answer_mode == math_llm 时也应触发转换。

    覆盖 classify_query_type 内部不同写入路径：某些情况下 is_math_problem 未置位、
    仅靠路由表 INTENT_TO_ANSWER_MODE 解析出 math_llm，此时仍需转换。
    """

    async def fake_word_to_latex(query):
        return "转换后文本"

    monkeypatch.setattr(WORD_TO_LATEX_PATH, fake_word_to_latex)

    state = {
        "user_query": "求该排列组合的方法数。",
        "classification_label": "general_knowledge",
        "is_math_problem": False,
        "answer_mode": AnswerMode.MATH_LLM.value,
    }

    new_state = await nodes.post_classification_preprocess(state)

    assert new_state["asr_latex_should_run"] is True
    assert new_state["asr_latex_converted"] is True
    assert new_state["user_query"] == "转换后文本"


@pytest.mark.asyncio
async def test_post_classification_preprocess_noop_when_unchanged(monkeypatch, nodes):
    """转换结果与原文相同时，不应标记为已转换（避免前端误判为预处理过）。"""

    async def fake_word_to_latex(query):
        return query  # 原样返回

    monkeypatch.setattr(WORD_TO_LATEX_PATH, fake_word_to_latex)

    original_query = "求x的值。"
    state = {
        "user_query": original_query,
        "classification_label": "math_problem",
        "is_math_problem": True,
        "answer_mode": AnswerMode.MATH_LLM.value,
    }

    new_state = await nodes.post_classification_preprocess(state)

    assert new_state["asr_latex_converted"] is False
    assert new_state["query_preprocessed"] is False
    assert new_state["user_query"] == original_query
