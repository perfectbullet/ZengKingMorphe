"""
上下文消歧拆分链路单测。

验证重构后的核心不变量（上下文消歧从 classify_query_type 独立为
resolve_context_query 节点）：

1. 完整数学题跳过上下文消歧（不调用 aresolve_standalone_query /
   aclassify_context_dependence，不改写题干）；
2. 数学追问使用历史上下文（math_context_text 非空），但不调用普通 resolver
   改写题干；
3. 非数学追问走普通 resolver 生成 standalone query（query_rewritten=True）；
4. word_to_latex 只在 post_classification_preprocess 调用，
   resolve_context_query / finalize_classification 均不调用；
5. workflow 顺序为 classify_query_type → resolve_context_query
   → finalize_classification → post_classification_preprocess。

被测节点（resolve_context_query / finalize_classification /
post_classification_preprocess）均不访问 self.workflow，故用
ConversationNodes(workflow_instance=None) 轻量构造，并通过 monkeypatch
替换 get_query_classifier / word_to_latex，避免外部 LLM 调用。
"""
import pytest

from app.services.query_classifier import ClassificationResult
from app.services.conversation.conversation_nodes import ConversationNodes

GET_CLASSIFIER_PATH = "app.services.conversation.conversation_nodes.get_query_classifier"
WORD_TO_LATEX_PATH = "app.services.conversation.conversation_nodes.word_to_latex"

# 完整数学题样本（长题干，含"已知/求"，确保被判为完整题而非追问）。
COMPLETE_MATH_QUERY = (
    "已知a向量、b向量、e向量是平面向量，e向量是单位向量。"
    "若非零向量a向量与e向量的夹角为π/3，求a向量在e向量方向上的投影数量。"
)
# 数学追问样本（短问句，含追问标记词）。
MATH_FOLLOWUP_QUERY = "第二问怎么做？"
# 非数学追问样本。
NON_MATH_FOLLOWUP_QUERY = "那怎么优化？"


class _FakeClassifier:
    """记录调用行为的假分类器，避免真实 LLM 调用。"""

    def __init__(self, dependence=None, resolved=None):
        # dependence: (is_related, reason) 或 None（默认 unrelated）
        self._dependence = dependence
        self._resolved = resolved
        self.calls = {
            "aclassify": 0,
            "aclassify_context_dependence": 0,
            "aresolve_standalone_query": 0,
        }

    async def aclassify(self, query, context_query=None):
        self.calls["aclassify"] += 1
        return ClassificationResult(label="other", confidence="medium", reason="fake")

    async def aclassify_context_dependence(self, query, dialog_text):
        self.calls["aclassify_context_dependence"] += 1
        return self._dependence or (False, "fake")

    async def aresolve_standalone_query(self, latest_query, dialog_text):
        self.calls["aresolve_standalone_query"] += 1
        return self._resolved or latest_query

    async def aneed_realtime(self, query):
        return False

    async def agen_search_query(self, query, hint=None):
        return query


@pytest.fixture
def nodes():
    """轻量构造 ConversationNodes：被测节点不访问 self.workflow，传 None 即可。"""
    return ConversationNodes(workflow_instance=None)


# =============================================================================
# 1. 完整数学题不调用 resolver
# =============================================================================
@pytest.mark.asyncio
async def test_complete_math_skips_resolver(monkeypatch, nodes):
    """完整数学题（raw_label=math_problem + 长题干）应跳过上下文消歧。"""
    fake = _FakeClassifier(dependence=(True, "llm_yes"), resolved="不应被改写")
    monkeypatch.setattr(GET_CLASSIFIER_PATH, lambda: fake)

    state = {
        "user_query": COMPLETE_MATH_QUERY,
        "raw_classification_label": "math_problem",
        "classification_label": "math_problem",
        "context": {
            "messages": [
                {"role": "user", "content": "上一题题干"},
                {"role": "assistant", "content": "上一题解答"},
            ]
        },
        "sources": [],
    }

    new_state = await nodes.resolve_context_query(state)

    assert new_state["context_resolution_mode"] == "skipped_complete_math"
    assert new_state["context_resolution_skipped_reason"] == "complete_math_problem"
    assert new_state["context_dependence"] == "unrelated"
    assert new_state["query_rewritten"] is False
    assert new_state["effective_query"] == COMPLETE_MATH_QUERY
    assert new_state["math_context_used"] is False
    assert new_state["math_context_text"] is None
    # 完整数学题既不调用普通 resolver，也不调用相关性判定
    assert fake.calls["aresolve_standalone_query"] == 0
    assert fake.calls["aclassify_context_dependence"] == 0


# =============================================================================
# 2. 数学追问使用上下文但不调用普通 resolver
# =============================================================================
@pytest.mark.asyncio
async def test_math_followup_uses_context_without_resolver(monkeypatch, nodes):
    """数学追问应设置 math_context_text，但不改写题干、不调用普通 resolver。"""
    fake = _FakeClassifier(dependence=(True, "llm_yes"), resolved="不应被改写")
    monkeypatch.setattr(GET_CLASSIFIER_PATH, lambda: fake)

    state = {
        "user_query": MATH_FOLLOWUP_QUERY,
        "raw_classification_label": "math_problem",
        "classification_label": "math_problem",
        "context": {
            "messages": [
                {"role": "user", "content": "已知圆锥底面半径为1，高为2，求侧面积。"},
                {"role": "assistant", "content": "侧面积是 sqrt(5)*pi。"},
            ]
        },
        "sources": [],
    }

    new_state = await nodes.resolve_context_query(state)

    assert new_state["context_resolution_mode"] == "math_context_only"
    assert new_state["math_context_used"] is True
    assert new_state["math_context_text"]  # 非空
    assert new_state["query_rewritten"] is False
    assert new_state["effective_query"] == MATH_FOLLOWUP_QUERY
    # 数学追问不调用普通 resolver，也不走相关性判定
    assert fake.calls["aresolve_standalone_query"] == 0
    assert fake.calls["aclassify_context_dependence"] == 0


# =============================================================================
# 3. 非数学追问可以调用 resolver
# =============================================================================
@pytest.mark.asyncio
async def test_non_math_followup_uses_normal_resolver(monkeypatch, nodes):
    """非数学追问走普通上下文消歧，调用 aresolve_standalone_query 改写。"""
    fake = _FakeClassifier(
        dependence=(True, "llm_yes"),
        resolved="Qwen3-32B推理速度慢怎么优化",
    )
    monkeypatch.setattr(GET_CLASSIFIER_PATH, lambda: fake)

    state = {
        "user_query": NON_MATH_FOLLOWUP_QUERY,
        "raw_classification_label": "general_knowledge",
        "classification_label": "general_knowledge",
        "context": {
            "messages": [
                {"role": "user", "content": "Qwen3-32B推理速度慢"},
                {"role": "assistant", "content": "可以考虑量化或减少批大小。"},
            ]
        },
        "sources": [],
    }

    new_state = await nodes.resolve_context_query(state)

    assert new_state["context_resolution_mode"] == "normal_resolver"
    assert new_state["query_rewritten"] is True
    assert new_state["effective_query"] == "Qwen3-32B推理速度慢怎么优化"
    assert fake.calls["aclassify_context_dependence"] == 1
    assert fake.calls["aresolve_standalone_query"] == 1


# =============================================================================
# 4. post_classification_preprocess 是数学格式转换唯一位置
# =============================================================================
@pytest.mark.asyncio
async def test_word_to_latex_only_in_post_classification(monkeypatch, nodes):
    """resolve_context_query / finalize_classification 均不调用 word_to_latex，
    仅 post_classification_preprocess 调用。"""
    called = {"count": 0}

    async def fake_word_to_latex(query):
        called["count"] += 1
        return f"\\({query}\\)"

    monkeypatch.setattr(WORD_TO_LATEX_PATH, fake_word_to_latex)

    fake = _FakeClassifier(dependence=(True, "llm_yes"), resolved="x")
    monkeypatch.setattr(GET_CLASSIFIER_PATH, lambda: fake)

    # resolve_context_query：完整数学题，不应调用 word_to_latex
    state_resolve = {
        "user_query": COMPLETE_MATH_QUERY,
        "raw_classification_label": "math_problem",
        "classification_label": "math_problem",
        "context": {"messages": []},
        "sources": [],
    }
    await nodes.resolve_context_query(state_resolve)
    assert called["count"] == 0

    # finalize_classification：基于已消歧 state 做最终分类，不应调用 word_to_latex
    state_finalize = {
        "user_query": COMPLETE_MATH_QUERY,
        "effective_query": COMPLETE_MATH_QUERY,
        "raw_classification_label": "math_problem",
        "raw_classification_confidence": "high",
        "raw_classification_reason": "test",
        "sources": [],
    }
    await nodes.finalize_classification(state_finalize)
    assert called["count"] == 0

    # post_classification_preprocess：数学题分类命中，应调用 word_to_latex
    state_post = {
        "user_query": COMPLETE_MATH_QUERY,
        "classification_label": "math_problem",
        "is_math_problem": True,
        "answer_mode": "math_llm",
    }
    await nodes.post_classification_preprocess(state_post)
    assert called["count"] == 1


# =============================================================================
# 5. workflow 顺序测试
# =============================================================================
def _extract_edges(compiled):
    """从 compiled graph 提取 (source, target) 顺序边集合。

    LangGraph 的 conditional edge 的 target 可能不是 str，这里只保留
    两端都是 str 的普通顺序边。
    """
    g = compiled.get_graph()
    pairs = set()
    for e in g.edges:
        src = getattr(e, "source", None)
        tgt = getattr(e, "target", None)
        if isinstance(src, str) and isinstance(tgt, str):
            pairs.add((src, tgt))
    return pairs


def test_workflow_order_includes_new_nodes():
    """确认 graph 顺序包含 classify_query_type → resolve_context_query
    → finalize_classification → post_classification_preprocess。

    复用模块级 conversation_workflow 单例（import 时已实例化），避免重建 LLM。
    """
    from app.services.conversation_service import conversation_workflow

    compiled = conversation_workflow.workflow

    node_names = set(compiled.get_graph().nodes.keys())
    for n in (
        "classify_query_type",
        "resolve_context_query",
        "finalize_classification",
        "post_classification_preprocess",
    ):
        assert n in node_names, f"workflow 缺少节点: {n}"

    edges = _extract_edges(compiled)
    assert ("classify_query_type", "resolve_context_query") in edges
    assert ("resolve_context_query", "finalize_classification") in edges
    assert ("finalize_classification", "post_classification_preprocess") in edges
