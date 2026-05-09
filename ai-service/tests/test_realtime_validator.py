"""
单元测试：实时查询二元 LLM 校验兜底（realtime_query_llm_fallback）

覆盖场景：
- aneed_realtime 自身的 yes/no 解析（含异常容错）
- classify_query_type 兜底分支：
  1. 主分类误判 general_knowledge / chit_chat / concept_explain / other
     + confidence != "high" + 二元 LLM 返 yes → 升级 realtime_query
  2. math_problem / greeting / noise / realtime_query 永远不触发兜底
  3. 主分类 confidence == "high" 时不触发
  4. 二元 LLM 返 no 时不升级
  5. 开关关闭时不调用 aneed_realtime
  6. preserve_original_intent 命中时不触发

运行:
    cd ai-service
    PYTHONPATH=. python -m pytest tests/test_realtime_validator.py -v
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.query_classifier import (
    ClassificationResult,
    QueryClassifier,
)


# ---------------------------------------------------------------------------
# Part 1: aneed_realtime 自身行为
# ---------------------------------------------------------------------------
class _FakeLLM:
    """最小化 ainvoke 的 mock。"""
    def __init__(self, content: str, raise_exc: Exception | None = None):
        self._content = content
        self._raise = raise_exc

    def with_config(self, **_kwargs):
        return self

    async def ainvoke(self, _messages):
        if self._raise:
            raise self._raise
        return SimpleNamespace(content=self._content)


@pytest.mark.parametrize(
    "raw_content, expected",
    [
        ("yes", True),
        ("Yes", True),
        ("YES", True),
        ("yes.", True),       # 容错：标点
        ("yes\n", True),      # 容错：换行
        ("yes，需要联网", True),  # 容错：模型多吐字
        ("no", False),
        ("No.", False),
        ("不需要", False),
        ("", False),         # 空响应保守判 False
        ("maybe", False),    # 不以 yes 开头一律 False
    ],
)
def test_aneed_realtime_parsing(raw_content: str, expected: bool):
    classifier = QueryClassifier(_FakeLLM(raw_content))
    out = asyncio.run(classifier.aneed_realtime("今天成都堵不堵？"))
    assert out is expected, f"raw={raw_content!r} -> {out}, expected {expected}"


def test_aneed_realtime_empty_query_short_circuit():
    """空 query 不应触发 LLM 调用，直接返回 False。"""
    fake = _FakeLLM("yes")
    fake.ainvoke = AsyncMock(side_effect=AssertionError("must not call LLM on empty query"))
    classifier = QueryClassifier(fake)
    assert asyncio.run(classifier.aneed_realtime("")) is False
    assert asyncio.run(classifier.aneed_realtime("   ")) is False


def test_aneed_realtime_llm_exception_returns_false():
    """LLM 调用异常时保守返回 False，不影响主流程。"""
    classifier = QueryClassifier(_FakeLLM("", raise_exc=RuntimeError("boom")))
    assert asyncio.run(classifier.aneed_realtime("今天成都堵不堵？")) is False


# ---------------------------------------------------------------------------
# Part 2: classify_query_type 兜底分支（直接构造 if 条件，避免依赖整个 workflow）
# ---------------------------------------------------------------------------
# 这部分不实例化整个 ConversationNodes（其依赖 db / employee config），
# 而是直接用 conversation_nodes.py 中相同的 if 条件做断言，
# 避免侵入 workflow 又能覆盖兜底分支的所有真值组合。
from app.services.conversation.conversation_nodes import _REALTIME_HEURISTIC_SKIP_LABELS


def _should_trigger_fallback(
    label: str, confidence: str, preserve_original_intent: bool, fallback_enabled: bool
) -> bool:
    """复刻节点中的 if 表达式（除最末的 await aneed_realtime() 之外的所有条件）。"""
    return (
        fallback_enabled
        and label not in _REALTIME_HEURISTIC_SKIP_LABELS
        and confidence != "high"
        and not preserve_original_intent
    )


@pytest.mark.parametrize(
    "label, expected_can_trigger",
    [
        ("general_knowledge", True),
        ("concept_explain", True),
        ("chit_chat", True),
        ("english_query", True),
        ("other", True),
        # 不应触发：math_problem 必须保持走 RAG / Phi-4
        ("math_problem", False),
        # 不应触发：greeting / noise / realtime_query 已被早期分支处理
        ("greeting", False),
        ("noise", False),
        ("realtime_query", False),
    ],
)
def test_skip_labels_protected(label: str, expected_can_trigger: bool):
    can = _should_trigger_fallback(
        label=label,
        confidence="medium",
        preserve_original_intent=False,
        fallback_enabled=True,
    )
    assert can is expected_can_trigger, (
        f"label={label} 期望 can_trigger={expected_can_trigger}，实际={can}"
    )


def test_high_confidence_skipped():
    """主分类高置信度时不应触发兜底，避免覆盖明确判断。"""
    can = _should_trigger_fallback(
        label="general_knowledge",
        confidence="high",
        preserve_original_intent=False,
        fallback_enabled=True,
    )
    assert can is False


def test_preserve_original_intent_skipped():
    """preserve_original_intent 命中时跳过兜底。"""
    can = _should_trigger_fallback(
        label="general_knowledge",
        confidence="medium",
        preserve_original_intent=True,
        fallback_enabled=True,
    )
    assert can is False


def test_disabled_switch_skipped():
    """开关关闭时彻底跳过。"""
    can = _should_trigger_fallback(
        label="general_knowledge",
        confidence="medium",
        preserve_original_intent=False,
        fallback_enabled=False,
    )
    assert can is False


@pytest.mark.parametrize(
    "validator_returns, prev_label, expected_promoted",
    [
        # 触发条件齐全 + 二元 yes → 升级
        (True, "general_knowledge", True),
        (True, "concept_explain", True),
        (True, "chit_chat", True),
        # 触发条件齐全 + 二元 no → 不升级
        (False, "general_knowledge", False),
    ],
)
def test_full_path_promotion_logic(validator_returns: bool, prev_label: str, expected_promoted: bool):
    """模拟节点完整 if 表达式 (含 await aneed_realtime) 的最终结论。"""
    can_trigger = _should_trigger_fallback(
        label=prev_label,
        confidence="medium",
        preserve_original_intent=False,
        fallback_enabled=True,
    )
    promoted = can_trigger and validator_returns
    assert promoted is expected_promoted


# ---------------------------------------------------------------------------
# Part 3: 兜底升级后的 ClassificationResult 形态
# ---------------------------------------------------------------------------
def test_promoted_result_shape():
    """升级后产生的结果应符合下游路由对 realtime_query 的期待。"""
    prev = ClassificationResult(label="general_knowledge", confidence="medium", reason="百科")
    promoted = ClassificationResult(label="realtime_query", confidence=prev.confidence, reason="general")
    assert promoted.label == "realtime_query"
    assert promoted.confidence == "medium"
    # reason="general" 经下游 _normalize_realtime_category 后会得到 "general"，
    # 后续路由会走 web_search 而非 RAG。
    assert promoted.reason == "general"


# ---------------------------------------------------------------------------
# Part 4: 配置默认值
# ---------------------------------------------------------------------------
def test_config_default_is_true():
    """config.py 中默认值应为 True，确保新部署开箱启用兜底。"""
    import importlib
    from app.core import config as config_module
    importlib.reload(config_module)
    # 取字段定义而不是 settings 单例，避免被 .env 覆盖
    field = config_module.Settings.model_fields["realtime_query_llm_fallback_enabled"]
    assert field.default is True, f"默认值应为 True，实际={field.default}"
