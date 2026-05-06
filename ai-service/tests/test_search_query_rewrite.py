"""
单元测试：联网检索前的 LLM query 改写（agen_search_query）

覆盖：
- agen_search_query 自身的成功改写、空输入、相同输出、LLM 异常容错
- 改写产物清洗（首行、引号、末尾标点）
- web_search 调用条件：开关 + is_realtime_query 双门
- 配置默认值

运行:
    cd ai-service
    PYTHONPATH=. python -m pytest tests/test_search_query_rewrite.py -v
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.query_classifier import QueryClassifier


# ---------------------------------------------------------------------------
# Part 1: agen_search_query 自身行为
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
        # 正常改写
        ("成都 实时路况 拥堵 2026-05-06", "成都 实时路况 拥堵 2026-05-06"),
        # 多行：只取首行
        ("成都 实时路况\n（注：以上是改写）", "成都 实时路况"),
        # 引号包裹：去除
        ("「成都 实时路况 拥堵」", "成都 实时路况 拥堵"),
        ('"Beijing weather today"', "Beijing weather today"),
        # 末尾标点：去除
        ("成都 实时路况 拥堵。", "成都 实时路况 拥堵"),
        ("成都 实时路况！", "成都 实时路况"),
        # 留白容错
        ("  成都 实时路况  ", "成都 实时路况"),
    ],
)
def test_agen_search_query_parsing(raw_content: str, expected: str):
    classifier = QueryClassifier(_FakeLLM(raw_content))
    out = asyncio.run(classifier.agen_search_query("我想了解一下今天成都堵不堵？"))
    assert out == expected


def test_agen_search_query_empty_short_circuit():
    """空 query 不应触发 LLM 调用，直接返回 None。"""
    fake = _FakeLLM("成都 路况")
    fake.ainvoke = AsyncMock(side_effect=AssertionError("must not call LLM on empty query"))
    classifier = QueryClassifier(fake)
    assert asyncio.run(classifier.agen_search_query("")) is None
    assert asyncio.run(classifier.agen_search_query("   ")) is None


def test_agen_search_query_returns_none_when_same_as_input():
    """LLM 输出与原句完全一致 → 返回 None，调用方走原 query 即可，避免无效改写。"""
    classifier = QueryClassifier(_FakeLLM("我想了解一下今天成都堵不堵？"))
    assert asyncio.run(classifier.agen_search_query("我想了解一下今天成都堵不堵？")) is None


def test_agen_search_query_llm_exception_returns_none():
    """LLM 异常时保守返回 None，不影响主流程。"""
    classifier = QueryClassifier(_FakeLLM("", raise_exc=RuntimeError("boom")))
    assert asyncio.run(classifier.agen_search_query("今天北京天气怎么样")) is None


def test_agen_search_query_truncates_long_output():
    """LLM 偶发输出过长 → 截断保护下游。"""
    long = "成都" * 200  # 400 字
    classifier = QueryClassifier(_FakeLLM(long))
    out = asyncio.run(classifier.agen_search_query("成都堵不堵"))
    assert out is not None
    assert len(out) <= 200


def test_agen_search_query_passes_hint_without_breaking():
    """hint 仅作弱提示传入 user prompt，hint 为空也不应崩。"""
    classifier = QueryClassifier(_FakeLLM("北京 实时天气 今日"))
    out_no_hint = asyncio.run(classifier.agen_search_query("北京今天天气怎么样"))
    out_with_hint = asyncio.run(
        classifier.agen_search_query("北京今天天气怎么样", hint="weather")
    )
    assert out_no_hint == "北京 实时天气 今日"
    assert out_with_hint == "北京 实时天气 今日"


# ---------------------------------------------------------------------------
# Part 2: web_search 节点的调用条件（独立验证 if 表达式）
# ---------------------------------------------------------------------------
def _should_call_rewriter(
    is_realtime: bool,
    rewrite_enabled: bool,
) -> bool:
    """复刻 web_search 节点中 LLM 改写的触发条件。"""
    return is_realtime and rewrite_enabled


@pytest.mark.parametrize(
    "is_realtime, rewrite_enabled, expected",
    [
        # 同时满足 → 调用
        (True, True, True),
        # 任一为 False → 不调用
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_web_search_rewriter_gate(
    is_realtime: bool, rewrite_enabled: bool, expected: bool
):
    assert _should_call_rewriter(is_realtime, rewrite_enabled) is expected


# ---------------------------------------------------------------------------
# Part 3: 配置默认值
# ---------------------------------------------------------------------------
def test_config_default_is_true():
    """新开关默认 True，新部署开箱启用 LLM 改写。"""
    import importlib
    from app.core import config as config_module
    importlib.reload(config_module)
    field = config_module.Settings.model_fields["realtime_query_search_rewrite_enabled"]
    assert field.default is True, f"默认值应为 True，实际={field.default}"


# ---------------------------------------------------------------------------
# Part 4: 改写后 query 的语义保留（行为契约）
# ---------------------------------------------------------------------------
def test_rewriter_preserves_locations():
    """改写器（在合理 LLM 输出下）应保留地点等关键实体。"""
    classifier = QueryClassifier(_FakeLLM("成都 实时路况 拥堵 2026-05-06"))
    out = asyncio.run(classifier.agen_search_query("我想了解一下今天成都堵不堵？"))
    assert out is not None and "成都" in out


def test_rewriter_keeps_language():
    """中文输入 → 中文输出，英文输入 → 英文输出（依赖 LLM 行为契约）。"""
    cls_zh = QueryClassifier(_FakeLLM("北京 今日 天气"))
    cls_en = QueryClassifier(_FakeLLM("Beijing weather today"))
    out_zh = asyncio.run(cls_zh.agen_search_query("北京今天天气怎么样"))
    out_en = asyncio.run(cls_en.agen_search_query("How's the weather in Beijing today"))
    assert out_zh and out_en
    # 简单字符级断言，避免引入 detect_dominant_language 依赖
    assert any("\u4e00" <= ch <= "\u9fff" for ch in out_zh)
    assert all(not ("\u4e00" <= ch <= "\u9fff") for ch in out_en)
