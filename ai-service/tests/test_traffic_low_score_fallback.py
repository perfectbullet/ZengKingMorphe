"""
单元测试：traffic 类查询在 web 召回相关性低于阈值时，跳过 LLM 直接走 direct_text 模板

覆盖点：
- 低 score 命中（如 0.26 这种 Tavily 召回的无关广告页）→ 走 direct_text 兜底；
- 高 score 命中 → 走 LLM 路径（不影响命中真有用资料的情形）；
- 阈值可配（settings.realtime_traffic_min_web_score）；
- 中英文 direct_text 模板均含等级 / 时间 / 出行建议 / 推荐高德或百度地图；
- web_search_used=False（联网未开/未走）→ 沿用现有兜底；
- web_search_error 非空 → 沿用现有"网络不可用"提示，不被新逻辑覆盖；
- 非 traffic 类（如 weather）即使 score 低也不走该兜底，避免回归。

运行:
    cd ai-service
    PYTHONPATH=. python -m pytest tests/test_traffic_low_score_fallback.py -v
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app.services.conversation.conversation_nodes import ConversationNodes


# ---------------------------------------------------------------------------
# 辅助：构造最小可用的 state + 调用 generate_answer 节点
# ---------------------------------------------------------------------------
def _build_state(
    *,
    web_search_used: bool,
    web_results: list[dict] | None,
    realtime_category: str = "traffic",
    web_search_error: str | None = None,
    prefer_zh_output: bool = True,
) -> dict:
    state: dict = {
        "user_query": "我想了解一下今天成都堵不堵？",
        "rewritten_query": "今天成都的交通状况怎么样",
        "intent": "general_query",
        "is_realtime_query": True,
        "realtime_category": realtime_category,
        "web_search_used": web_search_used,
        "web_search_results": web_results or [],
        "retrieved_docs": [],
        "context": {"messages": []},
        "compressed_context": "",
        "prefer_zh_output": prefer_zh_output,
        "employee_config": {
            "name": "测试员工",
            "role": "客服",
            "description": "用于单元测试",
            "personality": {"tone": "friendly", "style": "concise", "formality": "moderate"},
            "greeting": "你好",
        },
        "sources": [],
        "node_timings": {},
    }
    if web_search_error is not None:
        state["web_search_error"] = web_search_error
    return state


def _run_generate_answer(state: dict) -> dict:
    """实例化最小化的 ConversationNodes，调用 generate_answer。"""
    nodes = ConversationNodes.__new__(ConversationNodes)
    nodes.workflow = None  # type: ignore[attr-defined]
    return asyncio.run(nodes.generate_answer(state))


# ---------------------------------------------------------------------------
# Part 1: 低 score 命中 → direct_text 兜底
# ---------------------------------------------------------------------------
def test_traffic_low_score_triggers_direct_text():
    """复现真实 bug：max_score=0.26 < 0.5 阈值时跳过 LLM。"""
    state = _build_state(
        web_search_used=True,
        web_results=[
            {"title": "小鹏快讯", "score": 0.26, "url": "https://example.com", "content": "广告"},
            {"title": "福特快讯", "score": 0.22, "url": "https://example.com", "content": "广告"},
        ],
    )
    result = _run_generate_answer(state)
    assert result["streaming_type"] == "direct_text"
    assert result.get("streaming_llm") is None
    assert result["direct_text_answer"]
    txt = result["direct_text_answer"]
    # 模板必含：等级 + 时间锚点 + 出行建议 + 推荐专业地图
    assert "整体路况判断" in txt or "整体" in txt
    assert "出行建议" in txt
    assert ("高德地图" in txt or "百度地图" in txt)


def test_traffic_low_score_fallback_includes_estimate_fields():
    """direct_text 必须含基于"时段先验"的等级 + 当前时间 + 星期。"""
    state = _build_state(
        web_search_used=True,
        web_results=[{"title": "无关", "score": 0.10}],
    )
    result = _run_generate_answer(state)
    est = result.get("traffic_estimate") or {}
    assert est, "traffic_estimate 应被注入"
    assert est["level"] in {"较拥堵", "中等拥堵", "基本通畅"}
    txt = result["direct_text_answer"]
    assert est["level"] in txt
    assert est["now_iso"] in txt
    assert est["weekday_zh"] in txt


# ---------------------------------------------------------------------------
# Part 2: 高 score 命中 → 不走 direct_text，回到 LLM 路径
# ---------------------------------------------------------------------------
def test_traffic_high_score_keeps_llm_path():
    """max_score=0.85 ≥ 0.5 阈值时不应触发 direct_text 兜底。"""
    state = _build_state(
        web_search_used=True,
        web_results=[
            {"title": "成都今日实时路况", "score": 0.85, "url": "https://map.example.com",
             "content": "二环高架 14:00 起车流量增大，南三环局部缓行..."},
        ],
    )
    # 使 LLM 路径需要的下游函数被 patch 掉，避免真实调用 ChatOllama。
    with patch("app.services.conversation.conversation_nodes.build_generation_messages",
               return_value=[]) as m_build, \
         patch.object(ConversationNodes, "__init__", lambda self: None):
        nodes = ConversationNodes()
        # 给 workflow.get_streaming_llm 一个 mock
        class _W:
            def get_streaming_llm(self, _state):
                return ("FAKE_LLM", "fake-model")
        nodes.workflow = _W()  # type: ignore[attr-defined]
        result = asyncio.run(nodes.generate_answer(state))

    assert result.get("streaming_type") == "langchain_llm", \
        f"高分应走 LLM 路径，实际 type={result.get('streaming_type')}"
    assert m_build.called, "build_generation_messages 应被调用（LLM 路径）"


# ---------------------------------------------------------------------------
# Part 3: 阈值可配
# ---------------------------------------------------------------------------
def test_threshold_is_configurable_via_settings():
    """阈值降到 0.05 后，0.10 的 score 不再被视作'不可信'。"""
    state = _build_state(
        web_search_used=True,
        web_results=[{"title": "无关", "score": 0.10}],
    )
    with patch("app.services.conversation.conversation_nodes.settings") as m_settings, \
         patch("app.services.conversation.conversation_nodes.build_generation_messages",
               return_value=[]):
        m_settings.realtime_traffic_min_web_score = 0.05
        nodes = ConversationNodes.__new__(ConversationNodes)
        class _W:
            def get_streaming_llm(self, _state):
                return ("FAKE_LLM", "fake-model")
        nodes.workflow = _W()  # type: ignore[attr-defined]
        result = asyncio.run(nodes.generate_answer(state))
    # 阈值放宽后不应再走 direct_text；走 LLM 路径
    assert result.get("streaming_type") != "direct_text" or \
        "整体路况判断" not in (result.get("direct_text_answer") or ""), \
        "阈值放宽后不应再触发 traffic 兜底模板"


# ---------------------------------------------------------------------------
# Part 4: 中英文模板差异
# ---------------------------------------------------------------------------
def test_traffic_low_score_english_template():
    state = _build_state(
        web_search_used=True,
        web_results=[{"title": "irrelevant", "score": 0.10}],
        prefer_zh_output=False,
    )
    result = _run_generate_answer(state)
    assert result["streaming_type"] == "direct_text"
    txt = result["direct_text_answer"]
    assert "Today's overall traffic" in txt
    assert "Tips:" in txt
    assert ("Amap" in txt or "Baidu" in txt)


# ---------------------------------------------------------------------------
# Part 5: 沿用现有兜底分支（向后兼容回归）
# ---------------------------------------------------------------------------
def test_web_search_error_keeps_existing_message():
    state = _build_state(
        web_search_used=False,
        web_results=[],
        web_search_error="ConnectionTimeout",
    )
    result = _run_generate_answer(state)
    assert result["streaming_type"] == "direct_text"
    assert "网络检索不可用" in result["direct_text_answer"]


def test_web_search_not_used_keeps_traffic_estimate_fallback():
    """web_search_used=False 时仍走原 traffic 兜底，但现在是升级版完整模板。"""
    state = _build_state(
        web_search_used=False,
        web_results=[],
    )
    result = _run_generate_answer(state)
    assert result["streaming_type"] == "direct_text"
    txt = result["direct_text_answer"]
    assert "出行建议" in txt
    assert ("高德地图" in txt or "百度地图" in txt)


# ---------------------------------------------------------------------------
# Part 6: 非 traffic 类不受影响（不回归 weather/news/market）
# ---------------------------------------------------------------------------
def test_non_traffic_category_low_score_does_not_fallback():
    """weather 类即使 score 低也不应走 traffic 兜底模板。"""
    state = _build_state(
        web_search_used=True,
        web_results=[{"title": "无关", "score": 0.10}],
        realtime_category="weather",
    )
    with patch("app.services.conversation.conversation_nodes.build_generation_messages",
               return_value=[]):
        nodes = ConversationNodes.__new__(ConversationNodes)
        class _W:
            def get_streaming_llm(self, _state):
                return ("FAKE_LLM", "fake-model")
        nodes.workflow = _W()  # type: ignore[attr-defined]
        result = asyncio.run(nodes.generate_answer(state))
    # weather 不命中 traffic 兜底；不应注入 traffic_estimate
    assert "traffic_estimate" not in result or not result.get("traffic_estimate")
    direct = result.get("direct_text_answer") or ""
    assert "出行建议" not in direct  # 模板不应误用到 weather
