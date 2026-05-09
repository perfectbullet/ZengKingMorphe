"""
单元测试：traffic 类查询的"时段先验"注入与 prompt 拼接

覆盖点：
- _traffic_congestion_estimate 在不同时段/星期下输出对齐预期等级；
- traffic_estimate 字典字段齐全（level / reason / now_iso / weekday_zh / weekday_en）；
- build_generation_messages 在 traffic 分支注入"时段先验"段落；
- 无 traffic_estimate 时不渲染先验段落，保留原有 prompt 行为；
- 中英文 prompt 都注入先验段落（语言自动同频）。

运行:
    cd ai-service
    PYTHONPATH=. python -m pytest tests/test_traffic_estimate_prior.py -v
"""
from __future__ import annotations

from datetime import datetime

import pytest

from app.services.conversation.conversation_nodes import _traffic_congestion_estimate
from app.services.conversation.conversation_helpers import build_generation_messages


# ---------------------------------------------------------------------------
# Part 1: _traffic_congestion_estimate 自身（已有函数，回归）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "now, expect_level_keyword",
    [
        # 工作日早高峰
        (datetime(2026, 5, 6, 8, 30), "较拥堵"),
        # 工作日晚高峰
        (datetime(2026, 5, 6, 18, 0), "较拥堵"),
        # 工作日白天平峰
        (datetime(2026, 5, 6, 14, 12), "中等拥堵"),
        # 工作日深夜
        (datetime(2026, 5, 6, 23, 30), "基本通畅"),
        # 周末商圈时段（5月10日是周日）
        (datetime(2026, 5, 10, 15, 0), "中等拥堵"),
        # 周末凌晨
        (datetime(2026, 5, 10, 4, 0), "基本通畅"),
    ],
)
def test_traffic_congestion_estimate_levels(now: datetime, expect_level_keyword: str):
    level, reason = _traffic_congestion_estimate(now)
    assert level == expect_level_keyword, f"now={now}, level={level}, expected={expect_level_keyword}"
    assert reason  # 非空


# ---------------------------------------------------------------------------
# Part 2: build_generation_messages 的 traffic prompt 包含"时段先验"
# ---------------------------------------------------------------------------
def _build_state(prefer_zh_output: bool, traffic_estimate: dict | None) -> dict:
    """构造最小可用的 state，避免依赖 employee_config / context 等复杂字段。"""
    state: dict = {
        "user_query": "我想了解一下今天成都堵不堵？",
        "rewritten_query": "今天成都的交通拥堵情况怎么样",
        "is_realtime_query": True,
        "realtime_category": "traffic",
        "web_search_used": True,
        "web_search_results": [
            {"title": "小鹏快讯", "content": "无关汽车广告内容", "url": "https://example.com"}
        ],
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
    }
    if traffic_estimate is not None:
        state["traffic_estimate"] = traffic_estimate
    return state


_FAKE_ESTIMATE_ZH = {
    "level": "中等拥堵",
    "reason": "工作日白天车流较大",
    "now_iso": "2026-05-06 14:12",
    "weekday_zh": "周三",
    "weekday_en": "Wed",
}


def _system_prompt(messages: list) -> str:
    """从 build_generation_messages 返回的消息列表中取 system prompt 文本。"""
    return messages[0].content


# 条件渲染段独有的标识串（不会出现在常驻指引文本里）：
_ZH_PRIOR_MARKER = "系统本地推算"
_EN_PRIOR_MARKER = "locally derived"


def test_traffic_prompt_zh_with_estimate_contains_prior_block():
    state = _build_state(prefer_zh_output=True, traffic_estimate=_FAKE_ESTIMATE_ZH)
    sys = _system_prompt(build_generation_messages(state))
    # 条件渲染段必须出现
    assert _ZH_PRIOR_MARKER in sys
    assert "2026-05-06 14:12" in sys
    assert "周三" in sys
    assert "中等拥堵" in sys
    assert "工作日白天车流较大" in sys
    # 降级指引推荐专业地图
    assert "高德地图" in sys or "百度地图" in sys


def test_traffic_prompt_en_with_estimate_contains_prior_block():
    state = _build_state(prefer_zh_output=False, traffic_estimate=_FAKE_ESTIMATE_ZH)
    sys = _system_prompt(build_generation_messages(state))
    assert _EN_PRIOR_MARKER in sys
    assert "2026-05-06 14:12" in sys
    assert "Wed" in sys
    assert "中等拥堵" in sys  # level 字段直接渲染原文
    assert "Amap" in sys or "Baidu" in sys


def test_traffic_prompt_zh_without_estimate_skips_prior_block():
    """无 traffic_estimate 时，不应渲染条件先验段（向后兼容旧调用方）。"""
    state = _build_state(prefer_zh_output=True, traffic_estimate=None)
    sys = _system_prompt(build_generation_messages(state))
    assert _ZH_PRIOR_MARKER not in sys
    # 时间字段也不应被泄漏
    assert "2026-05-06 14:12" not in sys


def test_traffic_prompt_zh_with_empty_estimate_skips_prior_block():
    """level 为空字符串时也应跳过（防御空字段）。"""
    state = _build_state(
        prefer_zh_output=True,
        traffic_estimate={"level": "", "reason": "", "now_iso": "", "weekday_zh": "", "weekday_en": ""},
    )
    sys = _system_prompt(build_generation_messages(state))
    assert _ZH_PRIOR_MARKER not in sys


def test_traffic_prompt_zh_keeps_existing_anti_holiday_rules():
    """回归：禁止误判节假日的现有规则保留。"""
    state = _build_state(prefer_zh_output=True, traffic_estimate=_FAKE_ESTIMATE_ZH)
    sys = _system_prompt(build_generation_messages(state))
    assert "假期" in sys and "节日" in sys
    # 必须保留禁用语
    assert "禁止" in sys


# ---------------------------------------------------------------------------
# Part 3: 改动后的"降级指引"行为契约
# ---------------------------------------------------------------------------
def test_traffic_prompt_zh_includes_fallback_guidance():
    """资料不足时给出"等级 + 依据 + 出行建议 + 推荐专业地图"的降级指引。"""
    state = _build_state(prefer_zh_output=True, traffic_estimate=_FAKE_ESTIMATE_ZH)
    sys = _system_prompt(build_generation_messages(state))
    # 关键词级最弱断言（不锁死措辞）
    assert "出行建议" in sys
    assert "禁止说" in sys or "回避语" in sys


def test_traffic_prompt_en_includes_fallback_guidance():
    state = _build_state(prefer_zh_output=False, traffic_estimate=_FAKE_ESTIMATE_ZH)
    sys = _system_prompt(build_generation_messages(state))
    assert "travel tips" in sys.lower()
    assert "do not" in sys.lower()


# ---------------------------------------------------------------------------
# Part 4: 非 traffic 类不受影响（保护现有行为）
# ---------------------------------------------------------------------------
def test_non_traffic_prompt_unaffected():
    """weather 类查询不应渲染时段先验（traffic 专属字段）。"""
    state = _build_state(prefer_zh_output=True, traffic_estimate=_FAKE_ESTIMATE_ZH)
    state["realtime_category"] = "weather"
    sys = _system_prompt(build_generation_messages(state))
    assert _ZH_PRIOR_MARKER not in sys
