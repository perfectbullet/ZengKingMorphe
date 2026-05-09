"""
Web search 结果的新鲜度处理（通用）。

目标：
- 不依赖业务关键词匹配；
- 尽量从 title/content/url 里提取日期，按“更接近现在”的结果优先；
- 可配置：不同 realtime_category 使用不同的最大允许陈旧天数；
- 结果为空时不强行丢弃，保持回退能力。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime


_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 2026-04-28 / 2026/04/28 / 2026.04.28
    re.compile(r"(?P<y>20\d{2})[-/.](?P<m>0?[1-9]|1[0-2])[-/.](?P<d>0?[1-9]|[12]\d|3[01])"),
    # 2026年4月28日
    re.compile(r"(?P<y>20\d{2})\s*年\s*(?P<m>0?[1-9]|1[0-2])\s*月\s*(?P<d>0?[1-9]|[12]\d|3[01])\s*日"),
    # 20260428（常见于某些站点/路径）
    re.compile(r"(?P<y>20\d{2})(?P<m>0[1-9]|1[0-2])(?P<d>0[1-9]|[12]\d|3[01])"),
    # 2026/0428（URL 中常见）
    re.compile(r"(?P<y>20\d{2})[-/.](?P<m>0[1-9]|1[0-2])(?P<d>0[1-9]|[12]\d|3[01])"),
    # 04月28日（无年份：按当年推断）
    re.compile(r"(?P<m>0?[1-9]|1[0-2])\s*月\s*(?P<d>0?[1-9]|[12]\d|3[01])\s*日"),
)


def _extract_date(text: str, now: datetime) -> datetime | None:
    if not text:
        return None
    for p in _DATE_PATTERNS:
        m = p.search(text)
        if not m:
            continue
        gd = m.groupdict()
        y = int(gd.get("y") or now.year)
        month = int(gd["m"])
        day = int(gd["d"])
        try:
            return datetime(y, month, day)
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class RecencyPolicy:
    max_age_days: int
    prefer_today: bool = True
    query_hint: str = ""


DEFAULT_POLICIES: dict[str, RecencyPolicy] = {
    # 更强调“今天/刚更新”
    "traffic": RecencyPolicy(max_age_days=2, prefer_today=True, query_hint=" 实时 路况 拥堵 指数"),
    "weather": RecencyPolicy(max_age_days=2, prefer_today=True),
    # 允许稍旧但仍要近期；不强制日期锚点，避免把查询带偏到无关的当日新闻
    "news": RecencyPolicy(max_age_days=14, prefer_today=False),
    "market": RecencyPolicy(max_age_days=7, prefer_today=True),
    "time": RecencyPolicy(max_age_days=365, prefer_today=False),
    "general": RecencyPolicy(max_age_days=30, prefer_today=False),
}


def build_time_anchored_query(query: str, now: datetime, policy: RecencyPolicy | None = None) -> str:
    """
    为“需要今天信息”的检索增加时间锚点（不做关键词判断）。
    仅追加日期字符串，增强搜索引擎命中当天页面的概率。
    """
    if not query:
        return query
    date_iso = now.strftime("%Y-%m-%d")
    date_cn = f"{now.month}月{now.day}日"
    # 追加两种格式，兼容不同站点的日期写法
    hint = (policy.query_hint if policy else "").strip()
    if hint:
        return f"{query}{policy.query_hint} {date_iso} {date_cn}"
    return f"{query} {date_iso} {date_cn}"


def filter_and_sort_by_recency(
    results: list[dict],
    now: datetime,
    policy: RecencyPolicy,
) -> list[dict]:
    """
    过滤/排序：优先保留较新结果；如果完全提取不到日期则不强过滤（避免全空）。
    """
    if not results:
        return []

    scored: list[tuple[float, bool, dict]] = []
    dated_count = 0

    for r in results:
        title = str(r.get("title", "") or "")
        content = str(r.get("content", "") or "")
        url = str(r.get("url", "") or "")
        dt = _extract_date(title, now) or _extract_date(content, now) or _extract_date(url, now)
        if dt:
            dated_count += 1
            age_days = abs((now.date() - dt.date()).days)
            if age_days > policy.max_age_days:
                continue
            # 越接近现在越好；今天额外加成
            score = -float(age_days)
            if policy.prefer_today and age_days == 0:
                score += 1.0
            scored.append((score, True, r))
        else:
            # 无日期：先保留，稍后看是否需要整体降权
            scored.append((-9999.0, False, r))

    if dated_count == 0:
        # 完全无法判断新鲜度：保持原顺序
        return results

    # 对需要“今天/最新”的场景：当已存在可判定日期的结果时，丢弃无日期结果，降低误导风险
    if policy.prefer_today:
        scored = [x for x in scored if x[1]]

    # 若存在有日期的数据：把可判断的新鲜结果排前，无法判断的排后
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, __, r in scored]

