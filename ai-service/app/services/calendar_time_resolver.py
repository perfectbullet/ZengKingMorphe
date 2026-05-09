"""
日历/日期类查询的确定性解析与格式化。

设计目标：
- **可维护**：相对日偏移、解析规则集中在一处，便于扩展；
- **可扩展**：新增解析策略只需实现 Resolver 并在 RESOLVERS 注册；
- **与路由解耦**：返回 None 表示无法用本地规则确定，交由联网或其它路径。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from typing import Protocol

try:
    from zhdate import ZhDate
except ImportError:  # pragma: no cover - 可选依赖
    ZhDate = None  # type: ignore[misc, assignment]

# 农历数据表（1900-2099），用于 zhdate 不可用时的本地换算兜底。
# 数据含义沿用通用位编码：低4位闰月，0x10000 为闰月大月标记，其余 12 位表示正月至腊月大小月。
_LUNAR_INFO: tuple[int, ...] = (
    0x04BD8, 0x04AE0, 0x0A570, 0x054D5, 0x0D260, 0x0D950, 0x16554, 0x056A0, 0x09AD0, 0x055D2,
    0x04AE0, 0x0A5B6, 0x0A4D0, 0x0D250, 0x1D255, 0x0B540, 0x0D6A0, 0x0ADA2, 0x095B0, 0x14977,
    0x04970, 0x0A4B0, 0x0B4B5, 0x06A50, 0x06D40, 0x1AB54, 0x02B60, 0x09570, 0x052F2, 0x04970,
    0x06566, 0x0D4A0, 0x0EA50, 0x06E95, 0x05AD0, 0x02B60, 0x186E3, 0x092E0, 0x1C8D7, 0x0C950,
    0x0D4A0, 0x1D8A6, 0x0B550, 0x056A0, 0x1A5B4, 0x025D0, 0x092D0, 0x0D2B2, 0x0A950, 0x0B557,
    0x06CA0, 0x0B550, 0x15355, 0x04DA0, 0x0A5D0, 0x14573, 0x052D0, 0x0A9A8, 0x0E950, 0x06AA0,
    0x0AEA6, 0x0AB50, 0x04B60, 0x0AAE4, 0x0A570, 0x05260, 0x0F263, 0x0D950, 0x05B57, 0x056A0,
    0x096D0, 0x04DD5, 0x04AD0, 0x0A4D0, 0x0D4D4, 0x0D250, 0x0D558, 0x0B540, 0x0B5A0, 0x195A6,
    0x095B0, 0x049B0, 0x0A974, 0x0A4B0, 0x0B27A, 0x06A50, 0x06D40, 0x0AF46, 0x0AB60, 0x09570,
    0x04AF5, 0x04970, 0x064B0, 0x074A3, 0x0EA50, 0x06B58, 0x05AC0, 0x0AB60, 0x096D5, 0x092E0,
    0x0C960, 0x0D954, 0x0D4A0, 0x0DA50, 0x07552, 0x056A0, 0x0ABB7, 0x025D0, 0x092D0, 0x0CAB5,
    0x0A950, 0x0B4A0, 0x0BAA4, 0x0AD50, 0x055D9, 0x04BA0, 0x0A5B0, 0x15176, 0x052B0, 0x0A930,
    0x07954, 0x06AA0, 0x0AD50, 0x05B52, 0x04B60, 0x0A6E6, 0x0A4E0, 0x0D260, 0x0EA65, 0x0D530,
    0x05AA0, 0x076A3, 0x096D0, 0x04BD7, 0x04AD0, 0x0A4D0, 0x1D0B6, 0x0D250, 0x0D520, 0x0DD45,
    0x0B5A0, 0x056D0, 0x055B2, 0x049B0, 0x0A577, 0x0A4B0, 0x0AA50, 0x1B255, 0x06D20, 0x0ADA0,
    0x14B63, 0x09370, 0x049F8, 0x04970, 0x064B0, 0x168A6, 0x0EA50, 0x06AA0, 0x1A6C4, 0x0AAE0,
    0x092E0, 0x0D2E3, 0x0C960, 0x0D557, 0x0D4A0, 0x0DA50, 0x05D55, 0x056A0, 0x0A6D0, 0x055D4,
    0x052D0, 0x0A9B8, 0x0A950, 0x0B4A0, 0x0B6A6, 0x0AD50, 0x055A0, 0x0ABA4, 0x0A5B0, 0x052B0,
    0x0B273, 0x06930, 0x07337, 0x06AA0, 0x0AD50, 0x14B55, 0x04B60, 0x0A570, 0x054E4, 0x0D160,
    0x0E968, 0x0D520, 0x0DAA0, 0x16AA6, 0x056D0, 0x04AE0, 0x0A9D4, 0x0A2D0, 0x0D150, 0x0F252,
)
_LUNAR_BASE_SOLAR = date(1900, 1, 31)  # 对应农历 1900-01-01

# -----------------------------------------------------------------------------
# 数据：相对日历日偏移（列表顺序很重要——长的词必须排在前面）
# -----------------------------------------------------------------------------
_RELATIVE_DAY_OFFSET_ZH: tuple[tuple[str, int], ...] = (
    ("大前天", -3),
    ("前天", -2),
    ("昨天", -1),
    ("今天", 0),
    ("明天", 1),
    ("后天", 2),
    ("大后天", 3),
)
_RELATIVE_DAY_OFFSET_EN: tuple[tuple[str, int], ...] = (
    ("three days ago", -3),
    ("day before yesterday", -2),
    ("yesterday", -1),
    ("today", 0),
    ("tomorrow", 1),
    ("day after tomorrow", 2),
    ("three days later", 3),
)

_DAY_LABEL_ZH: dict[int, str] = {
    -3: "大前天",
    -2: "前天",
    -1: "昨天",
    0: "今天",
    1: "明天",
    2: "后天",
    3: "大后天",
}
_DAY_LABEL_EN: dict[int, str] = {
    -3: "Three days ago",
    -2: "The day before yesterday",
    -1: "Yesterday",
    0: "Today",
    1: "Tomorrow",
    2: "The day after tomorrow",
    3: "Three days from now",
}

_WEEKDAY_CN_INDEX: dict[str, int] = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_WEEK_PREFIX_OFFSET: dict[str, int] = {
    "上上上周": -3,
    "上上周": -2,
    "上周": -1,
    "本周": 0,
    "这周": 0,
    "本星期": 0,
    "这星期": 0,
    "下周": 1,
    "下下周": 2,
    "下下下周": 3,
}


def split_user_questions(text: str) -> list[str]:
    """按标点切分用户问题，返回清理后的问题列表。"""
    if not text:
        return []
    parts = re.split(r"[?？!！。；;\n]+", text)
    return [p.strip() for p in parts if p and p.strip()]


def normalize_calendar_user_query(text: str) -> str:
    """去除多余空白；修复中文短语被空格打断导致的匹配失败（如「几月几 号」）。"""
    if not text:
        return ""
    s = text.strip().replace("\u3000", "")
    # 统一常见 Unicode 标点（避免英文缩写/智能引号导致意图识别失败）
    s = (
        s.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    # 移除仅出现在汉字之间的空格/换行
    s = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_CALENDAR_INTENT_ZH = re.compile(
    r"(几月几号|几月几日|哪天|哪一日|日期|星期几|周几|礼拜几|几号|几日|号\?|号？)"
)
_CALENDAR_INTENT_EN = re.compile(
    r"(?:"
    r"\b("
    r"(what(?:'s|\s+is)?\s+the\s+date)"
    r"|what\s+date"
    r"|which\s+day"
    r"|day\s+of\s+the\s+week"
    r"|month\s+and\s+day"
    r"|today'?s\s+date"
    r"|the\s+date\s+today"
    r"|(?:what\s+was|what\s+is)\s+the\s+date"
    r"|what\s+day(?:\s+is\s+it)?"
    r")\b"
    r"|'s\s+date\b"
    r")",
    re.IGNORECASE,
)

# 明显不是“问日历”的实时场景（避免「昨天天气」被当成「昨天日期」）
_NON_CALENDAR_REALTIME_HINT_ZH = (
    "天气",
    "气温",
    "降雨",
    "下雪",
    "台风",
    "空气质量",
    "拥堵",
    "路况",
    "交通",
    "堵",
    "股价",
    "汇率",
    "黄金",
    "新闻",
)

# 英文版「明显不是问日历的实时场景」匹配。
#
# 设计要点：
#   - 中文 hint 用 ``in`` 子串匹配即可（汉字天然有词界）；英文必须加 ``\b``，
#     否则 ``"news"`` 会误中 ``"newspaper"``、``"market"`` 会误中
#     ``"supermarket"`` 等无关名词，在该函数里产生连锁误判。
#   - 这里只匹配「领域关键词」，**不**判定时间相对词（如 "today"）——后者本就
#     是日历推算的合法触发条件，是否真要走日历由领域关键词反向兜底。
#   - 词表与中文版语义对齐（天气 / 路况 / 行情 / 新闻 等）；按需扩展时
#     **只追加新分组**，避免把过于宽泛的词（如 "price"、"rate"）放进来——
#     它们可能出现在日历无关的非实时问题里（如 "exchange rate of yuan in 2010"）。
_NON_CALENDAR_REALTIME_HINT_EN = re.compile(
    r"\b("
    # 天气 / 气象
    r"weather|temperature|forecast|"
    r"rain(?:ing|y|fall)?|snow(?:ing|y|fall)?|"
    r"typhoon|hurricane|thunderstorm|storm|blizzard|"
    r"humid(?:ity)?|sunny|cloudy|windy|foggy|"
    r"air\s+quality|aqi|pollution|smog|haze|"
    # 路况 / 交通
    r"traffic|congestion|gridlock|road\s+condition|"
    # 行情 / 金融实时
    r"stock(?:s)?|share\s+price|exchange\s+rate|currency|gold\s+price|"
    # 新闻 / 时事
    r"news|headline(?:s)?|breaking"
    r")\b",
    re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# 农历「固定月日」传统节日 → 公历日期（表驱动扩展；日期由 zhdate 在公历年内搜索确定）
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LunarFixedFestival:
    """农历几月几日固定的节日（非节气/公历节日勿放入此表）。"""

    display_name: str
    lunar_month: int
    lunar_day: int
    aliases: tuple[str, ...]


LUNAR_FIXED_FESTIVALS: tuple[LunarFixedFestival, ...] = (
    LunarFixedFestival("春节", 1, 1, ("农历新年", "大年初一", "正月初一", "过年", "春节")),
    LunarFixedFestival("元宵节", 1, 15, ("元宵节", "上元节", "正月十五", "元宵")),
    LunarFixedFestival("端午节", 5, 5, ("端午节", "龙舟节", "端阳节", "端午")),
    LunarFixedFestival("七夕节", 7, 7, ("七夕节", "乞巧节", "七夕", "中国情人节")),
    LunarFixedFestival("中秋节", 8, 15, ("中秋节", "八月节", "中秋")),
    LunarFixedFestival("重阳节", 9, 9, ("重阳节", "老年节", "重阳", "九九重阳节")),
    LunarFixedFestival("腊八节", 12, 8, ("腊八节", "腊八")),
)

_ALIAS_TO_FESTIVAL_SORTED: list[tuple[str, LunarFixedFestival]] | None = None


@dataclass(frozen=True)
class SolarWeekdayFestival:
    """
    公历某月第 N 个星期几的节日（如父亲节：6月第3个星期日）。
    weekday: Monday=0 ... Sunday=6
    ordinal: 第几个（1~5）
    """

    display_name: str
    month: int
    weekday: int
    ordinal: int
    aliases: tuple[str, ...]


SOLAR_WEEKDAY_FESTIVALS: tuple[SolarWeekdayFestival, ...] = (
    SolarWeekdayFestival("父亲节", 6, 6, 3, ("父亲节", "father's day", "fathers day")),
)

_ALIAS_TO_SOLAR_WEEKDAY_SORTED: list[tuple[str, SolarWeekdayFestival]] | None = None


@dataclass(frozen=True)
class SolarTermFestival:
    """
    二十四节气中的节日型问法（通用公式，支持新增）。
    day = int(Y*0.2422 + C) - int((Y-1)/4), Y = year % 100
    """

    display_name: str
    month: int
    c_21st: float
    aliases: tuple[str, ...]


SOLAR_TERM_FESTIVALS: tuple[SolarTermFestival, ...] = (
    SolarTermFestival("清明节", 4, 4.81, ("清明节", "清明", "qingming")),
)

_ALIAS_TO_SOLAR_TERM_SORTED: list[tuple[str, SolarTermFestival]] | None = None


def _alias_festival_pairs() -> list[tuple[str, LunarFixedFestival]]:
    global _ALIAS_TO_FESTIVAL_SORTED
    if _ALIAS_TO_FESTIVAL_SORTED is None:
        pairs: list[tuple[str, LunarFixedFestival]] = []
        for fest in LUNAR_FIXED_FESTIVALS:
            for a in fest.aliases:
                pairs.append((a, fest))
        pairs.sort(key=lambda x: len(x[0]), reverse=True)
        _ALIAS_TO_FESTIVAL_SORTED = pairs
    return _ALIAS_TO_FESTIVAL_SORTED


def _alias_solar_weekday_pairs() -> list[tuple[str, SolarWeekdayFestival]]:
    global _ALIAS_TO_SOLAR_WEEKDAY_SORTED
    if _ALIAS_TO_SOLAR_WEEKDAY_SORTED is None:
        pairs: list[tuple[str, SolarWeekdayFestival]] = []
        for fest in SOLAR_WEEKDAY_FESTIVALS:
            for a in fest.aliases:
                pairs.append((a, fest))
        pairs.sort(key=lambda x: len(x[0]), reverse=True)
        _ALIAS_TO_SOLAR_WEEKDAY_SORTED = pairs
    return _ALIAS_TO_SOLAR_WEEKDAY_SORTED


def _alias_solar_term_pairs() -> list[tuple[str, SolarTermFestival]]:
    global _ALIAS_TO_SOLAR_TERM_SORTED
    if _ALIAS_TO_SOLAR_TERM_SORTED is None:
        pairs: list[tuple[str, SolarTermFestival]] = []
        for fest in SOLAR_TERM_FESTIVALS:
            for a in fest.aliases:
                pairs.append((a, fest))
        pairs.sort(key=lambda x: len(x[0]), reverse=True)
        _ALIAS_TO_SOLAR_TERM_SORTED = pairs
    return _ALIAS_TO_SOLAR_TERM_SORTED


def match_lunar_fixed_festival(user_query_normalized_lower: str) -> LunarFixedFestival | None:
    """数据表匹配别名（长词优先），非业务关键字路由。"""
    q = user_query_normalized_lower
    if not q:
        return None
    for alias, fest in _alias_festival_pairs():
        if alias in q:
            return fest
    return None


def match_solar_weekday_festival(
    user_query_normalized_lower: str,
) -> SolarWeekdayFestival | None:
    q = user_query_normalized_lower
    if not q:
        return None
    for alias, fest in _alias_solar_weekday_pairs():
        if alias in q:
            return fest
    return None


def match_solar_term_festival(user_query_normalized_lower: str) -> SolarTermFestival | None:
    q = user_query_normalized_lower
    if not q:
        return None
    for alias, fest in _alias_solar_term_pairs():
        if alias in q:
            return fest
    return None


def _solar_date_for_lunar_month_day_in_solar_year(
    solar_year: int, lunar_month: int, lunar_day: int
) -> date | None:
    """在给定公历年内找到「农历 lunar_month 月初 lunar_day」对应的公历日期（跳过闰月）。"""
    if ZhDate is not None:
        for month in range(1, 13):
            for day in range(1, 32):
                try:
                    d = date(solar_year, month, day)
                except ValueError:
                    continue
                z = ZhDate.from_datetime(datetime.combine(d, dt_time.min))
                if z.lunar_month == lunar_month and z.lunar_day == lunar_day and not z.leap_month:
                    return d
        return None
    # 兜底：内置农历算法（1900-2099）
    return _solar_date_for_lunar_month_day_fallback(solar_year, lunar_month, lunar_day)


def _lunar_leap_month(year: int) -> int:
    return _LUNAR_INFO[year - 1900] & 0xF


def _lunar_leap_days(year: int) -> int:
    leap = _lunar_leap_month(year)
    if leap == 0:
        return 0
    return 30 if (_LUNAR_INFO[year - 1900] & 0x10000) else 29


def _lunar_month_days(year: int, month: int) -> int:
    return 30 if (_LUNAR_INFO[year - 1900] & (0x10000 >> month)) else 29


def _solar_from_lunar_fallback(year: int, month: int, day: int, is_leap: bool = False) -> date | None:
    if year < 1900 or year > 2099 or month < 1 or month > 12 or day < 1 or day > 30:
        return None
    leap = _lunar_leap_month(year)
    if is_leap and leap != month:
        return None
    offset = 0
    for y in range(1900, year):
        year_days = sum(_lunar_month_days(y, m) for m in range(1, 13)) + _lunar_leap_days(y)
        offset += year_days
    for m in range(1, month):
        offset += _lunar_month_days(year, m)
        if leap == m:
            offset += _lunar_leap_days(year)
    if is_leap:
        offset += _lunar_month_days(year, month)
    mdays = _lunar_leap_days(year) if is_leap else _lunar_month_days(year, month)
    if day > mdays:
        return None
    offset += day - 1
    return _LUNAR_BASE_SOLAR + timedelta(days=offset)


def _solar_date_for_lunar_month_day_fallback(solar_year: int, lunar_month: int, lunar_day: int) -> date | None:
    for ly in (solar_year - 1, solar_year, solar_year + 1):
        d = _solar_from_lunar_fallback(ly, lunar_month, lunar_day, is_leap=False)
        if d and d.year == solar_year:
            return d
    return None


def _infer_target_solar_year_for_calendar(
    user_query: str, now: datetime, anchor_year: int | None
) -> int:
    """从问句推断目标公历年：显式年份 > 今年/明年/去年 > 会话锚定 > 当前公历年。"""
    q = (user_query or "").strip()
    m = re.search(r"(?<![0-9])(20\d{2}|19\d{2})年?", q)
    if m:
        y = int(m.group(1))
        if 1900 <= y <= 2100:
            return y
    ql = q.lower()
    if re.search(r"(今年|本年|this\s+year)", ql, re.IGNORECASE):
        return now.year
    rel_off = _parse_relative_year_offset(q)
    if rel_off is not None:
        return now.year + rel_off
    if anchor_year is not None:
        return int(anchor_year)
    return now.year


_ZH_NUM_MAP: dict[str, int] = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _parse_zh_small_num(s: str) -> int | None:
    t = (s or "").strip()
    if not t:
        return None
    if t.isdigit():
        return int(t)
    if t in _ZH_NUM_MAP:
        return _ZH_NUM_MAP[t]
    # 十三、二十、二十三
    if "十" in t:
        if t == "十":
            return 10
        if t.startswith("十"):
            right = _ZH_NUM_MAP.get(t[1:])
            return None if right is None else 10 + right
        if t.endswith("十"):
            left = _ZH_NUM_MAP.get(t[0])
            return None if left is None else left * 10
        left = _ZH_NUM_MAP.get(t[0])
        right = _ZH_NUM_MAP.get(t[-1])
        if left is None or right is None:
            return None
        return left * 10 + right
    return None


def _parse_relative_year_offset(q: str) -> int | None:
    """解析相对年份偏移（前年/后年/三年后/2年前/in 3 years 等）。"""
    t = (q or "").strip().lower()
    if not t:
        return None
    # 固定表达
    fixed = {
        "大前年": -3,
        "前年": -2,
        "去年": -1,
        "明年": 1,
        "后年": 2,
        "大后年": 3,
    }
    for k, v in fixed.items():
        if k in t:
            return v

    # 中文/数字：N 年前 / N 年后
    m = re.search(r"([0-9]+|[零一二两三四五六七八九十]+)\s*年\s*(前|后)", t)
    if m:
        n = _parse_zh_small_num(m.group(1))
        if n is not None:
            return -n if m.group(2) == "前" else n

    # 英文：N years ago / in N years
    m_en_ago = re.search(r"\b(\d+)\s+years?\s+ago\b", t, re.IGNORECASE)
    if m_en_ago:
        return -int(m_en_ago.group(1))
    m_en_in = re.search(r"\bin\s+(\d+)\s+years?\b", t, re.IGNORECASE)
    if m_en_in:
        return int(m_en_in.group(1))
    if re.search(r"\blast\s+year\b", t, re.IGNORECASE):
        return -1
    if re.search(r"\bnext\s+year\b", t, re.IGNORECASE):
        return 1

    return None


def _nth_weekday_of_month(year: int, month: int, weekday: int, ordinal: int) -> date | None:
    """返回 year/month 的第 ordinal 个 weekday 对应的日期。"""
    if ordinal < 1:
        return None
    first = date(year, month, 1)
    delta = (weekday - first.weekday()) % 7
    day = 1 + delta + (ordinal - 1) * 7
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _resolve_solar_weekday_festival(
    sub_q_lower: str, now: datetime, anchor_year: int | None
) -> ResolvedCalendarLine | None:
    qn = normalize_calendar_user_query(sub_q_lower).lower()
    fest = match_solar_weekday_festival(qn)
    if fest is None:
        return None
    solar_y = _infer_target_solar_year_for_calendar(sub_q_lower, now, anchor_year)
    d = _nth_weekday_of_month(solar_y, fest.month, fest.weekday, fest.ordinal)
    if d is None:
        return None
    target_dt = datetime.combine(d, dt_time.min)
    day_offset = (d - now.date()).days
    return ResolvedCalendarLine(
        sub_q_lower=sub_q_lower,
        target_dt=target_dt,
        day_offset=day_offset,
        relative_label=fest.display_name,
    )


def _resolve_solar_term_festival(
    sub_q_lower: str, now: datetime, anchor_year: int | None
) -> ResolvedCalendarLine | None:
    qn = normalize_calendar_user_query(sub_q_lower).lower()
    fest = match_solar_term_festival(qn)
    if fest is None:
        return None
    solar_y = _infer_target_solar_year_for_calendar(sub_q_lower, now, anchor_year)
    if solar_y < 2000 or solar_y > 2099:
        return None
    y2 = solar_y % 100
    day = int(y2 * 0.2422 + fest.c_21st) - int((y2 - 1) / 4)
    try:
        d = date(solar_y, fest.month, day)
    except ValueError:
        return None
    target_dt = datetime.combine(d, dt_time.min)
    day_offset = (d - now.date()).days
    return ResolvedCalendarLine(
        sub_q_lower=sub_q_lower,
        target_dt=target_dt,
        day_offset=day_offset,
        relative_label=fest.display_name,
    )


def _resolve_lunar_fixed_festival(
    sub_q_lower: str, now: datetime, anchor_year: int | None
) -> ResolvedCalendarLine | None:
    qn = normalize_calendar_user_query(sub_q_lower).lower()
    fest = match_lunar_fixed_festival(qn)
    if fest is None:
        return None
    solar_y = _infer_target_solar_year_for_calendar(sub_q_lower, now, anchor_year)
    d = _solar_date_for_lunar_month_day_in_solar_year(solar_y, fest.lunar_month, fest.lunar_day)
    if d is None:
        return None
    target_dt = datetime.combine(d, dt_time.min)
    day_offset = (d - now.date()).days
    return ResolvedCalendarLine(
        sub_q_lower=sub_q_lower,
        target_dt=target_dt,
        day_offset=day_offset,
        relative_label=fest.display_name,
    )


def should_attempt_calendar_resolution(user_query: str) -> bool:
    """
    是否应用本地日历推算：
    - 农历固定月日的传统节日（表驱动，由 zhdate 转公历）；或
    - 显式出现“几月几号/星期几/日期”等日历意图；或
    - 子问题几乎只有相对日词（如单独「昨天？」）；
    同时排除明显天气/路况/行情等实时检索意图。
    """
    q = normalize_calendar_user_query(user_query).lower()
    if not q:
        return False
    if any(h in q for h in _NON_CALENDAR_REALTIME_HINT_ZH):
        return False
    # 英文领域关键词反向兜底：例如 "what's the weather like today"
    # 含 "today" 会触发相对日匹配，但本质是天气问题——必须先短路掉，
    # 否则下游会被错误地归为 ``realtime_category="time"`` 并跳过 web search。
    if _NON_CALENDAR_REALTIME_HINT_EN.search(q):
        return False
    if match_lunar_fixed_festival(q):
        return True
    if match_solar_weekday_festival(q):
        return True
    if match_solar_term_festival(q):
        return True
    if _CALENDAR_INTENT_EN.search(q) or _CALENDAR_INTENT_ZH.search(q):
        return True

    subs = split_user_questions(q)
    if not subs:
        subs = [q]
    for sq in subs:
        s = sq.strip().lower()
        if not s:
            continue
        # 相对日：**仅含有「今天/明天」等词不足以启动日历**，须剥掉首个相对日短语后，
        # 残余为空（或仅存标点语气）或为显式历法问法——否则多半是「today + 实质主题」（行情、日程等）。
        if _relative_day_qualifies_calendar(s):
            return True
    return False


_RELATIVE_DAY_OFFSET_ZH_BY_LEN: tuple[tuple[str, int], ...] = tuple(
    sorted(_RELATIVE_DAY_OFFSET_ZH, key=lambda kv: len(kv[0]), reverse=True)
)
_RELATIVE_DAY_OFFSET_EN_BY_LEN: tuple[tuple[str, int], ...] = tuple(
    sorted(_RELATIVE_DAY_OFFSET_EN, key=lambda kv: len(kv[0]), reverse=True)
)


def _relative_calendar_day_offset(q_lower: str) -> int | None:
    """
    匹配相对日词条，**按词长降序匹配**——避免长词被同前缀短词截胡。

    典型反例（按声明顺序匹配会出错）：
      - ``大后天是几月几号？``：声明顺序中 "后天" 在 "大后天" 之前，
        ``in`` 子串扫描会先命中 "后天"（+2），导致结果比正确日期早 1 天。
      - ``three days ago`` vs ``yesterday``、``day after tomorrow`` vs
        ``tomorrow`` 等英文词条同样存在前缀重叠。
    用按长度倒序的副本匹配后，"大后天" / "day before yesterday" 这类长词必然先命中，
    数据声明顺序仍可按语义阅读习惯保留。
    """
    for phrase, off in _RELATIVE_DAY_OFFSET_ZH_BY_LEN:
        if phrase in q_lower:
            return off
    for phrase, off in _RELATIVE_DAY_OFFSET_EN_BY_LEN:
        if phrase in q_lower:
            return off
    return None


def _trim_calendar_followup(rem: str) -> str:
    t = normalize_calendar_user_query(rem).strip().lower()
    t = re.sub(
        r"^[`'\"“”，,。.．:：;；、!！…~\s_-]+|[`'\"“”，,。.．:：;；、!！…~\s_-]+$",
        "",
        t,
    )
    return t.strip()


def _relative_day_qualifies_calendar(sq_lower: str) -> bool:
    if _relative_calendar_day_offset(sq_lower) is None:
        return False
    remn_with_space = normalize_calendar_user_query(sq_lower).strip().lower()
    stripped = False
    for phrase, _ in _RELATIVE_DAY_OFFSET_ZH_BY_LEN:
        if phrase in remn_with_space:
            remn_with_space = normalize_calendar_user_query(
                remn_with_space.replace(phrase, " ", 1)
            ).strip().lower()
            stripped = True
            break
    if not stripped:
        for phrase, _ in _RELATIVE_DAY_OFFSET_EN_BY_LEN:
            m_en = re.search(r"\b" + re.escape(phrase) + r"\b", sq_lower)
            if m_en:
                remn_with_space = normalize_calendar_user_query(
                    sq_lower[: m_en.start()] + " " + sq_lower[m_en.end() :]
                ).strip().lower()
                stripped = True
                break
    if not stripped:
        return False
    rem = _trim_calendar_followup(remn_with_space)
    if not rem:
        return True
    if _CALENDAR_INTENT_ZH.search(rem) or _CALENDAR_INTENT_EN.search(rem):
        return True
    return False


def _parse_weekday_date_cn(q_lower: str, now: datetime) -> tuple[str, datetime] | None:
    """解析中文星期日期（上周六/下周一），返回标签+日期时刻。"""
    if re.search(r"(星期几|周几|礼拜几)", q_lower):
        return None
    if ("周" not in q_lower) and ("星期" not in q_lower) and not any(
        p in q_lower for p in _WEEK_PREFIX_OFFSET.keys()
    ):
        return None

    m = re.search(
        r"(?P<prefix>上上上周|上上周|上周|本周|这周|本星期|这星期|下周|下下周|下下下周|上星期|本星期|这星期|下星期|上上星期|下下星期)?"
        r"(?:(?:周|星期)?)(?P<wd>[一二三四五六日天])",
        q_lower,
    )
    if not m:
        return None
    prefix = m.group("prefix") or "本周"
    wd = m.group("wd")
    prefix_norm = prefix.replace("星期", "周")
    if prefix_norm not in _WEEK_PREFIX_OFFSET:
        prefix_norm = prefix_norm.replace("上上周", "上上周").replace("下下周", "下下周")
    week_off = _WEEK_PREFIX_OFFSET.get(prefix_norm)
    wd_idx = _WEEKDAY_CN_INDEX.get(wd)
    if week_off is None or wd_idx is None:
        return None

    monday = now.date() - timedelta(days=now.weekday())
    target_date = monday + timedelta(days=week_off * 7 + wd_idx)
    if "星期" in q_lower or "星期" in (prefix or ""):
        label = prefix_norm.replace("周", "星期") + wd
    else:
        label = prefix_norm + wd
    target_dt = datetime.combine(target_date, now.time())
    return label, target_dt


def _chinese_weekday(idx: int) -> str:
    return ["一", "二", "三", "四", "五", "六", "日"][idx]


@dataclass(frozen=True)
class ResolvedCalendarLine:
    """单条子问题的解析结果（用于格式化）。"""
    sub_q_lower: str
    target_dt: datetime
    day_offset: int
    relative_label: str | None  # 如「上周六」；相对日偏移时为 None，用 day_word
    explicit_date: bool = False


class CalendarResolver(Protocol):
    def __call__(self, sub_q_lower: str, now: datetime) -> ResolvedCalendarLine | None: ...


def _resolve_relative_day(sub_q_lower: str, now: datetime) -> ResolvedCalendarLine | None:
    off = _relative_calendar_day_offset(sub_q_lower)
    if off is None:
        return None
    target_dt = now + timedelta(days=off)
    return ResolvedCalendarLine(
        sub_q_lower=sub_q_lower, target_dt=target_dt, day_offset=off, relative_label=None
    )


def _resolve_cn_weekday(sub_q_lower: str, now: datetime) -> ResolvedCalendarLine | None:
    parsed = _parse_weekday_date_cn(sub_q_lower, now)
    if not parsed:
        return None
    label, target_dt = parsed
    day_offset = (target_dt.date() - now.date()).days
    return ResolvedCalendarLine(
        sub_q_lower=sub_q_lower, target_dt=target_dt, day_offset=day_offset, relative_label=label
    )


def _extract_explicit_month_day(sub_q_lower: str) -> tuple[int, int] | None:
    """提取显式月日（中文数字/阿拉伯数字）：五月一号、5月1日。"""
    m = re.search(
        r"([0-9零一二两三四五六七八九十]+)\s*月\s*([0-9零一二两三四五六七八九十]+)\s*(?:日|号)",
        sub_q_lower,
    )
    if not m:
        return None
    month = _parse_zh_small_num(m.group(1))
    day = _parse_zh_small_num(m.group(2))
    if month is None or day is None:
        return None
    if 1 <= month <= 12 and 1 <= day <= 31:
        return month, day
    return None


def _resolve_explicit_month_day(
    sub_q_lower: str, now: datetime, anchor_year: int | None
) -> ResolvedCalendarLine | None:
    md = _extract_explicit_month_day(sub_q_lower)
    if md is None:
        return None
    month, day = md
    solar_y = _infer_target_solar_year_for_calendar(sub_q_lower, now, anchor_year)
    try:
        d = date(solar_y, month, day)
    except ValueError:
        return None
    target_dt = datetime.combine(d, dt_time.min)
    day_offset = (d - now.date()).days
    return ResolvedCalendarLine(
        sub_q_lower=sub_q_lower,
        target_dt=target_dt,
        day_offset=day_offset,
        relative_label=None,
        explicit_date=True,
    )


# 解析器注册表：前者优先匹配（如需新增策略，追加函数即可）
RESOLVERS: tuple[CalendarResolver, ...] = (
    _resolve_cn_weekday,
    _resolve_relative_day,
)


def try_resolve_calendar_answer(
    user_query: str,
    now: datetime,
    anchor_year: int | None = None,
) -> list[ResolvedCalendarLine] | None:
    """
    尝试解析整段 query（支持一句多问）。
    任一子问题无法解析则返回 None，调用方走其它路径。
    """
    user_query = normalize_calendar_user_query(user_query)
    if not user_query:
        return None
    subs = split_user_questions(user_query)
    if not subs:
        subs = [user_query]

    lines: list[ResolvedCalendarLine] = []
    for sq in subs:
        sq_lower = sq.strip().lower()
        resolved: ResolvedCalendarLine | None = _resolve_explicit_month_day(
            sq_lower, now, anchor_year
        )
        if resolved is None:
            resolved = _resolve_lunar_fixed_festival(
            sq_lower, now, anchor_year
            )
        if resolved is None:
            resolved = _resolve_solar_weekday_festival(sq_lower, now, anchor_year)
        if resolved is None:
            resolved = _resolve_solar_term_festival(sq_lower, now, anchor_year)
        if resolved is None:
            for resolver in RESOLVERS:
                resolved = resolver(sq_lower, now)
                if resolved:
                    break
        if resolved is None:
            return None
        lines.append(resolved)
    return lines


def format_calendar_answer(lines: list[ResolvedCalendarLine], prefer_zh_output: bool, now: datetime) -> str:
    """将解析结果格式化为面向用户的自然语言。"""
    answers: list[str] = []
    for line in lines:
        sq_lower = line.sub_q_lower
        asks_clock_only = any(
            kw in sq_lower
            for kw in ("几点", "什么时间", "时刻", "当前时间", "现在几点", "what time", "clock")
        )
        asks_date_only = any(
            kw in sq_lower for kw in ("日期", "几号", "哪天", "星期", "礼拜", "周几", "星期几", "哪一日")
        )
        asks_month_day_cn = any(
            phrase in sq_lower for phrase in ("几月几号", "几月几日", "几月几")
        )

        if prefer_zh_output:
            md = f"{line.target_dt.month}月{line.target_dt.day}日"
            wk = f"星期{_chinese_weekday(line.target_dt.weekday())}"
            if line.explicit_date:
                text = f"{line.target_dt.year}年{md}，{wk}。"
                if asks_clock_only and not asks_date_only:
                    text = text[:-1] + f"，当前时间{now.strftime('%H:%M:%S')}。"
                elif asks_clock_only:
                    text += f"当前时间{now.strftime('%H:%M:%S')}。"
                answers.append(text)
                continue
            if line.relative_label:
                text = (
                    f"{line.relative_label}是{line.target_dt.year}年{md}"
                    f"（{wk}）。"
                )
            else:
                day_word = _DAY_LABEL_ZH.get(line.day_offset, "该日")
                # 「昨天是几月几号？」等：更贴近口语（昨天是4月27日，星期一）
                if asks_month_day_cn or asks_date_only:
                    text = f"{day_word}是{line.target_dt.year}年{md}，{wk}。"
                else:
                    text = (
                        f"{day_word}的日期是{line.target_dt.year}年{md}"
                        f"（{wk}）。"
                    )
            if asks_clock_only and not asks_date_only:
                text = text[:-1] + f"，当前时间{now.strftime('%H:%M:%S')}。"
            elif asks_clock_only:
                text += f"当前时间{now.strftime('%H:%M:%S')}。"
            answers.append(text)
        else:
            weekday_en = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ][line.target_dt.weekday()]
            if line.relative_label:
                day_word = line.relative_label
            else:
                day_word = _DAY_LABEL_EN.get(line.day_offset, "That day")
            en_verb = (
                "is"
                if line.day_offset == 0
                else ("was" if line.day_offset < 0 else "will be")
            )
            text = (
                f"{day_word} {en_verb} {line.target_dt.strftime('%B')} "
                f"{line.target_dt.day}, {line.target_dt.year} ({weekday_en})."
            )
            if asks_clock_only and not asks_date_only:
                text += f" The current time is {now.strftime('%H:%M:%S')}."
            elif asks_clock_only:
                text += f" Current time: {now.strftime('%H:%M:%S')}."
            answers.append(text)

    return "\n".join(answers)


def calendar_direct_text_answer(
    user_query: str,
    prefer_zh_output: bool,
    anchor_year: int | None = None,
    now: datetime | None = None,
    language_hint_query: str | None = None,
) -> str | None:
    """
    对外统一入口：能解析则返回格式化文本，否则 None。

    Args:
        user_query: 用于日历推算的查询（可能是被上下文消歧/改写过的版本）。
        prefer_zh_output: 上游计算的输出语言偏好。
        anchor_year: 会话锚定年份。
        now: 当前时间。
        language_hint_query: 用于语言判定的查询。**强烈建议传入用户原始问句**，
            避免改写后的 `user_query` 与原始语言不一致（例如英文问句被改写为中文）
            导致输出语言被错误覆盖。未传入时回退到 `user_query`，保持旧行为。
    """
    if not should_attempt_calendar_resolution(user_query):
        return None
    # 语言自适应兜底：避免上游 prefer_zh_output 丢失/传错导致“英文问中文答”。
    # 注意：以 `language_hint_query`（通常为用户原始问句）为准，避免被
    # rewriter/翻译产生的中介查询污染语言判定。
    lang_q = (language_hint_query if language_hint_query is not None else user_query) or ""
    lang_q = lang_q.strip()
    if lang_q:
        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in lang_q)
        has_alpha = bool(re.search(r"[A-Za-z]", lang_q))
        if has_cjk:
            prefer_zh_output = True
        elif has_alpha:
            prefer_zh_output = False
    if now is None:
        now = datetime.now()
    resolved = try_resolve_calendar_answer(user_query, now=now, anchor_year=anchor_year)
    if not resolved:
        return None
    return format_calendar_answer(resolved, prefer_zh_output, now)
