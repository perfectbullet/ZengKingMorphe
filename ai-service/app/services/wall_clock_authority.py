"""
墙上时钟权威源：用户问「北京时间/几点几分」时，直接用服务端 Asia/Shanghai 时间，
避免网页摘要的时区/缓存错误。
"""
from __future__ import annotations
import re
from datetime import datetime
from typing import Dict, Tuple
from zoneinfo import ZoneInfo
from app.core.logging import get_logger

logger = get_logger(__name__)
CN_TZ = ZoneInfo("Asia/Shanghai")

# 其他城市时区（不使用北京时间）
_OTHER_TZ_HINTS_ZH_EN: Tuple[str, ...] = (
    "纽约",
    "洛杉磯",
    "洛杉矶",
    "旧金山",
    "舊金山",
    "芝加哥",
    "倫敦",
    "伦敦",
    "巴黎",
    "柏林",
    "莫斯科",
    "東京",
    "东京",
    "首爾",
    "首尔",
    "悉尼",
    "雪梨",
    "墨爾本",
    "墨尔本",
    "迪拜",
    "new york",
    "los angeles",
    "san francisco",
    "chicago",
    "london",
    "paris",
    "berlin",
    "moscow",
    "tokyo",
    "seoul",
    "sydney",
    "melbourne",
    "dubai",
)

# 黄历相关（不使用时钟）
_SKIP_WEB_HINTS_ALMANAC: Tuple[str, ...] = ("农历", "黄历", "老黄历", "宜忌", "生肖运程")

_OTHER_TZ_REGEX = re.compile(
    r"|".join(re.escape(x) for x in _OTHER_TZ_HINTS_ZH_EN),
    re.IGNORECASE,
)

_ASKS_CLOCK_ZH_REGEX = re.compile(
    r"(几点|几多分|多少分|几分|几时|什么时间|啥时候|几时几分|現在幾點|现在几点|当前时间|現在時間|此时此刻|此刻|對表|对表)",
    re.IGNORECASE,
)

_ASKS_CLOCK_EN_REGEX = re.compile(
    r"\b(what\s+time(\s+is\s+it)?|current\s+time|time\s+now)\b",
    re.IGNORECASE,
)

_SCOPE_BEIJING_REGEX = re.compile(
    r"(北京|中国|中國|beijing|shanghai|china|CST|UTC\+8|UTC\+0800|utc\+8|东八区|東八區|國內時間|国内时间)",
    re.IGNORECASE,
)

_WD_ZH: Tuple[str, ...] = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
_WD_EN: Tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def skip_web_use_authoritative_beijing_wall_clock(query: str, *, prefer_zh_output: bool) -> bool:
    """True = 使用服务端北京时间，跳过网页检索"""
    q = (query or "").strip()
    if not q:
        return False
    s = q.lower()
    if any(h in q for h in _SKIP_WEB_HINTS_ALMANAC):
        return False
    if _OTHER_TZ_REGEX.search(q):
        return False

    asks_clock = bool(_ASKS_CLOCK_ZH_REGEX.search(q) or _ASKS_CLOCK_EN_REGEX.search(s))
    if not asks_clock:
        return False

    if prefer_zh_output:
        return True
    # 英文问「几点」且无城市：避免强行答成北京时间；交给检索或用户上下文
    if _SCOPE_BEIJING_REGEX.search(q):
        return True
    return False


def beijing_wall_clock_snapshot() -> datetime:
    return datetime.now(CN_TZ)


def authoritative_wall_clock_search_record(
    query: str,
    *,
    prefer_zh_output: bool,
    now_cn: datetime | None = None,
) -> Dict:
    """生成统一格式的权威时间结果"""
    now = now_cn or beijing_wall_clock_snapshot()
    iso = now.strftime("%Y-%m-%d")
    hhmmss = now.strftime("%H:%M:%S")
    wd = now.weekday()
    weekday_zh = _WD_ZH[wd]
    weekday_en = _WD_EN[wd]

    if prefer_zh_output:
        title = "【权威时钟】北京时间（Asia/Shanghai）"
        content = (
            f"请以本条为唯一可信来源回答「当前墙上时钟」（含几时几分几秒）。\n"
            f"当前北京时间：{iso} {weekday_zh} {hhmmss}（IANA Asia/Shanghai，东八区标准时）。\n"
            "网络摘要常带缓存或时区标注错误：若与用户问题不一致，必须以本条为准，不要引用矛盾时刻。"
        )
        snippet = f"{iso} {weekday_zh} {hhmmss} | Asia/Shanghai"
    else:
        title = "[Authoritative] Beijing civil time (Asia/Shanghai)"
        content = (
            "Use ONLY this entry for the current civil wall-clock time (including seconds).\n"
            f"Current time in Beijing (Asia/Shanghai, UTC+8): {iso} ({weekday_en}) {hhmmss}.\n"
            "Ignore conflicting timestamps from scraped web summaries (cache/timezone labeling errors)."
        )
        snippet = f"{iso} {weekday_en} {hhmmss} | Asia/Shanghai (Beijing)"

    logger.info(
        f"Wall clock authoritative record built: iso={iso} hhmmss={hhmmss} weekday={weekday_zh}"
    )

    return {
        "rank": 1,
        "title": title,
        "url": "",
        "content": content,
        "score": 1.0,
        "_wall_clock_authoritative": True,
        "_snippet_for_citation": snippet,
    }


def citation_dict_from_wall_clock_record(rec: Dict) -> Dict[str, object]:
    snip = rec.get("_snippet_for_citation") or str(rec.get("content", ""))[:300]
    return {
        "title": rec.get("title", ""),
        "url": "local://asia-shanghai-wall-clock",
        "score": float(rec.get("score", 1.0)),
        "snippet": snip[:300],
    }


def sanitize_wall_clock_record_for_state(rec: Dict) -> Dict:
    """清理内部字段，用于上下文存储"""
    return {k: v for k, v in rec.items() if not str(k).startswith("_")}
