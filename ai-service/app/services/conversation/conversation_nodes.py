"""
Node implementations for LangGraph conversation workflow.

This module contains all node functions that process the conversation state:
- Configuration loading nodes
- Input validation nodes
- Query classification nodes
- Complexity evaluation nodes
- Web search nodes
- Answer generation nodes
- Conversation saving nodes

Note: Simplified workflow using RAGAnything for RAG retrieval.
Removed nodes: intent_recognition, knowledge_retrieval, grade_documents,
               compress_context, match_faq, rewrite_query
"""
import asyncio
import hashlib
import time
from datetime import datetime
from datetime import timedelta
import re

import httpx
from bs4 import BeautifulSoup
from langchain_community.tools.tavily_search import TavilySearchResults

from app.core.config import settings
from app.core.database import get_database
from app.core.logging import get_logger
from app.models.database import ConversationModel, SessionModel
from app.services.conversation.conversation_state import (
    ConversationState,
    GREETING_KEYWORDS,
    DEFAULT_SENSITIVE_WORDS,
    DEFAULT_SENSITIVE_WORDS_LOWER,
    NOISE_PRESET_RESPONSE_TEXT,
    sensitive_term_matches_query,
)
from app.services.query_classifier import (
    ClassificationResult,
    augment_dialog_with_persisted_turns,
    format_dialog_for_resolver,
    get_query_classifier,
)
from app.services.realtime_intent_heuristic import heuristic_realtime_category
from app.services.math_intent_heuristic import (
    HEURISTIC_PROMOTABLE_LABELS,
    heuristic_concept_explain,
    heuristic_math_problem,
)
from app.services.conversation.conversation_helpers import (
    time_node,
    heuristic_complexity,
    build_generation_messages,
    build_math_generation_messages,
    resolve_prefer_zh_output,
    resolve_target_year_from_query,
)
from app.services.conversation.intent_routing import (
    AnswerMode,
    ROUTE_BRANCH_GENERAL,
    ROUTE_BRANCH_GREETING,
    ROUTE_BRANCH_MATH,
    ROUTE_BRANCH_RAG,
    ROUTE_BRANCH_REALTIME,
    apply_noise_preset_gate,
    resolve_answer_mode,
    resolve_route_branch,
)
from app.services.calendar_time_resolver import calendar_direct_text_answer
from app.services.web_search_recency import (
    DEFAULT_POLICIES,
    RecencyPolicy,
    build_time_anchored_query,
    filter_and_sort_by_recency,
    merge_tavily_result_lists,
)
from app.services.wall_clock_authority import (
    authoritative_wall_clock_search_record,
    citation_dict_from_wall_clock_record,
    sanitize_wall_clock_record_for_state,
    skip_web_use_authoritative_beijing_wall_clock,
)
from app.utils.common import has_language_drift

logger = get_logger(__name__)

# 启发式联网补位：不覆盖问候、噪声、数学题；也不重复覆盖已是实时的分支
_REALTIME_HEURISTIC_SKIP_LABELS = frozenset({"greeting", "noise", "math_problem", "realtime_query"})

# =============================================================================
# Employee Config Resolvers
# 员工 kb_ids / rag_disabled 字段历史上散落在多个集合 / 多种 schema 路径里：
#   - 新版（EmployeeSyncService 写入）: digital_employee_settings.knowledge_kb_ids
#   - 旧版 A: knowledge.kb_ids
#   - 旧版 B: 顶层 kb_ids
#   - 旧版 C: capabilities.kb_ids
# 用集中维护的“路径表”避免在节点里到处写 if/else，且新增路径只改这两个表即可。
# 注：RAGAnything 是单一全局知识库，kb_ids 仅作展示/审计用，不再作为路由开关。
# =============================================================================
_EMPLOYEE_KB_ID_PATHS: tuple[tuple[str, ...], ...] = (
    ("setting", "knowledge_kb_ids"),
    ("knowledge", "kb_ids"),
    ("kb_ids",),
    ("capabilities", "kb_ids"),
)
_EMPLOYEE_RAG_DISABLED_PATHS: tuple[tuple[str, ...], ...] = (
    ("setting", "rag_disabled"),
    ("rag_disabled",),
    ("capabilities", "rag_disabled"),
)


def _get_nested(cfg, path: tuple[str, ...]):
    """按路径取值，遇到非 dict 即兜底返回 None，避免脏数据 KeyError。"""
    cur = cfg
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _resolve_employee_kb_ids(cfg) -> list:
    """从员工配置多条新/旧路径解析 kb_ids；非 list 或空 list 自动跳到下一条。"""
    if not isinstance(cfg, dict):
        return []
    for path in _EMPLOYEE_KB_ID_PATHS:
        value = _get_nested(cfg, path)
        if isinstance(value, list) and value:
            return value
    return []


def _resolve_employee_rag_disabled(cfg) -> bool:
    """员工级 RAG 总开关：仅接受严格 ``True``，避免脏数据（"true"/1）误关 RAG。"""
    if not isinstance(cfg, dict):
        return False
    for path in _EMPLOYEE_RAG_DISABLED_PATHS:
        if _get_nested(cfg, path) is True:
            return True
    return False


def _extract_weather_location(query: str) -> str | None:
    """
    从自然语言天气问句中提取地点，失败时返回 None。

    分两条路径：
    1. **中文 / CJK 路径**：匹配「X天气」「X的天气」「X的气温」等常见结构，
       并清理常见礼貌前缀（请问 / 麻烦问下…）和时间后缀（今天 / 明天 / 当前…）。
    2. **英文路径**：匹配 ``in / at / for / of <城市>`` 这类介词短语提取专有名词。
       Open-Meteo 的 geocoding 接受任何城市名，所以提取出连续大写开头的英文词
       即可（如 "Beijing"、"New York"、"San Francisco"）。

    对 ``"what's the weather like in Beijing today"`` 这类问句，先前的实现因为
    既匹配不到「X天气」，整段去标点后又超过 20 字而返回 None，导致结构化天气源
    完全派不上用场。新实现保留旧的中文短路径，再叠加一条英文专有名词提取，
    兼顾双语查询的稳定性。
    """
    q = (query or "").strip()
    if not q:
        return None

    # 第 1 步：定位「城市 + 时间词 / 天气词」结构。
    #
    # 思路：城市名一般是 2-4 个连续汉字，紧跟在时间词或天气词之前。
    # 但仅靠 ``{2,4}?`` 非贪婪量词 + lookahead 不够——当用户问句开头有
    # 「请问」「我想了解一下」「麻烦问下」这类礼貌引导词时，礼貌前缀会和真正
    # 的城市拼成 4 字串（例：「请问北京」紧跟"今天"），lookahead 仍会成功命中。
    # 解决方案：先把句首已知的礼貌引导词剥离掉，再在剩余文本上做城市定位；
    # 这样两类输入都能被命中（「北京…」/「请问北京…」），同时不会把「请问」
    # 误并入城市名。词表集中维护，扩展只需追加新引导词。
    cn_polite_lead_pat = re.compile(
        r"^(?:请问|麻烦问下|麻烦下|麻烦您|麻烦|问下|请帮我查一下|帮我查一下|"
        r"我想了解一下|我想了解|想了解一下|想了解|了解一下|了解|想知道|"
        r"请告诉我|告诉我|请|查一下)\s*"
    )
    q_for_loc = cn_polite_lead_pat.sub("", q).strip()
    cn_anchor_pat = re.compile(
        r"([\u4e00-\u9fff]{2,4}?)"
        r"(?=(?:今天|今日|明天|后天|本周|这周|当前|现在|此时|目前|"
        r"的天气|的气温|的气候|天气|气温|气候|阴晴|降雨|下雨|下雪|气象))"
    )
    am = cn_anchor_pat.search(q_for_loc)
    if am:
        loc = am.group(1).strip()
        if len(loc) >= 2:
            return loc

    # 第 2 步：兜底——直接匹配「X 天气」结构（覆盖第 1 步未命中的边缘表达）。
    cn_loc_pat = re.compile(
        r"([\u4e00-\u9fffA-Za-z]{2,20})\s*(?:的)?\s*(?:天气|气温|气候|阴晴|降雨|下雨|下雪|气象)"
    )
    cn_polite_prefix = re.compile(
        r"^(请问|麻烦问下|问下|请帮我查一下|帮我查一下|今天|今日|明天|后天|当前|现在|的)"
    )
    cn_time_suffix = re.compile(r"(今天|今日|明天|后天|本周|这周|当前|现在|的)$")

    m = cn_loc_pat.search(q)
    if m:
        loc = m.group(1).strip()
        # 多次剥离前后缀，处理「请问北京今天天气」「成都的当前天气」这种叠加表达。
        for _ in range(3):
            loc_new = cn_polite_prefix.sub("", loc).strip()
            loc_new = cn_time_suffix.sub("", loc_new).strip()
            if loc_new == loc:
                break
            loc = loc_new
        if loc and len(loc) >= 2:
            return loc

    en_after_prep = re.compile(
        r"\b(?:in|at|for|of|near|around)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})",
    )
    em = en_after_prep.search(q)
    if em:
        loc_en = em.group(1).strip()
        # 把"the/today/tomorrow"等通用词从尾巴上剥掉，避免 geocoding 拿到无效输入。
        loc_en = re.sub(
            r"\s+(today|tomorrow|now|tonight|currently|right\s+now)\s*$",
            "",
            loc_en,
            flags=re.IGNORECASE,
        ).strip()
        if loc_en:
            return loc_en

    cjk_only = re.sub(r"[^\u4e00-\u9fff]", "", q)
    if 2 <= len(cjk_only) <= 20:
        return cjk_only
    return None


def _extract_traffic_location(query: str) -> str | None:
    """从路况问题提取城市，如“成都堵不堵”→“成都”"""
    q = (query or "").strip()
    if not q:
        return None
    m = re.search(r"([\u4e00-\u9fffA-Za-z]{2,20})(?:堵不堵|拥堵|路况|交通)", q)
    if m:
        loc = m.group(1).strip()
        loc = re.sub(r"^(请问|我想了解一下|想了解一下|帮我查一下|查一下|今天|今日|现在)", "", loc).strip()
        loc = re.sub(r"(今天|今日|现在)$", "", loc).strip()
        if loc:
            return loc
    simple = re.sub(r"[^\u4e00-\u9fffA-Za-z]", "", q)
    simple = re.sub(r"(堵不堵|拥堵|路况|交通|今天|今日|现在)", "", simple)
    return simple if 2 <= len(simple) <= 20 else None


# 世界气象组织（WMO）weather_code → 自然语言描述映射表。
# Open-Meteo 沿用 WMO Code 4677，通过把数字解码成「晴 / 多云 / 小雨 / 雷雨」等
# 文字短语，让下游 LLM 不需要自己做编号→描述的猜测（这是先前出现「23°C
# 全部解读成 cloudy」这类幻觉的主要原因）。
# 文案分别提供中英文，由 prefer_zh_output 决定用哪种；扩展只需在 dict 里追加值。
_WMO_WEATHER_CODE_DESCRIPTIONS: dict[int, dict[str, str]] = {
    0: {"zh": "晴朗", "en": "clear sky"},
    1: {"zh": "晴间多云", "en": "mainly clear"},
    2: {"zh": "局部多云", "en": "partly cloudy"},
    3: {"zh": "阴天", "en": "overcast"},
    45: {"zh": "雾", "en": "fog"},
    48: {"zh": "结冰雾", "en": "depositing rime fog"},
    51: {"zh": "小毛毛雨", "en": "light drizzle"},
    53: {"zh": "毛毛雨", "en": "moderate drizzle"},
    55: {"zh": "强毛毛雨", "en": "dense drizzle"},
    56: {"zh": "冻毛毛雨", "en": "light freezing drizzle"},
    57: {"zh": "强冻毛毛雨", "en": "dense freezing drizzle"},
    61: {"zh": "小雨", "en": "slight rain"},
    63: {"zh": "中雨", "en": "moderate rain"},
    65: {"zh": "大雨", "en": "heavy rain"},
    66: {"zh": "冻雨", "en": "light freezing rain"},
    67: {"zh": "强冻雨", "en": "heavy freezing rain"},
    71: {"zh": "小雪", "en": "slight snow"},
    73: {"zh": "中雪", "en": "moderate snow"},
    75: {"zh": "大雪", "en": "heavy snow"},
    77: {"zh": "雪粒", "en": "snow grains"},
    80: {"zh": "阵雨", "en": "slight rain showers"},
    81: {"zh": "中等阵雨", "en": "moderate rain showers"},
    82: {"zh": "强阵雨", "en": "violent rain showers"},
    85: {"zh": "阵雪", "en": "slight snow showers"},
    86: {"zh": "强阵雪", "en": "heavy snow showers"},
    95: {"zh": "雷阵雨", "en": "thunderstorm"},
    96: {"zh": "雷阵雨伴小冰雹", "en": "thunderstorm with slight hail"},
    99: {"zh": "雷阵雨伴大冰雹", "en": "thunderstorm with heavy hail"},
}


def _wmo_describe(code: int | float | None, prefer_zh_output: bool) -> str:
    """把 WMO weather_code 翻译成自然语言；未知 code 返回友好的兜底文案。"""
    if code is None:
        return "未知" if prefer_zh_output else "unknown"
    try:
        idx = int(code)
    except (TypeError, ValueError):
        return "未知" if prefer_zh_output else "unknown"
    entry = _WMO_WEATHER_CODE_DESCRIPTIONS.get(idx)
    if not entry:
        return "未知天气" if prefer_zh_output else "unknown conditions"
    return entry["zh" if prefer_zh_output else "en"]


def _wind_direction_label(deg: float | int | None, prefer_zh_output: bool) -> str:
    """把风向角度（0=北，顺时针）翻译为 8 方位文字（北 / 东北 / 东 ...）。"""
    if deg is None:
        return "未知" if prefer_zh_output else "unknown"
    try:
        d = float(deg) % 360.0
    except (TypeError, ValueError):
        return "未知" if prefer_zh_output else "unknown"
    sectors_zh = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    sectors_en = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((d + 22.5) // 45) % 8
    return (sectors_zh if prefer_zh_output else sectors_en)[idx]


async def _open_meteo_weather_fallback(query: str, prefer_zh_output: bool = True) -> dict | None:
    """
    免费结构化天气源（Open-Meteo）查询。

    与早期实现的差异：
    - 现在作为天气查询的**主源**而非 fallback：Tavily 抓回的 weatherapi.com 页面
      只是 JSON 字符串前若干字符（typically 300-500 字），LLM 看到的是被截断的
      ``{'temp_c': 23.1, 'tem...`` 这种残缺数据，会自行编造 condition / wind_dir /
      mph 等字段。改为使用 Open-Meteo 的 JSON API 直接拿到结构化字段，再把它
      格式化成「晴间多云、温度 23℃、东南风 4.3km/h」这种**完整自然语言**，
      让 LLM 没有「猜测/补全」的可乘之机。
    - 字段扩展：天气状况描述（来自 WMO weather_code 映射）、风向（角度 → 八方位）、
      湿度、降水量、最高 / 最低气温——全部一次拉齐，避免下游再二次拼接。
    - 文案中英文双语：依据 ``prefer_zh_output`` 切换；这样英文问句拿到英文 content，
      不会再次跨语言翻译/掉细节。
    """
    loc = _extract_weather_location(query)
    if not loc:
        return None
    geocode_lang = "zh" if prefer_zh_output else "en"
    timeout = httpx.Timeout(8.0, connect=4.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        geo_resp = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": loc, "count": 1, "language": geocode_lang, "format": "json"},
        )
        if geo_resp.status_code != 200:
            return None
        geo_data = geo_resp.json() or {}
        results = geo_data.get("results") or []
        if not results:
            return None
        top = results[0]
        lat = top.get("latitude")
        lon = top.get("longitude")
        if lat is None or lon is None:
            return None
        city_name = top.get("name") or loc
        country = top.get("country") or ""

        weather_resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                # 拉齐主要 current 字段，确保 LLM 不需要自己「猜测」缺失项：
                # - temperature_2m / apparent_temperature  → 实测气温 / 体感温度
                # - relative_humidity_2m                   → 相对湿度
                # - precipitation                          → 当前小时降水量
                # - weather_code                           → WMO 编号（再翻译成自然语言）
                # - wind_speed_10m / wind_direction_10m    → 风速 / 风向角度
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,weather_code,wind_speed_10m,wind_direction_10m"
                ),
                # 同时取当天最高 / 最低，便于回答「今天最高/最低气温」类问题。
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "Asia/Shanghai",
                "forecast_days": 1,
            },
        )
        if weather_resp.status_code != 200:
            return None
        weather_data = weather_resp.json() or {}
        current = weather_data.get("current") or {}
        daily = weather_data.get("daily") or {}
        temp = current.get("temperature_2m")
        feels = current.get("apparent_temperature")
        humidity = current.get("relative_humidity_2m")
        precip = current.get("precipitation")
        wind = current.get("wind_speed_10m")
        wind_dir = current.get("wind_direction_10m")
        code = current.get("weather_code")
        obs_time = current.get("time")
        if temp is None and feels is None and wind is None and code is None:
            return None

        # 当日极值（数组取首值）；若服务端没返回这些字段，用 None 占位，避免拼接报错。
        def _first(v):
            if isinstance(v, list) and v:
                return v[0]
            return v

        t_max = _first(daily.get("temperature_2m_max"))
        t_min = _first(daily.get("temperature_2m_min"))
        precip_sum = _first(daily.get("precipitation_sum"))

        weather_desc = _wmo_describe(code, prefer_zh_output)
        wind_dir_label = _wind_direction_label(wind_dir, prefer_zh_output)

        if prefer_zh_output:
            extras = []
            if humidity is not None:
                extras.append(f"湿度 {humidity}%")
            if precip is not None:
                extras.append(f"当前降水 {precip}mm")
            if t_max is not None and t_min is not None:
                extras.append(f"今日最高 {t_max}°C / 最低 {t_min}°C")
            if precip_sum is not None:
                extras.append(f"今日总降水 {precip_sum}mm")
            extras_text = ("，" + "，".join(extras)) if extras else ""
            content = (
                f"{city_name}{('·' + country) if country else ''} 当前天气：{weather_desc}，"
                f"气温 {temp}°C（体感 {feels}°C），{wind_dir_label}风 {wind}km/h"
                f"{extras_text}。观测时间：{obs_time}。"
            )
        else:
            extras = []
            if humidity is not None:
                extras.append(f"humidity {humidity}%")
            if precip is not None:
                extras.append(f"current precipitation {precip}mm")
            if t_max is not None and t_min is not None:
                extras.append(f"today high {t_max}°C / low {t_min}°C")
            if precip_sum is not None:
                extras.append(f"today total precipitation {precip_sum}mm")
            extras_text = (", " + ", ".join(extras)) if extras else ""
            content = (
                f"{city_name}{(', ' + country) if country else ''} current weather: "
                f"{weather_desc}, temperature {temp}°C (feels like {feels}°C), "
                f"wind from {wind_dir_label} at {wind}km/h{extras_text}. "
                f"Observation time: {obs_time}."
            )
        title = (
            f"{city_name} 实时天气（Open-Meteo）"
            if prefer_zh_output
            else f"{city_name} current weather (Open-Meteo)"
        )
        return {
            "rank": 1,
            "title": title,
            "url": f"https://open-meteo.com/en/docs?latitude={lat}&longitude={lon}",
            "content": content,
            # 给一个偏高的分数，以便天气类查询在排序中稳定排在前面。
            "score": 0.95,
        }


async def _duckduckgo_traffic_fallback(query: str) -> list[dict]:
    """交通检索兜底：使用 DuckDuckGo 公共搜索抓取摘要。"""
    try:
        loc = _extract_traffic_location(query) or "当地"
        search_q = f"{loc} 今天 路况 拥堵"
        timeout = httpx.Timeout(10.0, connect=4.0)
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
        async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
            resp = await client.get("https://duckduckgo.com/html/", params={"q": search_q})
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, "html.parser")
            items: list[dict] = []
            for i, node in enumerate(soup.select(".result"), 1):
                a = node.select_one("a.result__a")
                snip = node.select_one(".result__snippet")
                if not a:
                    continue
                title = (a.get_text(" ", strip=True) or "").strip()
                href = (a.get("href") or "").strip()
                content = (snip.get_text(" ", strip=True) if snip else "").strip()
                if not (title and href):
                    continue
                items.append(
                    {
                        "rank": i,
                        "title": title[:200],
                        "url": href,
                        "content": content[:2000],
                        "score": max(0.5, 0.9 - i * 0.1),
                    }
                )
                if len(items) >= 5:
                    break
            return items
    except Exception as e:
        logger.warning(f"DuckDuckGo traffic fallback failed: {e}")
        return []


def _traffic_congestion_estimate(now: datetime) -> tuple[str, str]:
    """无实时数据时，按时段估算拥堵等级"""
    h = now.hour
    weekday = now.weekday()  # Mon=0
    is_workday = weekday < 5
    # 早晚高峰
    if is_workday and (7 <= h <= 9 or 17 <= h <= 19):
        return "较拥堵", "工作日通勤高峰"
     # 工作日白天
    if is_workday and (10 <= h <= 16):
        return "中等拥堵", "工作日白天车流较大"
    # 工作日其他时间
    if is_workday:
        return "基本通畅", "非通勤高峰时段"
    # 周末白天
    if 10 <= h <= 20:
        return "中等拥堵", "周末商圈/景点出行时段"
    return "基本通畅", "周末非高峰时段"


def _normalize_realtime_category(reason: str | None) -> str:
    """将QueryClassifier的reason映射为内部统一枚举：weather/time/news/market/traffic/general"""
    if not reason:
        return "general"
    r = reason.strip().lower()
    # 别名映射表（集中维护）
    aliases = {
        # 英文 / 规范值
        "weather": "weather",
        "time": "time",
        "news": "news",
        "market": "market",
        "traffic": "traffic",
        "general": "general",
        # 常见中文别名（可按需追加）
        "天气": "weather",
        "气象": "weather",
        "时间": "time",
        "日期": "time",
        "时刻": "time",
        "时钟": "time",
        "新闻": "news",
        "时事": "news",
        "资讯": "news",
        "股市": "market",
        "股票": "market",
        "行情": "market",
        "汇率": "market",
        "金价": "market",
        "交通": "traffic",
        "路况": "traffic",
        "堵车": "traffic",
        "拥堵": "traffic",
    }
    return aliases.get(r, r if r in {"weather", "time", "news", "market", "traffic", "general"} else "general")


# =============================================================================
# Base class for node implementations
# =============================================================================
class ConversationNodes:
    """
    Container class for all workflow nodes.

    Each node is an async method that takes a ConversationState
    and returns an updated ConversationState.
    """

    def __init__(self, workflow_instance):
        """
        Initialize nodes with reference to parent workflow.

        Args:
            workflow_instance: The ConversationWorkflow instance
        """
        self.workflow = workflow_instance

    # -------------------------------------------------------------------------
    # Workflow Nodes - Configuration Loading
    # -------------------------------------------------------------------------
    async def load_employee_config(self, state: ConversationState) -> ConversationState:
        """
        Load digital employee configuration from MongoDB.

        Retrieves employee-specific settings including:
        - Basic info (name, role, description)
        - Personality (tone, style, formality)
        - Capabilities (web_search_enabled, kb_ids)
        - FAQ settings (faq_sim_threshold, faq_top_k)

        Args:
            state: Current conversation state

        Returns:
            Updated state with employee_config populated

        Raises:
            ValueError: If employee_id not found in database
        """
        async with time_node("load_employee_config", state):
            db = await get_database()
            employee = await db.digital_employee_configs.find_one({"employee_id": state["employee_id"]})

            if not employee:
                error_msg = f"Employee config not found: employee_id={state['employee_id']}"
                logger.error(f"{error_msg}", exc_info=False)
                raise ValueError(error_msg)

            employee.pop("_id", None)

            # EmployeeSyncService 把 knowledge_kb_ids 等运行时设置写到 digital_employee_settings 集合，
            # 这里合并进来供下游统一从 employee_config 读取，避免路由因找不到 kb_ids 而误降级。
            setting_doc = await db.digital_employee_settings.find_one({"employee_id": state["employee_id"]})
            if setting_doc:
                setting_doc.pop("_id", None)
                employee["setting"] = setting_doc

            # 解析 kb_ids 时按 _EMPLOYEE_KB_ID_PATHS 优先级取值，并规范化到 employee_config["kb_ids"]，
            # 给老调用方留向后兼容；路由本身不再依赖 kb_ids（RAGAnything 是全局单库）。
            kb_ids = _resolve_employee_kb_ids(employee)
            employee["kb_ids"] = kb_ids
            state["employee_config"] = employee

            logger.info(
                f"Employee config loaded: employee_id={state['employee_id']}, "
                f"name={employee.get('name')}, kb_ids={kb_ids}, kb_count={len(kb_ids)}"
            )

        return state

    async def load_session_context(self, state: ConversationState) -> ConversationState:
        """
        Load or create session context from MongoDB.

        Handles:
        - Loading existing session with message history
        - Creating new session for first-time users
        - Updating last_activity timestamp

        Args:
            state: Current conversation state

        Returns:
            Updated state with context populated
        """
        async with time_node("load_session_context", state):
            try:
                db = await get_database()
                session = await db.sessions.find_one({"session_id": state["session_id"]})

                if session:
                    # Load recent messages (last 10 for context)
                    state["context"] = {
                        "messages": session.get("context_messages", [])[-10:],
                        "message_count": session.get("message_count", 0)
                    }
                    # Update activity timestamp
                    await db.sessions.update_one(
                        {"session_id": state["session_id"]},
                        {"$set": {"last_activity": datetime.now()}}
                    )
                else:
                    # Create new session
                    session_model = SessionModel(
                        session_id=state["session_id"],
                        user_id=state["user_id"],
                        employee_id=state["employee_id"],
                        status="active",
                        message_count=0
                    )
                    await db.sessions.insert_one(session_model.model_dump())
                    state["context"] = {"messages": [], "message_count": 0}
                    state["sources"] = []  # 初始化 sources 列表

                logger.debug(
                    f"Session context loaded: session_id={state['session_id']}, "
                    f"message_count={state['context']['message_count']}"
                )

            except Exception as e:
                logger.error(f"Failed to load session context: {str(e)}", exc_info=True)
                state["context"] = {"messages": [], "message_count": 0}

        return state

    # -------------------------------------------------------------------------
    # Workflow Nodes - Input Processing
    # -------------------------------------------------------------------------
    async def validate_input(self, state: ConversationState) -> ConversationState:
        """
        Validate and sanitize user input.

        Performs basic validation:
        - Check for empty or malicious input
        - Detect sensitive content using default + employee-specific words

        Args:
            state: Current conversation state

        Returns:
            Updated state with validation results
        """
        async with time_node("validate_input", state):
            query = state["user_query"].strip().lower()

            employee_config = state.get("employee_config") or {}

            # Build sensitive words list (default + employee-specific)
            sensitive_words = set(DEFAULT_SENSITIVE_WORDS)

            # Add employee-specific sensitive words from safe_rule.sensitive_ids
            safe_rule = employee_config.get("safe_rule") or {}
            sensitive_ids = safe_rule.get("sensitive_ids") or []
            employee_words_lower: set = set()
            if sensitive_ids:
                # Load sensitive words from database
                db = get_database()
                if sensitive_ids:
                    cursor = db.thesaurus_sensitive.find(
                        {"thesaurus_id": {"$in": sensitive_ids}},
                        {"word": 1, "_id": 0}
                    )
                    sensitive_docs = await cursor.to_list(length=None)
                    employee_words = [doc.get("word", "") for doc in sensitive_docs if doc.get("word")]
                    sensitive_words.update(employee_words)
                    employee_words_lower = {w.lower() for w in employee_words if w}

            # Check if query contains any sensitive word (ASCII terms: whole-word only)
            check_words_lower = DEFAULT_SENSITIVE_WORDS_LOWER | employee_words_lower
            query_lower = query.lower()
            has_sensitive = any(
                sensitive_term_matches_query(query_lower, word) for word in check_words_lower
            )

            state["has_sensitive"] = has_sensitive

            # DEBUG: 输出敏感词检测结果
            matched_words = [
                w for w in sensitive_words if sensitive_term_matches_query(query_lower, w.lower())
            ]
            logger.info(
                f"[DEBUG] Sensitive check | query={query[:50]} | "
                f"words_count={len(sensitive_words)} | has_sensitive={has_sensitive} | "
                f"matched={matched_words[:5]}"
            )

            if has_sensitive:
                logger.info(
                    f"Sensitive word detected in query, user_id={state.get('user_id')}, "
                    f"employee_id={state.get('employee_id')}, query={query[:100]}"
                )
        return state

    # -------------------------------------------------------------------------
    # Workflow Nodes - Query Classification (Early Exit)
    # -------------------------------------------------------------------------
    async def classify_query_type(self, state: ConversationState) -> ConversationState:
        """
        Query Classification - 使用 LLM 进行细粒度查询分类并路由。

        使用 QueryClassifier 进行分类，支持 9 种类别：
        - math_problem, concept_explain, greeting, english_query
        - realtime_query, general_knowledge, chit_chat, noise, other

        分类结果映射到 workflow state：
        - greeting → intent="greeting", complexity_score=0.0
        - realtime_query → is_realtime_query=True
        - math_problem → is_math_problem=True
        - noise → 返回友好提示后结束
        - 其他 → 继续正常流程

        Args:
            state: Current conversation state

        Returns:
            Updated state with query_type classification
        """
        async with time_node("classify_query_type", state):
            # 1.基础变量初始化
            query = state["user_query"].strip()
            context_messages = (state.get("context") or {}).get("messages") or []
            # 获取上一轮用户问题
            last_user_query = next(
                (
                    (m.get("content") or "").strip()
                    for m in reversed(context_messages)
                    if m.get("role") == "user" and (m.get("content") or "").strip()
                ),
                "",
            )
            # 2.格式化对话上下文
            dialog_text = format_dialog_for_resolver(context_messages)
            # 统计上下文用户消息数量
            session_user_count = sum(
                1 for m in context_messages if m.get("role") == "user" and (m.get("content") or "").strip()
            )
            # 上下文不足2轮时，从DB补全历史对话
            if state.get("session_id") and session_user_count < 2:
                try:
                    db = await get_database()
                    recent_turns = await db.conversations.find(
                        {"session_id": state.get("session_id", "")},
                        {"_id": 0, "user_query": 1, "ai_response": 1},
                    ).sort("created_at", 1).limit(30).to_list(length=30)
                    dialog_text = augment_dialog_with_persisted_turns(
                        dialog_text, list(recent_turns), query
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to merge persisted dialog for resolver: session_id={state.get('session_id')}, error={e}",
                        exc_info=True,
                    )
            # 3.意图分类器初始化 + 原始意图预判 + 上下文相关性判定（并行）
            #
            # 性能优化（与原 3 节合并）：
            #   - 原实现把 ``aclassify`` 与 ``aclassify_context_dependence`` 串行调用，
            #     两次小模型推理叠加约 2s。两者输入彼此独立（前者用 query+last_user_query，
            #     后者用 query+dialog_text），完全可以 ``asyncio.gather`` 并行；
            #   - 仅当 ``dynamic_context_memory_enabled`` 启用且 ``dialog_text`` 非空时
            #     才需要真正发起依赖判定 LLM 调用，其余分支直接拿到 ``unrelated``，
            #     避免无意义的 LLM 等待；
            #   - 异常处理在原 helper 内部已退化为 ``False / "fallback"``，并行化不会
            #     放大故障范围。
            classifier = get_query_classifier()
            dynamic_ctx_enabled = bool(getattr(settings, "dynamic_context_memory_enabled", True))
            classify_task = asyncio.create_task(
                classifier.aclassify(query, context_query=last_user_query or None)
            )

            need_llm_dependence = dynamic_ctx_enabled and bool(dialog_text.strip())
            if need_llm_dependence:
                dependence_task = asyncio.create_task(
                    classifier.aclassify_context_dependence(query, dialog_text)
                )
                original_ctx_result, (is_related, judge_reason) = await asyncio.gather(
                    classify_task, dependence_task
                )
                context_dependence = "related" if is_related else "unrelated"
                ctx_reason = judge_reason
            else:
                original_ctx_result = await classify_task
                if not dynamic_ctx_enabled:
                    context_dependence = "related"
                    ctx_reason = "disabled"
                else:
                    # 无历史就没有"相关"可言；标记为 unrelated，下游一律不去拼历史。
                    context_dependence = "unrelated"
                    ctx_reason = "no_history"

            # 高置信度非实时查询，保留原始意图
            preserve_original_intent = (
                original_ctx_result.label != "realtime_query"
                and original_ctx_result.confidence == "high"
                and len(query) >= 8
            )
            state["context_dependence"] = context_dependence
            state["context_dependence_reason"] = ctx_reason
            logger.info(
                f"Dynamic context memory: dependence={context_dependence}, reason={ctx_reason}, "
                f"dialog_empty={not dialog_text.strip()}, query={query[:60]!r}"
            )

            # 4.上下文消歧，改写问句
            #   只有当 dependence == "related" 时才进行消歧；否则保持原句，避免把
            #   历史话题硬塞进改写，污染下游分类与 RAG 检索。
            resolved = query
            if context_dependence == "related" and dialog_text.strip():
                resolved = await classifier.aresolve_standalone_query(query, dialog_text)
                if not (resolved or "").strip():
                    resolved = query
            resolved = resolved.strip()
            state["rewritten_query"] = resolved
            state["query_rewritten"] = resolved != query
            # 显式记录改写结果，便于排查"代词追问被错判为 unrelated"或"改写器未触发"。
            logger.info(
                f"Standalone query resolution: original={query[:80]!r}, "
                f"resolved={resolved[:80]!r}, query_rewritten={resolved != query}, "
                f"context_dependence={context_dependence}"
            )

            # 5.对改写后的问句分类
            #
            # 性能优化：当 ``resolved == query``（无消歧改写发生）时，第二次
            # ``aclassify`` 与第一次 ``aclassify(query, last_user_query)`` 的输入
            # 实际等价（同一句话；上下文判定为 unrelated 时 LLM 已确认上一轮语境
            # 不会改变本句的语义类别）——直接复用 ``original_ctx_result`` 即可，
            # 节省一次 ~1.6s 的 14B 模型 RTT。仅在真的发生改写时才需要再次分类，
            # 用来检测"改写偏移"（rewrite drift），保留下游既有的偏移修正逻辑。
            if resolved == query:
                result_llm = original_ctx_result
                logger.info(
                    f"Skipping redundant classification (resolved == query): "
                    f"label={result_llm.label}, confidence={result_llm.confidence}"
                )
            else:
                result_llm = await classifier.aclassify(resolved, context_query=None)
            # 短问句场景：恢复原始实时查询意图
            if (
                result_llm.label != "realtime_query"
                and original_ctx_result.label == "realtime_query"
                and original_ctx_result.confidence == "high"
                and len(query) < 8
            ):
                logger.info(
                    "Restore realtime intent for short contextual follow-up: "
                    f"query={query[:80]}, resolved={resolved[:80]}"
                )
                result_llm = original_ctx_result
            # 高置信度原始意图：防止改写偏移
            if (
                result_llm.label == "realtime_query"
                and original_ctx_result.label != "realtime_query"
                and original_ctx_result.confidence == "high"
            ):
                logger.info(
                    "Keep original intent classification to avoid rewrite drift: "
                    f"original={original_ctx_result.label}, rewritten={result_llm.label}, "
                    f"query={query[:80]}, resolved={resolved[:80]}"
                )
                result_llm = original_ctx_result
            # 6.启发式规则：升级为实时查询
            boost = heuristic_realtime_category(resolved)
            if (
                boost is not None
                and result_llm.label not in _REALTIME_HEURISTIC_SKIP_LABELS
                and not preserve_original_intent
            ):
                logger.info(
                    f"Realtime heuristic upgrade: boost={boost}, llm_label={result_llm.label}, "
                    f"query={resolved[:80]}"
                )
                result = ClassificationResult(
                    label="realtime_query",
                    confidence=result_llm.confidence,
                    reason=boost,
                )
            else:
                result = result_llm

            # 6.5 二元 LLM 兜底：主分类未识别为 realtime 但置信度不 high 时，
            # 用一次"是否需要联网/最新信息"的 yes/no LLM 校验把漏检拉回 realtime_query。
            # 设计要点：
            # - 通过 settings.realtime_query_llm_fallback_enabled 开关控制（可在 .env 关闭）；
            # - _REALTIME_HEURISTIC_SKIP_LABELS 已包含 math_problem/greeting/noise/realtime_query，
            #   数学题/问候/噪声不会被升级（保护现有路由：数学题继续走 RAG / Phi-4）；
            # - preserve_original_intent 命中时跳过，避免覆盖高置信度的非实时原意图；
            # - aneed_realtime 内部异常一律返回 False，不让兜底机制反过来引入新故障。
            if (
                getattr(settings, "realtime_query_llm_fallback_enabled", False)
                and result.label not in _REALTIME_HEURISTIC_SKIP_LABELS
                and result.confidence != "high"
                and not preserve_original_intent
                and await classifier.aneed_realtime(resolved)
            ):
                logger.info(
                    "Realtime LLM fallback promoted to realtime_query: "
                    f"prev_label={result.label}, prev_confidence={result.confidence}, "
                    f"query={resolved[:80]}"
                )
                result = ClassificationResult(
                    label="realtime_query",
                    confidence=result.confidence,
                    reason="general",
                )

            # 6.6 数学题 / 教材概念题 启发式补位
            #
            # 背景：qwen2.5:7b 这类小分类器对没有"求/解/计算"动词的几何应用题
            # （"已知圆锥的底面半径为 1，高为 2，则圆锥的侧面积为多少?"），
            # 以及长篇教材式提问（"在数列的学习中…请分别说明…推导方法…比较异同"）
            # 存在系统性漏判，会落到 ``general_knowledge / chit_chat / other`` 这类兜底
            # 标签上，导致原本应走 Phi-4 的题目被通用 LLM 接住、原本应走 RAG 的教材
            # 题被通用 LLM 直答。
            #
            # 设计要点（与 realtime 启发式同思路）：
            # - 只对 LLM 弱标签（``HEURISTIC_PROMOTABLE_LABELS``）补位，保护 LLM 已识别准确的强分类；
            # - 数学题启发式优先（"已知…为多少" 这种模式比"教学语境"更具体）；
            # - 启发式自身彼此互斥（math 启发式内部已经把 ``请讲解 / 推导方法 / 异同``
            #   这类元语言信号当作排除项），不会把 Q4 类教材题误升为数学题。
            if result.label in HEURISTIC_PROMOTABLE_LABELS:
                if heuristic_math_problem(resolved):
                    logger.info(
                        "Math heuristic promoted to math_problem: "
                        f"prev_label={result.label}, prev_confidence={result.confidence}, "
                        f"query={resolved[:80]}"
                    )
                    result = ClassificationResult(
                        label="math_problem",
                        confidence=result.confidence,
                        reason="heuristic_math",
                    )
                elif heuristic_concept_explain(resolved):
                    logger.info(
                        "Concept heuristic promoted to concept_explain: "
                        f"prev_label={result.label}, prev_confidence={result.confidence}, "
                        f"query={resolved[:80]}"
                    )
                    result = ClassificationResult(
                        label="concept_explain",
                        confidence=result.confidence,
                        reason="heuristic_concept",
                    )

            logger.info(
                f"LLM classification: label={result.label}, confidence={result.confidence}, "
                f"reason={result.reason}, query={resolved[:50]}"
            )

            # 6.7 噪声预设话术安全护栏（"抱歉，我没有听清您的问题"路径）
            #
            # 背景：小分类器对短/口语化输入误判率高，命中 noise 后**完全跳过 LLM**
            # 直接返回预设话术，会让用户在数字人侧误以为"麦克风/ASR 故障"。
            # 这里在分类结果之上加三道独立闸门（置信度 / 长度 / 启发式），任一不过
            # 即把标签降级为 "other"（→ GENERAL_LLM 由 LLM 自己兜住）。
            #
            # 维护原则：闸门规则集中在 ``intent_routing.apply_noise_preset_gate``，
            # 这里只负责"传配置 + 写日志 + 改 result"，不在节点里重写规则；
            # 通过 ``settings.noise_preset_response_enabled=False`` 可一键回滚整条路径。
            gated_label, downgrade_reason = apply_noise_preset_gate(
                result.label,
                result.confidence,
                resolved,
                noise_preset_enabled=getattr(settings, "noise_preset_response_enabled", True),
                min_confidence=getattr(settings, "noise_preset_min_confidence", "high"),
                max_query_length=getattr(settings, "noise_preset_max_query_length", 12),
            )
            if gated_label != result.label:
                logger.info(
                    "Noise preset gate downgraded label: "
                    f"prev_label={result.label}, prev_confidence={result.confidence}, "
                    f"new_label={gated_label}, reason={downgrade_reason}, "
                    f"query={resolved[:80]!r}"
                )
                # 替换 label 但保留原始置信度与 reason，便于审计；
                # 下游的 ``case "noise":`` 不会被触发，自动落到默认 case → GENERAL_LLM。
                result = ClassificationResult(
                    label=gated_label or "other",
                    confidence=result.confidence,
                    reason=f"noise_gate:{downgrade_reason}",
                )

            # 7.解析目标年份，存入状态：用于后续生成阶段保持“今年/明年/去年”一致。
            target_year = resolve_target_year_from_query(resolved)
            if target_year is not None:
                state["target_year"] = target_year

            # 8.保存分类结果到状态
            state["classification_label"] = result.label
            state["classification_confidence"] = result.confidence
            state["classification_reason"] = result.reason

            # 8.5 由分类标签解析出 answer_mode（数据驱动，禁止在此处写硬编码 if）
            #     - 路由表集中维护在 intent_routing.INTENT_TO_ANSWER_MODE；
            #     - 下游 route_after_classification 与 generate_answer 仅依赖
            #       state["answer_mode"]，不再叠加 is_realtime / is_math_problem
            #       等一堆复合条件来决策；
            #     - 未识别标签自动落到 GENERAL_LLM，避免进入 RAG / Phi-4 等带外
            #       依赖的路径。
            answer_mode = resolve_answer_mode(result.label)
            state["answer_mode"] = answer_mode.value

            # 9.根据分类标签设置状态
            match result.label:
                # 问候语
                case "greeting":
                    state["intent"] = "greeting"
                    state["complexity_score"] = 0.0
                    state["complexity_reason"] = "greeting"
                    state["is_realtime_query"] = False
                    # 10. 添加问候语来源
                    state["sources"].append({
                        "type": "text",
                        "from": "greeting",
                        "text": query,
                        "citations": []
                    })
                # 实时查询
                case "realtime_query":
                    state["is_realtime_query"] = True
                    state["realtime_category"] = _normalize_realtime_category(result.reason)
                    state["realtime_detect_reason"] = f"llm:{result.confidence}"
                    state["intent"] = "general_query"
                # 数学题
                case "math_problem":
                    state["is_math_problem"] = True
                    state["intent"] = "general_query"
                # 无效噪声
                case "noise":
                    state["intent"] = "noise"
                    state["sources"].append({
                        "type": "text",
                        "from": "noise_response",
                        "text": NOISE_PRESET_RESPONSE_TEXT,
                        "citations": []
                    })
                # 默认通用查询（含 concept_explain / english_query / general_knowledge / chit_chat / other）
                case _:
                    state["is_realtime_query"] = False
                    state["intent"] = "general_query"

            # 10.日历日期直出答案（优先原问句，再用改写后问句）避免把“习俗/由来”等非日期问题误转为日期回答。
            # `language_hint_query` 始终传入用户原始问句，避免改写后的查询语言污染输出语言判定
            # （例如：英文问句被消歧/改写为中文，造成英文问、中文答的混语回复）。
            prefer_zh_output = resolve_prefer_zh_output(state)
            direct = calendar_direct_text_answer(
                query,
                prefer_zh_output,
                anchor_year=state.get("target_year"),
                language_hint_query=state.get("user_query") or query,
            )
            if not direct and result.label == "realtime_query":
                direct = calendar_direct_text_answer(
                    resolved,
                    prefer_zh_output,
                    anchor_year=state.get("target_year"),
                    language_hint_query=state.get("user_query") or query,
                )
            # 命中日历答案，设置本地计算状态
            if direct:
                state["direct_text_answer"] = direct
                state["is_realtime_query"] = True
                state["realtime_category"] = "time"
                state["realtime_detect_reason"] = "local_calendar_resolver"
        return state

    # -------------------------------------------------------------------------
    # Math Problem Detection
    # -------------------------------------------------------------------------
    # 概念性问题排除关键词（不算数学题）
    CONCEPT_KEYWORDS = ["是什么", "什么是", "介绍", "解释", "定义", "概念"]
    # 数学操作关键词
    MATH_OPERATION_KEYWORDS = ["求", "计算", "解", "证明", "推导", "化简"]
    # 数学对象关键词
    MATH_OBJECT_KEYWORDS = ["函数", "方程", "不等式", "集合", "数列", "三角函数", "导数", "积分", "极限", "椭圆", "双曲线", "抛物线"]

    async def check_math_problem(self, state: ConversationState) -> ConversationState:
        """
        数学问题检测节点 - 检测用户查询是否为数学问题。

        使用基于规则的模式匹配：
        - 排除概念性问题（"什么是函数"、"介绍一下三角函数"等）
        - 检测数学操作+数学对象（"求函数值域"、"解方程"等）

        Args:
            state: Current conversation state

        Returns:
            Updated state with is_math_problem populated
        """
        async with time_node("check_math_problem", state):
            query = state["user_query"].strip()

            # 执行数学问题检测
            is_math, reason = self._detect_math_problem(query)
            state["is_math_problem"] = is_math

            logger.info(
                f"Math detection result: is_math={is_math}, reason={reason}, "
                f"query={query[:50]}"
            )

            # 如果是数学问题，添加来源信息
            if is_math:
                state["sources"].append({
                    "type": "text",
                    "from": "phi4_math",
                    "text": query,
                    "citations": []
                })
                logger.info(f"Math problem source added | query={query[:50]}")

        return state

    @staticmethod
    def _detect_math_problem(query: str) -> tuple[bool, str]:
        """
        基于规则的模式匹配检测数学问题。

        规则:
        1. 概念性问题排除（"是什么"、"什么是"等 + 短问题）
        2. 同时包含数学操作关键词 + 数学对象关键词 → 数学题
        3. 其他 → 普通查询

        Args:
            query: 用户查询

        Returns:
            (is_math, reason) - 是否为数学问题及原因
        """
        query_lower = query.strip().lower()

        # 1. 概念性问题排除
        for concept_kw in ConversationNodes.CONCEPT_KEYWORDS:
            if concept_kw in query_lower and len(query) < 50:
                return (False, "concept_question")

        # 2. 同时包含数学操作+数学对象 → 数学题
        has_operation = any(kw in query_lower for kw in ConversationNodes.MATH_OPERATION_KEYWORDS)
        has_object = any(kw in query_lower for kw in ConversationNodes.MATH_OBJECT_KEYWORDS)

        if has_operation and has_object:
            return (True, "math_problem")

        return (False, "general_query")

    @staticmethod
    def route_after_math_check(state: ConversationState) -> str:
        """
        路由决策: 数学检测后的下一步。

        Args:
            state: Current conversation state

        Returns:
            目标节点名称 (math/normal)
        """
        if state.get("is_math_problem", False):
            return "math"
        return "normal"

    @staticmethod
    def route_after_classification(state: ConversationState) -> str:
        """
        路由决策: 查询分类后的下一步。

        路由优先级（高到低）：
            1. ``intent in {greeting, noise}`` 直接走 greeting 分支：
               - 这两类已在 classify_query_type 中预先生成 sources/preset 答案，
                 跳过 web/RAG 节省时延。
            2. ``is_realtime_query`` 为真 → realtime（含日历直出兜底，下游再判断）：
               - 日历问题、启发式升级、LLM 兜底都会把这个标志置真，
                 此处无需关心具体子类型。
            3. 其余情况按 ``state["answer_mode"]`` 数据驱动：
               - PHI4_MATH        → math
               - RAG_WITH_FALLBACK → rag
               - GENERAL_LLM / 默认 → general

        Args:
            state: Current conversation state

        Returns:
            目标分支名（``ROUTE_BRANCH_*`` 之一）
        """
        intent = state.get("intent")
        if intent in ("greeting", "noise"):
            return ROUTE_BRANCH_GREETING
        # 实时类必须先经过 web_search（日历直出在 generate_answer 内部短路），
        # 不能让 answer_mode 把这条路径降级成普通 LLM。
        if state.get("is_realtime_query"):
            return ROUTE_BRANCH_REALTIME
        # 表驱动：把分类标签 → 路由分支的所有判断集中到 intent_routing。
        return resolve_route_branch(state.get("answer_mode"))

    # -------------------------------------------------------------------------
    # Workflow Nodes - Complexity Evaluation
    # -------------------------------------------------------------------------
    async def evaluate_complexity(self, state: ConversationState) -> ConversationState:
        """
        Query Complexity Evaluation - 评估问题复杂度，决定使用本地还是外部模型。

        使用本地 Ollama 模型快速评估问题复杂度（0-10分）：
        - 0-3分：简单问题 - 本地 Ollama 足够
        - 4-6分：中等复杂 - 可用本地，必要时用外部
        - 7-10分：复杂问题 - 使用外部 API 模型

        复杂度评估维度：
        1. 问题长度（越长越复杂）
        2. 问题类型（简单问答 vs 复杂推理）
        3. 是否需要多步推理
        4. 是否需要综合多个信息源
        5. 意图是否清晰

        Args:
            state: Current conversation state

        Returns:
            Updated state with complexity_score and complexity_reason populated
        """
        async with time_node("evaluate_complexity", state):
            query = state["user_query"].strip()

            # 默认复杂度
            state["complexity_score"] = 3.0
            state["complexity_reason"] = "default"

            # 非混合模式：跳过复杂度评估
            routing_mode = getattr(settings, 'llm_routing_mode', 'local_only')
            if routing_mode != 'hybrid':
                logger.debug(
                    f"Complexity evaluation skipped (routing_mode={routing_mode})"
                )
                return state

            # 问候语检测已在 classify_query_type 节点中处理，此处无需重复

            # 快速启发式评估（跳过 LLM 调用以提高响应速度）
            # LLM 评估虽然更准确，但会增加 15+ 秒延迟，影响用户体验
            score = heuristic_complexity(query)

            # 根据查询特征优化 reason 分类
            reason = "heuristic"
            if any(kw in query for kw in ["集合", "函数", "定理", "公式", "定义", "什么是"]):
                reason = "数学概念"
            elif any(kw in query for kw in ["证明", "推导", "为什么"]):
                reason = "数学推理"
            elif any(kw in query for kw in ["分析", "比较", "总结"]):
                reason = "综合分析"

            state["complexity_score"] = score
            state["complexity_reason"] = reason

            logger.info(
                f"Complexity evaluated (heuristic): {score}/10 - {reason}"
            )

        return state

    # -------------------------------------------------------------------------
    # Workflow Nodes - Web Search
    # -------------------------------------------------------------------------
    async def web_search(self, state: ConversationState) -> ConversationState:
        """
        Web Search - Fetch realtime information from the internet.

        Uses Tavily Search API to get current data for:
        - Realtime queries (weather, news, prices)
        - Low relevance fallback (when knowledge base doesn't have answer)

        Checks:
        1. Web search enabled in settings
        2. Tavily API key configured
        3. Employee has web_search capability

        Args:
            state: Current conversation state

        Returns:
            Updated state with web_search_results populated
        """
        async with time_node("web_search", state):
            try:
                # Check if web search is enabled
                if not settings.web_search_enabled:
                    logger.info("Web search is disabled in settings")
                    state["web_search_results"] = []
                    state["web_search_used"] = False
                    state["web_search_error"] = None
                    return state

                if not settings.tavily_api_key:
                    logger.warning("Tavily API key is not configured")
                    state["web_search_results"] = []
                    state["web_search_used"] = False
                    state["web_search_error"] = "Tavily API key not configured"
                    return state

                # Check employee config for web search permission
                employee_config = state.get("employee_config", {})
                capabilities = employee_config.get("capabilities", {})
                if not capabilities.get("web_search_enabled", True):
                    logger.info(f"Web search disabled for employee: {state.get('employee_id')}")
                    state["web_search_results"] = []
                    state["web_search_used"] = False
                    state["web_search_error"] = None
                    return state

                # 日历问题：优先本地推算，避免分类器把 reason 标成 general 仍去联网抄错误示例
                if state.get("is_realtime_query"):
                    q_cal = (state.get("rewritten_query") or state.get("user_query") or "").strip()
                    direct_cal = calendar_direct_text_answer(
                        q_cal,
                        resolve_prefer_zh_output(state),
                        anchor_year=state.get("target_year"),
                        language_hint_query=state.get("user_query") or q_cal,
                    )
                    if direct_cal:
                        state["web_search_results"] = []
                        state["web_search_used"] = False
                        state["web_search_error"] = None
                        state["direct_text_answer"] = direct_cal
                        state["realtime_category"] = "time"
                        logger.info(
                            "Web search skipped: local_calendar_resolver, "
                            f"query={state.get('user_query', '')[:80]}"
                        )
                        return state

                query = state["user_query"]
                logger.info(
                    f"Web search started: query={query[:100]}, "
                    f"is_realtime={state.get('is_realtime_query')}"
                )

                # 天气类查询：优先用结构化天气源（Open-Meteo），跳过 Tavily。
                # 之前的实现把 Tavily 当主源，weatherapi.com 等页面被 Tavily 截成几百字
                # 的 JSON 残片（如 ``{'temp_c': 23.1, 'tem...``），LLM 看不到完整字段时
                # 会自行编造 "cloudy / 4.3 mph / from south" 这类幻觉。换成直接 API
                # 拿全字段（气象代码、风向角度、湿度、降水），让 LLM 只做"组织语句"
                # 而非"补全数据"，从源头消除幻觉。
                # Open-Meteo 失败时（地理编码无果 / 接口超时）才回退到 Tavily 通用搜索，
                # 保留旧路径作为兜底，避免天气查询彻底无结果。
                if state.get("realtime_category") == "weather":
                    structured = await _open_meteo_weather_fallback(
                        query, prefer_zh_output=resolve_prefer_zh_output(state)
                    )
                    if structured:
                        state["web_search_results"] = [structured]
                        state["web_search_used"] = True
                        state["web_search_error"] = None
                        state["sources"].append({
                            "type": "text",
                            "from": "web_search",
                            "text": state.get("rewritten_query", state["user_query"]),
                            "citations": [{
                                "title": structured.get("title", ""),
                                "url": structured.get("url", ""),
                                "score": structured.get("score", 0.0),
                                "snippet": structured.get("content", "")[:300],
                            }],
                        })
                        logger.info(
                            "Weather query handled by Open-Meteo (primary structured source); "
                            f"skipping Tavily. query={query[:80]}"
                        )
                        return state
                        logger.info(
                            "Open-Meteo unavailable for weather query; falling back to Tavily. "
                            f"query={query[:80]}"
                        )

                # 「当前北京时间/几点几分」：爬虫摘要常混入错误时区≈±8 小时，
                # 用 Asia/Shanghai 服务端时刻作为权威源，跳过 Tavily（与结构化天气同源思路）。
                if state.get("realtime_category") == "time":
                    q_time = (state.get("rewritten_query") or state.get("user_query") or "").strip()
                    _pref_zh = resolve_prefer_zh_output(state)
                    if skip_web_use_authoritative_beijing_wall_clock(
                        q_time, prefer_zh_output=_pref_zh
                    ):
                        _rec = authoritative_wall_clock_search_record(
                            q_time, prefer_zh_output=_pref_zh
                        )
                        _clean = sanitize_wall_clock_record_for_state(_rec)
                        state["web_search_results"] = [_clean]
                        state["web_search_used"] = True
                        state["web_search_error"] = None
                        state["sources"].append({
                            "type": "text",
                            "from": "web_search",
                            "text": state.get("rewritten_query", state["user_query"]),
                            "citations": [citation_dict_from_wall_clock_record(_rec)],
                        })
                        logger.info(
                            "Web search skipped: authoritative Beijing wall clock, "
                            f"query={q_time[:80]}"
                        )
                        return state

                now = datetime.now()
                realtime_category = state.get("realtime_category", "") or "general"
                policy = DEFAULT_POLICIES.get(realtime_category, DEFAULT_POLICIES["general"])
                if realtime_category == "market":
                    _mh = (getattr(settings, "web_search_market_anchor_hint_zh", None) or "").strip()
                    if _mh:
                        policy = RecencyPolicy(
                            max_age_days=policy.max_age_days,
                            prefer_today=policy.prefer_today,
                            query_hint=_mh,
                        )
                anchored_query = query
                used_duck_fallback = False

                # 检索短语 LLM 改写：把"我想了解一下今天成都堵不堵？"这类口语化原句
                # 改写为搜索引擎友好的精炼短语（不依赖任何关键字 / 模板）。
                # 失败/为空时保守回退到原 query，不会让现有行为变差。
                # 通过 settings.realtime_query_search_rewrite_enabled 控制（可在 .env 关闭）。
                #
                # 语言一致性守门（避免英文问句被改写成中文短语后污染 Tavily 召回，
                # 进而让下游 LLM 看到外文 context 跟着语言飘移）：改写产物的主导
                # 语言必须与原 query 一致，否则回退原 query。复用 has_language_drift。
                if (
                    state.get("is_realtime_query")
                    and getattr(settings, "realtime_query_search_rewrite_enabled", False)
                ):
                    rewritten = await get_query_classifier().agen_search_query(
                        query, hint=realtime_category
                    )
                    if rewritten and has_language_drift(query, rewritten):
                        logger.info(
                            "Search rewrite language drift, falling back to original query: "
                            f"original={query[:80]!r}, rewritten={rewritten[:80]!r}"
                        )
                        rewritten = None
                    if rewritten:
                        query = rewritten

                # 实时类查询统一追加时间锚点（通用）：避免“今年/当前/最近”等相对时间漂移。
                if state.get("is_realtime_query"):
                    anchored_query = build_time_anchored_query(query, now, policy)

                # 行情类：advanced 检索更容易带回含「收盘」「收报」数字的正文片段（可配置关闭）。
                _tavily_depth = (
                    "advanced"
                    if realtime_category == "market"
                    and getattr(settings, "web_search_market_tavily_advanced", True)
                    else "basic"
                )
                web_search_tool = TavilySearchResults(
                    max_results=10,  # 返回结果数量，默认 5
                    search_depth=_tavily_depth,
                    tavily_api_key=settings.tavily_api_key,
                )
                search_results = await web_search_tool.ainvoke({"query": anchored_query})
                # 锚定查询可能过窄：实时查询无结果时，用原始查询重试一次（通用回退）。
                if (
                    state.get("is_realtime_query")
                    and anchored_query != query
                    and (not search_results)
                ):
                    logger.info("Web search retry with raw query after anchored miss")
                    search_results = await web_search_tool.ainvoke({"query": query})
                # 交通类查询：首轮无结果时，使用更精准的关键词（地点+路况+拥堵）重试搜索
                if (
                    state.get("is_realtime_query")
                    and state.get("realtime_category") == "traffic"
                    and (not search_results)
                ):
                    loc = _extract_traffic_location(query) or "当地"
                    focused_query = f"{loc} 实时路况 拥堵 情况"
                    logger.info(f"Web search retry with traffic-focused query: {focused_query}")
                    search_results = await web_search_tool.ainvoke({"query": focused_query})
                # 重试仍无结果：启用 DuckDuckGo 兜底搜索
                if (
                    state.get("is_realtime_query")
                    and state.get("realtime_category") == "traffic"
                    and (not search_results)
                ):
                    logger.info("Web search fallback retry with DuckDuckGo for traffic")
                    search_results = await _duckduckgo_traffic_fallback(query)
                    used_duck_fallback = bool(search_results)

                if realtime_category == "market":
                    _mtpl = (
                        getattr(settings, "web_search_market_secondary_query_zh", None) or ""
                    ).strip()
                    if _mtpl:
                        try:
                            q_mk = _mtpl.format(
                                date_iso=now.strftime("%Y-%m-%d"),
                                date_cn=f"{now.month}月{now.day}日",
                            )
                        except Exception:
                            logger.warning(
                                "web_search_market_secondary_query_zh has invalid placeholders, "
                                "using template as literal"
                            )
                            q_mk = _mtpl
                        try:
                            extra_mk = await web_search_tool.ainvoke({"query": q_mk})
                            primary_mk = (
                                search_results if isinstance(search_results, list) else []
                            )
                            search_results = merge_tavily_result_lists(
                                primary_mk,
                                extra_mk if isinstance(extra_mk, list) else [],
                            )
                        except Exception as e:
                            logger.warning(f"Market secondary Tavily skipped: {e}")

                # Format results
                formatted_results = []
                api_error_message = None

                if search_results:
                    # 先判断 search_results 类型
                    if not isinstance(search_results, list):
                        # 处理非列表返回值（可能是错误字符串）
                        results_str = str(search_results)
                        if "401" in results_str or "Unauthorized" in results_str or "authentication" in results_str.lower():
                            api_error_message = "当前无法进行网络检索（API Key 可能过期或无效）"
                            logger.warning(
                                f"Web search API error: error_type={type(search_results).__name__}, "
                                f"error={results_str[:200]}"
                            )
                    else:
                        # 是列表，检查每个元素
                        for result in search_results:
                            if not isinstance(result, dict):
                                result_str = str(result)
                                if "401" in result_str or "Unauthorized" in result_str or "authentication" in result_str.lower():
                                    api_error_message = "当前无法进行网络检索（API Key 可能过期或无效）"
                                    logger.warning(
                                        f"Web search API error in result: error={result_str[:200]}"
                                    )
                                    break

                        # 只有没有 API 错误时才格式化结果
                        if not api_error_message:
                            selected_results = (
                                search_results[:settings.web_search_max_results]
                                if used_duck_fallback
                                else filter_and_sort_by_recency(search_results, now, policy)[:settings.web_search_max_results]
                            )
                            for i, result in enumerate(selected_results, 1):
                                if not isinstance(result, dict):
                                    logger.warning(
                                        f"Invalid search result type: result_type={type(result).__name__}, "
                                        f"result={str(result)[:200]}"
                                    )
                                    continue

                                formatted_results.append({
                                    "rank": i,
                                    "title": result.get("title", ""),
                                    "url": result.get("url", ""),
                                    "content": result.get("content", "")[:2000],
                                    "score": result.get("score", 0.0)
                                })

                state["web_search_results"] = formatted_results
                state["web_search_used"] = len(formatted_results) > 0
                state["web_search_error"] = api_error_message

                # 天气兜底：无搜索结果且无API错误时，调用免费天气接口
                if (
                    not state["web_search_used"]
                    and not api_error_message
                    and state.get("realtime_category") == "weather"
                ):
                    fallback = await _open_meteo_weather_fallback(query)
                    if fallback:
                        formatted_results = [fallback]
                        state["web_search_results"] = formatted_results
                        state["web_search_used"] = True
                        state["web_search_error"] = None
                        logger.info(
                            f"Web search fallback used: open_meteo, query={query[:80]}"
                        )

                # 添加 web_search source
                if formatted_results:
                    citations = []
                    for result in formatted_results[:3]:  # 最多3个
                        citations.append({
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "score": result.get("score", 0.0),
                            "snippet": result.get("content", "")[:300]
                        })

                    state["sources"].append({
                        "type": "text",
                        "from": "web_search",
                        "text": state.get("rewritten_query", state["user_query"]),
                        "citations": citations
                    })

                logger.info(
                    f"Web search completed: results_count={len(formatted_results)}, "
                    f"has_error={api_error_message is not None}"
                )

            except Exception as e:
                logger.error(f"Web search failed: {str(e)}", exc_info=True)
                state["web_search_results"] = []
                state["web_search_used"] = False
                state["web_search_error"] = f"Web search failed: {str(e)}"

        return state

    # -------------------------------------------------------------------------
    # Workflow Nodes - Answer Generation
    # -------------------------------------------------------------------------
    async def generate_answer(self, state: ConversationState) -> ConversationState:
        """
        配置流式输出对象 - 根据意图和数据源选择不同的流式输出方式

        设置的状态字段：
        - state["streaming_llm"]     : 流式输出对象（LLM 或 None）
        - state["streaming_type"]   : "langchain_llm" 或 "raganything_stream"
        - state["streaming_messages"]: messages（LangChain LLM 使用）
        - state["raganything_query"] : 查询文本（RAGAnything 使用）
        - state["raganything_mode"]  : 检索模式（RAGAnything 使用）
        - state["confidence"]        : 置信度分数

        实际的流式输出在 chat_stream_v1.py 中根据这些配置执行。
        """
        async with time_node("generate_answer", state):
            # If final_answer is already set (direct match), skip placeholder
            if state.get("final_answer"):
                logger.info(
                    f"Direct match answer already set, skipping LLM generation: answer_length={len(state['final_answer'])}"
                )
                return state

            intent = state.get("intent")
            web_search_used = state.get("web_search_used", False)

            # 通用"先验信息"注入：对于 web 资料天然不可靠的实时类（traffic 等），
            # 即便 web_search 命中无关结果，也给 LLM 留一份"基于时段/星期"的合理估算，
            # 避免 LLM 因资料无关而输出"无法回答"。当前仅 traffic 提供先验；
            # 下游 prompt 由 build_generation_messages 读取 state["traffic_estimate"] 拼接，
            # 不构成强约束（LLM 仍优先采用真实命中的 web 数据）。
            if state.get("realtime_category") == "traffic" and not state.get("traffic_estimate"):
                _now = datetime.now()
                _lvl, _reason = _traffic_congestion_estimate(_now)
                state["traffic_estimate"] = {
                    "level": _lvl,
                    "reason": _reason,
                    "now_iso": _now.strftime("%Y-%m-%d %H:%M"),
                    "weekday_zh": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][_now.weekday()],
                    "weekday_en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][_now.weekday()],
                }

            # 噪声输入：使用预设的友好响应，不需要调用 LLM
            # 注意：到达这里时已经通过了 ``apply_noise_preset_gate`` 三道闸门
            # （置信度 / 长度 / 启发式），可以放心直出预设话术。
            # 文案统一从 ``NOISE_PRESET_RESPONSE_TEXT`` 读取，避免散落硬编码且
            # 不再使用"听清/听见"等会让用户误认作 ASR 故障的字眼。
            if intent == "noise":
                noise_source = next(
                    (s for s in state.get("sources", []) if s.get("from") == "noise_response"),
                    None,
                )
                preset_text = (
                    noise_source.get("text") if noise_source else None
                ) or NOISE_PRESET_RESPONSE_TEXT
                state["final_answer"] = preset_text
                state["confidence"] = 0.99
                state["streaming_llm"] = None
                state["streaming_type"] = "text"  # 标记为纯文本输出
                logger.info("Noise input detected, using preset response")
                return state

            # 纯日期/星期类：本地确定性推算，直接输出（不依赖分类 reason=time，也不依赖联网）
            if state.get("is_realtime_query"):
                direct = state.get("direct_text_answer")
                if not direct:
                    q_cal = (state.get("rewritten_query") or state.get("user_query") or "").strip()
                    direct = calendar_direct_text_answer(
                        q_cal,
                        resolve_prefer_zh_output(state),
                        anchor_year=state.get("target_year"),
                        language_hint_query=state.get("user_query") or q_cal,
                    )
                if direct:
                    state["direct_text_answer"] = direct
                    state["realtime_category"] = "time"
                    state["streaming_llm"] = None
                    state["streaming_messages"] = None
                    state["streaming_type"] = "direct_text"
                    state["confidence"] = 0.99
                    state["final_answer"] = ""
                    logger.info(
                        "Streaming configured: type=direct_text, realtime_category=time (local_calendar_resolver)"
                    )
                    return state

            # 实时查询：判定 web 召回是否"可信"，不可信时走 direct_text 兜底，
            # 避免无关命中（如 Tavily 对 traffic 召回到汽车广告页）把小模型带偏成"无法判断"。
            #
            # 判定规则（通用框架）：
            #   - web_search_used=False                                   → 不可信（联网未开/未走）
            #   - web_search_error 非空                                   → 不可信（联网失败）
            #   - 已知"web 数据源天然不准"的 category（当前：traffic）：  →
            #     再看 web_search_results 的最高 score 是否低于
            #     settings.realtime_traffic_min_web_score（默认 0.5）
            if state.get("is_realtime_query"):
                web_used = state.get("web_search_used", False)
                err = state.get("web_search_error")
                category = state.get("realtime_category")

                web_unreliable = (not web_used) or bool(err)
                if not web_unreliable and category == "traffic":
                    web_results = state.get("web_search_results") or []
                    max_score = max(
                        (float(r.get("score", 0.0) or 0.0) for r in web_results),
                        default=0.0,
                    )
                    threshold = float(getattr(settings, "realtime_traffic_min_web_score", 0.5))
                    if max_score < threshold:
                        web_unreliable = True
                        logger.info(
                            f"Traffic web results below relevance threshold: "
                            f"max_score={max_score:.3f} < {threshold}, fallback to local prior"
                        )

                if web_unreliable:
                    prefer_zh_unreliable = resolve_prefer_zh_output(state)
                    # (A) 网络完全失败：固定提示
                    if err and not web_used:
                        direct = (
                            "当前网络检索不可用，暂时无法确认实时信息，请稍后重试。"
                            if prefer_zh_unreliable
                            else "Real-time web retrieval is unavailable; please try again later."
                        )
                        state["direct_text_answer"] = direct
                        state["streaming_llm"] = None
                        state["streaming_messages"] = None
                        state["streaming_type"] = "direct_text"
                        state["confidence"] = 0.7
                        state["final_answer"] = ""
                        logger.info(
                            "Streaming configured: type=direct_text, realtime_web_error"
                        )
                        return state

                    # (B) 交通类：拼接完整的"时段先验"模板（等级 + 时间锚点 + 出行建议 + 推荐地图）
                    #     模板对城市保持中性（不复述用户问题里的具体城市），保证不同城市通用。
                    if category == "traffic":
                        est = state.get("traffic_estimate") or {}
                        if not est:
                            _now = datetime.now()
                            _lvl, _reason = _traffic_congestion_estimate(_now)
                            est = {
                                "level": _lvl,
                                "reason": _reason,
                                "now_iso": _now.strftime("%Y-%m-%d %H:%M"),
                                "weekday_zh": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][_now.weekday()],
                                "weekday_en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][_now.weekday()],
                            }
                            state["traffic_estimate"] = est
                        prefer_zh = prefer_zh_unreliable
                        if prefer_zh:
                            direct = (
                                f"今日整体路况判断：{est['level']}（依据：{est['reason']}）。\n"
                                f"参考时间：{est['now_iso']}（{est['weekday_zh']}）。\n"
                                "出行建议：建议错峰出行，避开早晚高峰主干道；可优先选择环线辅道或公共交通。\n"
                                "精确实时数据请使用高德地图 / 百度地图查询。"
                            )
                        else:
                            direct = (
                                f"Today's overall traffic: {est['level']} (reason: {est['reason']}).\n"
                                f"Reference time: {est['now_iso']} ({est['weekday_en']}).\n"
                                "Tips: travel off-peak, avoid major arteries during rush hour; "
                                "consider ring-road service lanes or public transit.\n"
                                "For precise real-time data, please use Amap or Baidu Maps."
                            )
                        state["direct_text_answer"] = direct
                        state["streaming_llm"] = None
                        state["streaming_messages"] = None
                        state["streaming_type"] = "direct_text"
                        state["confidence"] = 0.7
                        state["final_answer"] = ""
                        logger.info(
                            f"Streaming configured: type=direct_text, "
                            f"realtime_traffic_estimate, prefer_zh={prefer_zh}"
                        )
                        return state

            # 计算置信度
            confidence = 0.5  # Base confidence
            if intent == "greeting":
                confidence = 0.98
            elif web_search_used:
                web_results = state.get("web_search_results", [])
                if web_results:
                    avg_web_score = sum(r.get("score", 0.5) for r in web_results) / len(web_results)
                    confidence = max(0.75, avg_web_score)
            else:  # normal query with RAGAnything
                confidence = 0.8

            # 根据 answer_mode（由 classify_query_type 写入）配置流式输出。
            #
            # 表驱动分发原则：
            # - 任何"该走哪条生成路径"的判断只看 state["answer_mode"]；
            # - intent / is_realtime_query / is_math_problem / web_search_used
            #   仅作为细分场景修饰（如 RAG 在已联网情况下退回 langchain_llm）；
            # - 新分支只在本 if/elif 链中追加一条，对应 INTENT_TO_ANSWER_MODE 表。
            answer_mode = state.get("answer_mode") or AnswerMode.GENERAL_LLM.value

            # web_search 已经命中网络资料：不论 answer_mode 原本是什么，统一交给
            # langchain_llm 用网络上下文生成（避免再去走 RAG，让"实时问题"行为
            # 与原实现一致）。这是 web_search 节点之后必经的修正点。
            if web_search_used:
                effective_mode = AnswerMode.GENERAL_LLM.value
            else:
                effective_mode = answer_mode

            if state.get("is_math_problem", False) or effective_mode == AnswerMode.PHI4_MATH.value:
                # 数学题：Phi-4 推理，明确不走 RAG
                messages = build_math_generation_messages(state)
                streaming_llm, model_name = self.workflow.get_phi4_streaming_llm(state)
                state["streaming_llm"] = streaming_llm
                state["streaming_messages"] = messages
                state["streaming_type"] = "phi4_math"
                logger.info(
                    f"Streaming configured: type=phi4_math, model={model_name}, "
                    f"answer_mode={answer_mode}, query={state['user_query'][:50]}..."
                )
            elif effective_mode == AnswerMode.RAG_WITH_FALLBACK.value:
                employee_config = state.get("employee_config", {})
                # RAGAnything 是全局单一知识库（Milvus + Neo4j），kb_ids 已是空也仍能召回到全局图谱/向量；
                # 因此路由仅看全局开关 raganything_enabled 与员工级 rag_disabled，不再用 kb_ids 作为门。
                raganything_enabled = getattr(settings, "raganything_enabled", True)
                rag_disabled = _resolve_employee_rag_disabled(employee_config)
                kb_ids = _resolve_employee_kb_ids(employee_config)

                # 全局未启用 或 员工显式禁用 RAG → 降级为普通 LLM 生成。
                # 这是 "RAG 没有 → 通用 LLM" 兜底逻辑的"前置门"分支：
                # 当 RAG 完全不可用时直接走通用 LLM，避免 RAGAnything 抛错或返回空。
                if (not raganything_enabled) or rag_disabled:
                    messages = build_generation_messages(state)
                    streaming_llm, model_name = self.workflow.get_streaming_llm(state)
                    state["streaming_llm"] = streaming_llm
                    state["streaming_messages"] = messages
                    state["streaming_type"] = "langchain_llm"
                    logger.info(
                        f"Streaming configured: type=langchain_llm (rag_fallback), "
                        f"reason={'raganything_disabled' if not raganything_enabled else 'employee_rag_disabled'}, "
                        f"answer_mode={answer_mode}, model={model_name}"
                    )
                else:
                    # 启用 RAG → 使用 RAGAnything 混合检索；
                    # 召回为空时由 chat_stream_v1.py 中的运行期兜底（rag_empty）转通用 LLM。
                    state["streaming_llm"] = None
                    state["streaming_messages"] = None
                    state["streaming_type"] = "raganything_stream"
                    # 优先使用上下文消歧后的改写问句作为 RAG 的检索锚点：
                    # 当用户用「他/这个/它」等指代承接上一轮（如先问「勾股定理」、再问
                    # 「举例说明他在生活中的应用?」）时，原始 user_query 中的代词无法
                    # 命中 KG/向量库里的具体实体，而 classify_query_type 节点已经把它
                    # 改写为「举例说明勾股定理在生活中的应用?」并存入 rewritten_query。
                    # 这里直接用改写后的版本，让 RAG 拿到含明确主语的 query，避免出现
                    # 「请问您指的是哪方面的应用？」这类失忆式回答。
                    # rewritten_query 缺失或为空时回退到原始 user_query，向后兼容老路径。
                    rewritten_for_rag = (state.get("rewritten_query") or "").strip()
                    state["raganything_query"] = rewritten_for_rag or state["user_query"]
                    state["raganything_mode"] = "hybrid"

                    logger.info(
                        f"Streaming configured: type=raganything_stream, "
                        f"answer_mode={answer_mode}, mode=hybrid, kb_ids_meta={kb_ids}, "
                        f"query={state['user_query'][:50]}..."
                    )
            else:
                # GENERAL_LLM（含 greeting / english_query / general_knowledge / chit_chat /
                # web_search 命中后的兜底）：使用通用 LLM，不走 RAG，不走 web。
                messages = build_generation_messages(state)
                streaming_llm, model_name = self.workflow.get_streaming_llm(state)
                state["streaming_llm"] = streaming_llm
                state["streaming_messages"] = messages
                state["streaming_type"] = "langchain_llm"
                logger.info(
                    f"Streaming configured: type=langchain_llm, "
                    f"answer_mode={answer_mode}, intent={intent}, "
                    f"web_search_used={web_search_used}, model={model_name}"
                )

            state["confidence"] = confidence
            state["final_answer"] = ""  # Placeholder for streaming

        return state

    # -------------------------------------------------------------------------
    # Workflow Nodes - Answer Verification
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Workflow Nodes - Save Conversation
    # -------------------------------------------------------------------------
    async def save_conversation(self, state: ConversationState) -> ConversationState:
        """
        Save Conversation Record - Persist conversation to MongoDB.

        Saves:
        - User query and AI response
        - Performance metrics (timing, confidence)
        - Data sources used (KB, web search)
        - Intent and verification results

        Also updates session with new messages.

        Args:
            state: Current conversation state

        Returns:
            Updated state with conversation_id populated
        """
        async with time_node("save_conversation", state):
            try:
                db = await get_database()

                # Generate conversation ID
                session_id = state["session_id"]
                timestamp = datetime.now().timestamp()
                conv_id = f"conv_{hashlib.md5(f'{session_id}_{timestamp}'.encode()).hexdigest()[:12]}"
                state["conversation_id"] = conv_id

                # Calculate total response time
                workflow_start_time = state.get("workflow_start_time", time.time())
                total_time_ms = int((time.time() - workflow_start_time) * 1000)
                state["response_time_ms"] = total_time_ms

                # Create conversation record
                employee_config = state.get("employee_config", {})
                conversation = ConversationModel(
                    conversation_id=conv_id,
                    session_id=state["session_id"],
                    user_id=state["user_id"],
                    user_name=state.get('user_name', ''),
                    head_url=state.get('head_url', ''),
                    employee_id=state["employee_id"],
                    employee_name=employee_config.get("name", ""),
                    user_query=state["user_query"],
                    ai_response=state["final_answer"],
                    is_realtime_query=state.get("is_realtime_query", False),
                    realtime_category=state.get("realtime_category"),
                    intent=state.get("intent"),
                    kb_used=state.get("kb_used", []),
                    web_search_used=state.get("web_search_used", False),
                    web_search_results=[
                        {
                            "rank": result.get("rank"),
                            "title": result.get("title"),
                            "url": result.get("url"),
                            "score": result.get("score", 0.0)
                        }
                        for result in state.get("web_search_results", [])[:5]
                    ],
                    retrieved_docs=[
                        {
                            "doc_id": doc.get("doc_id"),
                            "kb_id": doc.get("kb_id"),
                            "score": doc.get("rrf_score", 0.0)
                        }
                        for doc in state.get("retrieved_docs", [])[:3]
                    ],
                    relevance_score=state.get("relevance_score", 0.0),
                    confidence=state.get("confidence", 0.0),
                    response_time_ms=total_time_ms
                )

                await db.conversations.insert_one(conversation.model_dump())

                # Update session context messages
                await db.sessions.update_one(
                    {"session_id": state["session_id"]},
                    {
                        "$push": {
                            "context_messages": {
                                "$each": [
                                    {"role": "user", "content": state["user_query"]},
                                    {"role": "assistant", "content": state["final_answer"]}
                                ],
                                "$slice": -20  # Keep last 20 messages
                            }
                        },
                        "$inc": {"message_count": 1}
                    }
                )

                # Log timing summary
                node_timings = state.get("node_timings", {})
                ttfb_ms = state.get("ttfb_ms")

                logger.info(
                    f"[TIMING_SUMMARY] Conversation completed - "
                    f"total: {total_time_ms}ms, ttfb: {ttfb_ms}ms, "
                    f"nodes: {node_timings}"
                )

            except Exception as e:
                logger.error(f"Failed to save conversation: {str(e)}", exc_info=True)

        return state
