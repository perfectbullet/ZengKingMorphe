"""
补位 LLM 分类：识别「会随时间变化的在任人事 / 职务归属」类事实查询。

LLM 常把此类问题判成 general_knowledge，导致仅用静态知识作答（过时或回避）。
本模块规则全部数据化（正则与词表），便于扩展；仅在命中明确模式时返回推荐的
realtime reason，与 QueryClassifier 的分支协作，不替代主分类器。
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 排除：明确在问史实 / 已故人物，不应强制走「时事」联网
# ---------------------------------------------------------------------------
_HISTORICAL_HINT = re.compile(
    r"(历史上|历代|历任|第一任|第二任|第三任|曾任|已故|去世|逝世|生前|"
    r"古代|清朝|明朝|民国|公元前)",
    re.I,
)

# 「当前在任」语义锚点（可按业务扩展，勿绑定具体人名）
_TIME_SENSITIVE = re.compile(
    r"(目前|当前|现任|现阶段|本届|现今|现在在任)",
)

# 指向「何人担任」的问法（刻意不收单独的「哪个」，以免命中「哪个省最大」类百科题）
_WHO_FOCUS = re.compile(r"(谁|哪位|哪一个|是谁|何人)")

# 公共事务 / 组织人事场景常见职务词（扩展时只加词，不写死国别或姓名）
_PUBLIC_OFFICE = re.compile(
    r"(?:"
    r"总理|主席|总统|首相|省长|市长|县长|区长|州长|"
    r"部长|司长|厅长|局长|处长|科长|主任|书记|阁员|内阁|"
    r"领导人|领导|元首|大使|代表|议员|議員"
    r")",
    re.I,
)

# English: minimal set; extend via same-style tuples if product needs more locales
_EN_TIME = re.compile(r"\b(current|present|incumbent|now)\b", re.I)
_EN_WHO = re.compile(r"\bwho(?:'s|\s+is|\s+are|\s+was|\s+were)\b", re.I)
_EN_OFFICE = re.compile(
    r"\b("
    r"president|premier|prime\s+minister|mayor|governor|secretary|chancellor|minister"
    r")\b",
    re.I,
)


def heuristic_realtime_category(query: str) -> str | None:
    """
    若 query 命中「时间敏感 + 人事/职务」模式，返回建议的 realtime reason（英文枚举）；
    否则返回 None。

    当前返回 "news"：与下游 _normalize_realtime_category 及联网摘要场景一致。
    """
    if not query or not query.strip():
        return None
    text = query.strip()

    if _HISTORICAL_HINT.search(text):
        return None

    if (
        _TIME_SENSITIVE.search(text)
        and _WHO_FOCUS.search(text)
        and _PUBLIC_OFFICE.search(text)
    ):
        return "news"

    if _EN_TIME.search(text) and _EN_WHO.search(text) and _EN_OFFICE.search(text):
        return "news"

    return None
