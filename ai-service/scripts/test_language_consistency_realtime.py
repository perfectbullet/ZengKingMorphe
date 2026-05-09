#!/usr/bin/env python3
"""
英文 realtime + web_search 路径下的语言一致性回归测试。

复现并防回归 Bug：
- 现象：英文问句（如 "what is the current time and date in beijing"）
        触发 realtime_query → web_search → langchain_llm 路径时，
        系统返回中文（如「北京时间为2026年5月6日…」），
        典型“英文问、中英文混合答”。
- 根因：``ConversationState`` TypedDict 未声明 ``prefer_zh_output``，
        LangGraph 1.x 按 schema 严格过滤，导致该字段在节点执行前被丢，
        ``state.get("prefer_zh_output", True)`` 静默回到中文兜底。

覆盖矩阵：
- Q1 EN time   ：英文时间类（命中 _CALENDAR_INTENT_EN，触发本地直出）
- Q2 EN time   ：上面那条用户原句（明确 realtime_query / web_search 流）
- Q3 EN news   ：英文新闻类，必走 web_search
- Q4 EN market ：英文行情类，必走 web_search
- Q5 EN traffic：英文路况类，常因 web 数据噪声进入 traffic 兜底模板
- Q6 ZH time   ：中文时间类（保证修改没把中文路径搞坏）

每条用例：
- 单独使用一个全新 session_id（避免历史污染）；
- 同一问题连发 2 次（覆盖 raganything 索引/缓存暖机后的稳态）；
- body 主导语言必须与本轮 query 语言完全一致，否则 FAIL。
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/Users/summer/Documents/metahuman_work/ZengKingMorphe/ai-service/scripts")
from test_language_consistency import detect_lang, run_query, strip_preface  # type: ignore

HOST = "http://127.0.0.1:8100"


CASES: list[tuple[str, str, str]] = [
    # tag, query, expected lang
    ("EN time/beijing-clock", "what is the current time and date in beijing", "en"),
    ("EN time/general-clock", "what time is it now", "en"),
    ("EN news", "any breaking world news today", "en"),
    ("EN market", "what is the current usd to cny exchange rate", "en"),
    ("EN traffic", "is there heavy traffic in chengdu now", "en"),
    ("ZH time/beijing-clock", "北京现在几点", "zh"),
]


def main() -> int:
    fail = 0
    total = 0
    for tag, query, expected in CASES:
        for round_idx in (1, 2):  # 同一 query 跑 2 轮，覆盖暖机后稳态
            total += 1
            sid = f"sess_lang_rt_{int(time.time())}_{round_idx}_{tag.split('/')[0].split()[-1]}"
            print(f"\n--- {tag} #{round_idx} | expected={expected} | session={sid} | query={query!r} ---")
            try:
                raw = run_query(HOST, sid, query, timeout=180)
            except Exception as e:
                print(f"  ERROR: {e}")
                fail += 1
                continue
            body = strip_preface(raw)
            lang = detect_lang(body)
            ok = (lang == expected)
            if not ok:
                fail += 1
            preview = body.replace("\n", " ")[:240]
            print(f"  [{'PASS' if ok else 'FAIL'}] detected={lang}")
            print(f"  body: {preview}")

    print()
    print("=" * 60)
    print(f"FINAL: total={total}, fail={fail}")
    print("=" * 60)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
