#!/usr/bin/env python3
"""
扩展回归测试：除了用户原报告的 EN/EN/ZH/EN/EN 场景外，再覆盖：
  Pattern A: ZH/ZH/EN/ZH/ZH（镜像方向）
  Pattern B: 复用用户原来的 sess_28_2373824_68 会话（已有累积历史）
            按 EN/EN/ZH/EN/EN 顺序再跑一遍，确认即使历史里已经有
            来自历史 bug 的“中英混合”消息，本轮回复仍然能保持纯英文。
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/Users/summer/Documents/metahuman_work/ZengKingMorphe/ai-service/scripts")
from test_language_consistency import detect_lang, run_query, strip_preface  # type: ignore

HOST = "http://127.0.0.1:8100"


def run_plan(label: str, session_id: str, plan: list[tuple[str, str, str]]) -> int:
    print(f"\n========== {label}  session_id={session_id} ==========")
    fail = 0
    for tag, query, expected in plan:
        print(f"\n--- {tag} | expected={expected} | query={query!r} ---")
        try:
            raw = run_query(HOST, session_id, query)
        except Exception as e:
            print(f"  ERROR: {e}")
            fail += 1
            continue
        body = strip_preface(raw)
        lang = detect_lang(body)
        ok = lang == expected
        if not ok:
            fail += 1
        preview = body.replace("\n", " ")[:240]
        print(f"  [{'PASS' if ok else 'FAIL'}] detected={lang}")
        print(f"  body: {preview}")
    return fail


def main() -> int:
    fail = 0

    # Pattern A: 反向（中文为主，中间夹一个英文）
    plan_a = [
        ("A1 ZH", "介绍一下集合的基本运算", "zh"),
        ("A2 ZH", "什么是函数的定义域", "zh"),
        ("A3 EN", "what is the most popular hobby among young people", "en"),
        ("A4 ZH", "再讲讲单调性", "zh"),
        ("A5 ZH", "那奇偶性呢", "zh"),
    ]
    fail += run_plan("PATTERN A (ZH/ZH/EN/ZH/ZH)", f"sess_lang_extra_a_{int(time.time())}", plan_a)

    # Pattern B: 复用用户原 session_id（已有历史污染）
    plan_b = [
        ("B1 EN", "what is the most popular hobby among young people", "en"),
        ("B2 EN", "what is the most popular hobby among young people", "en"),
        ("B3 ZH", "我想了解一下今天成都堵不堵？", "zh"),
        ("B4 EN", "what is the most popular hobby among young people", "en"),
        ("B5 EN", "what is the most popular hobby among young people", "en"),
    ]
    fail += run_plan("PATTERN B (reuse user's polluted session)", "sess_28_2373824_68", plan_b)

    print()
    print("=" * 60)
    print(f"FINAL: total_fail={fail}")
    print("=" * 60)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
