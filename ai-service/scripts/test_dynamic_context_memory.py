"""
真实 LLM 烟雾测试：验证动态上下文记忆判定在生产环境下的准确性。

运行：
    cd ai-service && PYTHONPATH=. python scripts/test_dynamic_context_memory.py

会调用 ``settings.ollama_base_url`` 配置的 Ollama 模型；如果端点不可达
（或 .env 未配置），脚本会打印错误并退出。
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Iterable

from app.services.query_classifier import format_dialog_for_resolver, get_query_classifier


# 测试用例：每条 (描述, dialog_text, query, expected_related)
# expected_related 为 None 表示对结果不做硬断言（仅打印观察），True/False 表示我们对结论的预期。
CASES: list[tuple[str, str, str, bool | None]] = [
    # 无历史 → 应判 unrelated（节点会短路，不调 LLM）
    (
        "首轮无历史",
        "",
        "你好",
        False,
    ),
    # 指代追问 → related
    (
        "明确指代追问 / 求继续",
        "用户：导数怎么求？\n助手：对函数求导有四种基本方法……\n用户：再举个例子",
        "它的几何意义是什么",
        True,
    ),
    # 同一话题延续 → related
    (
        "同主题数学追问",
        "用户：什么是等差数列？\n助手：等差数列是相邻两项之差为常数的数列……",
        "通项公式怎么写",
        True,
    ),
    # 切换到完全不同的领域 → unrelated
    (
        "突然切到另一个话题",
        "用户：什么是等差数列？\n助手：等差数列是相邻两项之差为常数的数列……",
        "今天上海天气怎么样",
        False,
    ),
    # 新的问候语 → unrelated
    (
        "聊完技术后新一轮问候",
        "用户：导数怎么求？\n助手：对函数求导有四种基本方法……",
        "你好",
        False,
    ),
    # 完全独立的事实查询 → unrelated
    (
        "完全独立的事实查询",
        "用户：等差数列的求和公式是什么？\n助手：S_n = n(a_1 + a_n)/2",
        "中国的首都在哪里",
        False,
    ),
    # 纠错或评价上一轮 → related
    (
        "纠错评价",
        "用户：1+1 等于多少？\n助手：等于 3。",
        "不对，你算错了",
        True,
    ),
    # 用户用代词回顾上文 → related
    (
        "代词指代上文",
        "用户：什么是导数？\n助手：导数是函数变化率的极限。",
        "它有什么实际应用",
        True,
    ),
]


async def _run() -> int:
    classifier = get_query_classifier()
    total = len(CASES)
    matched = 0
    failures: list[str] = []
    for desc, dialog, query, expected in CASES:
        try:
            is_related, reason = await classifier.aclassify_context_dependence(query, dialog)
        except Exception as e:
            failures.append(f"[CASE EXC] {desc}: {e!r}")
            continue
        verdict_zh = "相关" if is_related else "无关"
        if expected is None:
            print(f"[OBS] {desc}: result={verdict_zh}({reason})  query={query!r}")
            continue
        ok = is_related is expected
        prefix = "[OK]" if ok else "[NG]"
        print(
            f"{prefix} {desc}: result={verdict_zh}({reason}), "
            f"expected={'相关' if expected else '无关'}  query={query!r}"
        )
        if ok:
            matched += 1
        else:
            failures.append(
                f"[NG] {desc}: result={is_related} reason={reason}, "
                f"expected={expected} query={query!r}"
            )

    asserted = sum(1 for c in CASES if c[3] is not None)
    print()
    print(f"Total cases   : {total}")
    print(f"Asserted cases: {asserted}")
    print(f"Matched       : {matched}")
    print(f"Failed        : {len(failures)}")

    if failures:
        print("\n--- Failures ---")
        for f in failures:
            print(f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
