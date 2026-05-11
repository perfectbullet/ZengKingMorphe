"""独立测试 is_math_problem 函数的脚本。

用法:
    conda activate morphe
    cd ai-service
    python scripts/test_heuristic_math.py
"""
import sys
from pathlib import Path

# 让 import 能找到 app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.math_intent_heuristic import (
    is_math_problem,
    heuristic_concept_explain,
)


# ── 测试用例：(query, 期望结果, 说明) ──────────────────────────────
CASES = [
    # ──── 应返回 True（数学题）────
    # 模式 A：几何应用题
    ("已知圆锥的底面半径为 1，高为 2，则圆锥的侧面积为多少?", True, "模式A: 几何应用题"),
    ("已知正方体的棱长为 2，则其对角线的长度为多少?", True, "模式A: 正方体对角线"),
    ("已知球的半径为 3，则球的体积为多少", True, "模式A: 球体积"),
    # 模式 B：数学动词 + 数学对象
    ("求不等式 x²-5x+6<0", True, "模式B: 求不等式"),
    ("证明：若 a>b>0，则 1/a < 1/b", True, "模式B: 证明题"),
    ("计算函数 f(x)=x²+2x+1 的导数", True, "模式B: 计算导数"),
    ("解方程 x²-3x+2=0", True, "模式B: 解方程"),
    ("化简 (x+1)²-(x-1)²", True, "模式B: 化简"),
    ("判断下列不等式是否成立", True, "模式B: 判断不等式"),
    # 模式 C：数列递推/求值
    (
        "已知数列A下标N，满足A下标N加一等于二乘以A的下标N加一，A的下标一等于一的A下标四的值为多少",
        True,
        "模式C: 数列递推ASR",
    ),
    ("已知等差数列的首项为3，公差为2，前n项和为多少", True, "模式C: 等差数列"),
    ("已知数列的通项公式，求公比", True, "模式C: 数列通项+求"),

    # ──── 应返回 False（非数学题 / 概念题）────
    ("", False, "空字符串"),
    ("你好", False, "闲聊"),
    ("今天天气怎么样", False, "闲聊-天气"),
    ("请讲解等差数列前 n 项和的推导方法", False, "概念题: 推导方法"),
    ("什么是导数", False, "概念题: 是什么"),
    ("在数列的学习中，请分别说明等差数列和等比数列的推导方法，比较异同", False, "概念题: 教学语境+异同"),
    ("请详细讲解一下二项式定理的推导过程及其在组合数学中的应用", False, "概念题: 长问句+讲解"),
    ("足球比赛的结果是多少", False, "非数学: 球员/球场"),
    ("球场旁边有个餐厅", False, "非数学: 球场噪声"),
    ("高矮不齐的篮球运动员", False, "非数学: 高(中考)排除"),
]


def main():
    passed = 0
    failed = 0

    for query, expected, desc in CASES:
        result = is_math_problem(query)
        ok = result is expected

        if not ok:
            failed += 1
            tag = "FAIL"
        else:
            passed += 1
            tag = "PASS"

        short_q = query[:60] + ("…" if len(query) > 60 else "")
        print(f"  [{tag}] {desc}")
        print(f"        query={short_q!r}")
        print(f"        expected={expected}, got={result}")
        print()

    # ── 交互模式 ──
    total = passed + failed
    print("=" * 50)
    print(f"结果: {passed}/{total} 通过, {failed} 失败")
    print("=" * 50)

    print("\n进入交互模式 (输入 q 退出):\n")
    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in ("q", "quit", "exit"):
            break
        if not query:
            continue

        is_math = is_math_problem(query)
        is_concept = heuristic_concept_explain(query)
        print(f"  is_math_problem           → {is_math}")
        print(f"  heuristic_concept_explain → {is_concept}")
        print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
