#!/usr/bin/env python3
"""数学检测功能的单元测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class MathDetectionTester:
    """数学检测测试类"""

    def __init__(self):
        # 导入检测逻辑
        from app.services.conversation.conversation_nodes import ConversationNodes

        # 创建节点实例（需要 mock workflow）
        class MockWorkflow:
            pass

        self.nodes = ConversationNodes(MockWorkflow())
        self._detect_math_problem = self.nodes._detect_math_problem

    def test_concept_questions(self):
        """测试概念性问题（应该返回 False）"""
        print("\n=== 测试概念性问题 ===")

        test_cases = [
            "什么是函数",
            "介绍一下三角函数",
            "解释集合的定义",
            "概念是什么",
            "定义是什么",
            "介绍一下"
        ]

        all_passed = True
        for query in test_cases:
            is_math, reason = self._detect_math_problem(query)
            if is_math:
                print(f"✗ 失败: '{query}' - 概念性问题被误判为数学题")
                all_passed = False
            else:
                print(f"✓ 通过: '{query}' - 正确识别为概念问题")

        return all_passed

    def test_math_problems(self):
        """测试数学问题（应该返回 True）"""
        print("\n=== 测试数学问题 ===")

        test_cases = [
            "求函数的值域",
            "计算 sin(30°) + cos(60°)",
            "解方程 x² + 2x + 1 = 0",
            "证明三角函数的和角公式",
            "推导椭圆的标准方程",
            "化简表达式 (x²-4)/(x-2)",
            "求数列 1, 2, 4, 8, 16 的通项公式"
        ]

        all_passed = True
        for query in test_cases:
            is_math, reason = self._detect_math_problem(query)
            if not is_math:
                print(f"✗ 失败: '{query}' - 数学问题未被识别")
                all_passed = False
            else:
                print(f"✓ 通过: '{query}' - 正确识别为数学题")

        return all_passed

    def test_boundary_cases(self):
        """测试边界案例"""
        print("\n=== 测试边界案例 ===")

        test_cases = [
            # 应该返回 True
            ("如何求函数的值域", True, "包含操作+对象"),
            ("解三角形", True, "隐含操作+对象"),
            ("求极限", True, "短查询，但操作明确"),

            # 应该返回 False
            ("三角函数的应用", False, "只有对象，无操作"),
            ("函数的类型", False, "概念性问题但较长"),
            ("椭圆和双曲线的区别", False, "比较，非求/解")
        ]

        all_passed = True
        for query, expected, description in test_cases:
            is_math, reason = self._detect_math_problem(query)
            if is_math != expected:
                print(f"✗ 失败: '{query}' - 预期 {expected}, 实际 {is_math}")
                all_passed = False
            else:
                print(f"✓ 通过: '{query}' - {description}")

        return all_passed

    def test_edge_cases(self):
        """测试边界极端案例"""
        print("\n=== 测试极端案例 ===")

        test_cases = [
            # 空查询
            ("", False, "空查询"),
            # 只有关键词
            ("求", False, "只有操作词"),
            ("函数", False, "只有对象词"),
            # 超长查询（防止误判）
            ("这是一个非常长的查询，用来测试边界情况，看起来像在问概念但实际上不是数学问题，这个问题很长，超过了50个字符，不应该被误判为数学题", False, "超长概念查询"),
        ]

        all_passed = True
        for query, expected, description in test_cases:
            is_math, reason = self._detect_math_problem(query)
            if is_math != expected:
                print(f"✗ 失败: '{query}' - 预期 {expected}, 实际 {is_math}")
                all_passed = False
            else:
                print(f"✓ 通过: '{query}' - {description}")

        return all_passed

    def test_reason_accuracy(self):
        """测试返回的原因准确性"""
        print("\n=== 测试原因准确性 ===")

        test_cases = [
            ("求函数的值域", "math_problem"),
            ("什么是函数", "concept_question"),
            ("天气怎么样", "general_query"),
            ("", "general_query")
        ]

        all_passed = True
        for query, expected_reason in test_cases:
            _, reason = self._detect_math_problem(query)
            if reason != expected_reason:
                print(f"✗ 失败: '{query}' - 预期原因 '{expected_reason}', 实际 '{reason}'")
                all_passed = False
            else:
                print(f"✓ 通过: '{query}' - 原因正确: {reason}")

        return all_passed

    def run_all_tests(self):
        """运行所有测试"""
        print("=== 数学检测功能单元测试 ===\n")

        tests = [
            self.test_concept_questions,
            self.test_math_problems,
            self.test_boundary_cases,
            self.test_edge_cases,
            self.test_reason_accuracy
        ]

        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                print(f"✗ 测试异常: {e}")
                results.append(False)

        # 汇总结果
        print("\n=== 测试结果汇总 ===")
        passed = sum(results)
        total = len(results)

        for i, (test, result) in enumerate(zip(tests, results)):
            test_name = test.__name__.replace("test_", "").replace("_", " ")
            status = "✓ 通过" if result else "✗ 失败"
            print(f"{status}: {test_name}")

        print(f"\n总体结果: {passed}/{total} 测试通过")
        overall_success = passed == total

        if overall_success:
            print("🎉 所有测试通过！数学检测功能正常工作。")
        else:
            print("⚠️  部分测试失败，请检查检测逻辑。")

        return overall_success


def main():
    """主函数"""
    tester = MathDetectionTester()
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()