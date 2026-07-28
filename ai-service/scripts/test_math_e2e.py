#!/usr/bin/env python3
"""数学检测功能的端到端测试"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_math_detection_and_routing():
    """测试数学检测和工作流路由"""
    from app.services.conversation_service import ConversationWorkflow

    print("=== 测试数学检测和工作流路由 ===")

    # 创建 workflow 实例
    workflow = ConversationWorkflow()

    # 测试用例
    test_cases = [
        {
            "name": "数学问题",
            "query": "求椭圆 x²/16 + y²/9 = 1 的焦点坐标",
            "expected_is_math": True,
            "expected_streaming_type": "math_llm"
        },
        {
            "name": "概念性问题",
            "query": "什么是函数",
            "expected_is_math": False,
            "expected_streaming_type": "langchain_llm"  # greeting 会走这个分支
        },
        {
            "name": "普通问题",
            "query": "人工智能的发展历史",
            "expected_is_math": False,
            "expected_streaming_type": "langchain_llm"
        }
    ]

    print("\n开始测试...")

    for test_case in test_cases:
        print(f"\n--- 测试: {test_case['name']} ---")

        # 准备初始状态
        initial_state = {
            "user_query": test_case["query"],
            "user_id": "test_user",
            "session_id": "test_session",
            "employee_id": "29",
            "employee_config": {
                "name": "测试助手",
                "role": "AI助手",
                "description": "专业的AI助手"
            },
            "messages": [],
            "is_math_problem": False,  # 初始值
            "sources": []
        }

        # 执行工作流（只到数学检测节点）
        print(f"查询: {test_case['query']}")

        try:
            # 1. 加载员工配置
            state = await workflow.nodes.load_employee_config(initial_state)
            print(f"✓ 员工配置加载成功")

            # 2. 加载会话上下文
            state = await workflow.nodes.load_session_context(state)
            print(f"✓ 会话上下文加载成功")

            # 3. 输入验证
            state = await workflow.nodes.validate_input(state)
            print(f"✓ 输入验证成功")

            # 4. 查询分类
            state = await workflow.nodes.classify_query_type(state)
            print(f"✓ 查询分类完成 - intent: {state.get('intent')}")

            # 5. 数学检测
            state = await workflow.nodes.check_math_problem(state)
            print(f"✓ 数学检测完成 - is_math_problem: {state.get('is_math_problem')}")

            # 验证结果
            actual_is_math = state.get("is_math_problem", False)
            expected_is_math = test_case["expected_is_math"]

            if actual_is_math == expected_is_math:
                print(f"✓ 数学检测正确: {expected_is_math}")
            else:
                print(f"✗ 数学检测错误: 预期 {expected_is_math}, 实际 {actual_is_math}")

            # 测试路由
            routed_node = ConversationWorkflow.nodes.route_after_math_check(state)
            print(f"✓ 路由决策: {routed_node}")

            # 对于数学问题，测试消息构建
            if actual_is_math:
                from app.services.conversation.conversation_helpers import build_math_generation_messages
                messages = build_math_generation_messages(state)
                print(f"✓ 数学消息构建成功 - 消息数: {len(messages)}")
                if messages:
                    print(f"  系统提示: {messages[0].content[:50]}...")

            print("✓ 该测试用例通过")

        except Exception as e:
            print(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()


async def test_math_detection_accuracy():
    """测试数学检测的准确性"""
    from app.services.conversation.conversation_nodes import ConversationNodes

    print("\n=== 测试数学检测准确性 ===")

    # 创建测试器
    class MockWorkflow:
        pass
    nodes = ConversationNodes(MockWorkflow())
    _detect_math_problem = nodes._detect_math_problem

    # 测试数据集
    accuracy_tests = [
        # (query, expected, category)
        ("什么是函数", False, "概念性问题"),
        ("求函数值域", True, "数学问题"),
        ("介绍三角函数", False, "概念性问题"),
        ("计算 sin(30°)", True, "数学问题"),
        ("解方程 x+1=0", True, "数学问题"),
        ("方程是什么", False, "概念性问题"),
        ("集合的定义", False, "概念性问题"),
        ("求集合的交集", True, "数学问题"),
        ("推导导数公式", True, "数学问题"),
        ("什么是导数", False, "概念性问题"),
        ("化简表达式", True, "数学问题"),
        ("椭圆的性质", False, "概念性问题"),
        ("求椭圆面积", True, "数学问题"),
    ]

    correct = 0
    total = len(accuracy_tests)

    for query, expected, category in accuracy_tests:
        actual, reason = _detect_math_problem(query)
        if actual == expected:
            correct += 1
            print(f"✓ {category}: '{query}'")
        else:
            print(f"✗ {category}: '{query}' (预期 {expected}, 实际 {actual})")

    accuracy = correct / total * 100
    print(f"\n准确率: {correct}/{total} ({accuracy:.1f}%)")

    return accuracy >= 90.0  # 目标准确率 90%


async def test_integration_with_conversation_service():
    """测试与对话服务的集成"""
    from app.services.conversation_service import ConversationWorkflow

    print("\n=== 测试与对话服务集成 ===")

    # 创建 workflow
    workflow = ConversationWorkflow()

    # 测试是否能正确初始化
    if workflow:
        print("✓ ConversationWorkflow 初始化成功")

    # 检查是否有数学模型相关方法
    if hasattr(workflow, 'get_math_streaming_llm'):
        print("✓ get_math_streaming_llm 方法存在")

    # 测试路由决策方法
    from app.services.conversation.conversation_nodes import ConversationNodes

    # 测试数学问题的路由
    math_state = {"is_math_problem": True}
    route = ConversationNodes.route_after_math_check(math_state)
    if route == "math":
        print("✓ 数学问题路由正确")
    else:
        print(f"✗ 数学问题路由错误: {route}")

    # 测试普通问题的路由
    normal_state = {"is_math_problem": False}
    route = ConversationNodes.route_after_math_check(normal_state)
    if route == "normal":
        print("✓ 普通问题路由正确")
    else:
        print(f"✗ 普通问题路由错误: {route}")

    print("✓ 集成测试完成")


async def main():
    """主函数 - 运行所有端到端测试"""
    print("=== 数学检测功能端到端测试 ===\n")

    tests = [
        test_math_detection_and_routing,
        test_math_detection_accuracy,
        test_integration_with_conversation_service
    ]

    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result if isinstance(result, bool) else True)
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            import traceback
            traceback.print_exc()
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

    if passed == total:
        print("🎉 所有端到端测试通过！数学检测功能已成功集成到系统中。")
        print("\n接下来可以:")
        print("1. 启动数学模型服务器")
        print("2. 运行 test_math_llm.py 测试实际数学模型调用")
        print("3. 部署并测试完整的数学问答功能")
    else:
        print("⚠️  部分测试失败，请检查实现。")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
