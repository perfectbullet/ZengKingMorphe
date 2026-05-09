#!/usr/bin/env python3
"""测试 get_phi4_streaming_llm() 方法返回的 LLM 是否可用"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.conversation_service import ConversationWorkflow


async def test_phi4_llm():
    """测试 Phi-4 LLM 创建和基本调用"""
    # 创建 workflow 实例
    workflow = ConversationWorkflow()

    # 创建最小 state
    state = {
        "user_query": "求椭圆 x²/16 + y²/9 = 1 的焦点坐标",
        "employee_id": "test_employee",
        "employee_config": {},
    }

    try:
        # 获取 Phi-4 LLM
        phi4_llm, model_name = workflow.get_phi4_streaming_llm(state)

        print(f"✓ Phi-4 LLM 创建成功")
        print(f"  模型: {model_name}")
        print(f"  类型: {type(phi4_llm)}")

        # 测试简单调用
        messages = [{"role": "user", "content": "1+1=?"}]

        # 使用 async for 处理流式响应
        response_content = ""
        print("\n✓ 测试流式输出:")

        try:
            # 同步方式测试（不使用流式）
            from langchain_core.messages import HumanMessage

            # 创建一个简单的测试消息
            test_message = HumanMessage(content="1+1=?")

            # 调用 LLM
            response = await phi4_llm.ainvoke([test_message])

            if hasattr(response, 'content') and response.content:
                response_content = response.content
                print(f"✓ Phi-4 调用成功")
                print(f"  响应: {response_content[:100]}...")
                return True
            else:
                print(f"✗ Phi-4 响应格式异常")
                print(f"  响应对象: {response}")
                return False

        except Exception as e:
            print(f"✗ Phi-4 调用失败: {str(e)}")
            if "Connection refused" in str(e):
                print("\n💡 提示: 请确保 Phi-4 服务器正在运行")
                print(f"   检查: http://192.168.8.235:8000/v1")
            return False

    except Exception as e:
        print(f"✗ Phi-4 LLM 创建失败: {str(e)}")
        return False


async def test_fallback_behavior():
    """测试禁用 Phi-4 时的降级行为"""
    # 设置环境变量禁用 Phi-4
    import os
    os.environ["PHI4_ENABLED"] = "false"

    # 创建 workflow 实例
    workflow = ConversationWorkflow()

    state = {
        "user_query": "简单问题",
        "employee_id": "test_employee",
        "employee_config": {},
    }

    try:
        # 应该降级到默认 LLM
        llm, model_name = workflow.get_phi4_streaming_llm(state)

        print(f"\n✓ 降级测试成功 - 使用默认 LLM: {model_name}")
        print(f"  类型: {type(llm)}")
        return True

    except Exception as e:
        print(f"✗ 降级测试失败: {str(e)}")
        return False
    finally:
        # 恢复设置
        os.environ["PHI4_ENABLED"] = "true"


if __name__ == "__main__":
    print("=== Phi-4 LLM 测试 ===\n")

    # 测试 1: Phi-4 基本功能
    success1 = asyncio.run(test_phi4_llm())

    # 测试 2: 降级行为
    success2 = asyncio.run(test_fallback_behavior())

    # 汇总结果
    print("\n=== 测试结果 ===")
    print(f"Phi-4 测试: {'✓ 通过' if success1 else '✗ 失败'}")
    print(f"降级测试: {'✓ 通过' if success2 else '✗ 失败'}")

    overall_success = success1 and success2
    print(f"\n总体结果: {'✓ 所有测试通过' if overall_success else '✗ 有测试失败'}")

    sys.exit(0 if overall_success else 1)