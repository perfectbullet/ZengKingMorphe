"""
查询扩展功能测试示例

演示 QueryExpander 的使用方法（仅 LLM 语义扩展）
"""

import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.query_expansion import QueryExpander
from src.retrieval.llm_client import create_llm_client


async def test_llm_semantic_expand():
    """测试 LLM 语义扩展（包含查询分解和术语扩展）"""
    print("\n" + "="*60)
    print("测试 LLM 语义扩展（包含查询分解和术语扩展）")
    print("="*60)

    try:
        llm_client = create_llm_client(
            provider="ollama",
            base_url="http://192.168.8.233:11434",
            model="qwen2.5:14b"
        )

        expander = QueryExpander(llm_client=llm_client)

        test_queries = [
            # 复合查询（需要分解）
            "导数和积分的关系",
            "函数与方程的区别",

            # 单概念查询（同义词扩展）
            "什么是导数？",
            "如何求微商？",

            # 方法论查询
            "如何求函数的极值？",

            # 术语扩展查询
            "f(x) 怎么求导数",
            "极限和收敛的联系",
        ]

        for query in test_queries:
            print(f"\n原始查询: {query}")
            try:
                expansions = await expander.expand(query, max_expansions=3)
                print(f"扩展结果 ({len(expansions)} 个):")
                for i, exp in enumerate(expansions, 1):
                    print(f"  {i}. {exp}")
            except Exception as e:
                print(f"  扩展失败: {e}")

    except Exception as e:
        print(f"\n无法连接 LLM 服务: {e}")
        print("请确保 Ollama 服务运行在 http://192.168.8.233:11434")


async def main():
    """主函数"""
    print("\n" + "="*60)
    print("查询扩展功能测试（仅 LLM 语义扩展）")
    print("="*60)

    await test_llm_semantic_expand()

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
