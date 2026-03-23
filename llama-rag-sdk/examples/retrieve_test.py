"""
检索测试示例

演示如何使用 Retriever 进行文档检索。

功能说明：
- 创建 Retriever 实例
- 执行单一查询检索
- 执行多查询检索（带去重）

注意：当前所有 Retriever 实例都使用统一的混合检索 + Rerank 策略。
"""

import asyncio

from llama_rag_sdk.config import settings
from llama_rag_sdk.document_indexer.indexer import DocumentIndexer
from llama_rag_sdk.document_indexer.storage import VectorStore
from llama_rag_sdk.retrieval.retriever import Retriever
from llama_rag_sdk.utils import setup_logger


async def setup_test_data():
    """准备测试数据"""
    # 示例文档
    documents = [
        """
# 数学：函数

函数是数学中的基本概念。一个函数 f: A → B 表示从集合 A 到集合 B 的映射。
对于集合 A 中的每一个元素 a，在集合 B 中都有唯一的一个元素 b 与之对应。

函数的定义包含三个要素：定义域、值域和对应法则。
常见的函数类型包括：一次函数、二次函数、指数函数、对数函数等。
""",
        """
# 数学：导数

导数是函数在某一点的变化率。如果函数 f 在点 x 处可导，则其导数为：
f'(x) = lim(h→0) [f(x+h) - f(x)] / h

导数的几何意义是函数图像在某一点的切线斜率。
常见的导数公式包括：
- (x^n)' = nx^(n-1)
- (sin x)' = cos x
- (e^x)' = e^x
""",
        """
# 数学：积分

积分是导数的逆运算。定积分计算曲线下的面积，不定积分求原函数。
牛顿-莱布尼茨公式将定积分与原函数联系起来：
∫[a,b] f(x)dx = F(b) - F(a)

其中 F 是 f 的一个原函数。
""",
        """
# 物理：牛顿定律

牛顿第一定律（惯性定律）：物体在没有外力作用时保持静止或匀速直线运动。

牛顿第二定律：物体的加速度与作用力成正比，与质量成反比。
F = ma

牛顿第三定律：作用力与反作用力大小相等、方向相反。
""",
        """
# 物理：能量

能量是描述物体状态的物理量。常见的能量形式包括：
- 动能：Ek = 1/2 mv²
- 势能：重力势能 Ep = mgh
- 机械能守恒：Ek + Ep = 常数

能量守恒定律是物理学的基本定律之一。
"""
    ]

    return documents


async def main():
    """主函数"""
    # 配置日志
    logger = setup_logger(
        log_level=settings.log_level,
        log_file=settings.log_file
    )
    logger.info("检索测试示例")

    # 确保必要的目录存在
    settings.ensure_directories()

    # 1. 准备测试数据
    print("\n=== 步骤 1: 准备测试数据 ===")
    test_documents = await setup_test_data()
    print(f"准备了 {len(test_documents)} 篇测试文档")

    # 2. 创建索引
    print("\n=== 步骤 2: 创建索引 ===")

    try:
        indexer = DocumentIndexer(collection_name="retrieve_test")

        await indexer.create_index(
            documents=test_documents,
            collection_name="retrieve_test"
        )
        print("✓ 索引创建成功")

    except Exception as e:
        print(f"✗ 索引创建失败: {e}")
        print("  请确保 vLLM embedding 服务和 ChromaDB 服务正在运行")
        return

    # 3. 测试检索功能
    print("\n=== 步骤 3: 测试检索 ===")

    try:
        vector_store = VectorStore(collection_name="retrieve_test")

        # 创建 Retriever（统一使用混合检索 + Rerank 策略）
        retriever = Retriever(vector_store=vector_store)

        # 示例查询
        queries = [
            "什么是导数？",
            "牛顿第二定律的内容是什么？",
            "动能的公式是什么？"
        ]

        for query in queries:
            print(f"\n--- 查询: {query} ---")
            results = await retriever.retrieve(query, top_k=3)

            for i, doc in enumerate(results, 1):
                print(f"  结果 {i} (得分: {doc.score:.3f}):")
                print(f"    {doc.text[:80]}...")

    except Exception as e:
        print(f"✗ 检索测试失败: {e}")
        print("  请确保 vLLM embedding、ChromaDB 和 BGE Reranker 服务正在运行")

    # 4. 多查询检索
    print("\n=== 步骤 4: 多查询检索 ===")

    try:
        vector_store = VectorStore(collection_name="retrieve_test")
        retriever = Retriever(vector_store=vector_store)

        queries = [
            "函数的定义",
            "导数公式",
            "牛顿定律",
            "能量守恒"
        ]

        all_results = await retriever.retrieve_multiple(
            queries=queries,
            top_k=2,
            deduplicate=True
        )

        print(f"多查询检索: {len(queries)} 个查询 -> {len(all_results)} 个唯一结果")
        for i, doc in enumerate(all_results[:5], 1):
            print(f"  {i}. {doc.text[:60]}... (得分: {doc.score:.3f})")

    except Exception as e:
        print(f"✗ 多查询检索失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())
