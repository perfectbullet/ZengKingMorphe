"""
RAG 系统集成示例

展示如何使用 RAGSystem 类进行完整的 RAG 流程
"""

import asyncio
from pathlib import Path

from src.rag_system import RAGSystem
from src.utils import setup_logger


async def main():
    """主函数"""
    # 配置日志
    logger = setup_logger(
        log_level="INFO",
        log_file="./logs/rag_demo.log"
    )
    logger.info("RAG 系统集成示例")

    # 创建 RAG 系统
    async with RAGSystem(
        collection_name="demo_collection",
        use_hybrid_retrieval=False,
        use_rerank=True
    ) as rag:
        # 1. 添加示例文档
        print("\n=== 步骤 1: 添加文档 ===")

        sample_documents = [
            {
                "text": """
# 数学：函数

函数是数学中的基本概念。一个函数 f: A → B 表示从集合 A 到集合 B 的映射。
对于集合 A 中的每一个元素 a，在集合 B 中都有唯一的一个元素 b 与之对应。

函数的定义包含三个要素：定义域、值域和对应法则。
常见的函数类型包括：一次函数、二次函数、指数函数、对数函数等。
""",
                "metadata": {
                    "subject": "数学",
                    "chapter": "函数",
                    "level": "高中"
                }
            },
            {
                "text": """
# 数学：导数

导数是函数在某一点的变化率。如果函数 f 在点 x 处可导，则其导数为：
f'(x) = lim(h→0) [f(x+h) - f(x)] / h

导数的几何意义是函数图像在某一点的切线斜率。
常见的导数公式包括：
- (x^n)' = nx^(n-1)
- (sin x)' = cos x
- (e^x)' = e^x
""",
                "metadata": {
                    "subject": "数学",
                    "chapter": "导数",
                    "level": "高中"
                }
            },
            {
                "text": """
# 物理：牛顿定律

牛顿第一定律（惯性定律）：物体在没有外力作用时保持静止或匀速直线运动。

牛顿第二定律：物体的加速度与作用力成正比，与质量成反比。
F = ma

牛顿第三定律：作用力与反作用力大小相等、方向相反。
""",
                "metadata": {
                    "subject": "物理",
                    "chapter": "牛顿定律",
                    "level": "高中"
                }
            },
            {
                "text": """
# 物理：能量

能量是描述物体状态的物理量。常见的能量形式包括：
- 动能：Ek = 1/2 mv²
- 势能：重力势能 Ep = mgh
- 机械能守恒：Ek + Ep = 常数

能量守恒定律是物理学的基本定律之一。
""",
                "metadata": {
                    "subject": "物理",
                    "chapter": "能量",
                    "level": "高中"
                }
            }
        ]

        try:
            # 添加文本文档
            texts = [doc["text"] for doc in sample_documents]
            metadatas = [doc["metadata"] for doc in sample_documents]

            doc_ids = await rag.add_text_documents(texts, metadatas)
            print(f"✓ 成功添加 {len(doc_ids)} 个文档块")

        except Exception as e:
            print(f"✗ 添加文档失败: {e}")
            print("  请确保 Ollama 和 ChromaDB 服务正在运行")
            return

        # 2. 索引 PDF 文档（如果有）
        print("\n=== 步骤 2: 索引 PDF 文档 ===")
        pdf_path = "data/sample_pdf/example.pdf"

        if Path(pdf_path).exists():
            try:
                doc_ids = await rag.index_document(pdf_path)
                print(f"✓ 成功索引 PDF 文档: {len(doc_ids)} 个块")
            except Exception as e:
                print(f"✗ 索引 PDF 失败: {e}")
        else:
            print(f"  未找到 PDF 文件: {pdf_path}")

        # 3. 检索文档
        print("\n=== 步骤 3: 检索文档 ===")

        queries = [
            "什么是导数？",
            "牛顿第二定律的内容是什么？",
            "动能的计算公式",
            "函数的定义包含哪些要素？"
        ]

        for query in queries:
            print(f"\n查询: {query}")
            try:
                results = await rag.retrieve(query, top_k=2)

                if results:
                    for i, doc in enumerate(results, 1):
                        print(f"  结果 {i} (得分: {doc.score:.3f}):")
                        print(f"    {doc.text[:100]}...")

                        # 显示元数据
                        if doc.metadata:
                            subject = doc.metadata.get("subject", "未知")
                            chapter = doc.metadata.get("chapter", "未知")
                            print(f"    [科目: {subject}, 章节: {chapter}]")
                else:
                    print("  未找到相关结果")

            except Exception as e:
                print(f"  检索失败: {e}")

        # 4. 多查询检索
        print("\n=== 步骤 4: 多查询检索 ===")

        try:
            all_results = await rag.retrieve_multiple(
                queries=queries[:3],
                top_k=2,
                deduplicate=True
            )
            print(f"从 3 个查询中检索到 {len(all_results)} 个唯一结果")

            for i, doc in enumerate(all_results[:5], 1):
                print(f"  {i}. {doc.text[:60]}... (得分: {doc.score:.3f})")

        except Exception as e:
            print(f"✗ 多查询检索失败: {e}")

        # 5. 使用过滤器检索
        print("\n=== 步骤 5: 使用过滤器检索 ===")

        try:
            # 只检索数学相关文档
            results = await rag.retrieve(
                query="导数的定义",
                top_k=3,
                filters={"subject": "数学"}
            )

            print(f"数学相关结果: {len(results)} 个")
            for i, doc in enumerate(results, 1):
                print(f"  {i}. {doc.text[:80]}... (得分: {doc.score:.3f})")

        except Exception as e:
            print(f"✗ 过滤检索失败: {e}")

        # 6. 获取统计信息
        print("\n=== 步骤 6: 系统统计 ===")

        try:
            stats = await rag.get_stats()
            print(f"集合名称: {stats.get('collection_name')}")
            print(f"文档数量: {stats.get('count')}")
            print(f"缓存文档数: {stats.get('document_cache_size')}")

        except Exception as e:
            print(f"✗ 获取统计信息失败: {e}")

        print("\n=== 示例完成 ===")


if __name__ == "__main__":
    asyncio.run(main())
