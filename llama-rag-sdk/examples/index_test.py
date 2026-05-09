"""
文档索引测试示例
"""

import asyncio

from llama_rag_sdk.config import settings
from llama_rag_sdk.document_indexer.indexer import DocumentIndexer
from llama_rag_sdk.document_indexer.storage import VectorStore
from llama_rag_sdk.utils import setup_logger


async def main():
    """主函数"""
    # 配置日志
    logger = setup_logger(
        log_level=settings.log_level,
        log_file=settings.log_file
    )
    logger.info("文档索引测试示例")

    # 确保必要的目录存在
    settings.ensure_directories()

    # 1. 测试不同分块策略
    print("\n=== 测试 1: 不同分块策略 ===")

    from llama_rag_sdk.document_indexer.base import ChunkStrategy
    from llama_rag_sdk.document_indexer.chunker import FixedSizeChunker, SemanticChunker, HybridChunker

    test_text = """
# 第一章 函数

函数是数学中的基本概念。一个函数 f: A → B 表示从集合 A 到集合 B 的映射。

## 1.1 函数的定义

对于集合 A 中的每一个元素 a，在集合 B 中都有唯一的一个元素 b 与之对应。
我们将 b 称为 a 的像，记作 b = f(a)。

## 1.2 函数的性质

函数可以是单射、满射或双射。单射函数保证不同的输入有不同的输出。

# 第二章 导数

导数是函数在某一点的变化率。如果函数 f 在点 x 处可导，则其导数为：
f'(x) = lim(h→0) [f(x+h) - f(x)] / h

## 2.1 导数的几何意义

导数表示函数图像在某一点的切线斜率。
"""

    # 固定大小分块
    fixed_strategy = ChunkStrategy(
        type="fixed",
        chunk_size=100,
        chunk_overlap=20
    )
    fixed_chunker = FixedSizeChunker(fixed_strategy)
    fixed_chunks = fixed_chunker.chunk(test_text)
    print(f"固定大小分块: {len(fixed_chunks)} 个块")

    # 语义分块
    semantic_strategy = ChunkStrategy(type="semantic")
    semantic_chunker = SemanticChunker(semantic_strategy)
    semantic_chunks = semantic_chunker.chunk(test_text)
    print(f"语义分块: {len(semantic_chunks)} 个块")

    # 混合分块
    hybrid_strategy = ChunkStrategy(
        type="hybrid",
        chunk_size=50,
        chunk_overlap=10,
        max_chunk_size=150
    )
    hybrid_chunker = HybridChunker(hybrid_strategy)
    hybrid_chunks = hybrid_chunker.chunk(test_text)
    print(f"混合分块: {len(hybrid_chunks)} 个块")

    # 2. 创建索引
    print("\n=== 测试 2: 创建索引 ===")

    sample_documents = [
        test_text,
        """
这是一篇关于物理学的内容。

牛顿第二定律告诉我们，力等于质量乘以加速度：
F = ma

这个公式描述了力、质量和加速度之间的关系。
""",
        """
化学反应速率与温度有关。阿伦尼乌斯方程描述了这种关系：
k = Ae^(-Ea/RT)

其中 k 是反应速率常数，A 是指前因子，Ea 是活化能。
"""
    ]

    try:
        # 创建索引器
        indexer = DocumentIndexer(collection_name="test_collection")

        # 创建索引
        index_id = await indexer.create_index(
            documents=sample_documents,
            collection_name="test_collection"
        )
        print(f"✓ 索引创建成功: {index_id}")

        # 添加更多文档
        new_documents = [
            "这是另一篇关于数学的内容。微积分包括微分和积分两部分。",
            "这是另一篇关于物理的内容。能量守恒定律是物理学的基本定律之一。"
        ]

        doc_ids = await indexer.add_documents(
            documents=new_documents,
            collection_name="test_collection"
        )
        print(f"✓ 添加 {len(doc_ids)} 个文档块到索引")

        # 获取统计信息
        stats = await indexer.get_collection_stats("test_collection")
        print(f"✓ 集合统计: {stats}")

    except Exception as e:
        print(f"✗ 索引操作失败: {e}")
        print("  请确保 Ollama 和 ChromaDB 服务正在运行")
        return

    # 3. 测试向量存储
    print("\n=== 测试 3: 向量存储 ===")

    try:
        vector_store = VectorStore(collection_name="test_collection")
        stats = vector_store.get_stats()
        print(f"✓ 向量存储统计: {stats}")

    except Exception as e:
        print(f"✗ 向量存储操作失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())
