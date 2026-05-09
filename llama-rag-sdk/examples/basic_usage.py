"""
LlamaRAG SDK 基本使用示例

展示如何使用 RAG SDK 进行文档解析、索引和检索
"""

import asyncio
from pathlib import Path

from llama_rag_sdk.config import settings
from llama_rag_sdk.document_parser.mineru_client import MinerUParser
from llama_rag_sdk.document_parser.image_processor import ImageDescriptor
from llama_rag_sdk.document_indexer.indexer import DocumentIndexer
from llama_rag_sdk.retrieval.retriever import Retriever
from llama_rag_sdk.document_indexer.storage import VectorStore
from llama_rag_sdk.utils import setup_logger


async def main():
    """主函数"""
    # 1. 配置日志
    logger = setup_logger(
        log_level=settings.log_level,
        log_file=settings.log_file
    )
    logger.info("LlamaRAG SDK 基本使用示例")

    # 确保必要的目录存在
    settings.ensure_directories()

    # 2. 文档解析
    print("\n=== 步骤 1: 文档解析 ===")
    async with MinerUParser() as parser:
        async with ImageDescriptor() as descriptor:
            # 解析文档（需要真实的 PDF 文件）
            pdf_path = "data/sample_pdf/example.pdf"

            if Path(pdf_path).exists():
                document = await parser.parse(pdf_path)
                print(f"文档标题: {document.title}")
                print(f"文本块数量: {len(document.chunks)}")
                print(f"图片数量: {len(document.images)}")

                # 生成图片描述
                if document.images and settings.enable_image_description:
                    print("\n生成图片描述...")
                    document.images = await descriptor.describe_images_batch(
                        document.images,
                        concurrency=3
                    )
                    for img in document.images:
                        print(f"  - {Path(img.path).name}: {img.description[:50]}...")
            else:
                print(f"PDF 文件不存在: {pdf_path}")
                print("请将示例 PDF 放到 data/sample_pdf/ 目录下")

    # 3. 创建索引
    print("\n=== 步骤 2: 创建索引 ===")

    # 示例文档（如果没有 PDF，使用示例文本）
    sample_documents = [
        "这是一篇关于数学的教材内容。它包含了函数、导数和积分的概念。",
        "导数是函数在某一点的变化率，表示函数在该点的瞬时变化率。",
        "积分是导数的逆运算，用于计算曲线下的面积和累积量。",
        "函数是数学中的基本概念，描述了两个集合之间的对应关系。"
    ]

    try:
        indexer = DocumentIndexer(collection_name="example_collection")

        # 添加文档到索引
        doc_ids = await indexer.add_documents(
            documents=sample_documents,
            collection_name="example_collection"
        )
        print(f"成功添加 {len(doc_ids)} 个文档块到索引")

        # 获取统计信息
        stats = await indexer.get_collection_stats("example_collection")
        print(f"集合统计: {stats}")

    except Exception as e:
        print(f"索引创建失败（可能需要 Ollama 服务）: {e}")
        return

    # 4. 检索
    print("\n=== 步骤 3: 检索 ===")

    try:
        # 创建向量存储
        vector_store = VectorStore(collection_name="example_collection")

        # 创建检索器
        retriever = Retriever(
            vector_store=vector_store,
            use_hybrid=False,  # 使用纯向量检索
            use_rerank=False  # 不使用重排序
        )

        # 执行检索
        queries = [
            "什么是导数？",
            "函数的定义是什么？",
            "积分和导数的关系"
        ]

        for query in queries:
            print(f"\n查询: {query}")
            documents = await retriever.retrieve(query, top_k=2)

            for i, doc in enumerate(documents, 1):
                print(f"  结果 {i} (得分: {doc.score:.3f}):")
                print(f"    {doc.text[:100]}...")

        # 多查询检索
        print("\n=== 多查询检索 ===")
        all_results = await retriever.retrieve_multiple(
            queries=queries,
            top_k=2,
            deduplicate=True
        )
        print(f"共检索到 {len(all_results)} 个唯一结果")

    except Exception as e:
        print(f"检索失败（可能需要 Ollama 服务）: {e}")


if __name__ == "__main__":
    asyncio.run(main())
