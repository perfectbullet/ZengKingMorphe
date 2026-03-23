"""
文档索引模块测试
"""

import pytest

from llama_rag_sdk.document_indexer.base import ChunkStrategy
from llama_rag_sdk.document_indexer.chunker import FixedSizeChunker, SemanticChunker, HybridChunker
from llama_rag_sdk.document_indexer.indexer import DocumentIndexer


@pytest.fixture
def chunk_strategy():
    """分块策略"""
    return ChunkStrategy(
        type="fixed",
        chunk_size=100,
        chunk_overlap=20
    )


@pytest.fixture
def sample_documents():
    """示例文档"""
    return [
        "这是第一篇文档的内容。它包含多个段落。每个段落都有不同的信息。",
        "这是第二篇文档的内容。它讨论了索引和检索的基本概念。"
    ]


class TestChunkers:
    """分块器测试"""

    def test_fixed_size_chunker(self, chunk_strategy):
        """测试固定大小分块器"""
        chunker = FixedSizeChunker(chunk_strategy)

        text = "这是一段测试文本。它将被分成多个块。每个块都有固定的大小。"
        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_semantic_chunker(self):
        """测试语义分块器"""
        strategy = ChunkStrategy(type="semantic")
        chunker = SemanticChunker(strategy)

        text = """# 第一章

这是第一章的内容。

## 第一节

这是第一节的内容。

# 第二章

这是第二章的内容。
"""

        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        # 语义分块应该根据标题分割
        assert any("第一章" in chunk for chunk in chunks)

    def test_hybrid_chunker(self):
        """测试混合分块器"""
        strategy = ChunkStrategy(
            type="hybrid",
            chunk_size=50,
            chunk_overlap=10
        )
        chunker = HybridChunker(strategy)

        text = "# 标题\n" + "这是一段很长的文本。" * 20
        chunks = chunker.chunk(text)

        assert len(chunks) > 0


class TestDocumentIndexer:
    """文档索引器测试"""

    def test_indexer_initialization(self):
        """测试索引器初始化"""
        indexer = DocumentIndexer()

        assert indexer.chunk_strategy is not None
        assert indexer.vector_store is not None
        # Embedding 模型可能初始化失败（如果 Ollama 不可用）

    def test_create_chunker(self):
        """测试分块器创建"""
        strategy = ChunkStrategy(type="fixed")
        indexer = DocumentIndexer(chunk_strategy=strategy)

        assert isinstance(indexer.chunker, FixedSizeChunker)

        strategy = ChunkStrategy(type="semantic")
        indexer = DocumentIndexer(chunk_strategy=strategy)

        assert isinstance(indexer.chunker, SemanticChunker)

    @pytest.mark.asyncio
    async def test_chunk_documents(self):
        """测试文档分块"""
        indexer = DocumentIndexer()

        documents = [
            "这是第一篇文档",
            "这是第二篇文档"
        ]

        chunks, metadatas = indexer._chunk_documents(documents)

        assert len(chunks) > 0
        assert len(metadatas) == len(chunks)
        assert all("chunk_id" in m for m in metadatas)
        assert all("chunk_index" in m for m in metadatas)

    @pytest.mark.asyncio
    async def test_create_index(self, sample_documents):
        """测试创建索引"""
        # 注意：此测试需要 Ollama 服务运行
        pytest.skip("需要 Ollama 服务运行")

        indexer = DocumentIndexer()

        index_id = await indexer.create_index(
            documents=sample_documents,
            collection_name="test_collection"
        )

        assert index_id == "test_collection"

    @pytest.mark.asyncio
    async def test_add_documents(self, sample_documents):
        """测试添加文档"""
        # 注意：此测试需要 Ollama 服务运行
        pytest.skip("需要 Ollama 服务运行")

        indexer = DocumentIndexer()

        doc_ids = await indexer.add_documents(
            documents=sample_documents,
            collection_name="test_collection"
        )

        assert len(doc_ids) > 0
        assert all(isinstance(doc_id, str) for doc_id in doc_ids)

    @pytest.mark.asyncio
    async def test_get_collection_stats(self):
        """测试获取集合统计"""
        pytest.skip("需要 ChromaDB 服务运行")

        indexer = DocumentIndexer()

        stats = await indexer.get_collection_stats("test_collection")

        assert "collection_name" in stats
        assert "count" in stats
