"""
检索模块测试
"""

import pytest

from src.retrieval.base import RetrievedDocument
from src.retrieval.strategies import VectorRetrieval, HybridRetrieval, RerankRetrieval
from src.retrieval.retriever import Retriever
from src.document_indexer.storage import VectorStore


@pytest.fixture
def vector_store():
    """向量存储实例"""
    # 使用临时集合进行测试
    return VectorStore(collection_name="test_retrieval")


@pytest.fixture
def sample_retrieved_documents():
    """示例检索结果"""
    return [
        RetrievedDocument(
            text="这是第一个相关文档",
            metadata={"source": "doc1.pdf", "page": 1},
            score=0.95,
            source="doc1.pdf",
            chunk_id="chunk_1"
        ),
        RetrievedDocument(
            text="这是第二个相关文档",
            metadata={"source": "doc2.pdf", "page": 2},
            score=0.85,
            source="doc2.pdf",
            chunk_id="chunk_2"
        ),
        RetrievedDocument(
            text="这是第三个相关文档",
            metadata={"source": "doc1.pdf", "page": 3},
            score=0.75,
            source="doc1.pdf",
            chunk_id="chunk_3"
        )
    ]


class TestVectorRetrieval:
    """向量检索测试"""

    def test_vector_retrieval_initialization(self, vector_store):
        """测试向量检索初始化"""
        retrieval = VectorRetrieval(vector_store=vector_store)

        assert retrieval.vector_store == vector_store

    @pytest.mark.asyncio
    async def test_retrieve(self, vector_store):
        """测试向量检索"""
        pytest.skip("需要 Ollama 和 ChromaDB 服务运行")

        retrieval = VectorRetrieval(vector_store=vector_store)

        documents = await retrieval.retrieve(
            query="测试查询",
            top_k=5
        )

        assert isinstance(documents, list)
        assert all(isinstance(doc, RetrievedDocument) for doc in documents)


class TestHybridRetrieval:
    """混合检索测试"""

    def test_hybrid_retrieval_initialization(self, vector_store):
        """测试混合检索初始化"""
        vector_retrieval = VectorRetrieval(vector_store=vector_store)
        retrieval = HybridRetrieval(vector_retrieval=vector_retrieval)

        assert retrieval.vector_retrieval == vector_retrieval
        assert retrieval.alpha == 0.7

    def test_hybrid_retrieval_with_custom_alpha(self, vector_store):
        """测试自定义 alpha 的混合检索"""
        vector_retrieval = VectorRetrieval(vector_store=vector_store)
        retrieval = HybridRetrieval(
            vector_retrieval=vector_retrieval,
            alpha=0.5
        )

        assert retrieval.alpha == 0.5

    @pytest.mark.asyncio
    async def test_retrieve(self, vector_store):
        """测试混合检索"""
        pytest.skip("需要 Ollama 和 ChromaDB 服务运行")

        vector_retrieval = VectorRetrieval(vector_store=vector_store)
        retrieval = HybridRetrieval(vector_retrieval=vector_retrieval)

        documents = await retrieval.retrieve(
            query="测试查询",
            top_k=5
        )

        assert isinstance(documents, list)
        assert all(isinstance(doc, RetrievedDocument) for doc in documents)


class TestRerankRetrieval:
    """重排序检索测试"""

    def test_rerank_initialization(self, vector_store):
        """测试重排序初始化"""
        vector_retrieval = VectorRetrieval(vector_store=vector_store)
        retrieval = RerankRetrieval(base_retrieval=vector_retrieval)

        assert retrieval.base_retrieval == vector_retrieval

    @pytest.mark.asyncio
    async def test_rerank_sorts_by_score(self, sample_retrieved_documents, vector_store):
        """测试重排序按分数排序"""
        # 创建一个模拟的基础检索器
        class MockRetrieval:
            async def retrieve(self, query, top_k, filters=None):
                # 返回乱序的结果
                return [
                    sample_retrieved_documents[2],
                    sample_retrieved_documents[0],
                    sample_retrieved_documents[1]
                ]

        base_retrieval = MockRetrieval()
        retrieval = RerankRetrieval(base_retrieval=base_retrieval)

        results = await retrieval.retrieve(
            query="测试查询",
            top_k=3
        )

        # 应该按分数降序排列
        assert results[0].score >= results[1].score >= results[2].score


class TestRetriever:
    """检索器测试"""

    def test_retriever_initialization(self, vector_store):
        """测试检索器初始化"""
        retriever = Retriever(vector_store=vector_store)

        assert retriever.vector_store == vector_store
        assert retriever.strategy is not None

    def test_retriever_with_hybrid(self, vector_store):
        """测试混合检索模式"""
        retriever = Retriever(
            vector_store=vector_store,
            use_hybrid=True
        )

        assert retriever.use_hybrid is True

    def test_retriever_with_rerank(self, vector_store):
        """测试重排序模式"""
        retriever = Retriever(
            vector_store=vector_store,
            use_rerank=True
        )

        assert retriever.use_rerank is True

    @pytest.mark.asyncio
    async def test_retrieve(self, vector_store):
        """测试检索"""
        pytest.skip("需要 Ollama 和 ChromaDB 服务运行")

        retriever = Retriever(vector_store=vector_store)

        documents = await retriever.retrieve(
            query="测试查询",
            top_k=5
        )

        assert isinstance(documents, list)

    @pytest.mark.asyncio
    async def test_retrieve_multiple(self, vector_store):
        """测试多次检索"""
        pytest.skip("需要 Ollama 和 ChromaDB 服务运行")

        retriever = Retriever(vector_store=vector_store)

        queries = ["查询1", "查询2", "查询3"]
        documents = await retriever.retrieve_multiple(
            queries=queries,
            top_k=3,
            deduplicate=True
        )

        assert isinstance(documents, list)


class TestRetrievedDocument:
    """检索文档模型测试"""

    def test_retrieved_document_creation(self):
        """测试检索文档创建"""
        doc = RetrievedDocument(
            text="文档内容",
            metadata={"source": "doc.pdf", "page": 1},
            score=0.9,
            source="doc.pdf",
            chunk_id="chunk_1"
        )

        assert doc.text == "文档内容"
        assert doc.score == 0.9
        assert doc.source == "doc.pdf"
        assert doc.chunk_id == "chunk_1"
        assert doc.metadata["page"] == 1

    def test_retrieved_document_defaults(self):
        """测试检索文档默认值"""
        doc = RetrievedDocument(text="文档内容")

        assert doc.text == "文档内容"
        assert doc.score == 0.0
        assert doc.metadata == {}
        assert doc.source is None
        assert doc.chunk_id is None
