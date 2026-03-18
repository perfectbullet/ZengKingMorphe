"""
检索模块测试
"""

import pytest

from src.retrieval.base import RetrievedDocument
from src.retrieval.strategies import HybridRerankRetrieval
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import BGERerankerClient, BGERerankerClientError
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


class TestHybridRerankRetrieval:
    """混合检索 + Rerank 测试"""

    def test_hybrid_rerank_initialization(self, vector_store):
        """测试混合检索 + Rerank 初始化"""
        # 创建禁用的 reranker 客户端（仅用于测试，实际使用时会报错）
        class MockReranker:
            pass

        retrieval = HybridRerankRetrieval(
            vector_store=vector_store,
            rerank_client=MockReranker()
        )

        assert retrieval.vector_store == vector_store
        assert retrieval.rerank_client is not None
        assert retrieval.candidate_multiplier == 4  # 默认值

    def test_hybrid_rerank_with_custom_multiplier(self, vector_store):
        """测试自定义倍数的混合检索"""
        class MockReranker:
            pass

        retrieval = HybridRerankRetrieval(
            vector_store=vector_store,
            rerank_client=MockReranker(),
            candidate_multiplier=6
        )

        assert retrieval.candidate_multiplier == 6

    def test_hybrid_rerank_without_reranker_raises_error(self, vector_store):
        """测试没有 Reranker 时抛出 ValueError"""
        with pytest.raises(ValueError, match="rerank_client 是必需参数"):
            HybridRerankRetrieval(vector_store=vector_store, rerank_client=None)

    @pytest.mark.asyncio
    async def test_hybrid_rerank_with_empty_candidates(self, vector_store):
        """测试空候选列表"""
        class MockReranker:
            pass

        retrieval = HybridRerankRetrieval(
            vector_store=vector_store,
            rerank_client=MockReranker()
        )

        # 应该返回空列表
        result = await retrieval._rerank_documents("查询", [])
        assert result == []

    @pytest.mark.skip(reason="需要 Ollama 和 ChromaDB 服务运行")
    def test_retrieve_with_services(self, vector_store):
        """测试完整检索流程"""
        pytest.skip("需要 Ollama 和 ChromaDB 服务运行")


class TestBGERerankerClient:
    """BGE Reranker 客户端测试"""

    def test_reranker_initialization_disabled(self):
        """测试 Reranker 客户端初始化（禁用状态不应存在）"""
        # 修改后的 BGERerankerClient 不再接受 enable 参数
        # 如果服务不可用，初始化时会直接抛出异常
        pass

    def test_reranker_initialization_fails_on_invalid_url(self):
        """测试无效 URL 初始化失败"""
        with pytest.raises(BGERerankerClientError, match="服务连接失败"):
            BGERerankerClient(
                base_url="http://invalid-host-99999:9999",
                timeout=1
            )

    def test_rerank_with_empty_documents(self):
        """测试空文档列表的 rerank"""
        # 由于初始化时会检查服务，无法创建禁用状态的客户端
        # 此测试需要模拟服务或使用实际服务
        pytest.skip("需要 BGE Reranker 服务运行")

    @pytest.mark.skip(reason="需要 BGE Reranker 服务运行")
    def test_real_rerank_call(self):
        """测试真实的 Rerank API 调用"""
        client = BGERerankerClient(
            base_url="http://192.168.8.233:8091"
        )

        documents = [
            "人工智能是指由人制造出来的机器所表现出来的智能。",
            "今天天气很好，适合出去散步。",
            "机器学习是人工智能的一个分支。"
        ]

        results = client.rerank("什么是人工智能？", documents, top_n=2)

        assert len(results) <= 2
        # AI 相关的文档应该排在前面
        assert results[0][2] > results[1][2]

    def test_get_model_info_with_invalid_url(self):
        """测试无效 URL 的模型信息获取"""
        with pytest.raises(BGERerankerClientError, match="获取模型信息失败|服务连接失败"):
            client = BGERerankerClient(
                base_url="http://invalid-host-99999:9999",
                timeout=1
            )
            # 初始化时就会检查服务，所以这里会直接失败
            # 不需要额外调用 get_model_info


class TestRetriever:
    """检索器测试"""

    def test_retriever_initialization(self, vector_store):
        """测试检索器初始化（需要有效的 Reranker 服务）"""
        # 由于初始化需要连接 BGE Reranker 服务
        # 如果服务不可用会抛出 RuntimeError
        pytest.skip("需要 BGE Reranker 服务运行")

    def test_retriever_fails_without_reranker_service(self, vector_store):
        """测试没有 Reranker 服务时初始化失败"""
        from src.config import settings
        original_url = settings.rerank_base_url

        try:
            # 临时修改 settings 的 base_url
            settings.rerank_base_url = "http://invalid-host-99999:9999"

            with pytest.raises(RuntimeError, match="无法连接到 BGE Reranker 服务"):
                Retriever(
                    vector_store=vector_store,
                    use_rerank=True
                )
        finally:
            # 恢复原始值
            settings.rerank_base_url = original_url

    @pytest.mark.skip(reason="需要 Ollama 和 ChromaDB 服务运行")
    def test_retrieve(self, vector_store):
        """测试检索"""
        pytest.skip("需要 Ollama 和 ChromaDB 服务运行")

    @pytest.mark.skip(reason="需要 Ollama 和 ChromaDB 服务运行")
    def test_retrieve_multiple(self, vector_store):
        """测试多次检索"""
        pytest.skip("需要 Ollama 和 ChromaDB 服务运行")


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
