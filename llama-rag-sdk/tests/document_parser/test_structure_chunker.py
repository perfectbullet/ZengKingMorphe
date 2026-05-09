"""
MinerU 结构感知分块器测试
"""

import pytest
from llama_rag_sdk.document_parser.mineru_structure_aware_chunker import MinerUStructureAwareChunker
from llama_rag_sdk.document_parser.base import TextChunk


@pytest.fixture
def chunker():
    """分块器实例"""
    return MinerUStructureAwareChunker()


@pytest.fixture
def sample_content_list():
    """示例 content_list"""
    return [
        {
            "type": "title",
            "text": "第一章 简介",
            "text_level": 1,
            "page_idx": 0,
        },
        {
            "type": "text",
            "text": "这是第一章的内容",
            "page_idx": 0,
        },
        {
            "type": "title",
            "text": "1.1 小节",
            "text_level": 2,
            "page_idx": 1,
        },
        {
            "type": "text",
            "text": "这是小节的内容" * 50,  # 较长文本测试分割
            "page_idx": 1,
        },
    ]


class TestMinerUStructureAwareChunker:
    """MinerU 结构感知分块器测试"""

    def test_chunker_initialization(self, chunker):
        """测试分块器初始化"""
        assert chunker.max_chunk_size > 0
        assert chunker.chunk_overlap >= 0
        assert chunker.min_chunk_size > 0

    def test_chunk_content_list_basic(self, chunker, sample_content_list):
        """测试基本分块功能"""
        chunks = chunker.chunk_content_list(
            content_list=sample_content_list,
            pdf_name="test.pdf"
        )

        assert len(chunks) > 0
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)

    def test_chunk_with_metadata(self, chunker, sample_content_list):
        """测试分块元数据"""
        chunks = chunker.chunk_content_list(
            content_list=sample_content_list,
            pdf_name="test.pdf"
        )

        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert "page_idx" in chunk.metadata
            assert "title_path" in chunk.metadata

    def test_empty_content_list(self, chunker):
        """测试空 content_list"""
        chunks = chunker.chunk_content_list(
            content_list=[],
            pdf_name="test.pdf"
        )

        assert len(chunks) == 0

    def test_chunk_size_limits(self, chunker):
        """测试分块大小限制"""
        # 创建一个超长的 content_list
        long_content = [
            {
                "type": "title",
                "text": "长章节",
                "text_level": 1,
                "page_idx": 0,
            },
            {
                "type": "text",
                "text": "这是一个很长的段落" * 200,
                "page_idx": 0,
            },
        ]

        chunks = chunker.chunk_content_list(
            content_list=long_content,
            pdf_name="test.pdf"
        )

        # 验证所有 chunk 都不超过最大限制
        for chunk in chunks:
            assert len(chunk.text) <= chunker.max_chunk_size + 100  # 允许小误差


class TestTOCFilter:
    """目录过滤功能测试"""

    @pytest.fixture
    def toc_content_list(self):
        """模拟包含目录的 content_list"""
        return [
            # 第3页 - 第一章目录
            {
                "type": "title",
                "text": "第一章 集合与常用逻辑用语",
                "text_level": 1,
                "page_idx": 3,
            },
            {
                "type": "text",
                "text": "1.1 集合的概念  2  \n1.2集合间的基本关系  \n1.3集合的基本运算 10  \n阅读与思考 集合中元素的个数 15",
                "page_idx": 3,
            },
            # 第5页 - 正文内容
            {
                "type": "title",
                "text": "第一章集合与常用逻辑用语",
                "text_level": 1,
                "page_idx": 5,
            },
            {
                "type": "text",
                "text": "我们知道，方程 $x ^ { 2 } = 2$ 在有理数范围内无解，但在实数范围内有解。" * 10,
                "page_idx": 5,
            },
        ]

    def test_toc_filter_enabled(self, toc_content_list):
        """测试启用目录过滤时，目录内容被过滤"""
        chunker = MinerUStructureAwareChunker()
        chunker.enable_toc_filter = True
        chunker.max_toc_pages = 4

        chunks = chunker.chunk_content_list(
            content_list=toc_content_list,
            pdf_name="test.pdf"
        )

        # 目录应该被过滤，只保留正文
        assert len(chunks) == 1
        assert "我们知道，方程" in chunks[0].text
        assert "1.1 集合的概念" not in chunks[0].text

    def test_toc_filter_disabled(self, toc_content_list):
        """测试禁用目录过滤时，目录内容被保留"""
        chunker = MinerUStructureAwareChunker()
        chunker.enable_toc_filter = False

        chunks = chunker.chunk_content_list(
            content_list=toc_content_list,
            pdf_name="test.pdf"
        )

        # 目录应该被保留
        assert len(chunks) >= 1
        toc_chunks = [c for c in chunks if "1.1 集合的概念" in c.text]
        assert len(toc_chunks) > 0

    def test_sequential_relations(self, toc_content_list):
        """测试顺序关系（prev/next_chunk_id）"""
        chunker = MinerUStructureAwareChunker()
        chunker.enable_toc_filter = True

        chunks = chunker.chunk_content_list(
            content_list=toc_content_list,
            pdf_name="test.pdf"
        )

        if len(chunks) >= 2:
            # 第一个 chunk 没有 prev，有 next
            assert chunks[0].metadata.get("prev_chunk_id") is None
            assert chunks[0].metadata.get("next_chunk_id") == chunks[1].metadata["chunk_id"]

            # 中间 chunk 既有 prev 也有 next
            assert chunks[1].metadata.get("prev_chunk_id") == chunks[0].metadata["chunk_id"]

            # 最后一个 chunk 有 prev，没有 next
            assert chunks[-1].metadata.get("next_chunk_id") is None
            assert chunks[-1].metadata.get("prev_chunk_id") == chunks[-2].metadata["chunk_id"]

    def test_is_toc_content_method(self):
        """测试 _is_toc_content 方法"""
        chunker = MinerUStructureAwareChunker()

        # 目录内容（页码密度高）
        toc_text = "1.1 集合的概念  2  \n1.2集合间的基本关系  \n1.3集合的基本运算 10"
        assert chunker._is_toc_content(toc_text, page_idx=3, title_path=["第一章"]) is True

        # 正常内容（页码密度低）
        normal_text = "这是正常的正文内容，不包含大量页码引用。" * 5
        assert chunker._is_toc_content(normal_text, page_idx=3, title_path=["第一章"]) is False

        # 超过页码范围的内容不会被识别为目录
        assert chunker._is_toc_content(toc_text, page_idx=10, title_path=["第一章"]) is False
