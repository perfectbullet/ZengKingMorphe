"""
MinerU 结构感知分块器测试
"""

import pytest
from src.document_parser.mineru_structure_aware_chunker import MinerUStructureAwareChunker
from src.document_parser.base import TextChunk


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
