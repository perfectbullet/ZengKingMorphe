"""
文档解析模块测试
"""

import pytest
from pathlib import Path

from src.document_parser.base import ParsedDocument, TextChunk, ImageInfo
from src.document_parser.mineru_client import MinerUParser
from src.document_parser.image_processor import ImageDescriptor


@pytest.fixture
def sample_pdf_path():
    """测试 PDF 文件路径"""
    return "data/sample_pdf/test.pdf"


@pytest.fixture
def mineru_parser():
    """MinerU 解析器实例"""
    return MinerUParser()


@pytest.fixture
def image_descriptor():
    """图片描述生成器实例"""
    return ImageDescriptor()


class TestMinerUParser:
    """MinerU 解析器测试"""

    def test_parser_initialization(self, mineru_parser):
        """测试解析器初始化"""
        assert mineru_parser.mcp_url is not None
        assert mineru_parser.output_dir is not None

    @pytest.mark.asyncio
    async def test_parse_document(self, mineru_parser, sample_pdf_path):
        """测试文档解析"""
        # 注意：此测试需要 MinerU 服务运行
        # 如果服务不可用，跳过测试
        pytest.skip("需要 MinerU 服务运行")

        document = await mineru_parser.parse(sample_pdf_path)

        assert isinstance(document, ParsedDocument)
        assert document.title is not None
        assert document.content is not None
        assert len(document.chunks) > 0

    @pytest.mark.asyncio
    async def test_parse_batch(self, mineru_parser):
        """测试批量解析"""
        pytest.skip("需要 MinerU 服务运行")

        file_paths = ["data/sample_pdf/test1.pdf", "data/sample_pdf/test2.pdf"]
        documents = await mineru_parser.parse_batch(file_paths)

        assert len(documents) == len(file_paths)
        for doc in documents:
            assert isinstance(doc, ParsedDocument)


class TestImageDescriptor:
    """图片描述生成器测试"""

    def test_descriptor_initialization(self, image_descriptor):
        """测试描述生成器初始化"""
        assert image_descriptor.model_name is not None

    @pytest.mark.asyncio
    async def test_describe_image(self, image_descriptor):
        """测试图片描述生成"""
        # 注意：此测试需要 Ollama Qwen2-VL 服务
        pytest.skip("需要 Qwen2-VL 服务运行")

        image_path = "data/output/test_image.png"
        description = await image_descriptor.describe_image(image_path)

        assert isinstance(description, str)
        assert len(description) > 0


class TestDataModels:
    """数据模型测试"""

    def test_parsed_document(self):
        """测试 ParsedDocument 模型"""
        doc = ParsedDocument(
            title="测试文档",
            content="测试内容",
            chunks=[
                TextChunk(text="块1", index=0),
                TextChunk(text="块2", index=1)
            ],
            images=[
                ImageInfo(path="image1.png"),
                ImageInfo(path="image2.png", description="图片描述")
            ],
            metadata={"source": "test.pdf"}
        )

        assert doc.title == "测试文档"
        assert len(doc.chunks) == 2
        assert len(doc.images) == 2
        assert doc.images[1].description == "图片描述"

    def test_text_chunk(self):
        """测试 TextChunk 模型"""
        chunk = TextChunk(
            text="测试文本",
            page=1,
            section="第一章",
            index=0,
            metadata={"key": "value"}
        )

        assert chunk.text == "测试文本"
        assert chunk.page == 1
        assert chunk.section == "第一章"
        assert chunk.metadata["key"] == "value"

    def test_image_info(self):
        """测试 ImageInfo 模型"""
        image = ImageInfo(
            path="test.png",
            page=2,
            description="测试图片",
            position="top-left"
        )

        assert image.path == "test.png"
        assert image.page == 2
        assert image.description == "测试图片"
        assert image.position == "top-left"
