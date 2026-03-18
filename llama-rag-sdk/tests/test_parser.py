"""
文档解析模块测试
"""

import pytest
from pathlib import Path

from src.document_parser.base import ParsedDocument, TextChunk, ImageInfo
from src.document_parser import (
    MinerUParser,
    ParseOptions,
    ReturnOptions,
)
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
        assert mineru_parser.api_url is not None
        assert mineru_parser.endpoint is not None
        assert mineru_parser.output_dir is not None

    def test_parse_options(self):
        """测试解析选项"""
        options = ParseOptions(
            backend="pipeline",
            parse_method="auto",
            lang="ch",
            formula_enable=True,
            table_enable=True,
            start_page=0,
            end_page=10,
        )

        assert options.backend == "pipeline"
        assert options.parse_method == "auto"
        assert options.lang == "ch"
        assert options.formula_enable is True
        assert options.table_enable is True
        assert options.start_page == 0
        assert options.end_page == 10

    def test_return_options(self):
        """测试返回选项"""
        options = ReturnOptions(
            return_md=True,
            return_middle_json=False,
            return_model_output=False,
            return_content_list=True,
            return_images=True,
        )

        assert options.return_md is True
        assert options.return_middle_json is False
        assert options.return_content_list is True
        assert options.return_images is True

    @pytest.mark.asyncio
    async def test_health_check(self, mineru_parser):
        """测试健康检查"""
        is_healthy = await mineru_parser.health_check()
        # 不断言结果，因为服务可能不可用
        assert isinstance(is_healthy, bool)

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
    async def test_parse_with_options(self, mineru_parser, sample_pdf_path):
        """测试使用自定义选项解析文档"""
        pytest.skip("需要 MinerU 服务运行")

        parse_options = ParseOptions(
            backend="pipeline",
            parse_method="auto",
            start_page=0,
            end_page=2,  # 只解析前 2 页
        )
        return_options = ReturnOptions(
            return_md=True,
            return_content_list=True,
        )

        document = await mineru_parser.parse(
            sample_pdf_path,
            parse_options=parse_options,
            return_options=return_options,
        )

        assert isinstance(document, ParsedDocument)

    @pytest.mark.asyncio
    async def test_parse_batch(self, mineru_parser):
        """测试批量解析"""
        pytest.skip("需要 MinerU 服务运行")

        file_paths = ["data/sample_pdf/test1.pdf", "data/sample_pdf/test2.pdf"]
        documents = await mineru_parser.parse_batch(file_paths)

        assert len(documents) == len(file_paths)
        for doc in documents:
            assert isinstance(doc, ParsedDocument)

    @pytest.mark.asyncio
    async def test_parse_to_memory(self, mineru_parser, sample_pdf_path):
        """测试直接返回 JSON"""
        pytest.skip("需要 MinerU 服务运行")

        result = await mineru_parser.parse_to_memory(sample_pdf_path)

        assert isinstance(result, dict)


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
