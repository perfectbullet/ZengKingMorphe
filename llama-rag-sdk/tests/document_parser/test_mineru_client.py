"""
MinerU 客户端测试
"""

import pytest
from llama_rag_sdk.document_parser import (
    MinerUParser,
    ParseOptions,
    ReturnOptions,
)


@pytest.fixture
def sample_pdf_path():
    """测试 PDF 文件路径"""
    return "data/sample_pdf/test.pdf"


@pytest.fixture
def mineru_parser():
    """MinerU 解析器实例"""
    return MinerUParser()


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

        assert document.title is not None
        assert document.content is not None
        assert len(document.chunks) > 0
