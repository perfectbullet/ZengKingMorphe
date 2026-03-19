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
from src.document_parser.mineru_structure_aware_chunker import MinerUStructureAwareChunker


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


class TestMinerUStructureAwareChunker:
    """MinerU 结构感知分块器测试"""

    def test_chunker_initialization(self):
        """测试分块器初始化"""
        chunker = MinerUStructureAwareChunker()

        # 配置从环境变量读取，默认值：CHUNK_SIZE=1000, CHUNK_OVERLAP=150
        # min_chunk_size 自动计算为 max_chunk_size 的 15%
        assert chunker.max_chunk_size == 1000
        assert chunker.min_chunk_size == 150  # 1000 * 0.15
        assert chunker.chunk_overlap == 150
        assert chunker.strategy == "hybrid"

    def test_chunk_simple_content_list(self):
        """测试简单的 content_list 分块"""
        content_list = [
            {"type": "title", "text": "第一章 导数", "text_level": 1, "page_idx": 0},
            {
                "type": "text",
                "text": "导数是描述变化率的概念，它表示函数在某一点的瞬时变化率。",
                "page_idx": 0,
            },
            {
                "type": "text",
                "text": "导数在几何上可以表示为曲线切线的斜率。",
                "page_idx": 0,
            },
            {"type": "title", "text": "1.1 导数定义", "text_level": 2, "page_idx": 1},
            {
                "type": "text",
                "text": "设函数 y = f(x) 在点 x0 的某个邻域内有定义。",
                "page_idx": 1,
            },
        ]

        chunker = MinerUStructureAwareChunker()
        chunks = chunker.chunk_content_list(content_list, "test.pdf")

        # 验证分块结果（应该生成 2 个 chunk，因为内容都足够长）
        assert len(chunks) >= 1  # 至少一个章节

        # 第一个 chunk 应该包含"第一章 导数"
        first_chunk = chunks[0]
        assert "第一章 导数" in first_chunk.text
        assert first_chunk.metadata["title_path"][0] == "第一章 导数"

        # 第二个 chunk 应该包含"1.1 导数定义"
        second_chunk = chunks[1]
        assert "1.1 导数定义" in second_chunk.text
        assert "1.1 导数定义" in second_chunk.metadata["title_path"]

    def test_chunk_with_images(self):
        """测试包含图片的 content_list 分块"""
        content_list = [
            {"type": "title", "text": "第一章", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "这是一段文字", "page_idx": 0},
            {
                "type": "image",
                "img_path": "images/test.png",
                "page_idx": 0
            },
            {"type": "text", "text": "图片说明文字", "page_idx": 0},
        ]

        chunker = MinerUStructureAwareChunker()
        chunks = chunker.chunk_content_list(
            content_list,
            "test.pdf",
            image_captions={"images/test.png": "测试图片描述"}
        )

        assert len(chunks) >= 1

        # 验证图片信息被记录
        first_chunk = chunks[0]
        assert "images/test.png" in first_chunk.metadata.get("image_references", [])

    def test_chunk_long_section_split(self):
        """测试超长章节分割"""
        # 创建一个超长章节（超过 400 字符）
        long_text = "这是一段很长的文字。" * 100  # 约 400+ 字符

        content_list = [
            {"type": "title", "text": "第一章", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": long_text, "page_idx": 0},
        ]

        chunker = MinerUStructureAwareChunker()
        chunks = chunker.chunk_content_list(content_list, "test.pdf")

        # 超长章节应该被分割成多个 chunk
        assert len(chunks) >= 1

        # 每个 chunk 的长度不应超过 max_chunk_size
        for chunk in chunks:
            # 允许小误差（标题前缀等）
            assert len(chunk.text) <= chunker.max_chunk_size + 100

    def test_chunk_metadata_structure(self):
        """测试分块元数据结构"""
        content_list = [
            {"type": "title", "text": "第一章 导数", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "导数是描述变化率的重要概念，表示函数在某一点的瞬时变化率。", "page_idx": 0},
        ]

        chunker = MinerUStructureAwareChunker()
        chunks = chunker.chunk_content_list(content_list, "test.pdf")

        assert len(chunks) >= 1

        chunk = chunks[0]

        # 验证必需的元数据字段
        assert "chunk_id" in chunk.metadata
        assert "type" in chunk.metadata
        assert "page_idx" in chunk.metadata
        assert "page_indices" in chunk.metadata
        assert "title_path" in chunk.metadata
        assert "structure_level" in chunk.metadata
        assert "block_types" in chunk.metadata

        # 验证标题路径
        assert isinstance(chunk.metadata["title_path"], list)
        assert len(chunk.metadata["title_path"]) > 0

    def test_chunk_empty_content_list(self):
        """测试空 content_list"""
        chunker = MinerUStructureAwareChunker()
        chunks = chunker.chunk_content_list([], "test.pdf")

        assert len(chunks) == 0

    def test_chunk_filter_short_sections(self):
        """测试过滤过短章节"""
        content_list = [
            {"type": "title", "text": "第一章", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "短", "page_idx": 0},
            {"type": "title", "text": "第二章", "text_level": 1, "page_idx": 1},
            {"type": "text", "text": "这是一个足够长的章节内容，应该被保留。" * 10, "page_idx": 1},
        ]

        chunker = MinerUStructureAwareChunker()
        chunks = chunker.chunk_content_list(content_list, "test.pdf")

        # 短章节应该被过滤掉
        for chunk in chunks:
            assert len(chunk.text) >= chunker.MIN_VALID_CHUNK_LENGTH

    def test_chunk_with_text_level(self):
        """测试使用 text_level 识别标题"""
        content_list = [
            {"type": "text", "text": "第一章 集合", "text_level": 1, "page_idx": 0},
            {
                "type": "text",
                "text": "集合是数学中的基本概念，具有确定性、互异性和无序性。",
                "page_idx": 0,
            },
            {
                "type": "text",
                "text": "集合中的元素必须是确定的，互不相同的，且无序的。",
                "page_idx": 0,
            },
            {"type": "text", "text": "1.1 集合的含义", "text_level": 2, "page_idx": 1},
            {
                "type": "text",
                "text": "一般地，我们把研究对象统称为元素，把一些元素组成的总体称为集合。",
                "page_idx": 1,
            },
        ]

        chunker = MinerUStructureAwareChunker()
        chunks = chunker.chunk_content_list(content_list, "test.pdf")

        # 验证分块结果
        assert len(chunks) >= 1, "应该至少生成一个章节"

        # 第一个 chunk 应该包含"第一章 集合"
        assert "第一章 集合" in chunks[0].text
        assert chunks[0].metadata["title_path"][0] == "第一章 集合"

    def test_chunk_with_discarded(self):
        """测试跳过 discarded 类型"""
        content_list = [
            {"type": "discarded", "text": "页眉文字", "page_idx": 0},
            {
                "type": "text",
                "text": "第一章 内容",
                "text_level": 1,
                "page_idx": 0
            },
            {
                "type": "text",
                "text": "这是一段较长的正文内容，应该被保留下来。"
                "集合是数学中的基本概念，具有确定性、互异性和无序性。",
                "page_idx": 0
            },
            {"type": "discarded", "text": "页脚文字", "page_idx": 0},
        ]

        chunker = MinerUStructureAwareChunker()
        chunks = chunker.chunk_content_list(content_list, "test.pdf")

        # 验证 discarded 内容被跳过
        assert len(chunks) >= 1
        assert "页眉文字" not in chunks[0].text
        assert "页脚文字" not in chunks[0].text
        assert "第一章 内容" in chunks[0].text
        assert "这是一段较长的正文内容" in chunks[0].text

    def test_chunk_real_math_textbook(self):
        """测试真实数学教材的 content_list 分块"""
        import json

        content_list_path = (
            "data/output/01高中数学必修第一册-40pages-part1-page1-40/"
            "01高中数学必修第一册-40pages-part1-page1-40_content_list.json"
        )

        # 检查文件是否存在
        path = Path(content_list_path)
        if not path.exists():
            pytest.skip(f"测试文件不存在: {content_list_path}")

        # 读取 content_list
        with open(path, encoding="utf-8") as f:
            content_list = json.load(f)

        print(f"\n读取到 {len(content_list)} 个内容项")

        # 使用结构感知分块
        chunker = MinerUStructureAwareChunker()
        chunks = chunker.chunk_content_list(
            content_list,
            "01高中数学必修第一册-40pages-part1-page1-40.pdf",
        )

        print(f"生成了 {len(chunks)} 个 chunk")

        # 验证分块结果
        assert len(chunks) > 0, "应该至少生成一个 chunk"

        # 统计信息
        chunk_lengths = [len(c.text) for c in chunks]
        print("分块长度统计:")
        print(f"  最大: {max(chunk_lengths)} 字符")
        print(f"  最小: {min(chunk_lengths)} 字符")
        print(f"  平均: {sum(chunk_lengths) / len(chunk_lengths):.1f} 字符")

        # 验证每个 chunk 都有必要的元数据
        for i, chunk in enumerate(chunks):
            assert "chunk_id" in chunk.metadata, f"Chunk {i} 缺少 chunk_id"
            assert "title_path" in chunk.metadata, f"Chunk {i} 缺少 title_path"
            assert "page_idx" in chunk.metadata, f"Chunk {i} 缺少 page_idx"

        # 保存 chunks 到文件
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_dir / "01高中数学必修第一册-40pages-part1-page1-40_chunks.json"
        )

        # 转换为可序列化的字典
        chunks_data = []
        for chunk in chunks:
            chunk_dict = {
                "text": chunk.text,
                "index": chunk.index,
                "page": chunk.page,
                "section": chunk.section,
                "metadata": chunk.metadata,
                "length": len(chunk.text),
            }
            chunks_data.append(chunk_dict)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        print(f"\nChunks 已保存到: {output_path}")

        # 显示前 3 个 chunk 的预览
        print("\n前 3 个 chunk 预览:")
        for i, chunk in enumerate(chunks[:3], 1):
            title_path_str = ' > '.join(chunk.metadata.get('title_path', []))
            print(f"\nChunk {i}:")
            print(f"  长度: {len(chunk.text)} 字符")
            print(f"  页码: {chunk.page}")
            print(f"  标题路径: {title_path_str}")
            print(f"  预览: {chunk.text[:100]}...")


