"""
MinerU 客户端

基于 mineru-client SDK 实现 PDF 文档解析。
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Callable

from loguru import logger

from src.document_parser.base import DocumentParser, ParsedDocument, TextChunk, ImageInfo
from src.config import settings

# SDK 导入（直接使用 SDK 类，保持向后兼容）
from mineru_client import (
    AsyncMinerUClient,
    ParseOptions as SDKParseOptions,
    ReturnOptions as SDKReturnOptions,
    ZipResult,
)

# SDK 的 STAGE_NAMES 也重新导出
STAGE_NAMES = {
    "uploading": "上传中",
    "parsing": "解析中",
    "downloading": "下载中",
    "extracting": "解压中",
    "complete": "完成",
}

# 从 SDK 导入异常类并重新导出，保持向后兼容
from mineru_client.exceptions import (
    MinerUClientError,
    ConfigurationError,
    FileUploadError as SDKFileUploadError,
    ParseError as SDKParseError,
    DownloadError as SDKDownloadError,
    ExtractionError as SDKExtractionError,
    TimeoutError as SDKTimeoutError,
    FileNotFoundError as SDKFileNotFoundError,
)

# 重新导出 SDK 类（保持向后兼容）
ParseOptions = SDKParseOptions
ReturnOptions = SDKReturnOptions

# 重新导出异常类
MinerUError = MinerUClientError
FileUploadError = SDKFileUploadError
ParseError = SDKParseError
DownloadError = SDKDownloadError
ExtractionError = SDKExtractionError
TimeoutError = SDKTimeoutError
FileNotFoundError = SDKFileNotFoundError


# ==================== 主客户端 ====================


class MinerUParser(DocumentParser):
    """MinerU API 客户端解析器（基于 mineru-client SDK）

    使用 mineru-client SDK 进行 PDF 文档解析，支持异步调用。

    示例:
        >>> async with MinerUParser() as parser:
        ...     doc = await parser.parse("document.pdf")
        ...     print(doc.title)
        ...     print(len(doc.chunks))
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        output_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        """
        初始化 MinerU 解析器

        Args:
            api_url: MinerU API 服务器地址（默认从配置读取）
            output_dir: 输出目录（默认从配置读取）
            timeout: 请求超时时间（秒）
        """
        # 配置
        self.api_url = (api_url or settings.mineru_api_url).rstrip("/")
        self.output_dir = Path(output_dir or settings.mineru_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout or settings.mineru_timeout

        # 创建 SDK 客户端
        self._sdk_client = AsyncMinerUClient(
            api_url=self.api_url,
            timeout=self.timeout,
        )

        # 兼容性属性（保持向后兼容）
        self.endpoint = f"{self.api_url}/file_parse"

    async def health_check(self) -> bool:
        """
        检查 MinerU API 服务是否可访问

        Returns:
            服务器可访问返回 True，否则返回 False
        """
        return await self._sdk_client.health_check()

    def _extract_title(self, zip_result: ZipResult) -> str:
        """从 ZipResult 提取文档标题

        优先级：
        1. content_list 中的第一个标题 (text_level=1)
        2. PDF 文件名
        """
        if zip_result.content_list:
            for item in zip_result.content_list:
                if item.get("type") == "title":
                    level = item.get("text_level", item.get("level", 0))
                    if level == 1:
                        return item.get("text", "").strip()

        return zip_result.pdf_name.replace("_", " ").strip()

    def _convert_images_to_image_info(
        self,
        images: List[Path],
        output_dir: Path,
    ) -> List[ImageInfo]:
        """将 SDK 返回的图片路径列表转换为 ImageInfo"""
        result: List[ImageInfo] = []

        for img_path in images:
            # 尝试从文件名提取页码: {prefix}_{page}.{ext}
            page = None
            stem = img_path.stem
            if "_" in stem:
                parts = stem.split("_")
                if parts[-1].isdigit():
                    page = int(parts[-1])

            result.append(ImageInfo(
                path=str(img_path),
                page=page,
                metadata={
                    "relative_path": str(img_path.relative_to(output_dir)),
                    "file_name": img_path.name,
                },
            ))

        return result

    def _convert_to_parsed_document(
        self,
        zip_result: ZipResult,
        file_path: str,
        parse_options: Optional[ParseOptions] = None,
    ) -> ParsedDocument:
        """将 SDK ZipResult 转换为 ParsedDocument（使用结构感知分块）"""
        # 提取标题
        title = self._extract_title(zip_result)

        # 提取 Markdown 内容
        content = zip_result.md_content or ""

        # 转换 content_list 为 TextChunk（使用结构感知分块）
        chunks = []
        if zip_result.content_list:
            from src.document_parser.mineru_structure_aware_chunker import MinerUStructureAwareChunker
            chunker = MinerUStructureAwareChunker()
            chunks = chunker.chunk_content_list(
                content_list=zip_result.content_list,
                pdf_name=zip_result.pdf_name
            )

        # 转换图片为 ImageInfo
        images = self._convert_images_to_image_info(
            zip_result.images,
            zip_result.output_dir,
        )

        # 构建 ParsedDocument
        metadata = {
            "source": file_path,
            "parser": "mineru",
            "pdf_name": zip_result.pdf_name,
            "output_dir": str(zip_result.output_dir),
        }

        if parse_options:
            metadata["parse_options"] = {
                "backend": parse_options.backend,
                "parse_method": parse_options.parse_method,
                "lang": parse_options.lang,
            }

        return ParsedDocument(
            title=title,
            content=content,
            chunks=chunks,
            images=images,
            metadata=metadata,
        )

    def _try_load_cached_result(self, file_path: str) -> Optional["ZipResult"]:
        """尝试加载已有的解析结果（缓存）

        根据 file_path 生成 pdf_name，检查 output_dir/pdf_name/ 目录下
        是否已有解析结果文件（.md, _content_list.json, images/）。

        Args:
            file_path: PDF 文件路径

        Returns:
            如果缓存存在且完整，返回 ZipResult；否则返回 None
        """
        # 从 file_path 提取文件名（不含扩展名），作为 pdf_name
        pdf_name = Path(file_path).stem

        # 构造预期的输出目录路径
        result_dir = self.output_dir / pdf_name

        # 检查关键文件是否存在
        md_path = result_dir / f"{pdf_name}.md"
        content_list_path = result_dir / f"{pdf_name}_content_list.json"
        images_dir = result_dir / "images"

        if not (md_path.exists() and content_list_path.exists() and images_dir.exists()):
            logger.debug(f"缓存不完整或不存在: {result_dir}")
            return None

        try:
            # 构造 ZipResult 对象（@property 会自动读取文件内容）
            zip_result = ZipResult(
                pdf_name=pdf_name,
                output_dir=result_dir,
                md_path=md_path,
                middle_json_path=None,
                content_list_path=content_list_path,
                images_dir=images_dir,
            )
            # 验证文件可读（触发 @property 访问）
            _ = zip_result.md_content
            _ = zip_result.content_list
            _ = zip_result.images

            logger.info(f"使用缓存解析结果: {result_dir}")
            return zip_result

        except Exception as e:
            logger.warning(f"读取缓存失败，将重新解析: {e}")
            return None

    async def parse(
        self,
        file_path: str,
        parse_options: Optional[ParseOptions] = None,
        return_options: Optional[ReturnOptions] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ParsedDocument:
        """
        解析 PDF 文档

        Args:
            file_path: PDF 文件路径
            parse_options: 解析选项
            return_options: 返回内容选项
            progress_callback: 进度回调函数

        Returns:
            解析后的文档对象

        Raises:
            FileNotFoundError: 文件不存在
            FileUploadError: 文件上传失败
            ParseError: 服务器解析失败
            DownloadError: ZIP 下载失败
            ExtractionError: ZIP 解压失败
            TimeoutError: 请求超时
        """
        logger.info(f"开始解析文档: {file_path}")

        # 先尝试加载缓存
        zip_result = self._try_load_cached_result(file_path)

        if zip_result is None:
            # 缓存不存在或不完整，调用 SDK 解析
            try:
                zip_result = await self._sdk_client.parse_pdf(
                    file_path=file_path,
                    output_dir=str(self.output_dir),
                    parse_options=parse_options,
                    return_options=return_options,
                    progress_callback=progress_callback,
                )
            except SDKFileNotFoundError as e:
                raise FileNotFoundError(f"文件不存在: {file_path}") from e
            except SDKFileUploadError as e:
                raise FileUploadError(f"文件上传失败: {e}") from e
            except SDKParseError as e:
                raise ParseError(f"服务器解析失败: {e}") from e
            except SDKDownloadError as e:
                raise DownloadError(f"ZIP 下载失败: {e}") from e
            except SDKExtractionError as e:
                raise ExtractionError(f"ZIP 解压失败: {e}") from e
            except SDKTimeoutError as e:
                raise TimeoutError(f"请求超时: {e}") from e

        # 转换为 ParsedDocument
        document = self._convert_to_parsed_document(
            zip_result, file_path, parse_options
        )

        logger.info(
            f"文档解析完成: 标题={document.title}, 文本块={len(document.chunks)}, 图片={len(document.images)}"
        )

        return document

    async def parse_to_memory(
        self,
        file_path: str,
        parse_options: Optional[ParseOptions] = None,
        return_options: Optional[ReturnOptions] = None,
    ) -> Dict[str, Any]:
        """
        解析 PDF 并直接返回 JSON 结果（不下载 ZIP）

        适用于需要直接获取结构化数据而不需要文件的场景。

        Args:
            file_path: PDF 文件路径
            parse_options: 解析选项
            return_options: 返回内容选项

        Returns:
            包含解析结果的字典

        Raises:
            FileNotFoundError: 文件不存在
            FileUploadError: 文件上传失败
            ParseError: 服务器解析失败
        """
        if return_options is None:
            # JSON 模式下默认返回更多内容
            return_options = ReturnOptions(
                return_md=True,
                return_content_list=True,
            )

        return await self._sdk_client.parse_pdf_to_memory(
            file_path=file_path,
            parse_options=parse_options,
            return_options=return_options,
        )

    async def parse_batch(
        self,
        file_paths: List[str],
        parse_options: Optional[ParseOptions] = None,
        return_options: Optional[ReturnOptions] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> List[ParsedDocument]:
        """
        批量解析 PDF 文档（并发）

        Args:
            file_paths: PDF 文件路径列表
            parse_options: 解析选项
            return_options: 返回内容选项
            progress_callback: 进度回调函数

        Returns:
            解析后的文档对象列表
        """
        # SDK 并发处理
        results = await self._sdk_client.parse_multiple_pdfs(
            file_paths=file_paths,
            output_dir=str(self.output_dir),
            parse_options=parse_options,
            return_options=return_options,
            progress_callback=progress_callback,
        )

        # 转换为 ParsedDocument 列表
        documents: List[ParsedDocument] = []
        for fp, zip_result in results.items():
            try:
                doc = self._convert_to_parsed_document(zip_result, fp, parse_options)
                documents.append(doc)
            except Exception as e:
                logger.error(f"转换解析结果失败 {fp}: {e}")

        return documents

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        # SDK 客户端无需特殊清理
        pass
