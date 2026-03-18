"""
MinerU 客户端

用于调用 MinerU API 进行 PDF 文档解析。
基于 mineru-client 项目实现，支持完整的 MinerU API 功能。
"""

import asyncio
import zipfile
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal, Callable
from dataclasses import dataclass
from loguru import logger

import httpx

from src.document_parser.base import DocumentParser, ParsedDocument, TextChunk, ImageInfo
from src.config import settings


# ==================== 数据模型 ====================


@dataclass
class ParseOptions:
    """MinerU PDF 解析选项

    Args:
        backend: 解析后端
            - "pipeline": 传统后端，显存占用低 (~6GB)，兼容性好
            - "vlm-auto-engine": VLM 后端，精度高 (~8GB)
            - "hybrid-auto-engine": 混合后端，默认选项 (~10GB)
        parse_method: 解析方法
            - "auto": 自动判断
            - "txt": 纯文本提取
            - "ocr": OCR 识别
        lang: 语言代码，ch=中文，en=英文
        formula_enable: 是否启用公式解析
        table_enable: 是否启用表格解析
        start_page: 起始页码（从 0 开始）
        end_page: 结束页码（不包含），None 表示到最后一页
    """

    backend: Literal["pipeline", "vlm-auto-engine", "hybrid-auto-engine"] = "pipeline"
    parse_method: Literal["auto", "txt", "ocr"] = "auto"
    lang: str = "ch"
    formula_enable: bool = True
    table_enable: bool = True
    start_page: int = 0
    end_page: Optional[int] = None


@dataclass
class ReturnOptions:
    """MinerU 返回选项

    控制解析结果中包含哪些内容。

    Args:
        return_md: 是否返回 Markdown 内容
        return_middle_json: 是否返回中间 JSON
        return_model_output: 是否返回模型输出
        return_content_list: 是否返回内容列表
        return_images: 是否返回提取的图片
    """

    return_md: bool = True
    return_middle_json: bool = False
    return_model_output: bool = False
    return_content_list: bool = True
    return_images: bool = True


@dataclass
class ParseProgress:
    """解析进度信息

    Args:
        stage: 当前阶段 (uploading, parsing, downloading, extracting, complete)
        percent: 进度百分比 (0-100)
        message: 状态消息
    """

    stage: str = "uploading"
    percent: float = 0.0
    message: str = "正在上传..."

    # 回调函数
    callback: Optional[Callable[[str, float], None]] = None

    def update(self, stage: str, percent: float, message: str) -> None:
        """更新进度并触发回调"""
        self.stage = stage
        self.percent = percent
        self.message = message
        if self.callback:
            self.callback(stage, percent)


# 进度阶段名称映射
STAGE_NAMES = {
    "uploading": "上传中",
    "parsing": "解析中",
    "downloading": "下载中",
    "extracting": "解压中",
    "complete": "完成",
}


# ==================== 自定义异常 ====================


class MinerUError(Exception):
    """MinerU 客户端基础异常类"""

    pass


class FileNotFoundError(MinerUError):
    """文件不存在错误"""

    pass


class FileUploadError(MinerUError):
    """文件上传错误"""

    pass


class ParseError(MinerUError):
    """解析错误"""

    pass


class DownloadError(MinerUError):
    """下载错误"""

    pass


class ExtractionError(MinerUError):
    """解压错误"""

    pass


class TimeoutError(MinerUError):
    """超时错误"""

    pass


# ==================== 主客户端 ====================


class MinerUParser(DocumentParser):
    """MinerU API 客户端解析器

    使用 MinerU API 服务进行 PDF 文档解析，支持异步调用。

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
        self.api_url = (api_url or settings.mineru_api_url).rstrip("/")
        self.endpoint = f"{self.api_url}/file_parse"
        self.output_dir = Path(output_dir or settings.mineru_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 超时配置
        self.connect_timeout = 30
        self.read_timeout = timeout or settings.mineru_timeout

        # httpx timeout 配置
        self._httpx_timeout = httpx.Timeout(
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=self.read_timeout,
            pool=self.read_timeout,
        )

    async def health_check(self) -> bool:
        """
        检查 MinerU API 服务是否可访问

        Returns:
            服务器可访问返回 True，否则返回 False
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self.api_url)
                return response.status_code < 500
        except httpx.RequestError:
            return False

    async def _build_form_data(
        self,
        output_dir: str,
        parse_options: ParseOptions,
        return_options: ReturnOptions,
        use_zip: bool = True,
    ) -> dict:
        """构建表单数据"""
        return {
            "backend": parse_options.backend,
            "output_dir": output_dir,
            "parse_method": parse_options.parse_method,
            "lang_list": parse_options.lang,
            "formula_enable": str(parse_options.formula_enable).lower(),
            "table_enable": str(parse_options.table_enable).lower(),
            "return_md": str(return_options.return_md).lower(),
            "return_middle_json": str(return_options.return_middle_json).lower(),
            "return_model_output": str(return_options.return_model_output).lower(),
            "return_content_list": str(return_options.return_content_list).lower(),
            "return_images": str(return_options.return_images).lower(),
            "response_format_zip": str(use_zip).lower(),
            "start_page_id": str(parse_options.start_page),
            "end_page_id": str(parse_options.end_page) if parse_options.end_page is not None else "99999",
        }

    async def _download_zip(
        self,
        response: httpx.Response,
        output_dir: Path,
        zip_name: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Path:
        """
        异步下载 ZIP 文件（流式下载）

        Args:
            response: httpx 响应对象
            output_dir: 输出目录
            zip_name: ZIP 文件名（不含扩展名）
            progress_callback: 进度回调

        Returns:
            ZIP 文件路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        zip_path = output_dir / f"{zip_name}.zip"

        # 获取文件大小
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        # 流式写入文件
        with open(zip_path, "wb") as f:
            async for chunk in response.aiter_bytes():
                f.write(chunk)
                downloaded += len(chunk)

                # 回调进度
                if progress_callback and total_size > 0:
                    progress_callback("downloading", 50 + (downloaded / total_size) * 30)

        return zip_path

    def _extract_zip(
        self,
        zip_path: Path,
        output_dir: Path,
        pdf_name: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """
        解压 ZIP 文件到输出目录

        Args:
            zip_path: ZIP 文件路径
            output_dir: 输出目录
            pdf_name: PDF 文件名（不含扩展名）
            progress_callback: 可选的进度回调函数

        Returns:
            包含解析结果路径的字典

        Raises:
            ExtractionError: 解压失败时抛出
        """
        result = {
            "pdf_name": pdf_name,
            "output_dir": output_dir,
            "md_path": None,
            "middle_json_path": None,
            "content_list_path": None,
            "images_dir": None,
        }

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = zf.namelist()
                total = len(members)

                # 为 PDF 创建结果目录
                pdf_dir = output_dir / pdf_name
                pdf_dir.mkdir(parents=True, exist_ok=True)

                # 解压文件
                for i, name in enumerate(members):
                    # 跳过目录条目
                    if name.endswith("/"):
                        continue

                    # 解压文件
                    extracted_path = output_dir / name
                    zf.extract(name, output_dir)

                    # 更新进度
                    if progress_callback:
                        progress = ((i + 1) / total) * 100
                        progress_callback(progress)

                    # 更新结果路径
                    self._update_result_path(result, extracted_path, name, output_dir)

            return result

        except zipfile.BadZipFile as e:
            raise ExtractionError(f"ZIP 文件已损坏: {e}") from e
        except Exception as e:
            raise ExtractionError(f"解压失败: {e}") from e

    def _update_result_path(
        self, result: Dict[str, Any], extracted_path: Path, original_name: str, output_dir: Path
    ) -> None:
        """
        根据解压的文件路径更新结果

        Args:
            result: 结果字典
            extracted_path: 解压后的文件路径
            original_name: ZIP 中的原始文件名
            output_dir: 输出目录
        """
        parts = Path(original_name).parts
        if len(parts) < 2:
            return

        filename = parts[-1]

        # 根据文件名更新对应路径
        if filename.endswith(".md") and not filename.startswith("_"):
            result["md_path"] = extracted_path
        elif filename.endswith("_middle.json"):
            result["middle_json_path"] = extracted_path
        elif filename.endswith("_content_list.json"):
            result["content_list_path"] = extracted_path
        elif filename == "images" or (len(parts) > 2 and parts[1] == "images"):
            # 更新 images 目录
            result["images_dir"] = extracted_path.parent if extracted_path.is_file() else extracted_path

    def _parse_content_list(
        self,
        content_list_path: Optional[Path],
        markdown_content: Optional[str] = None,
    ) -> tuple[str, List[TextChunk]]:
        """
        解析 content_list.json 为文本块

        Args:
            content_list_path: content_list.json 文件路径
            markdown_content: Markdown 内容（备用）

        Returns:
            (文档标题, 文本块列表)
        """
        chunks = []
        title = "未命名文档"

        if content_list_path and content_list_path.exists():
            try:
                with open(content_list_path, "r", encoding="utf-8") as f:
                    content_list = json.load(f)

                # 遍历内容列表构建文本块
                for idx, item in enumerate(content_list):
                    content_type = item.get("type", "text")

                    if content_type == "text":
                        text = item.get("text", "").strip()
                        if text:
                            chunk = TextChunk(
                                text=text,
                                index=idx,
                                page=item.get("page_id"),
                                section=item.get("section_title"),
                                metadata={
                                    "type": "text",
                                    "page_id": item.get("page_id"),
                                    "section_title": item.get("section_title"),
                                    "paragraph_id": item.get("para_id"),
                                },
                            )
                            chunks.append(chunk)
                    elif content_type == "title":
                        # 标题信息，用于提取文档标题
                        text = item.get("text", "").strip()
                        level = item.get("level", 0)
                        if text and level == 1:
                            title = text
                        # 也可以将标题作为单独的块
                        if text:
                            chunk = TextChunk(
                                text=text,
                                index=idx,
                                page=item.get("page_id"),
                                section=item.get("section_title", text),
                                metadata={
                                    "type": "title",
                                    "level": level,
                                    "page_id": item.get("page_id"),
                                },
                            )
                            chunks.append(chunk)

                logger.info(f"从 content_list.json 解析了 {len(chunks)} 个文本块")

            except Exception as e:
                logger.warning(f"解析 content_list.json 失败: {e}，使用 Markdown 分块")
                title, chunks = self._parse_markdown_to_chunks(markdown_content or "")

        else:
            # 如果没有 content_list.json，使用 Markdown 分块
            title, chunks = self._parse_markdown_to_chunks(markdown_content or "")

        return title, chunks

    def _parse_markdown_to_chunks(
        self,
        markdown_content: str,
    ) -> tuple[str, List[TextChunk]]:
        """
        将 Markdown 内容按标题分块

        Args:
            markdown_content: Markdown 内容

        Returns:
            (文档标题, 文本块列表)
        """
        if not markdown_content:
            return "未命名文档", []

        chunks = []
        lines = markdown_content.split("\n")
        current_chunk_lines = []
        current_section = None
        chunk_index = 0
        title = "未命名文档"

        for line in lines:
            # 检测标题
            if line.startswith("#"):
                # 保存之前的块
                if current_chunk_lines:
                    text = "\n".join(current_chunk_lines).strip()
                    if text:
                        chunks.append(
                            TextChunk(
                                text=text,
                                index=chunk_index,
                                section=current_section,
                                metadata={"type": "section", "section": current_section},
                            )
                        )
                        chunk_index += 1
                    current_chunk_lines = []

                # 解析标题
                level = len(line) - len(line.lstrip("#"))
                title_text = line.lstrip("#").strip()

                # 一级标题作为文档标题
                if level == 1 and not title_text.startswith("_"):
                    title = title_text

                current_section = title_text

            current_chunk_lines.append(line)

        # 保存最后一个块
        if current_chunk_lines:
            text = "\n".join(current_chunk_lines).strip()
            if text:
                chunks.append(
                    TextChunk(
                        text=text,
                        index=chunk_index,
                        section=current_section,
                        metadata={"type": "section", "section": current_section},
                    )
                )

        logger.info(f"从 Markdown 解析了 {len(chunks)} 个文本块")

        return title, chunks

    def _extract_images(
        self,
        images_dir: Optional[Path],
        output_dir: Path,
    ) -> List[ImageInfo]:
        """
        从图片目录提取图片信息

        Args:
            images_dir: 图片目录路径
            output_dir: 输出目录（用于相对路径）

        Returns:
            图片信息列表
        """
        images = []

        if not images_dir or not images_dir.exists():
            return images

        # 查找所有图片文件
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
        for image_file in images_dir.iterdir():
            if image_file.suffix.lower() in image_extensions and image_file.is_file():
                # 尝试从文件名提取页码
                page = None
                name_parts = image_file.stem.split("_")
                if len(name_parts) > 0 and name_parts[-1].isdigit():
                    try:
                        page = int(name_parts[-1])
                    except ValueError:
                        pass

                images.append(
                    ImageInfo(
                        path=str(image_file),
                        page=page,
                        metadata={
                            "relative_path": str(image_file.relative_to(output_dir)),
                            "file_size": image_file.stat().st_size,
                        },
                    )
                )

        logger.info(f"提取了 {len(images)} 个图片")

        return images

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
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if parse_options is None:
            parse_options = ParseOptions()
        if return_options is None:
            return_options = ReturnOptions()

        logger.info(f"开始解析文档: {file_path}")

        # 上传文件
        if progress_callback:
            progress_callback("uploading", 10)

        data = await self._build_form_data(
            str(self.output_dir),
            parse_options,
            return_options,
            use_zip=True,
        )

        try:
            async with httpx.AsyncClient(timeout=self._httpx_timeout) as client:
                with open(file_path, "rb") as f:
                    files = {"files": (file_path_obj.name, f, "application/pdf")}
                    response = await client.post(
                        self.endpoint,
                        files=files,
                        data=data,
                    )

                    if response.status_code >= 400:
                        error_text = response.text
                        raise ParseError(f"MinerU API 返回错误: {response.status_code} - {error_text}")

                # 检查响应类型
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    json_data = response.json()
                    if "error" in json_data:
                        raise ParseError(f"服务器返回错误: {json_data['error']}")

                # 下载 ZIP
                if progress_callback:
                    progress_callback("downloading", 50)

                pdf_name = file_path_obj.stem
                zip_path = await self._download_zip(
                    response, self.output_dir, pdf_name, progress_callback
                )

                # 解压
                if progress_callback:
                    progress_callback("extracting", 85)

                extracted_result = self._extract_zip(
                    zip_path,
                    self.output_dir,
                    pdf_name,
                    lambda p: progress_callback("extracting", 85 + p * 0.15) if progress_callback else None,
                )

                if progress_callback:
                    progress_callback("complete", 100)

        except httpx.TimeoutException as e:
            raise TimeoutError(f"请求超时: {e}") from e
        except httpx.RequestError as e:
            raise FileUploadError(f"请求失败: {e}") from e

        # 解析结果
        md_path = Path(extracted_result.get("md_path"))
        content_list_path = Path(extracted_result.get("content_list_path"))
        images_dir = Path(extracted_result.get("images_dir"))

        # 读取 Markdown 内容
        markdown_content = None
        if md_path and md_path.exists():
            markdown_content = md_path.read_text(encoding="utf-8")

        # 解析内容列表
        title, chunks = self._parse_content_list(content_list_path, markdown_content)

        # 如果没有从 content_list 提取到标题，从 Markdown 第一行提取
        if title == "未命名文档" and markdown_content:
            for line in markdown_content.split("\n"):
                if line.strip() and not line.startswith("#"):
                    title = line.strip().split("\n")[0][:100]
                    break

        # 提取图片
        images = self._extract_images(images_dir, self.output_dir)

        # 构建文档
        document = ParsedDocument(
            title=title,
            content=markdown_content or "",
            chunks=chunks,
            images=images,
            metadata={
                "source": file_path,
                "parser": "mineru",
                "pdf_name": pdf_name,
                "total_chunks": len(chunks),
                "total_images": len(images),
                "output_dir": str(extracted_result["output_dir"]),
                "parse_options": {
                    "backend": parse_options.backend,
                    "parse_method": parse_options.parse_method,
                    "lang": parse_options.lang,
                },
            },
        )

        logger.info(
            f"文档解析完成: 标题={title}, 文本块={len(chunks)}, 图片={len(images)}"
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
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if parse_options is None:
            parse_options = ParseOptions()
        if return_options is None:
            # JSON 模式下默认返回更多内容
            return_options = ReturnOptions(return_md=True, return_content_list=True)

        data = await self._build_form_data(
            str(self.output_dir),
            parse_options,
            return_options,
            use_zip=False,
        )

        try:
            async with httpx.AsyncClient(timeout=self._httpx_timeout) as client:
                with open(file_path, "rb") as f:
                    files = {"files": (file_path_obj.name, f, "application/pdf")}
                    response = await client.post(
                        self.endpoint,
                        files=files,
                        data=data,
                    )
                    response.raise_for_status()
                    return response.json()

        except httpx.TimeoutException as e:
            raise TimeoutError(f"请求超时: {e}") from e
        except httpx.RequestError as e:
            raise FileUploadError(f"请求失败: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ParseError(f"API 请求失败: {e}") from e

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
        tasks = [
            self.parse(fp, parse_options, return_options, progress_callback)
            for fp in file_paths
        ]
        documents = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        results = []
        for i, doc in enumerate(documents):
            if isinstance(doc, Exception):
                logger.error(f"解析文档 {file_paths[i]} 失败: {doc}")
            else:
                results.append(doc)

        return results

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        # 如果需要清理资源，可以在这里添加
        pass
