"""
MinerU MCP 客户端

用于调用 MinerU MCP 服务解析文档
"""

import asyncio
import aiohttp
from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger

from src.document_parser.base import DocumentParser, ParsedDocument, TextChunk, ImageInfo
from src.config import settings


class MinerUParser(DocumentParser):
    """MinerU MCP 客户端解析器"""

    def __init__(
        self,
        mcp_url: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        初始化 MinerU 解析器

        Args:
            mcp_url: MinerU MCP 服务地址
            output_dir: 输出目录
        """
        self.mcp_url = mcp_url or settings.mineru_mcp_url
        self.output_dir = Path(output_dir or settings.mineru_output_dir)
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP 会话"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=300)  # 5分钟超时
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def close(self) -> None:
        """关闭 HTTP 会话"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def _call_mineru_api(
        self,
        file_path: str
    ) -> Dict[str, Any]:
        """
        调用 MinerU MCP API

        Args:
            file_path: 输入文件路径

        Returns:
            API 响应数据
        """
        session = await self._get_session()

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 准备请求数据
        payload = {
            "file_path": file_path,
            "output_dir": str(self.output_dir),
        }

        try:
            async with session.post(
                f"{self.mcp_url}/parse",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"MinerU API 调用失败: {response.status} - {error_text}"
                    )

                return await response.json()

        except aiohttp.ClientError as e:
            logger.error(f"MinerU API 请求错误: {e}")
            raise
        except Exception as e:
            logger.error(f"MinerU 解析错误: {e}")
            raise

    def _parse_markdown(
        self,
        markdown_content: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> List[TextChunk]:
        """
        解析 Markdown 内容为文本块

        Args:
            markdown_content: Markdown 内容
            chunk_size: 分块大小
            chunk_overlap: 重叠大小

        Returns:
            文本块列表
        """
        chunks = []
        lines = markdown_content.split('\n')
        current_chunk = ""
        chunk_index = 0

        for line in lines:
            # 检查是否是标题
            is_heading = line.startswith('#')

            # 如果当前块已达到大小限制，保存并创建新块
            if len(current_chunk) + len(line) > chunk_size and current_chunk:
                chunks.append(TextChunk(
                    text=current_chunk.strip(),
                    index=chunk_index
                ))
                chunk_index += 1
                # 保留重叠部分
                if chunk_overlap > 0 and chunks:
                    last_chunk_text = chunks[-1].text
                    current_chunk = last_chunk_text[-min(chunk_overlap, len(last_chunk_text)):]
                else:
                    current_chunk = ""
                # 新块从新行开始
                if is_heading:
                    current_chunk = line + "\n"
                else:
                    current_chunk += line + "\n"
            else:
                current_chunk += line + "\n"

        # 添加最后一个块
        if current_chunk.strip():
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                index=chunk_index
            ))

        return chunks

    def _extract_images(
        self,
        output_dir: Path
    ) -> List[ImageInfo]:
        """
        从输出目录提取图片信息

        Args:
            output_dir: MinerU 输出目录

        Returns:
            图片信息列表
        """
        images = []

        # 查找所有图片文件
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
        for image_file in output_dir.rglob('*'):
            if image_file.suffix.lower() in image_extensions:
                images.append(ImageInfo(
                    path=str(image_file),
                    page=None  # MinerU 可能不直接提供页码信息
                ))

        return images

    async def parse(self, file_path: str) -> ParsedDocument:
        """
        解析文档

        Args:
            file_path: 文档文件路径

        Returns:
            解析后的文档对象
        """
        logger.info(f"开始解析文档: {file_path}")

        # 调用 MinerU API
        result = await self._call_mineru_api(file_path)

        # 获取输出目录
        doc_output_dir = self.output_dir / Path(file_path).stem

        # 读取 Markdown 内容
        markdown_file = doc_output_dir / "output.md"
        if not markdown_file.exists():
            markdown_file = doc_output_dir / f"{Path(file_path).stem}.md"

        if markdown_file.exists():
            with open(markdown_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
        else:
            # 如果没有找到 Markdown 文件，尝试从 API 响应获取
            markdown_content = result.get('content', '')

        # 提取文档标题
        title = result.get('title', Path(file_path).stem)

        # 分块文本
        chunks = self._parse_markdown(
            markdown_content,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        # 提取图片
        images = self._extract_images(doc_output_dir)

        # 构建解析结果
        document = ParsedDocument(
            title=title,
            content=markdown_content,
            chunks=chunks,
            images=images,
            metadata={
                "source": file_path,
                "parser": "mineru",
                "total_chunks": len(chunks),
                "total_images": len(images)
            }
        )

        logger.info(
            f"文档解析完成: 标题={title}, 文本块={len(chunks)}, 图片={len(images)}"
        )

        return document

    async def parse_batch(self, file_paths: List[str]) -> List[ParsedDocument]:
        """
        批量解析文档

        Args:
            file_paths: 文档文件路径列表

        Returns:
            解析后的文档对象列表
        """
        tasks = [self.parse(path) for path in file_paths]
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
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
