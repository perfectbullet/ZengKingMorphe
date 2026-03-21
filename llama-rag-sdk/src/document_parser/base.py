"""
文档解析器基类和数据模型
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ImageInfo(BaseModel):
    """图片信息"""

    path: str = Field(..., description="图片文件路径")
    page: Optional[int] = Field(None, description="所在页码")
    description: Optional[str] = Field(None, description="图片描述")
    position: Optional[str] = Field(None, description="在文档中的位置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class TextChunk(BaseModel):
    """文本块"""

    text: str = Field(..., description="文本内容")
    page: Optional[int] = Field(None, description="所在页码")
    section: Optional[str] = Field(None, description="所在章节")
    index: int = Field(0, description="块索引")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class ParsedDocument(BaseModel):
    """解析后的文档"""

    title: str = Field(..., description="文档标题")
    content: str = Field(..., description="完整 Markdown 内容")
    chunks: List[TextChunk] = Field(default_factory=list, description="分块后的文本")
    images: List[ImageInfo] = Field(default_factory=list, description="图片信息列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")

    def add_chunk(self, chunk: TextChunk) -> None:
        """添加文本块"""
        self.chunks.append(chunk)

    def add_image(self, image: ImageInfo) -> None:
        """添加图片信息"""
        self.images.append(image)

    def save_to_json(
        self,
        pdf_filename: str,
        output_dir: Optional[str] = None,
        ensure_ascii: bool = False,
        indent: int = 2
    ) -> str:
        """
        将文档保存为 JSON 文件

        Args:
            pdf_filename: PDF 文件名（用于生成 JSON 文件名）
            output_dir: 输出目录（默认为 None，保存在当前目录）
            ensure_ascii: 是否确保 ASCII 编码（默认 False，支持中文）
            indent: JSON 缩进空格数（默认 2）

        Returns:
            保存的 JSON 文件路径
        """
        # 生成 JSON 文件名：使用 PDF 文件名，替换扩展名为 .json
        json_filename = Path(pdf_filename).stem + "-chunk.json"

        # 确定输出路径
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            json_path = output_path / json_filename
        else:
            json_path = Path(json_filename)

        # 转换为字典（Pydantic 模型自动支持）
        document_dict = self.model_dump(mode='json')

        # 写入 JSON 文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(document_dict, f, ensure_ascii=ensure_ascii, indent=indent)

        return str(json_path)


class DocumentParser(ABC):
    """文档解析器基类"""

    @abstractmethod
    async def parse(self, file_path: str) -> ParsedDocument:
        """
        解析文档

        Args:
            file_path: 文档文件路径

        Returns:
            解析后的文档对象
        """
        pass

    @abstractmethod
    async def parse_batch(self, file_paths: List[str]) -> List[ParsedDocument]:
        """
        批量解析文档

        Args:
            file_paths: 文档文件路径列表

        Returns:
            解析后的文档对象列表
        """
        pass
