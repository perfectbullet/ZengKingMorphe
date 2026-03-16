"""
文档解析器基类和数据模型
"""

from abc import ABC, abstractmethod
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
