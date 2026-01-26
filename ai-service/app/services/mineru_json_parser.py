"""
MinerU JSON Parser - 解析MinerU导出的JSON文件

该模块用于解析MinerU PDF转换工具生成的JSON文件，提取结构化数据：
- 文本内容及其类型（text/title/list/table等）
- 图片信息（URL、位置）
- 文档结构（标题层级）
- 页面信息（页码、尺寸）

使用示例:
    parser = MinerUJsonParser()
    doc = parser.parse_file("path/to/mineru_output.json")

    # 获取所有标题
    titles = doc.get_titles()

    # 获取所有图片
    images = doc.get_images()

    # 生成增强Markdown
    md = doc.to_enhanced_markdown()
"""
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class BlockType(str, Enum):
    """块类型枚举"""
    TEXT = "text"
    TITLE = "title"
    IMAGE = "image"
    LIST = "list"
    TABLE = "table"
    FOOTER = "footer"
    IMAGE_BODY = "image_body"


@dataclass
class Span:
    """文本跨度"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    span_type: str
    content: Optional[str] = None
    image_path: Optional[str] = None


@dataclass
class Line:
    """文本行"""
    bbox: Tuple[float, float, float, float]
    spans: List[Span] = field(default_factory=list)

    def get_text(self) -> str:
        """获取行的文本内容"""
        return " ".join(s.content for s in self.spans if s.content)


@dataclass
class Block:
    """段落块"""
    block_type: BlockType
    page_idx: int
    bbox: Tuple[float, float, float, float]
    index: int
    angle: float = 0
    lines: List[Line] = field(default_factory=list)
    children: List['Block'] = field(default_factory=list)  # 用于image类型的嵌套块
    image_path: Optional[str] = None

    def get_text(self) -> str:
        """获取块的文本内容"""
        if self.block_type == BlockType.IMAGE:
            return f"[图片: {self.image_path or '无URL'}]"

        texts = []
        for line in self.lines:
            text = line.get_text()
            if text:
                texts.append(text)
        return "\n".join(texts)

    def get_image_paths(self) -> List[str]:
        """获取块中的图片路径"""
        if self.block_type == BlockType.IMAGE and self.image_path:
            return [self.image_path]

        paths = []
        for child in self.children:
            if child.block_type == BlockType.IMAGE_BODY:
                for line in child.lines:
                    for span in line.spans:
                        if span.image_path:
                            paths.append(span.image_path)
        return paths


@dataclass
class Page:
    """页面"""
    page_idx: int
    page_size: Tuple[int, int]  # [width, height]
    blocks: List[Block] = field(default_factory=list)
    discarded_blocks: List[Dict] = field(default_factory=list)

    def get_blocks_by_type(self, block_type: BlockType) -> List[Block]:
        """获取指定类型的块"""
        return [b for b in self.blocks if b.block_type == block_type]

    def get_text_blocks(self) -> List[Block]:
        """获取所有文本块（包括text、title、list）"""
        return [
            b for b in self.blocks
            if b.block_type in (BlockType.TEXT, BlockType.TITLE, BlockType.LIST)
        ]

    def get_images(self) -> List[Tuple[Block, str]]:
        """获取所有图片（块，URL）"""
        images = []
        for block in self.blocks:
            if block.block_type == BlockType.IMAGE:
                for url in block.get_image_paths():
                    images.append((block, url))
        return images


@dataclass
class MinerUDocument:
    """MinerU解析的完整文档"""
    pdf_info: List[Page]
    backend: str
    version: str
    source_file: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_page(self, page_idx: int) -> Optional[Page]:
        """获取指定页面"""
        for page in self.pdf_info:
            if page.page_idx == page_idx:
                return page
        return None

    def get_total_pages(self) -> int:
        """获取总页数"""
        return len(self.pdf_info)

    def get_all_titles(self) -> List[Dict[str, Any]]:
        """获取所有标题"""
        titles = []
        for page in self.pdf_info:
            for block in page.get_blocks_by_type(BlockType.TITLE):
                titles.append({
                    "page_idx": page.page_idx,
                    "bbox": block.bbox,
                    "text": block.get_text(),
                    "index": block.index
                })
        return titles

    def get_all_images(self) -> List[Dict[str, Any]]:
        """获取所有图片"""
        images = []
        for page in self.pdf_info:
            for block, url in page.get_images():
                images.append({
                    "page_idx": page.page_idx,
                    "bbox": block.bbox,
                    "url": url,
                    "index": block.index
                })
        return images

    def get_text_by_page(self, page_idx: int) -> str:
        """获取指定页的文本内容"""
        page = self.get_page(page_idx)
        if not page:
            return ""

        texts = []
        for block in page.blocks:
            if block.block_type != BlockType.IMAGE:
                text = block.get_text()
                if text:
                    texts.append(text)
        return "\n\n".join(texts)

    def get_full_text(self) -> str:
        """获取完整文本内容"""
        texts = []
        for page in self.pdf_info:
            page_text = self.get_text_by_page(page.page_idx)
            if page_text:
                texts.append(f"=== 第{page.page_idx + 1}页 ===\n{page_text}")
        return "\n\n".join(texts)

    def get_title_hierarchy(self) -> List[Dict[str, Any]]:
        """获取标题层级结构"""
        titles = self.get_all_titles()
        # 按页码和索引排序
        titles.sort(key=lambda x: (x["page_idx"], x["index"]))

        # 简单的层级推断（可根据实际需求改进）
        hierarchy = []
        for title in titles:
            text = title["text"].strip()
            # 根据标题特征推断层级
            level = 1
            if text.startswith("第") and ("章" in text or "节" in text):
                level = 1
            elif text.startswith("#"):
                level = min(text.count("#"), 6)
            elif text[0].isdigit() and "." in text:
                level = text.count(".") + 1

            hierarchy.append({
                "page_idx": title["page_idx"],
                "text": text,
                "level": level,
                "bbox": title["bbox"]
            })
        return hierarchy

    def to_enhanced_markdown(self, include_images: bool = True) -> str:
        """生成增强版Markdown（带类型标记）"""
        lines = []

        for page in self.pdf_info:
            lines.append(f"\n<!-- PAGE={page.page_idx + 1} -->")

            for block in page.blocks:
                if block.block_type == BlockType.TITLE:
                    text = block.get_text().strip()
                    lines.append(f"\n<!-- TYPE=title PAGE={page.page_idx + 1} -->")
                    lines.append(f"## {text}\n")

                elif block.block_type == BlockType.TEXT:
                    text = block.get_text().strip()
                    if text:
                        lines.append(f"<!-- TYPE=text PAGE={page.page_idx + 1} -->")
                        lines.append(f"{text}\n")

                elif block.block_type == BlockType.LIST:
                    text = block.get_text().strip()
                    if text:
                        lines.append(f"<!-- TYPE=list PAGE={page.page_idx + 1} -->")
                        # 将列表项转换为markdown格式
                        for line in text.split("\n"):
                            line = line.strip()
                            if line and not line.startswith("-"):
                                lines.append(f"- {line}")
                            else:
                                lines.append(line)
                        lines.append("")

                elif block.block_type == BlockType.IMAGE and include_images:
                    images = block.get_image_paths()
                    for img_url in images:
                        lines.append(f"<!-- TYPE=image PAGE={page.page_idx + 1} SRC={img_url} -->")
                        lines.append(f"![图片]({img_url})\n")

                elif block.block_type == BlockType.TABLE:
                    text = block.get_text().strip()
                    if text:
                        lines.append(f"<!-- TYPE=table PAGE={page.page_idx + 1} -->")
                        lines.append(f"{text}\n")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "backend": self.backend,
            "version": self.version,
            "source_file": self.source_file,
            "total_pages": self.get_total_pages(),
            "titles": self.get_all_titles(),
            "images": self.get_all_images(),
            "title_hierarchy": self.get_title_hierarchy(),
        }


class MinerUJsonParser:
    """MinerU JSON解析器"""

    def __init__(self):
        self._current_doc: Optional[MinerUDocument] = None

    def parse_file(self, file_path: str) -> MinerUDocument:
        """解析JSON文件

        Args:
            file_path: JSON文件路径

        Returns:
            MinerUDocument: 解析后的文档对象
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._current_doc = self._parse_data(data, source_file=str(path))
        return self._current_doc

    def parse_dict(self, data: Dict[str, Any]) -> MinerUDocument:
        """解析字典数据

        Args:
            data: MinerU JSON数据字典

        Returns:
            MinerUDocument: 解析后的文档对象
        """
        self._current_doc = self._parse_data(data)
        return self._current_doc

    def _parse_data(
        self,
        data: Dict[str, Any],
        source_file: Optional[str] = None
    ) -> MinerUDocument:
        """解析MinerU JSON数据"""
        backend = data.get("_backend", "unknown")
        version = data.get("_version_name", "unknown")

        pdf_info = []
        for page_data in data.get("pdf_info", []):
            page = self._parse_page(page_data)
            pdf_info.append(page)

        return MinerUDocument(
            pdf_info=pdf_info,
            backend=backend,
            version=version,
            source_file=source_file
        )

    def _parse_page(self, page_data: Dict[str, Any]) -> Page:
        """解析单个页面"""
        page_idx = page_data.get("page_idx", 0)
        page_size = tuple(page_data.get("page_size", [0, 0]))

        blocks = []
        for block_data in page_data.get("para_blocks", []):
            block = self._parse_block(block_data, page_idx)
            if block:
                blocks.append(block)

        discarded = page_data.get("discarded_blocks", [])

        return Page(
            page_idx=page_idx,
            page_size=page_size,
            blocks=blocks,
            discarded_blocks=discarded
        )

    def _parse_block(
        self,
        block_data: Dict[str, Any],
        page_idx: int
    ) -> Optional[Block]:
        """解析单个块"""
        block_type_str = block_data.get("type", "text")
        try:
            block_type = BlockType(block_type_str)
        except ValueError:
            block_type = BlockType.TEXT

        bbox = tuple(block_data.get("bbox", [0, 0, 0, 0]))
        index = block_data.get("index", 0)
        angle = block_data.get("angle", 0)

        # 解析lines
        lines = []
        for line_data in block_data.get("lines", []):
            line = self._parse_line(line_data)
            if line:
                lines.append(line)

        # 解析嵌套的blocks（用于image类型）
        children = []
        for child_data in block_data.get("blocks", []):
            child = self._parse_block(child_data, page_idx)
            if child:
                children.append(child)

        # 提取图片路径
        image_path = None
        if block_type == BlockType.IMAGE:
            for child in children:
                for line in child.lines:
                    for span in line.spans:
                        if span.image_path:
                            image_path = span.image_path
                            break
                    if image_path:
                        break
                if image_path:
                    break

        return Block(
            block_type=block_type,
            page_idx=page_idx,
            bbox=bbox,
            index=index,
            angle=angle,
            lines=lines,
            children=children,
            image_path=image_path
        )

    def _parse_line(self, line_data: Dict[str, Any]) -> Optional[Line]:
        """解析文本行"""
        bbox = tuple(line_data.get("bbox", [0, 0, 0, 0]))

        spans = []
        for span_data in line_data.get("spans", []):
            span = self._parse_span(span_data)
            if span:
                spans.append(span)

        if not spans:
            return None

        return Line(bbox=bbox, spans=spans)

    def _parse_span(self, span_data: Dict[str, Any]) -> Optional[Span]:
        """解析文本跨度"""
        bbox = tuple(span_data.get("bbox", [0, 0, 0, 0]))
        span_type = span_data.get("type", "text")
        content = span_data.get("content")
        image_path = span_data.get("image_path")

        return Span(
            bbox=bbox,
            span_type=span_type,
            content=content,
            image_path=image_path
        )


def parse_mineru_json(file_path: str) -> MinerUDocument:
    """便捷函数：解析MinerU JSON文件

    Args:
        file_path: JSON文件路径

    Returns:
        MinerUDocument: 解析后的文档对象
    """
    parser = MinerUJsonParser()
    return parser.parse_file(file_path)
