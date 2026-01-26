"""
MinerU-Aware Chunking - 基于MinerU结构信息的智能分块

该模块利用MinerU JSON提供的结构信息进行智能分块：
1. 按标题（title）边界进行文档分段
2. 保持列表（list）完整性
3. 关联图片与其上下文
4. 保留页面信息用于溯源

与传统分块相比的优势：
- 更好的语义完整性（按章节分块）
- 更丰富的元数据（页码、块类型、图片）
- 支持多模态检索（图片描述 + 文本）
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from app.services.mineru_json_parser import (
    MinerUDocument,
    MinerUJsonParser,
    Block,
    BlockType,
    Page
)
from app.core.logging import logger


@dataclass
class MultiModalChunk:
    """多模态分块"""
    chunk_id: str
    doc_id: str
    kb_id: str
    content: str                          # 文本内容
    chunk_index: int

    # MinerU结构信息
    page_idx: int                         # 起始页码
    page_indices: List[int] = field(default_factory=list)  # 包含的所有页码
    block_types: List[str] = field(default_factory=list)   # 包含的块类型

    # 图片信息
    image_references: List[str] = field(default_factory=list)  # 图片URL列表
    image_captions: List[str] = field(default_factory=list)    # 图片描述列表

    # 结构信息
    title_path: List[str] = field(default_factory=list)  # 标题路径（面包屑）
    structure_level: int = 0            # 在文档结构中的层级

    # 元数据
    bbox: Optional[Tuple[float, float, float, float]] = None  # 边界框
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "kb_id": self.kb_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "page_idx": self.page_idx,
            "page_indices": self.page_indices,
            "block_types": self.block_types,
            "image_references": self.image_references,
            "image_captions": self.image_captions,
            "title_path": self.title_path,
            "structure_level": self.structure_level,
            "bbox": self.bbox,
            "metadata": self.metadata,
        }

    def get_enhanced_content(self) -> str:
        """获取增强内容（包含图片描述）"""
        enhanced = self.content

        if self.image_captions:
            enhanced += "\n\n[图片描述: "
            enhanced += " | ".join(self.image_captions)
            enhanced += "]"

        return enhanced

    def get_embedding_content(self) -> str:
        """获取用于向量化的内容（标题前置）

        将标题路径拼接到内容开头，增强向量的语义表达。
        检索返回时直接使用 content 字段，无需去除标题标记。
        """
        if self.title_path:
            title_str = " > ".join(self.title_path)
            return f"{title_str}\n\n{self.content}"
        return self.content


class ChunkingStrategy:
    """分块策略枚举"""
    BY_TITLE = "by_title"           # 按标题分块
    BY_PAGE = "by_page"             # 按页分块
    HYBRID = "hybrid"               # 混合策略（标题优先，超长再分）
    SEMANTIC_STRUCTURE = "semantic_structure"  # 语义+结构混合


class MinerUAwareChunker:
    """基于MinerU结构信息的智能分块器"""

    def __init__(
        self,
        max_chunk_size: int = 1000,      # 最大分块大小（字符数）
        min_chunk_size: int = 100,       # 最小分块大小
        chunk_overlap: int = 100,        # 分块重叠大小
        strategy: str = ChunkingStrategy.HYBRID
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    async def chunk_document(
        self,
        doc: MinerUDocument,
        doc_id: str,
        kb_id: str,
        image_captions: Optional[Dict[str, str]] = None
    ) -> List[MultiModalChunk]:
        """对文档进行分块

        Args:
            doc: MinerU解析的文档对象
            doc_id: 文档ID
            kb_id: 知识库ID
            image_captions: 图片URL到描述的映射（可选）

        Returns:
            多模态分块列表
        """
        if self.strategy == ChunkingStrategy.BY_TITLE:
            return await self._chunk_by_title(doc, doc_id, kb_id, image_captions)
        elif self.strategy == ChunkingStrategy.BY_PAGE:
            return await self._chunk_by_page(doc, doc_id, kb_id, image_captions)
        else:
            return await self._chunk_hybrid(doc, doc_id, kb_id, image_captions)

    async def _chunk_by_title(
        self,
        doc: MinerUDocument,
        doc_id: str,
        kb_id: str,
        image_captions: Optional[Dict[str, str]] = None
    ) -> List[MultiModalChunk]:
        """按标题边界分块

        将文档按标题分割，每个标题及其内容作为一个分块。
        """
        chunks = []
        current_chunk_text = []
        current_chunk_images = []
        current_chunk_page_idx = 0
        current_page_indices = set()
        current_title_path = []
        chunk_index = 0

        for page in doc.pdf_info:
            current_page_indices.add(page.page_idx)

            for block in page.blocks:
                # 处理标题
                if block.block_type == BlockType.TITLE:
                    # 如果当前chunk有内容，先保存
                    if current_chunk_text and self._get_text_length(current_chunk_text) >= self.min_chunk_size:
                        chunks.append(self._create_chunk(
                            doc_id=doc_id,
                            kb_id=kb_id,
                            content="\n".join(current_chunk_text),
                            chunk_index=chunk_index,
                            page_idx=current_chunk_page_idx,
                            page_indices=list(current_page_indices),
                            image_references=current_chunk_images,
                            image_captions=image_captions,
                            title_path=current_title_path.copy(),
                            block_types=["title", "text"]
                        ))
                        chunk_index += 1
                        current_chunk_text = []
                        current_chunk_images = []
                        current_page_indices = set()

                    # 更新标题路径
                    title_text = block.get_text().strip()
                    current_title_path = self._update_title_path(current_title_path, title_text)

                    # 添加标题到新chunk
                    current_chunk_text.append(f"## {title_text}")
                    current_chunk_page_idx = page.page_idx

                # 处理文本
                elif block.block_type == BlockType.TEXT:
                    text = block.get_text().strip()
                    if text:
                        current_chunk_text.append(text)

                # 处理列表
                elif block.block_type == BlockType.LIST:
                    text = block.get_text().strip()
                    if text:
                        # 转换为markdown列表格式
                        list_items = []
                        for line in text.split("\n"):
                            line = line.strip()
                            if line and not line.startswith("-"):
                                list_items.append(f"- {line}")
                            else:
                                list_items.append(line)
                        current_chunk_text.extend(list_items)

                # 处理图片
                elif block.block_type == BlockType.IMAGE:
                    for img_url in block.get_image_paths():
                        current_chunk_images.append(img_url)
                        # 如果有图片描述，添加到文本中
                        if image_captions and img_url in image_captions:
                            current_chunk_text.append(f"[图片: {image_captions[img_url]}]")

                # 检查是否需要分块（避免chunk过大）
                if self._get_text_length(current_chunk_text) >= self.max_chunk_size:
                    chunks.append(self._create_chunk(
                        doc_id=doc_id,
                        kb_id=kb_id,
                        content="\n".join(current_chunk_text),
                        chunk_index=chunk_index,
                        page_idx=current_chunk_page_idx,
                        page_indices=list(current_page_indices),
                        image_references=current_chunk_images,
                        image_captions=image_captions,
                        title_path=current_title_path.copy(),
                        block_types=self._infer_block_types(current_chunk_text)
                    ))
                    chunk_index += 1
                    current_chunk_text = []
                    current_chunk_images = []
                    current_page_indices = set()

        # 保存最后一个chunk
        if current_chunk_text:
            chunks.append(self._create_chunk(
                doc_id=doc_id,
                kb_id=kb_id,
                content="\n".join(current_chunk_text),
                chunk_index=chunk_index,
                page_idx=current_chunk_page_idx,
                page_indices=list(current_page_indices),
                image_references=current_chunk_images,
                image_captions=image_captions,
                title_path=current_title_path.copy(),
                block_types=self._infer_block_types(current_chunk_text)
            ))

        return chunks

    async def _chunk_by_page(
        self,
        doc: MinerUDocument,
        doc_id: str,
        kb_id: str,
        image_captions: Optional[Dict[str, str]] = None
    ) -> List[MultiModalChunk]:
        """按页分块

        每页作为一个或多个分块（取决于内容长度）。
        """
        chunks = []
        chunk_index = 0

        for page in doc.pdf_info:
            page_text = []
            page_images = []
            title_path = []

            for block in page.blocks:
                if block.block_type == BlockType.TITLE:
                    title_text = block.get_text().strip()
                    title_path = self._update_title_path(title_path, title_text)
                    page_text.append(f"## {title_text}")

                elif block.block_type == BlockType.TEXT:
                    text = block.get_text().strip()
                    if text:
                        page_text.append(text)

                elif block.block_type == BlockType.LIST:
                    text = block.get_text().strip()
                    if text:
                        page_text.append(text)

                elif block.block_type == BlockType.IMAGE:
                    for img_url in block.get_image_paths():
                        page_images.append(img_url)

            # 如果页面内容过长，进行分割
            full_text = "\n".join(page_text)
            if len(full_text) <= self.max_chunk_size:
                chunks.append(self._create_chunk(
                    doc_id=doc_id,
                    kb_id=kb_id,
                    content=full_text,
                    chunk_index=chunk_index,
                    page_idx=page.page_idx,
                    page_indices=[page.page_idx],
                    image_references=page_images,
                    image_captions=image_captions,
                    title_path=title_path,
                    block_types=["text"]
                ))
                chunk_index += 1
            else:
                # 长页面分割
                sub_chunks = self._split_long_text(
                    full_text,
                    doc_id,
                    kb_id,
                    chunk_index,
                    page.page_idx,
                    [page.page_idx],
                    page_images,
                    image_captions,
                    title_path
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)

        return chunks

    async def _chunk_hybrid(
        self,
        doc: MinerUDocument,
        doc_id: str,
        kb_id: str,
        image_captions: Optional[Dict[str, str]] = None
    ) -> List[MultiModalChunk]:
        """混合分块策略

        优先按标题分块，但如果某个章节过长，则进一步分割。
        """
        chunks = []
        chunk_index = 0

        # 先按标题分块
        title_chunks = await self._chunk_by_title(doc, doc_id, kb_id, image_captions)

        # 检查每个chunk的长度，过长的进行分割
        for chunk in title_chunks:
            if len(chunk.content) <= self.max_chunk_size:
                chunks.append(chunk)
            else:
                # 分割长chunk
                sub_chunks = self._split_long_text(
                    chunk.content,
                    doc_id,
                    kb_id,
                    chunk_index,
                    chunk.page_idx,
                    chunk.page_indices,
                    chunk.image_references,
                    image_captions,
                    chunk.title_path
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)

        # 重新编号
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        return chunks

    def _create_chunk(
        self,
        doc_id: str,
        kb_id: str,
        content: str,
        chunk_index: int,
        page_idx: int,
        page_indices: List[int],
        image_references: List[str],
        image_captions: Optional[Dict[str, str]],
        title_path: List[str],
        block_types: List[str]
    ) -> MultiModalChunk:
        """创建一个分块"""
        import uuid

        # 提取图片描述
        captions = []
        if image_captions:
            for img_url in image_references:
                if img_url in image_captions:
                    captions.append(image_captions[img_url])

        # 使用 UUID 确保 chunk_id 全局唯一，同时保留可读性
        unique_id = uuid.uuid4().hex[:8]
        chunk_id = f"{doc_id}_chunk_{chunk_index}_{unique_id}"

        return MultiModalChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            kb_id=kb_id,
            content=content,
            chunk_index=chunk_index,
            page_idx=page_idx,
            page_indices=page_indices,
            block_types=block_types,
            image_references=image_references,
            image_captions=captions,
            title_path=title_path,
            structure_level=len(title_path)
        )

    def _split_long_text(
        self,
        text: str,
        doc_id: str,
        kb_id: str,
        start_index: int,
        page_idx: int,
        page_indices: List[int],
        image_references: List[str],
        image_captions: Optional[Dict[str, str]],
        title_path: List[str]
    ) -> List[MultiModalChunk]:
        """分割长文本"""
        chunks = []
        paragraphs = text.split("\n\n")

        current_chunk = ""
        chunk_index = start_index

        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                # 保存当前chunk
                if current_chunk.strip():
                    chunks.append(self._create_chunk(
                        doc_id=doc_id,
                        kb_id=kb_id,
                        content=current_chunk.strip(),
                        chunk_index=chunk_index,
                        page_idx=page_idx,
                        page_indices=page_indices,
                        image_references=image_references,
                        image_captions=image_captions,
                        title_path=title_path,
                        block_types=["text"]
                    ))
                    chunk_index += 1

                # 添加重叠部分
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + "\n\n" + para + "\n\n"
                else:
                    current_chunk = para + "\n\n"

        # 保存最后一个chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                doc_id=doc_id,
                kb_id=kb_id,
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                page_idx=page_idx,
                page_indices=page_indices,
                image_references=image_references,
                image_captions=image_captions,
                title_path=title_path,
                block_types=["text"]
            ))

        return chunks

    def _get_text_length(self, texts: List[str]) -> int:
        """计算文本列表的总长度"""
        return sum(len(t) for t in texts)

    def _infer_block_types(self, texts: List[str]) -> List[str]:
        """推断块类型"""
        types = set()
        text_str = "\n".join(texts)

        if "## " in text_str or "# " in text_str:
            types.add("title")
        if "- " in text_str or "* " in text_str:
            types.add("list")
        if any(c.isalnum() for c in text_str):
            types.add("text")

        return list(types) if types else ["text"]

    def _update_title_path(self, current_path: List[str], new_title: str) -> List[str]:
        """更新标题路径"""
        # 简单的层级推断
        level = 1

        # 检测标题级别
        if new_title.startswith("第") and ("章" in new_title or "节" in new_title):
            level = 1
        elif re.match(r"^\d+\.\d+", new_title):
            level = 2
        elif re.match(r"^\d+\.\d+\.\d+", new_title):
            level = 3
        elif new_title.startswith("#"):
            level = min(new_title.count("#"), 6)

        # 更新路径
        if level == 1:
            return [new_title]
        elif level <= len(current_path):
            return current_path[:level - 1] + [new_title]
        else:
            return current_path + [new_title]


async def chunk_mineru_document(
    json_file_path: str,
    doc_id: str,
    kb_id: str,
    strategy: str = ChunkingStrategy.HYBRID,
    max_chunk_size: int = 1000,
    image_captions: Optional[Dict[str, str]] = None
) -> List[MultiModalChunk]:
    """便捷函数：对MinerU JSON文档进行分块

    Args:
        json_file_path: MinerU JSON文件路径
        doc_id: 文档ID
        kb_id: 知识库ID
        strategy: 分块策略
        max_chunk_size: 最大分块大小
        image_captions: 图片描述映射

    Returns:
        多模态分块列表
    """
    parser = MinerUJsonParser()
    doc = parser.parse_file(json_file_path)

    chunker = MinerUAwareChunker(
        max_chunk_size=max_chunk_size,
        strategy=strategy
    )

    return await chunker.chunk_document(doc, doc_id, kb_id, image_captions)
