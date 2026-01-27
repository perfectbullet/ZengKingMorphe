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
    """基于MinerU结构信息的智能分块器

    注意：以下参数已写死为默认值，调用方传入的参数将被忽略：
    - max_chunk_size: 512 (适配 bge-large-zh-v1.5-2k 的 2048 tokens)
    - min_chunk_size: 60 (约 max 的 12%)
    - chunk_overlap: 50 (约 max 的 10%)
    - strategy: "hybrid" (混合策略)

    Token 计算：
    - BGE-large-zh-v1.5-2k 最大 2048 tokens
    - 中文字符约等于 2-2.5 tokens
    - 400 字符 ≈ 800-1000 tokens（安全边界）
    """

    # 写死的默认配置
    # BGE-large-zh-v1.5-2k: 2048 tokens 限制
    # 中文字符约等于 2-2.5 tokens，使用 400 字符安全边界 (约 800-1000 tokens)
    DEFAULT_MAX_CHUNK_SIZE = 400
    DEFAULT_MIN_CHUNK_SIZE = 60  # 约 max 的 12%
    DEFAULT_CHUNK_OVERLAP = 50  # 约 max 的 10%
    DEFAULT_STRATEGY = ChunkingStrategy.HYBRID

    def __init__(
        self,
        max_chunk_size: int = 400,       # 参数被忽略，使用默认值
        min_chunk_size: int = 60,        # 参数被忽略，使用默认值
        chunk_overlap: int = 50,         # 参数被忽略，使用默认值
        strategy: str = ChunkingStrategy.HYBRID  # 参数被忽略，使用默认值
    ):
        # 强制使用写死的默认值，忽略调用方传入的参数
        # (pylint: disable=unused-argument)
        _ = (max_chunk_size, min_chunk_size, chunk_overlap, strategy)  # 显式忽略
        self.max_chunk_size = self.DEFAULT_MAX_CHUNK_SIZE
        self.min_chunk_size = self.DEFAULT_MIN_CHUNK_SIZE
        self.chunk_overlap = self.DEFAULT_CHUNK_OVERLAP
        self.strategy = self.DEFAULT_STRATEGY

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
                # 长页面分割 - 使用段落分割方法保持结构
                sub_chunks = self._split_long_text_preserve_structure(
                    texts=page_text,  # 使用原始段落列表
                    doc_id=doc_id,
                    kb_id=kb_id,
                    start_index=chunk_index,
                    page_idx=page.page_idx,
                    page_indices=[page.page_idx],
                    image_references=page_images,
                    image_captions=image_captions,
                    title_path=title_path,
                    block_types=["text"]
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
        """混合分块策略 - 充分利用 MinerU JSON 结构化数据

        策略：
        1. 按标题边界自然分割，保持章节语义完整性
        2. 标题不作为独立内容，必须与后续内容合并
        3. 单个章节超长时，在段落边界分割（保持段落完整）
        4. 每个分割后的子块保留标题路径，便于溯源
        5. 图片与上下文关联，增强检索效果
        6. 过滤掉内容过少的无效章节（纯标题章节）

        最大 chunk 限制为 400 字符（约 800-1000 tokens，适配 bge-large-zh-v1.5-2k）。
        最小有效 chunk 限制为 20 字符（过滤掉只有标题的无效章节）。
        """
        chunks = []
        chunk_index = 0

        # 按标题层级组织内容
        # 每个 section 包含：标题路径、段落列表、图片列表、页码范围
        sections = []  # List[Dict]
        current_section = {
            "title_path": [],
            "texts": [],
            "images": [],
            "page_indices": set(),
            "start_page": 0,
            "block_types": set(),
            "current_title": None  # 当前章节的标题（不放入 texts，等有内容时再添加）
        }

        for page in doc.pdf_info:
            current_section["page_indices"].add(page.page_idx)

            for block in page.blocks:
                # 遇到标题：结束当前章节，开始新章节
                if block.block_type == BlockType.TITLE:
                    # 保存当前章节（如果有实际内容）
                    # 只有 texts 非空（不只是标题）或包含图片时才保存
                    if current_section["texts"] or current_section["images"]:
                        sections.append({
                            "title_path": current_section["title_path"].copy(),
                            "texts": current_section["texts"].copy(),
                            "images": current_section["images"].copy(),
                            "page_indices": list(current_section["page_indices"]),
                            "start_page": current_section["start_page"],
                            "block_types": list(current_section["block_types"])
                        })

                    # 更新标题路径
                    title_text = block.get_text().strip()
                    current_section["title_path"] = self._update_title_path(
                        current_section["title_path"], title_text
                    )

                    # 重置当前章节（标题暂存，等有内容时再添加到 texts）
                    current_section["texts"] = []
                    current_section["images"] = []
                    current_section["page_indices"] = {page.page_idx}
                    current_section["start_page"] = page.page_idx
                    current_section["block_types"] = {"title"}
                    current_section["current_title"] = title_text

                # 处理文本段落
                elif block.block_type == BlockType.TEXT:
                    text = block.get_text().strip()
                    if text:
                        # 如果这是第一个文本块且有标题，先添加标题
                        if not current_section["texts"] and current_section["current_title"]:
                            current_section["texts"].append(f"## {current_section['current_title']}")
                        current_section["texts"].append(text)
                        current_section["block_types"].add("text")

                # 处理列表
                elif block.block_type == BlockType.LIST:
                    text = block.get_text().strip()
                    if text:
                        # 如果这是第一个内容且有标题，先添加标题
                        if not current_section["texts"] and current_section["current_title"]:
                            current_section["texts"].append(f"## {current_section['current_title']}")
                        # 保持原始列表格式
                        current_section["texts"].append(text)
                        current_section["block_types"].add("list")

                # 处理图片
                elif block.block_type == BlockType.IMAGE:
                    for img_url in block.get_image_paths():
                        current_section["images"].append(img_url)
                        current_section["block_types"].add("image")
                        # 如果有图片描述，作为文本添加
                        if image_captions and img_url in image_captions:
                            # 如果这是第一个内容且有标题，先添加标题
                            if not current_section["texts"] and current_section["current_title"]:
                                current_section["texts"].append(f"## {current_section['current_title']}")
                            current_section["texts"].append(f"[图片: {image_captions[img_url]}]")

        # 保存最后一个章节（过滤掉只有标题没有实际内容的章节）
        if current_section["texts"] or current_section["images"]:
            sections.append({
                "title_path": current_section["title_path"].copy(),
                "texts": current_section["texts"].copy(),
                "images": current_section["images"].copy(),
                "page_indices": list(current_section["page_indices"]),
                "start_page": current_section["start_page"],
                "block_types": list(current_section["block_types"])
            })

        # 最小有效 chunk 长度（过滤掉只有标题的无效章节）
        MIN_VALID_CHUNK_LENGTH = 20

        # 将章节转换为 chunks，处理超长章节
        split_count = 0
        skipped_count = 0
        for section in sections:
            # 合并章节文本
            section_text = "\n".join(section["texts"])
            section_length = len(section_text)

            # 过滤掉内容过少的无效章节
            if section_length < MIN_VALID_CHUNK_LENGTH:
                skipped_count += 1
                continue

            if section_length <= self.max_chunk_size:
                # 章节长度合适，直接作为一个 chunk
                chunks.append(self._create_chunk(
                    doc_id=doc_id,
                    kb_id=kb_id,
                    content=section_text,
                    chunk_index=chunk_index,
                    page_idx=section["start_page"],
                    page_indices=section["page_indices"],
                    image_references=section["images"],
                    image_captions=image_captions,
                    title_path=section["title_path"],
                    block_types=section["block_types"]
                ))
                chunk_index += 1
            else:
                # 章节超长，按段落边界分割
                split_count += 1
                sub_chunks = self._split_long_text_preserve_structure(
                    texts=section["texts"],
                    doc_id=doc_id,
                    kb_id=kb_id,
                    start_index=chunk_index,
                    page_idx=section["start_page"],
                    page_indices=section["page_indices"],
                    image_references=section["images"],
                    image_captions=image_captions,
                    title_path=section["title_path"],
                    block_types=section["block_types"]
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)

        # 重新编号
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        # 验证并记录结果
        if chunks:
            chunk_lengths = [len(c.content) for c in chunks]
            max_length = max(chunk_lengths)
            min_length = min(chunk_lengths)
            avg_length = sum(chunk_lengths) / len(chunk_lengths)
            over_limit = [length for length in chunk_lengths if length > self.max_chunk_size]

            if over_limit:
                logger.error(
                    f"分块完成但有{len(over_limit)}个超长chunk! "
                    f"最大: {max_length}, 最小: {min_length}, 平均: {avg_length:.1f}, "
                    f"限制: {self.max_chunk_size}, 总数: {len(chunks)}"
                )
            else:
                skip_msg = f", 跳过{skipped_count}个过短章节" if skipped_count > 0 else ""
                logger.info(
                    f"分块完成: 共{len(chunks)}个chunk, "
                    f"最大: {max_length}, 最小: {min_length}, 平均: {avg_length:.1f}, "
                    f"限制: {self.max_chunk_size}, 章节数: {len(sections)}, 分割了{split_count}个长章节"
                    f"{skip_msg}"
                )
        else:
            logger.warning("分块完成但生成了0个chunk")

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

    def _split_long_text_preserve_structure(
        self,
        texts: List[str],
        doc_id: str,
        kb_id: str,
        start_index: int,
        page_idx: int,
        page_indices: List[int],
        image_references: List[str],
        image_captions: Optional[Dict[str, str]],
        title_path: List[str],
        block_types: List[str]
    ) -> List[MultiModalChunk]:
        """按段落边界分割长文本，保持段落完整性和结构信息

        策略：
        1. 计算 title 的长度，在判断是否超限时要把 title 长度算上
        2. 如果单个段落超过限制，对半分，递归处理直到不超过限制
        3. 过滤掉过短的分割结果（小于 20 字符）
        """
        # 计算 title 长度（标题会作为前缀添加到 content 中）
        title_prefix = "\n".join(title_path) if title_path else ""
        title_length = len(title_prefix) + 2 if title_prefix else 0  # +2 for possible newlines

        # 实际可用于内容的长度
        available_length = self.max_chunk_size - title_length

        chunks = []
        chunk_index = start_index
        MIN_VALID_CHUNK_LENGTH = 20  # 最小有效 chunk 长度

        current_paragraphs = []
        current_length = 0

        def split_half(text: str) -> List[str]:
            """对半分文本，如果还超长则递归分割"""
            if len(text) <= available_length:
                return [text]
            # 对半分
            mid = len(text) // 2
            # 尝试在标点符号处分割（在中点附近50字符范围内搜索）
            best_pos = -1
            for sep in ["。", "！", "？", "\n\n", "；", ";", "，", ",", "、"]:
                pos = text.find(sep, max(0, mid - 50), min(len(text), mid + 50))
                if pos != -1:
                    best_pos = pos + len(sep)
                    break
            if best_pos > 0:
                split_pos = best_pos
            else:
                split_pos = mid

            part1 = text[:split_pos].strip()
            part2 = text[split_pos:].strip()

            result = []
            if part1:
                result.extend(split_half(part1))
            if part2:
                result.extend(split_half(part2))
            return result

        for para in texts:
            para_length = len(para)

            # 如果单个段落就超长（考虑 title 长度），对半分
            if para_length > available_length:
                # 先保存当前累积的内容
                if current_paragraphs:
                    content = "\n".join(current_paragraphs)
                    chunks.append(self._create_chunk(
                        doc_id=doc_id,
                        kb_id=kb_id,
                        content=content,
                        chunk_index=chunk_index,
                        page_idx=page_idx,
                        page_indices=page_indices,
                        image_references=image_references,
                        image_captions=image_captions,
                        title_path=title_path,
                        block_types=block_types
                    ))
                    chunk_index += 1
                    current_paragraphs = []
                    current_length = 0

                # 对半分超长段落
                parts = split_half(para)
                for part in parts:
                    chunks.append(self._create_chunk(
                        doc_id=doc_id,
                        kb_id=kb_id,
                        content=part,
                        chunk_index=chunk_index,
                        page_idx=page_idx,
                        page_indices=page_indices,
                        image_references=image_references,
                        image_captions=image_captions,
                        title_path=title_path,
                        block_types=block_types
                    ))
                    chunk_index += 1
                continue

            # 检查添加这个段落是否会超限（考虑 title 长度）
            if current_length + para_length + 1 <= available_length:
                current_paragraphs.append(para)
                current_length += para_length + 1  # +1 for newline
            else:
                # 保存当前 chunk
                if current_paragraphs:
                    chunks.append(self._create_chunk(
                        doc_id=doc_id,
                        kb_id=kb_id,
                        content="\n".join(current_paragraphs),
                        chunk_index=chunk_index,
                        page_idx=page_idx,
                        page_indices=page_indices,
                        image_references=image_references,
                        image_captions=image_captions,
                        title_path=title_path,
                        block_types=block_types
                    ))
                    chunk_index += 1

                # 开始新 chunk
                current_paragraphs = [para]
                current_length = para_length

        # 保存最后一个 chunk
        if current_paragraphs:
            chunks.append(self._create_chunk(
                doc_id=doc_id,
                kb_id=kb_id,
                content="\n".join(current_paragraphs),
                chunk_index=chunk_index,
                page_idx=page_idx,
                page_indices=page_indices,
                image_references=image_references,
                image_captions=image_captions,
                title_path=title_path,
                block_types=block_types
            ))

        # 过滤掉过短的 chunk（分割时可能产生极短片段）
        valid_chunks = [c for c in chunks if len(c.content) >= MIN_VALID_CHUNK_LENGTH]

        # 重新编号
        for i, chunk in enumerate(valid_chunks):
            chunk.chunk_index = start_index + i

        return valid_chunks

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
    max_chunk_size: int = 400,  # 最大分块大小（字符）- 适配 bge-large-zh-v1.5-2k (2048 tokens)
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
