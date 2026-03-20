"""
MinerU Structure-Aware Chunker - 基于 MinerU content_list 的结构感知分块

该模块利用 MinerU SDK content_list 提供的结构信息进行智能分块：
1. 按标题（title）边界进行文档分段
2. 保持列表（list）完整性
3. 关联图片与其上下文
4. 保留页面信息用于溯源

与传统分块相比的优势：
- 更好的语义完整性（按章节分块）
- 更丰富的元数据（页码、块类型、图片、标题路径）
- 支持多模态检索（图片描述 + 文本）

使用示例:
    >>> from src.document_parser.mineru_structure_aware_chunker import MinerUStructureAwareChunker
    >>> chunker = MinerUStructureAwareChunker()
    >>> chunks = await chunker.chunk_content_list(content_list, "document.pdf")
    >>> for chunk in chunks:
    ...     print(f"{chunk.metadata['title_path']}: {chunk.text[:50]}...")
"""

import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from src.document_parser.base import TextChunk
from src.constants import ChunkingStrategy, MinerUChunkingDefaults


class MinerUStructureAwareChunker:
    """基于 MinerU content_list 的结构感知分块器

    配置来源（优先级从高到低）：
    1. 环境变量 CHUNK_SIZE, CHUNK_OVERLAP（推荐）
    2. settings 配置（从 .env 读取）

    Token 计算：
    - BGE-m3: 8192 tokens
    - 中文字符约等于 2-2.5 tokens
    - 512 字符 ≈ 1024-1280 tokens（安全边界）
    """

    def __init__(
        self,
        max_chunk_size: Optional[int] = None,
        min_chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: Optional[str] = None,
    ):
        """
        初始化结构感知分块器

        配置优先级（从高到低）：
        1. 环境变量 CHUNK_SIZE, CHUNK_OVERLAP（推荐）
        2. settings 配置（从 .env 读取）
        """
        # 导入配置（延迟导入避免循环依赖）
        try:
            from src.config import settings
            self.max_chunk_size = settings.chunk_size
            self.chunk_overlap = settings.chunk_overlap
        except Exception:
            # 如果配置加载失败，使用兜底值
            logger.warning("无法加载配置，使用兜底值")
            self.max_chunk_size = 512
            self.chunk_overlap = 150

        # min_chunk_size 自动计算为 max_chunk_size 的 15%
        self.min_chunk_size = int(self.max_chunk_size * MinerUChunkingDefaults.MIN_CHUNK_SIZE_RATIO / 100)
        self.min_valid_chunk_length = MinerUChunkingDefaults.MIN_VALID_CHUNK_LENGTH

        # 强制使用 hybrid 策略
        self.strategy = ChunkingStrategy.HYBRID

        # 显式忽略传入的参数（保持兼容性但给出警告）
        if any([max_chunk_size, min_chunk_size, chunk_overlap, strategy]):
            logger.warning(
                f"MinerUStructureAwareChunker 初始化参数已被废弃，"
                f"请使用环境变量 CHUNK_SIZE={self.max_chunk_size} 和 "
                f"CHUNK_OVERLAP={self.chunk_overlap}"
            )

    def chunk_content_list(
        self,
        content_list: List[Dict[str, Any]],
        pdf_name: str,
        image_captions: Optional[Dict[str, str]] = None
    ) -> List[TextChunk]:
        """
        将 MinerU SDK 的 content_list 转换为结构化的 TextChunk 列表

        策略（hybrid）：
        1. 按标题层级组织内容
        2. 标题不作为独立 chunk，必须与后续内容合并
        3. 单个章节超长时，在段落边界分割
        4. 每个chunk保留：页码、标题路径、块类型、图片信息

        Args:
            content_list: MinerU SDK content_list
            pdf_name: PDF 文件名
            image_captions: 图片描述映射（可选）

        Returns:
            结构化的 TextChunk 列表
        """
        if not content_list:
            return []

        return self._chunk_by_sections(content_list, pdf_name, image_captions)

    def _chunk_by_sections(
        self,
        content_list: List[Dict[str, Any]],
        pdf_name: str,
        image_captions: Optional[Dict[str, str]] = None
    ) -> List[TextChunk]:
        """
        混合分块策略 - 充分利用 MinerU content_list 结构化数据

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

        for item in content_list:
            content_type = item.get("type", "text")
            page_idx = item.get("page_idx", item.get("page_id", 0))

            # 跳过 discarded 类型（页眉页脚等）
            if content_type == "discarded":
                continue

            # 添加页码到当前章节（跳过 discarded 后）
            current_section["page_indices"].add(page_idx)

            # 判断是否为标题（支持两种方式）
            # 1. type == "title"
            # 2. type == "text" 且有 text_level > 0
            is_title = (
                content_type == "title" or
                (content_type == "text" and item.get("text_level", 0) > 0)
            )

            # 遇到标题：结束当前章节，开始新章节
            if is_title:
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

                # 提取标题文本
                title_text = item.get("text", "").strip()
                text_level = item.get("text_level")

                # 更新标题路径
                current_section["title_path"] = self._update_title_path(
                    current_section["title_path"], title_text, text_level
                )

                # 重置当前章节（标题暂存，等有内容时再添加到 texts）
                current_section["texts"] = []
                current_section["images"] = []
                current_section["page_indices"] = {page_idx}
                current_section["start_page"] = page_idx
                current_section["block_types"] = {"title"}
                current_section["current_title"] = title_text

            # 处理文本段落（排除已识别为标题的）
            elif content_type == "text" and not is_title:
                text = item.get("text", "").strip()
                if text:
                    # 如果这是第一个文本块且有标题，先添加标题
                    if not current_section["texts"] and current_section["current_title"]:
                        current_section["texts"].append(f"## {current_section['current_title']}")
                    current_section["texts"].append(text)
                    current_section["block_types"].add("text")

            # 处理列表
            elif content_type == "list":
                text = item.get("text", "").strip()
                if text:
                    # 如果这是第一个内容且有标题，先添加标题
                    if not current_section["texts"] and current_section["current_title"]:
                        current_section["texts"].append(f"## {current_section['current_title']}")
                    # 保持原始列表格式
                    current_section["texts"].append(text)
                    current_section["block_types"].add("list")

            # 处理图片
            elif content_type == "image":
                img_path = item.get("img_path")
                if img_path:
                    current_section["images"].append(img_path)
                    current_section["block_types"].add("image")
                    # 如果有图片描述，作为文本添加
                    if image_captions and img_path in image_captions:
                        # 如果这是第一个内容且有标题，先添加标题
                        if not current_section["texts"] and current_section["current_title"]:
                            current_section["texts"].append(f"## {current_section['current_title']}")
                        current_section["texts"].append(f"[图片: {image_captions[img_path]}]")

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
        split_count = 0
        skipped_count = 0

        for section in sections:
            # 合并章节文本
            section_text = "\n".join(section["texts"])
            section_length = len(section_text)

            # 过滤掉内容过少的无效章节
            if section_length < self.min_valid_chunk_length:
                skipped_count += 1
                continue

            if section_length <= self.max_chunk_size:
                # 章节长度合适，直接作为一个 chunk
                chunks.append(self._create_chunk(
                    pdf_name=pdf_name,
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
                    pdf_name=pdf_name,
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
            chunk.index = i

        # 验证并记录结果
        if chunks:
            chunk_lengths = [len(c.text) for c in chunks]
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
        pdf_name: str,
        content: str,
        chunk_index: int,
        page_idx: int,
        page_indices: List[int],
        image_references: List[str],
        image_captions: Optional[Dict[str, str]],
        title_path: List[str],
        block_types: List[str]
    ) -> TextChunk:
        """创建一个 TextChunk"""
        # 提取图片描述
        captions = []
        if image_captions:
            for img_url in image_references:
                if img_url in image_captions:
                    captions.append(image_captions[img_url])

        # 使用 UUID 确保 chunk_id 全局唯一
        unique_id = uuid.uuid4().hex[:8]
        chunk_id = f"{pdf_name}_chunk_{chunk_index}_{unique_id}"

        # 构建 section（当前章节标题）
        section = title_path[-1] if title_path else None

        # 构建增强的 metadata
        metadata = {
            "chunk_id": chunk_id,
            "type": block_types[0] if block_types else "text",
            "page_idx": page_idx,
            "page_indices": page_indices,
            "title_path": title_path,
            "structure_level": len(title_path),
            "block_types": block_types,
            "image_references": image_references,
            "image_captions": captions,
            "created_at": datetime.utcnow().isoformat(),
        }

        return TextChunk(
            text=content,
            index=chunk_index,
            page=page_idx,
            section=section,
            metadata=metadata
        )

    def _split_long_text_preserve_structure(
        self,
        texts: List[str],
        pdf_name: str,
        start_index: int,
        page_idx: int,
        page_indices: List[int],
        image_references: List[str],
        image_captions: Optional[Dict[str, str]],
        title_path: List[str],
        block_types: List[str]
    ) -> List[TextChunk]:
        """
        按段落边界分割长文本，保持段落完整性和结构信息

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
                        pdf_name=pdf_name,
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
                        pdf_name=pdf_name,
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
                        pdf_name=pdf_name,
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
                pdf_name=pdf_name,
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
        valid_chunks = [c for c in chunks if len(c.text) >= self.min_valid_chunk_length]

        # 重新编号
        for i, chunk in enumerate(valid_chunks):
            chunk.index = start_index + i

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

    def _update_title_path(
        self,
        current_path: List[str],
        new_title: str,
        text_level: Optional[int] = None
    ) -> List[str]:
        """
        更新标题路径

        Args:
            current_path: 当前标题路径
            new_title: 新标题文本
            text_level: 标题级别（如果有的话，优先使用）

        根据标题特征推断层级：
        - 优先使用显式的 text_level
        - "第X章/节" → level 1
        - "1.1", "2.3" → level 2
        - "1.1.1", "2.3.4" → level 3
        - "# 标题" → 根据 # 数量
        """
        # 优先使用显式的 text_level
        if text_level is not None and text_level > 0:
            level = text_level
        else:
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


# 便捷函数
def chunk_mineru_content_list(
    content_list: List[Dict[str, Any]],
    pdf_name: str,
    strategy: str = ChunkingStrategy.HYBRID,
    max_chunk_size: int = 400,
    image_captions: Optional[Dict[str, str]] = None
) -> List[TextChunk]:
    """
    便捷函数：对 MinerU content_list 进行分块

    Args:
        content_list: MinerU content_list
        pdf_name: PDF 文件名
        strategy: 分块策略
        max_chunk_size: 最大分块大小
        image_captions: 图片描述映射

    Returns:
        分块后的 TextChunk 列表
    """
    chunker = MinerUStructureAwareChunker(
        max_chunk_size=max_chunk_size,
        strategy=strategy
    )

    return chunker.chunk_content_list(content_list, pdf_name, image_captions)
