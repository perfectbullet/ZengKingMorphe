"""
文本分块器

提供多种分块策略
"""

from typing import List
from abc import ABC, abstractmethod
import re
from loguru import logger

from src.document_indexer.base import ChunkStrategy


class BaseChunker(ABC):
    """分块器基类"""

    def __init__(self, strategy: ChunkStrategy):
        """
        初始化分块器

        Args:
            strategy: 分块策略
        """
        self.strategy = strategy

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """
        分块文本

        Args:
            text: 输入文本

        Returns:
            分块后的文本列表
        """
        pass


class FixedSizeChunker(BaseChunker):
    """固定大小分块器"""

    def chunk(self, text: str) -> List[str]:
        """
        按固定大小分块

        Args:
            text: 输入文本

        Returns:
            分块后的文本列表
        """
        chunks = []
        start = 0
        chunk_size = self.strategy.chunk_size
        overlap = self.strategy.chunk_overlap

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())

            # 计算下一块的起始位置
            start = end - overlap if end < len(text) else end

            # 防止无限循环
            if start >= len(text):
                break

        logger.debug(f"固定大小分块: 共 {len(chunks)} 块")
        return chunks


class SemanticChunker(BaseChunker):
    """语义分块器"""

    def chunk(self, text: str) -> List[str]:
        """
        按语义结构分块（基于段落和标题）

        Args:
            text: 输入文本

        Returns:
            分块后的文本列表
        """
        chunks = []
        current_chunk = ""
        min_size = self.strategy.min_chunk_size
        max_size = self.strategy.max_chunk_size

        # 按行分割
        lines = text.split('\n')

        for line in lines:
            line_stripped = line.strip()

            # 检查是否是标题（以 # 开头）
            is_heading = line_stripped.startswith('#')

            # 如果当前块不为空且遇到标题，保存当前块
            if is_heading and current_chunk:
                # 确保当前块满足最小大小要求
                if len(current_chunk) >= min_size or not chunks:
                    chunks.append(current_chunk.strip())
                    current_chunk = line + "\n"
                else:
                    # 合并到前一块
                    if chunks:
                        chunks[-1] += "\n" + current_chunk
                    current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"

                # 检查是否超过最大大小
                if len(current_chunk) >= max_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

        # 添加最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        logger.debug(f"语义分块: 共 {len(chunks)} 块")
        return chunks


class HybridChunker(BaseChunker):
    """混合分块器（语义 + 固定大小）"""

    def __init__(self, strategy: ChunkStrategy):
        super().__init__(strategy)
        self.semantic_chunker = SemanticChunker(strategy)
        self.fixed_size_chunker = FixedSizeChunker(strategy)

    def chunk(self, text: str) -> List[str]:
        """
        混合分块策略

        先按语义分块，对过长的块再按固定大小分块

        Args:
            text: 输入文本

        Returns:
            分块后的文本列表
        """
        # 先按语义分块
        semantic_chunks = self.semantic_chunker.chunk(text)

        # 对过长的块再分块
        final_chunks = []
        max_size = self.strategy.max_size

        for chunk in semantic_chunks:
            if len(chunk) > max_size:
                # 使用固定大小分块
                sub_chunks = self.fixed_size_chunker.chunk(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        logger.debug(f"混合分块: 共 {len(final_chunks)} 块")
        return final_chunks
