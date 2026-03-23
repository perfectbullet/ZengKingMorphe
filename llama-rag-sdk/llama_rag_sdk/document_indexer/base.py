"""
索引器基类和数据模型
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field


class ChunkStrategyType(str, Enum):
    """分块策略类型枚举"""

    FIXED = "fixed"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class ChunkStrategy(BaseModel):
    """分块策略配置"""

    type: ChunkStrategyType = Field(default=ChunkStrategyType.FIXED, description="策略类型")
    chunk_size: int = Field(default=512, description="分块大小")
    chunk_overlap: int = Field(default=50, description="重叠大小")
    min_chunk_size: int = Field(default=100, description="最小分块大小")
    max_chunk_size: int = Field(default=1024, description="最大分块大小")
    separator: str = Field(default="\n\n", description="分隔符")


class Indexer(ABC):
    """索引器基类"""

    @abstractmethod
    async def create_index(
        self,
        documents: List[Any],
        collection_name: Optional[str] = None
    ) -> str:
        """
        创建索引

        Args:
            documents: 文档列表
            collection_name: 集合名称

        Returns:
            索引 ID
        """
        pass

    @abstractmethod
    async def add_documents(
        self,
        documents: List[Any],
        collection_name: str
    ) -> List[str]:
        """
        添加文档到索引

        Args:
            documents: 文档列表
            collection_name: 集合名称

        Returns:
            文档 ID 列表
        """
        pass

    @abstractmethod
    async def delete_documents(
        self,
        document_ids: List[str],
        collection_name: str
    ) -> bool:
        """
        从索引中删除文档

        Args:
            document_ids: 文档 ID 列表
            collection_name: 集合名称

        Returns:
            是否成功
        """
        pass

    @abstractmethod
    async def get_collection_stats(
        self,
        collection_name: str
    ) -> Dict[str, Any]:
        """
        获取集合统计信息

        Args:
            collection_name: 集合名称

        Returns:
            统计信息字典
        """
        pass
