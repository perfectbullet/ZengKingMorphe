"""
检索基类和数据模型
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    """检索到的文档"""

    text: str = Field(..., description="文档内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    score: float = Field(0.0, description="相关性得分")
    source: Optional[str] = Field(None, description="来源文档")
    chunk_id: Optional[str] = Field(None, description="块 ID")


class RetrievalStrategy(ABC):
    """检索策略基类"""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        执行检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            检索到的文档列表
        """
        pass
