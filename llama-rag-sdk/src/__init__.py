"""
LlamaRAG SDK - 基于 LlamaIndex 的 RAG 检索增强生成系统

支持文档解析、索引创建和智能检索，专为教材文档优化。
"""

__version__ = "0.1.0"
__author__ = "ZengKing"

from src.config import Settings, settings
from src.rag_system import RAGSystem

__all__ = [
    "Settings",
    "settings",
    "RAGSystem",
]
