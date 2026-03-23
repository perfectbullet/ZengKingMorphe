"""
Document service utilities.

文档处理相关工具函数和常量。

注意：DocumentProcessor 已被移除，所有文档处理功能已迁移到：
- llama-rag-sdk RAGSystem（核心功能）
- app.services.task_processor（MongoDB 文档记录管理）
"""

from pathlib import Path

# 临时文件下载目录
UPLOAD_DIR = Path(__file__).parent.parent.parent.parent / "upload_docs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
