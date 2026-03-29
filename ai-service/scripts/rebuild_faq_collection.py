#!/usr/bin/env python3
"""删除旧的FAQ collection以应用新的cosine距离配置"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# from app.core.chroma import chroma_db
from app.core.logging import get_logger

logger = get_logger(__name__)

chroma_db.connect()

# 检查并删除旧collection（兼容ChromaDB v0.6.0 API）
collections = chroma_db.client.list_collections()
logger.info(f"当前ChromaDB collections: {collections}")

if "faq_knowledge" in collections:
    # Get collection to check metadata
    old_col = chroma_db.client.get_collection("faq_knowledge")
    count = old_col.count()
    metadata = old_col.metadata
    
    logger.warning(f"删除旧collection: {count}个向量, metadata={metadata}")
    
    # Delete the collection
    chroma_db.client.delete_collection("faq_knowledge")
    
    logger.info("✅ 已删除faq_knowledge，重启服务后将自动创建新collection（使用cosine距离）")
else:
    logger.info("faq_knowledge不存在，无需删除")

# 同时检查其他两个collection的距离配置
for col_name in ["rag_documents", "custom_dictionary"]:
    if col_name in collections:
        col = chroma_db.client.get_collection(col_name)
        hnsw_space = col.metadata.get("hnsw:space", "l2 (default)")
        logger.info(f"Collection {col_name}: 距离度量={hnsw_space}, 向量数={col.count()}")