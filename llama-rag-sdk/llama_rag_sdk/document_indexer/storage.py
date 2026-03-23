"""
向量存储层

封装 ChromaDB 操作
"""

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from llama_rag_sdk.config import settings


class VectorStore:
    """向量存储类"""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        persist_dir: Optional[str] = None,
        use_remote: Optional[bool] = None
    ) -> None:
        """
        初始化向量存储

        Args:
            collection_name: 集合名称
            host: ChromaDB 主机地址
            port: ChromaDB 端口
            persist_dir: 本地持久化目录
            use_remote: 是否使用远程服务
        """
        self.collection_name = collection_name or settings.chroma_collection_name
        self.host = host or settings.chroma_host
        self.port = port or settings.chroma_port
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.use_remote = use_remote if use_remote is not None else settings.chroma_use_remote

        self.client = self._create_client()
        self.collection = self._get_or_create_collection()

    def _create_client(self) -> chromadb.Client:
        """创建 ChromaDB 客户端"""
        if self.use_remote:
            logger.info(f"连接到远程 ChromaDB: {self.host}:{self.port}")
            return chromadb.HttpClient(host=self.host, port=self.port)

        logger.info(f"使用本地 ChromaDB: {self.persist_dir}")
        return chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

    def _get_or_create_collection(self) -> chromadb.Collection:
        """获取或创建集合"""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"使用现有集合: {self.collection_name}")
        except Exception:
            logger.info(f"创建新集合: {self.collection_name}")
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG 文档集合"}
            )
        return collection

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        添加文档到向量存储

        Args:
            ids: 文档 ID 列表
            embeddings: 嵌入向量列表
            documents: 文档内容列表
            metadatas: 元数据列表

        Raises:
            chromadb.errors.ChromaError: 添加文档失败时抛出
        """
        if metadatas is None:
            metadatas = [{} for _ in range(len(ids))]

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"添加 {len(ids)} 个文档到集合 {self.collection_name}")
        except Exception as e:
            logger.error(f"添加文档失败: {e}", exc_info=True)
            raise

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        查询向量存储

        Args:
            query_embeddings: 查询向量列表
            n_results: 返回结果数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件

        Returns:
            查询结果字典，包含 ids、documents、metadatas、distances 等

        Raises:
            chromadb.errors.ChromaError: 查询失败时抛出
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            result_count = len(results.get('ids', [[]])[0])
            logger.debug(f"查询返回 {result_count} 个结果")
            return results
        except Exception as e:
            logger.error(f"查询失败: {e}", exc_info=True)
            raise

    def delete(
        self,
        ids: List[str],
        where: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        从向量存储中删除文档

        Args:
            ids: 文档 ID 列表
            where: 元数据过滤条件

        Raises:
            chromadb.errors.ChromaError: 删除文档失败时抛出
        """
        try:
            self.collection.delete(ids=ids, where=where)
            logger.info(f"删除 {len(ids)} 个文档")
        except Exception as e:
            logger.error(f"删除文档失败: {e}", exc_info=True)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息

        Returns:
            统计信息字典，包含 collection_name、count、metadata

        Raises:
            chromadb.errors.ChromaError: 获取统计信息失败时抛出
        """
        try:
            count = self.collection.count()
            logger.info(f"集合 {self.collection_name} 包含 {count} 个文档")
            return {
                "collection_name": self.collection_name,
                "count": count,
                "metadata": self.collection.metadata
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}", exc_info=True)
            raise

    def reset_collection(self) -> None:
        """
        重置集合（删除所有数据）

        Raises:
            chromadb.errors.ChromaError: 重置集合失败时抛出
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._get_or_create_collection()
            logger.info(f"集合 {self.collection_name} 已重置")
        except Exception as e:
            logger.error(f"重置集合失败: {e}", exc_info=True)
            raise
