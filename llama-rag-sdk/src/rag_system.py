"""
RAG 系统集成类

提供统一的 RAG 系统接口，整合解析、索引和检索功能
"""

import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from src.config import settings
from src.document_parser.mineru_client import MinerUParser
from src.document_parser.image_processor import ImageDescriptor
from src.document_parser.base import ParsedDocument
from src.document_indexer.indexer import DocumentIndexer
from src.document_indexer.storage import VectorStore
from src.retrieval.retriever import Retriever
from src.retrieval.base import RetrievedDocument


class RAGSystem:
    """
    RAG 系统集成类

    整合文档解析、索引创建和智能检索功能
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        enable_image_description: Optional[bool] = None,
        use_hybrid_retrieval: Optional[bool] = None,
        use_rerank: Optional[bool] = None
    ):
        """
        初始化 RAG 系统

        Args:
            collection_name: 集合名称
            enable_image_description: 是否启用图片描述
            use_hybrid_retrieval: 是否使用混合检索
            use_rerank: 是否使用重排序
        """
        self.collection_name = collection_name or settings.chroma_collection_name
        self.enable_image_description = (
            enable_image_description if enable_image_description is not None
            else settings.enable_image_description
        )
        self.use_hybrid_retrieval = (
            use_hybrid_retrieval if use_hybrid_retrieval is not None
            else settings.use_hybrid_retrieval
        )
        self.use_rerank = (
            use_rerank if use_rerank is not None
            else settings.use_rerank
        )

        # 确保必要的目录存在
        settings.ensure_directories()

        # 初始化组件（延迟初始化）
        self._parser: Optional[MinerUParser] = None
        self._image_descriptor: Optional[ImageDescriptor] = None
        self._indexer: Optional[DocumentIndexer] = None
        self._retriever: Optional[Retriever] = None
        self._vector_store: Optional[VectorStore] = None

        # 追踪已处理的文档
        self._document_cache: Dict[str, ParsedDocument] = {}

    @property
    def parser(self) -> MinerUParser:
        """获取文档解析器"""
        if self._parser is None:
            self._parser = MinerUParser()
        return self._parser

    @property
    def image_descriptor(self) -> ImageDescriptor:
        """获取图片描述生成器"""
        if self._image_descriptor is None:
            self._image_descriptor = ImageDescriptor()
        return self._image_descriptor

    @property
    def indexer(self) -> DocumentIndexer:
        """获取文档索引器"""
        if self._indexer is None:
            self._indexer = DocumentIndexer(
                collection_name=self.collection_name
            )
        return self._indexer

    @property
    def vector_store(self) -> VectorStore:
        """获取向量存储"""
        if self._vector_store is None:
            self._vector_store = VectorStore(collection_name=self.collection_name)
        return self._vector_store

    @property
    def retriever(self) -> Retriever:
        """获取检索器"""
        if self._retriever is None:
            self._retriever = Retriever(
                vector_store=self.vector_store,
                use_rerank=self.use_rerank
            )
        return self._retriever

    @classmethod
    def from_config(cls) -> 'RAGSystem':
        """
        从配置文件创建 RAG 系统

        Returns:
            RAG 系统实例
        """
        return cls()

    async def parse_document(
        self,
        file_path: str,
        generate_image_descriptions: Optional[bool] = None
    ) -> ParsedDocument:
        """
        解析文档

        Args:
            file_path: 文档文件路径
            generate_image_descriptions: 是否生成图片描述

        Returns:
            解析后的文档对象
        """
        generate_image_descriptions = (
            generate_image_descriptions if generate_image_descriptions is not None
            else self.enable_image_description
        )

        logger.info(f"解析文档: {file_path}")

        # 解析文档
        document = await self.parser.parse(file_path)

        # 缓存文档
        self._document_cache[file_path] = document

        # 生成图片描述
        if generate_image_descriptions and document.images:
            logger.info(f"生成 {len(document.images)} 个图片描述...")
            document.images = await self.image_descriptor.describe_images_batch(
                document.images,
                concurrency=3
            )

        logger.info(
            f"文档解析完成: {document.title}, "
            f"{len(document.chunks)} 块, {len(document.images)} 图片"
        )

        return document

    async def index_document(
        self,
        file_path: str,
        generate_image_descriptions: Optional[bool] = None
    ) -> List[str]:
        """
        索引文档

        Args:
            file_path: 文档文件路径
            generate_image_descriptions: 是否生成图片描述

        Returns:
            文档 ID 列表
        """
        # 解析文档
        document = await self.parse_document(file_path, generate_image_descriptions)

        # 提取文本块
        chunks = [chunk.text for chunk in document.chunks]
        metadata_list = [
            {
                "source": file_path,
                "title": document.title,
                "page": chunk.page,
                "section": chunk.section,
                "chunk_index": chunk.index,
                **chunk.metadata
            }
            for chunk in document.chunks
        ]

        # 添加到索引
        doc_ids = await self.indexer.add_documents(
            documents=chunks,
            collection_name=self.collection_name,
            metadata_list=metadata_list
        )

        logger.info(f"文档索引完成: {len(doc_ids)} 个块")
        return doc_ids

    async def index_documents_batch(
        self,
        file_paths: List[str],
        generate_image_descriptions: Optional[bool] = None
    ) -> Dict[str, List[str]]:
        """
        批量索引文档

        Args:
            file_paths: 文档文件路径列表
            generate_image_descriptions: 是否生成图片描述

        Returns:
            文件路径到文档 ID 列表的映射
        """
        results = {}

        for file_path in file_paths:
            try:
                doc_ids = await self.index_document(file_path, generate_image_descriptions)
                results[file_path] = doc_ids
            except Exception as e:
                logger.error(f"索引文档失败 {file_path}: {e}")
                results[file_path] = []

        return results

    async def add_text_documents(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        添加文本文档到索引

        Args:
            texts: 文本内容列表
            metadata_list: 元数据列表

        Returns:
            文档 ID 列表
        """
        doc_ids = await self.indexer.add_documents(
            documents=texts,
            collection_name=self.collection_name,
            metadata_list=metadata_list
        )

        logger.info(f"添加文本文档完成: {len(doc_ids)} 个块")
        return doc_ids

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        检索文档

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            检索到的文档列表
        """
        if top_k is None:
            top_k = settings.top_k

        documents = await self.retriever.retrieve(query, top_k, filters)
        return documents

    async def retrieve_multiple(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        deduplicate: bool = True
    ) -> List[RetrievedDocument]:
        """
        多查询检索

        Args:
            queries: 查询文本列表
            top_k: 每次查询返回结果数量
            filters: 过滤条件
            deduplicate: 是否去重

        Returns:
            检索到的文档列表
        """
        if top_k is None:
            top_k = settings.top_k

        documents = await self.retriever.retrieve_multiple(
            queries=queries,
            top_k=top_k,
            filters=filters,
            deduplicate=deduplicate
        )
        return documents

    async def get_stats(self) -> Dict[str, Any]:
        """
        获取系统统计信息

        Returns:
            统计信息字典
        """
        stats = await self.indexer.get_collection_stats(self.collection_name)
        stats["document_cache_size"] = len(self._document_cache)
        return stats

    async def clear_collection(self) -> None:
        """
        清空集合
        """
        self.vector_store.reset_collection()
        self._document_cache.clear()
        logger.info(f"集合已清空: {self.collection_name}")

    async def close(self) -> None:
        """关闭资源"""
        # MinerUParser 和 ImageDescriptor 不需要显式关闭（基于 SDK）
        # VectorStore 会自动处理连接清理
        self._document_cache.clear()
        logger.info("RAG 系统资源已关闭")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
