"""
RAG 系统集成类

提供统一的 RAG 系统接口，整合解析、索引和检索功能
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

from loguru import logger

from src.config import settings
from src.document_indexer.docstore import (
    DocStoreDocument,
    MongoDBDocStore,
    create_docstore,
)
from src.document_indexer.indexer import DocumentIndexer
from src.document_indexer.storage import VectorStore
from src.document_indexer.summarizer import DocumentSummarizer
from src.document_parser.base import ParsedDocument
from src.document_parser.image_processor import ImageDescriptor
from src.document_parser.mineru_client import MinerUParser
from src.retrieval.base import RetrievedDocument
from src.retrieval.context_expander import ContextExpander
from src.retrieval.retriever import Retriever


class RAGSystem:
    """
    RAG 系统集成类

    整合文档解析、索引创建和智能检索功能

    新增功能：
    - DocStore 支持：存储完整文档内容和元数据
    - 上下文扩展：自动扩展检索结果的上下文
    - 文档摘要：LLM 驱动的双重索引（全文+摘要）
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        enable_image_description: Optional[bool] = None,
        enable_summarization: Optional[bool] = None,
    ):
        """
        初始化 RAG 系统

        Args:
            collection_name: 集合名称
            enable_image_description: 是否启用图片描述
            enable_summarization: 是否启用文档摘要生成
        """
        self.collection_name = collection_name or settings.chroma_collection_name
        self.enable_image_description = (
            enable_image_description if enable_image_description is not None
            else settings.enable_image_description
        )
        self.enable_summarization = (
            enable_summarization if enable_summarization is not None
            else settings.enable_summarization
        )

        # 确保必要的目录存在
        settings.ensure_directories()

        # 初始化组件（延迟初始化）
        self._parser: Optional[MinerUParser] = None
        self._image_descriptor: Optional[ImageDescriptor] = None
        self._indexer: Optional[DocumentIndexer] = None
        self._retriever: Optional[Retriever] = None
        self._vector_store: Optional[VectorStore] = None
        self._docstore: Optional[MongoDBDocStore] = None
        self._context_expander: Optional[ContextExpander] = None
        self._summarizer: Optional["DocumentSummarizer"] = None

        # 追踪已处理的文档
        self._document_cache: dict[str, ParsedDocument] = {}

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
                vector_store=self.vector_store
            )
        return self._retriever

    @property
    def docstore(self) -> MongoDBDocStore:
        """获取 MongoDB DocStore"""
        if self._docstore is None:
            self._docstore = create_docstore(
                uri=settings.mongodb_uri,
                db_name=settings.mongodb_db_name,
                collection_name=settings.docstore_collection
            )
        return self._docstore

    @property
    def context_expander(self) -> Optional[ContextExpander]:
        """获取上下文扩展器（延迟初始化）"""
        if self._context_expander is None:
            self._context_expander = ContextExpander(
                docstore=self.docstore,
                window=settings.context_expansion_window,
                include_parent=settings.context_expansion_include_parent
            )
        return self._context_expander

    @property
    def summarizer(self) -> "DocumentSummarizer":
        """获取文档摘要生成器"""
        if self._summarizer is None:
            from src.document_indexer.summarizer import DocumentSummarizer
            self._summarizer = DocumentSummarizer()
        return self._summarizer

    @classmethod
    def from_config(cls) -> 'RAGSystem':
        """
        从配置文件创建 RAG 系统

        Returns:
            RAG 系统实例
        """
        return cls()

    def _clean_metadata_for_chromadb(
        self,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        清理元数据，只保留 ChromaDB 支持的类型

        ChromaDB 支持: str, int, float, bool
        ChromaDB 不支持: list, dict, None

        Args:
            metadata: 原始元数据

        Returns:
            清理后的元数据
        """
        return {
            k: v for k, v in metadata.items()
            if v is not None and isinstance(v, (str, int, float, bool))
        }

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

    async def index_parsed_document(
        self,
        document: ParsedDocument,
        source_path: Optional[str] = None,
        generate_summaries: Optional[bool] = None
    ) -> list[str]:
        """
        索引已解析的文档（同时同步到 ChromaDB 和 DocStore）

        双重索引策略：
        - chunk.text → 向量化 → ChromaDB
        - chunk.summary → 向量化 → ChromaDB (如果启用摘要)

        Args:
            document: 已解析的文档对象
            source_path: 源文件路径（用于元数据）
            generate_summaries: 是否生成摘要（默认使用配置值）

        Returns:
            文档 ID 列表
        """
        # 确定是否生成摘要
        should_summarize = (
            generate_summaries if generate_summaries is not None
            else self.enable_summarization
        )

        # 如果启用摘要生成
        summary_map = {}  # chunk_index → summary
        if should_summarize:
            min_length = settings.summarization_min_length
            logger.info(f"开始生成摘要（最小长度: {min_length} 字符）...")

            # 筛选需要摘要的 chunks
            chunks_to_summarize = [
                (i, chunk) for i, chunk in enumerate(document.chunks)
                if len(chunk.text) >= min_length
            ]

            logger.info(
                f"需要摘要的 chunk: {len(chunks_to_summarize)}/{len(document.chunks)}"
            )

            # 为每个 chunk 生成摘要（只对长 chunks）
            if chunks_to_summarize:
                texts = [chunk.text for _, chunk in chunks_to_summarize]
                summaries = await self.summarizer.summarize_batch(
                    texts,
                    summary_type="chunk",
                    concurrent=settings.summarization_concurrent
                )

                # 保存摘要映射
                for (idx, chunk), summary in zip(chunks_to_summarize, summaries):
                    if summary:
                        chunk.metadata["summary"] = summary
                        summary_map[idx] = summary
                    else:
                        chunk.metadata["summary"] = None

            # 短 chunk 的 summary 设为 None
            for chunk in document.chunks:
                if "summary" not in chunk.metadata:
                    chunk.metadata["summary"] = None

            logger.info(
                f"摘要生成完成: {len(summary_map)}/{len(document.chunks)} 成功"
            )

        # ========== 双重索引：全文 + 摘要 ==========
        all_chunks_text = []
        all_metadata = []
        docstore_docs = []

        for i, chunk in enumerate(document.chunks):
            raw_metadata = {
                "source": source_path or "unknown",
                "title": document.title,
                "page": chunk.page,
                "section": chunk.section,
                "chunk_index": chunk.index,
                **chunk.metadata
            }

            # 清理元数据给 ChromaDB
            chromadb_metadata = self._clean_metadata_for_chromadb(raw_metadata)
            chromadb_metadata["entry_type"] = "full"  # 标记为全文条目

            # 添加全文条目
            all_chunks_text.append(chunk.text)
            all_metadata.append(chromadb_metadata)

            # DocStore 保留完整元数据（包含所有列表字段）
            docstore_docs.append(DocStoreDocument(
                id=chunk.metadata.get("chunk_id", str(chunk.index)),
                text=chunk.text,
                metadata=raw_metadata,
                prev_id=chunk.metadata.get("prev_chunk_id"),
                next_id=chunk.metadata.get("next_chunk_id"),
                level="leaf"
            ))

            # 如果有摘要，添加摘要条目
            if i in summary_map:
                summary = summary_map[i]
                summary_metadata = chromadb_metadata.copy()
                summary_metadata["entry_type"] = "summary"  # 标记为摘要条目
                summary_metadata["original_chunk_id"] = chunk.metadata.get("chunk_id", str(chunk.index))

                all_chunks_text.append(summary)
                all_metadata.append(summary_metadata)
                # 摘要条目不需要单独的 DocStore entry，关联到原文即可

        # 添加到向量索引（全文 + 摘要）
        doc_ids = await self.indexer.add_documents(
            documents=all_chunks_text,
            collection_name=self.collection_name,
            metadata_list=all_metadata
        )

        # 添加到 DocStore
        await self.docstore.add_many(docstore_docs)
        logger.info(f"DocStore 存储: {len(docstore_docs)} 个文档")

        logger.info(
            f"双重索引完成: {len(all_chunks_text)} 个条目"
            f"（全文 {len(document.chunks)} + 摘要 {len(summary_map)}）"
        )
        return doc_ids

    async def index_document(
        self,
        file_path: str,
        generate_image_descriptions: Optional[bool] = None
    ) -> list[str]:
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
        return await self.index_parsed_document(document, source_path=file_path)

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

    async def _enrich_results_from_docstore(
        self,
        results: list[RetrievedDocument]
    ) -> list[RetrievedDocument]:
        """
        从 DocStore 获取完整元数据并增强检索结果

        Args:
            results: ChromaDB 检索结果（包含有限元数据）

        Returns:
            增强后的结果（包含完整元数据）
        """
        if not results:
            return results

        # 收集所有 chunk_id
        chunk_ids = [r.chunk_id for r in results if r.chunk_id]

        if not chunk_ids:
            return results

        # 批量从 DocStore 获取完整文档
        docstore_docs = await self.docstore.get_many(chunk_ids)
        docstore_map = {doc.id: doc for doc in docstore_docs}

        # 用完整元数据增强每个结果
        for result in results:
            if result.chunk_id and result.chunk_id in docstore_map:
                doc = docstore_map[result.chunk_id]
                # 用完整元数据替换有限元数据
                result.metadata = doc.metadata

        return results

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None
    ) -> list[RetrievedDocument]:
        """
        检索文档（支持双重索引结果合并）

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            检索到的文档列表
        """
        if top_k is None:
            top_k = settings.top_k

        # 获取双重结果（全文 + 摘要），请求更多以便合并
        documents = await self.retriever.retrieve(query, top_k * 2, filters)

        # 合并去重：同一 chunk 的全文和摘要条目，保留分数高的
        documents = self._merge_dual_results(documents, top_k)

        # 从 DocStore 获取完整元数据
        documents = await self._enrich_results_from_docstore(documents)

        # 应用上下文扩展
        if self.context_expander:
            documents = await self.context_expander.expand(documents)

        return documents

    def _merge_dual_results(
        self,
        results: list[RetrievedDocument],
        top_k: int
    ) -> list[RetrievedDocument]:
        """
        合并双重索引结果

        如果同一 chunk 有全文和摘要两个条目，保留分数高的。

        Args:
            results: 原始检索结果
            top_k: 返回数量

        Returns:
            合并后的结果
        """
        # 按 chunk_id 分组
        grouped = defaultdict(list)
        for result in results:
            chunk_id = result.chunk_id

            # 摘要条目：获取 original_chunk_id
            if result.metadata.get("entry_type") == "summary":
                chunk_id = result.metadata.get("original_chunk_id")

            grouped[chunk_id].append(result)

        # 每组保留分数最高的
        merged = []
        for chunk_id, items in grouped.items():
            best = max(items, key=lambda x: x.score)
            merged.append(best)

        # 按分数排序并限制数量
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:top_k]

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
        # 关闭 DocStore 连接
        if self._docstore:
            await self._docstore.close()
        # 清理摘要器
        self._summarizer = None
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
