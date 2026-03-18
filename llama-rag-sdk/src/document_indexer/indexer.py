"""
文档索引器

整合分块、向量化和存储功能
"""

import uuid
from typing import List, Dict, Any, Optional
from loguru import logger

from src.document_indexer.base import Indexer, ChunkStrategy
from src.document_indexer.chunker import (
    FixedSizeChunker,
    SemanticChunker,
    HybridChunker
)
from src.document_indexer.storage import VectorStore
from src.config import settings

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
except ImportError:
    logger.warning("llama-index-embeddings-openai 未安装，embedding 功能不可用")
    OpenAIEmbedding = None


class DocumentIndexer(Indexer):
    """文档索引器"""

    def __init__(
        self,
        chunk_strategy: Optional[ChunkStrategy] = None,
        embedding_model: Optional[str] = None,
        api_base: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        """
        初始化文档索引器

        Args:
            chunk_strategy: 分块策略
            embedding_model: Embedding 模型名称
            api_base: vLLM Embedding API 地址
            collection_name: ChromaDB 集合名称
        """
        self.chunk_strategy = chunk_strategy or ChunkStrategy(
            type="fixed",
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        self.embedding_model = embedding_model or settings.vllm_embedding_model
        self.api_base = api_base or settings.vllm_embedding_api_base

        # 初始化分块器
        self.chunker = self._create_chunker()

        # 初始化向量存储
        self.vector_store = VectorStore(collection_name=collection_name)

        # 初始化 Embedding 模型
        self.embed_model = self._create_embedding_model()

    def _create_chunker(self) -> Any:
        """创建分块器"""
        strategy_type = self.chunk_strategy.type

        if strategy_type == "fixed":
            return FixedSizeChunker(self.chunk_strategy)
        elif strategy_type == "semantic":
            return SemanticChunker(self.chunk_strategy)
        elif strategy_type == "hybrid":
            return HybridChunker(self.chunk_strategy)
        else:
            logger.warning(f"未知的分块策略: {strategy_type}，使用固定大小分块")
            return FixedSizeChunker(self.chunk_strategy)

    def _create_embedding_model(self) -> Optional[Any]:
        """创建 Embedding 模型"""
        if OpenAIEmbedding is None:
            logger.error("OpenAIEmbedding 不可用，请安装 llama-index-embeddings-openai")
            return None

        try:
            embed_model = OpenAIEmbedding(
                model_name=self.embedding_model,
                api_base=self.api_base,
                api_key=settings.vllm_api_key,
                embed_batch_size=32,
                timeout=300,
            )
            logger.info(f"Embedding 模型初始化成功: {self.embedding_model} @ {self.api_base}")
            return embed_model
        except Exception as e:
            logger.error(f"初始化 Embedding 模型失败: {e}")
            return None

    def _chunk_documents(
        self,
        documents: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        分块文档

        Args:
            documents: 文档内容列表
            metadata_list: 元数据列表

        Returns:
            (分块后的文档列表, 元数据列表)
        """
        chunks = []
        metadatas = []

        for i, doc in enumerate(documents):
            base_metadata = metadata_list[i] if metadata_list else {}

            # 分块
            doc_chunks = self.chunker.chunk(doc)

            for j, chunk in enumerate(doc_chunks):
                chunks.append(chunk)

                # 合并元数据
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_id": f"{uuid.uuid4()}",
                    "chunk_index": j,
                    "total_chunks": len(doc_chunks)
                })
                metadatas.append(chunk_metadata)

        logger.info(f"文档分块完成: {len(documents)} 个文档 -> {len(chunks)} 个块")
        return chunks, metadatas

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        生成嵌入向量

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        if self.embed_model is None:
            raise RuntimeError("Embedding 模型未初始化")

        try:
            # 批量调用 vLLM API
            embeddings = self.embed_model.get_text_embedding_batch(texts)
            logger.debug(f"批量生成 {len(embeddings)} 个嵌入向量")
            return embeddings
        except Exception as e:
            logger.warning(f"批量 embedding 失败: {e}，改用逐个调用")
            embeddings = []
            failed_count = 0
            for i, text in enumerate(texts):
                # 跳过空文本或过短文本
                if not text or len(text.strip()) < 3:
                    logger.warning(f"跳过空文本或过短文本 (索引 {i})")
                    embeddings.append(None)  # 保持索引对应关系
                    failed_count += 1
                    continue

                # 清理文本：移除多余空白和控制字符
                cleaned_text = text.strip()
                if not cleaned_text:
                    logger.warning(f"跳过空白文本 (索引 {i})")
                    embeddings.append(None)
                    failed_count += 1
                    continue

                try:
                    emb = self.embed_model.get_text_embedding(cleaned_text)
                    embeddings.append(emb)
                except Exception as e:
                    logger.warning(f"单个 embedding 失败 (索引 {i}): {e}, 文本长度: {len(cleaned_text)}")
                    embeddings.append(None)
                    failed_count += 1

                if (i + 1) % 10 == 0:
                    logger.debug(f"已处理 {i + 1}/{len(texts)} 个嵌入向量")

            logger.info(f"逐个生成完成: 成功 {len(embeddings) - failed_count}/{len(embeddings)}, 失败 {failed_count}")

            # 移除失败的 embedding，保持 ids、embeddings、documents 对应
            # ChromaDB 可以处理 None embedding，但为了安全起见使用零向量
            zero_emb = [0.0] * 1024  # bge-m3 默认维度
            embeddings = [emb if emb is not None else zero_emb for emb in embeddings]

            return embeddings
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            raise

    async def create_index(
        self,
        documents: List[str],
        collection_name: Optional[str] = None
    ) -> str:
        """
        创建索引

        Args:
            documents: 文档内容列表
            collection_name: 集合名称

        Returns:
            索引 ID（使用集合名称）
        """
        logger.info(f"开始创建索引，文档数量: {len(documents)}")

        if collection_name:
            self.vector_store = VectorStore(collection_name=collection_name)

        # 分块
        chunks, metadatas = self._chunk_documents(documents)

        # 生成嵌入向量
        embeddings = self._generate_embeddings(chunks)

        # 生成文档 ID
        doc_ids = [str(uuid.uuid4()) for _ in chunks]

        # 添加到向量存储
        self.vector_store.add(
            ids=doc_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )

        logger.info(f"索引创建完成: {self.vector_store.collection_name}")
        return self.vector_store.collection_name

    async def add_documents(
        self,
        documents: List[str],
        collection_name: str,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        添加文档到索引

        Args:
            documents: 文档内容列表
            collection_name: 集合名称
            metadata_list: 元数据列表

        Returns:
            文档 ID 列表
        """
        # 确保使用指定的集合
        if collection_name != self.vector_store.collection_name:
            self.vector_store = VectorStore(collection_name=collection_name)

        logger.info(f"开始添加文档，数量: {len(documents)}")

        # 分块
        chunks, metadatas = self._chunk_documents(documents, metadata_list)

        # 生成嵌入向量
        embeddings = self._generate_embeddings(chunks)

        # 生成文档 ID
        doc_ids = [str(uuid.uuid4()) for _ in chunks]

        # 添加到向量存储
        self.vector_store.add(
            ids=doc_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )

        logger.info(f"文档添加完成: {len(doc_ids)} 个块")
        return doc_ids

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
        # 确保使用指定的集合
        if collection_name != self.vector_store.collection_name:
            self.vector_store = VectorStore(collection_name=collection_name)

        try:
            self.vector_store.delete(ids=document_ids)
            logger.info(f"文档删除成功: {len(document_ids)} 个")
            return True
        except Exception as e:
            logger.error(f"文档删除失败: {e}")
            return False

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
        # 确保使用指定的集合
        if collection_name != self.vector_store.collection_name:
            self.vector_store = VectorStore(collection_name=collection_name)

        return self.vector_store.get_stats()
