"""
Chroma vector database connection and operations.
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from app.core.config import settings
from app.core.logging import get_logger
from app.utils.embeddings import (
    OpenAIStyleEmbeddings,
    ChromaEmbeddingWrapper,
)

logger = get_logger(__name__)


class ChromaDB:
    """Chroma vector database manager."""

    def __init__(self):
        self.client: Optional[chromadb.Client] = None
        self.faq_collection = None
        self.doc_collection = None
        self.major_collection = None
        self.sensitive_collection = None

    def connect(self) -> None:
        """Connect to Chroma."""
        try:
            # Initialize Chroma client using REST API (Chroma >=0.5)
            logger.info(
                f"Initializing Chroma client: host={settings.chroma_host}, port={settings.chroma_port}"
            )

            # Create HttpClient with telemetry disabled
            # Use tenant and database to avoid telemetry issues
            self.client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                tenant="default_tenant",
                database="default_database",
            )

            # Create embedding function based on SDK configuration
            # SDK 使用 vLLM embeddings (OpenAI 兼容 API)
            base_url = os.getenv("VLLM_EMBEDDING_BASE_URL", settings.embedding_base_url)
            model = os.getenv("VLLM_EMBEDDING_MODEL", settings.embedding_model)
            api_key = os.getenv("VLLM_API_KEY", settings.embedding_api_key or "not-needed")

            logger.info(
                f"Creating embedding function: model={model}, base_url={base_url}"
            )

            embedder = OpenAIStyleEmbeddings(
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout=30.0,
            )
            logger.info(
                f"Using vLLM embeddings: model={model}, base_url={base_url}"
            )

            embedding_function = ChromaEmbeddingWrapper(embedder)

            # Get or create collections
            self.faq_collection = self.client.get_or_create_collection(
                name="faq_knowledge",
                metadata={
                    "description": "FAQ问答库",
                    "embedding_model": settings.embedding_model,
                    "hnsw:space": "cosine",  # ✅ 使用余弦距离
                },
                embedding_function=embedding_function,
            )

            self.doc_collection = self.client.get_or_create_collection(
                name="rag_documents",
                metadata={
                    "description": "RAG文档库",
                    "embedding_model": settings.embedding_model,
                    "hnsw:space": "cosine",  # ✅ 使用余弦距离
                },
                embedding_function=embedding_function,
            )

            self.major_collection = self.client.get_or_create_collection(
                name="thesaurus_major",
                metadata={
                    "description": "专业词库",
                    "embedding_model": settings.embedding_model,
                    "hnsw:space": "cosine",  # ✅ 使用余弦距离
                },
                embedding_function=embedding_function,
            )

            self.sensitive_collection = self.client.get_or_create_collection(
                name="thesaurus_sensitive",
                metadata={
                    "description": "敏感词库",
                    "embedding_model": settings.embedding_model,
                    "hnsw:space": "cosine",  # ✅ 使用余弦距离
                },
                embedding_function=embedding_function,
            )

            logger.info(
                f"Connected to Chroma: host={settings.chroma_host}, port={settings.chroma_port}"
            )

        except Exception as e:
            logger.error(f"Failed to connect to Chroma: error={str(e)}")
            raise

    def disconnect(self) -> None:
        """Disconnect from Chroma."""
        if self.client:
            # HttpClient (远程连接) 不支持 persist() 方法
            # ChromaDB 服务器会自动持久化数据,无需手动调用
            # 只需要清空客户端引用即可
            self.client = None
            logger.info("Disconnected from Chroma")

    async def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        """
        Add documents to a collection.

        Args:
            collection_name: Collection name (faq/doc/major/sensitive)
            documents: List of document texts
            metadatas: List of metadata dicts
            ids: List of document IDs
        """
        try:
            collection = self._get_collection(collection_name)
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(
                f"Added documents to Chroma: collection={collection_name}, count={len(documents)}"
            )
        except Exception as e:
            logger.error(
                f"Failed to add documents to Chroma: collection={collection_name}, error={str(e)}"
            )
            raise

    async def query_documents(
        self,
        collection_name: str,
        query_texts: List[str],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query documents from a collection.

        Args:
            collection_name: Collection name (faq/doc/major/sensitive)
            query_texts: List of query texts
            n_results: Number of results to return
            where: Filter conditions

        Returns:
            Query results
        """
        try:
            collection = self._get_collection(collection_name)
            results = collection.query(
                query_texts=query_texts, n_results=n_results, where=where
            )
            # Truncate documents for logging (only show first 50 chars)
            truncated_results = results.copy()
            if "documents" in truncated_results and truncated_results["documents"]:
                truncated_docs = [
                    [doc[:50] + "..." if len(doc) > 50 else doc for doc in doc_list]
                    for doc_list in truncated_results["documents"]
                ]
                truncated_results["documents"] = truncated_docs
            if truncated_results:
                logger.info(
                    f"Queried documents from Chroma: collection={collection_name}, results len={len(truncated_results)}"
                )
            else:
                logger.info("Queried documents from Chroma, but the truncated_results is empty")
            return results
        except Exception as e:
            logger.error(
                f"Failed to query documents from Chroma: collection={collection_name}, error={str(e)}"
            )
            logger.exception(e)
            raise

    async def delete_documents(self, collection_name: str, ids: List[str]) -> None:
        """
        Delete documents from a collection.

        Args:
            collection_name: Collection name (faq/doc/major/sensitive)
            ids: List of document IDs to delete
        """
        try:
            collection = self._get_collection(collection_name)
            collection.delete(ids=ids)
            logger.info(
                f"Deleted documents from Chroma: collection={collection_name}, count={len(ids)}"
            )
        except Exception as e:
            logger.error(
                f"Failed to delete documents from Chroma: collection={collection_name}, error={str(e)}"
            )
            raise

    def _get_collection(self, collection_name: str):
        """Get collection by name."""
        if collection_name == "faq":
            return self.faq_collection
        elif collection_name == "doc":
            return self.doc_collection
        elif collection_name == "major":
            return self.major_collection
        elif collection_name == "sensitive":
            return self.sensitive_collection
        else:
            raise ValueError(f"Unknown collection: {collection_name}")


# Global Chroma instance
chroma_db = ChromaDB()
