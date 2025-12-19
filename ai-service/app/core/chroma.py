"""
Chroma vector database connection and operations.
"""

from typing import List, Dict, Any, Optional
import chromadb
from app.core.config import settings
from app.core.logging import get_logger
from app.utils.embeddings import (
    OpenAIStyleEmbeddings,
    SiliconFlowEmbeddings,
    ChromaEmbeddingWrapper,
    OllamaEmbeddings,
)

logger = get_logger(__name__)


class ChromaDB:
    """Chroma vector database manager."""

    def __init__(self):
        self.client: Optional[chromadb.Client] = None
        self.faq_collection = None
        self.doc_collection = None
        self.dict_collection = None

    def connect(self) -> None:
        """Connect to Chroma."""
        try:
            # Initialize Chroma client using REST API (Chroma >=0.5)
            logger.info(
                "Initializing Chroma client",
                host=settings.chroma_host,
                port=settings.chroma_port
            )
            
            # Create settings with telemetry disabled to avoid signature mismatch errors
            chroma_settings = chromadb.config.Settings(
                chroma_api_impl="chromadb.api.fastapi.FastAPI",
                chroma_server_host=settings.chroma_host,
                chroma_server_http_port=settings.chroma_port,
                anonymized_telemetry=False
            )
            
            self.client = chromadb.HttpClient(
                host=settings.chroma_host, 
                port=settings.chroma_port,
                settings=chroma_settings
            )

            # Create embedding function based on configuration
            logger.info(
                "Creating embedding function",
                embedding_type=settings.embedding_type,
                embedding_model=settings.embedding_model,
                embedding_base_url=settings.embedding_base_url,
                embedding_api_url=settings.embedding_api_url
            )
            
            if settings.embedding_type == "siliconflow":
                embedder = SiliconFlowEmbeddings(
                    model=settings.embedding_model,
                    api_key=settings.embedding_api_key,
                    base_url=settings.siliconflow_embedding_api_url,
                    max_tokens=512  # BGE model limit
                )
                logger.info(
                    "Using SiliconFlow embeddings",
                    model=settings.embedding_model,
                    base_url=settings.siliconflow_embedding_api_url,
                    has_api_key=bool(settings.siliconflow_api_key or settings.embedding_api_key)
                )
            elif settings.embedding_type == "ollama":
                embedder = OllamaEmbeddings(
                    model=settings.embedding_ollama_model,
                    base_url=settings.ollama_base_url,
                    max_tokens=512  # BGE model limit
                )
                logger.info(
                    "Using Ollama embeddings",
                    model=settings.embedding_ollama_model,
                    base_url=settings.ollama_base_url
                )
            else:
                # Default to OpenAI-style embeddings
                embedder = OpenAIStyleEmbeddings(
                    model=settings.embedding_model,
                    base_url=settings.embedding_base_url,
                    api_key=settings.embedding_api_key,
                    timeout=30.0,
                )
                logger.info(
                    "Using OpenAI-style embeddings",
                    model=settings.embedding_model,
                    base_url=settings.embedding_base_url,
                    has_api_key=bool(settings.embedding_api_key)
                )

            embedding_function = ChromaEmbeddingWrapper(embedder)

            # Get or create collections
            self.faq_collection = self.client.get_or_create_collection(
                name="faq_knowledge",
                metadata={
                    "description": "FAQ问答库",
                    "embedding_model": settings.embedding_model,
                },
                embedding_function=embedding_function,
            )

            self.doc_collection = self.client.get_or_create_collection(
                name="rag_documents",
                metadata={
                    "description": "RAG文档库",
                    "embedding_model": settings.embedding_model,
                },
                embedding_function=embedding_function,
            )

            self.dict_collection = self.client.get_or_create_collection(
                name="custom_dictionary",
                metadata={
                    "description": "专业词库",
                    "embedding_model": settings.embedding_model,
                },
                embedding_function=embedding_function,
            )

            logger.info(
                "Connected to Chroma",
                host=settings.chroma_host,
                port=settings.chroma_port,
            )

        except Exception as e:
            logger.error("Failed to connect to Chroma", error=str(e))
            raise

    def disconnect(self) -> None:
        """Disconnect from Chroma."""
        if self.client:
            # Persist data
            self.client.persist()
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
            collection_name: Collection name (faq/doc/dict)
            documents: List of document texts
            metadatas: List of metadata dicts
            ids: List of document IDs
        """
        try:
            collection = self._get_collection(collection_name)
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(
                "Added documents to Chroma",
                collection=collection_name,
                count=len(documents),
            )
        except Exception as e:
            logger.error(
                "Failed to add documents to Chroma",
                collection=collection_name,
                error=str(e),
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
            collection_name: Collection name (faq/doc/dict)
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
            logger.info(
                "Queried documents from Chroma",
                collection=collection_name,
                n_results=n_results,
            )
            return results
        except Exception as e:
            logger.error(
                "Failed to query documents from Chroma",
                collection=collection_name,
                error=str(e),
            )
            logger.exception(e)
            raise

    async def delete_documents(self, collection_name: str, ids: List[str]) -> None:
        """
        Delete documents from a collection.

        Args:
            collection_name: Collection name (faq/doc/dict)
            ids: List of document IDs to delete
        """
        try:
            collection = self._get_collection(collection_name)
            collection.delete(ids=ids)
            logger.info(
                "Deleted documents from Chroma",
                collection=collection_name,
                count=len(ids),
            )
        except Exception as e:
            logger.error(
                "Failed to delete documents from Chroma",
                collection=collection_name,
                error=str(e),
            )
            raise

    def _get_collection(self, collection_name: str):
        """Get collection by name."""
        if collection_name == "faq":
            return self.faq_collection
        elif collection_name == "doc":
            return self.doc_collection
        elif collection_name == "dict":
            return self.dict_collection
        else:
            raise ValueError(f"Unknown collection: {collection_name}")


# Global Chroma instance
chroma_db = ChromaDB()
