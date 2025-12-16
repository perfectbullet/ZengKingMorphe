"""
Chroma vector database connection and operations.
"""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from app.core.config import settings
from app.core.logging import get_logger

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
            # Initialize Chroma client
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=settings.chroma_persist_dir
            ))
            
            # Create embedding function
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.openai_api_key,
                model_name=settings.openai_embedding_model,
                api_base=settings.openai_api_base
            )
            
            # Get or create collections
            self.faq_collection = self.client.get_or_create_collection(
                name="faq_knowledge",
                metadata={
                    "description": "FAQ问答库",
                    "embedding_model": settings.openai_embedding_model
                },
                embedding_function=openai_ef
            )
            
            self.doc_collection = self.client.get_or_create_collection(
                name="rag_documents",
                metadata={
                    "description": "RAG文档库",
                    "embedding_model": settings.openai_embedding_model
                },
                embedding_function=openai_ef
            )
            
            self.dict_collection = self.client.get_or_create_collection(
                name="custom_dictionary",
                metadata={
                    "description": "专业词库",
                    "embedding_model": settings.openai_embedding_model
                },
                embedding_function=openai_ef
            )
            
            logger.info("Connected to Chroma", persist_dir=settings.chroma_persist_dir)
            
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
        ids: List[str]
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
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(
                "Added documents to Chroma",
                collection=collection_name,
                count=len(documents)
            )
        except Exception as e:
            logger.error(
                "Failed to add documents to Chroma",
                collection=collection_name,
                error=str(e)
            )
            raise
    
    async def query_documents(
        self,
        collection_name: str,
        query_texts: List[str],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
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
                query_texts=query_texts,
                n_results=n_results,
                where=where
            )
            logger.info(
                "Queried documents from Chroma",
                collection=collection_name,
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(
                "Failed to query documents from Chroma",
                collection=collection_name,
                error=str(e)
            )
            raise
    
    async def delete_documents(
        self,
        collection_name: str,
        ids: List[str]
    ) -> None:
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
                count=len(ids)
            )
        except Exception as e:
            logger.error(
                "Failed to delete documents from Chroma",
                collection=collection_name,
                error=str(e)
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
