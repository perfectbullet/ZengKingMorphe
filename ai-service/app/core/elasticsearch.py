"""
ElasticSearch connection and operations.
"""
from typing import List, Dict, Any, Optional
from elasticsearch import AsyncElasticsearch
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ElasticSearchDB:
    """ElasticSearch database manager."""
    
    def __init__(self):
        self.client: Optional[AsyncElasticsearch] = None
        self.faq_index = f"{settings.es_index_prefix}_faq"
        self.doc_index = f"{settings.es_index_prefix}_doc"
    
    async def connect(self) -> None:
        """Connect to ElasticSearch."""
        try:
            # Create client
            es_config = {"hosts": [settings.es_url]}
            if settings.es_username and settings.es_password:
                es_config["basic_auth"] = (settings.es_username, settings.es_password)
            
            self.client = AsyncElasticsearch(**es_config)
            
            # Test connection
            info = await self.client.info()
            logger.info(f"Connected to ElasticSearch: version={info['version']['number']}")
            
            # Create indexes
            await self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to ElasticSearch: error={str(e)}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from ElasticSearch."""
        if self.client:
            await self.client.close()
            logger.info("Disconnected from ElasticSearch")
    
    async def _create_indexes(self) -> None:
        """Create ElasticSearch indexes."""
        try:
            # FAQ index mapping
            faq_mapping = {
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "ik_max_word_analyzer": {
                                "type": "standard"  # Use standard analyzer if IK not available
                            },
                            "ik_smart_analyzer": {
                                "type": "standard"
                            }
                        }
                    },
                    "number_of_shards": 3,
                    "number_of_replicas": 1
                },
                "mappings": {
                    "properties": {
                        "faq_id": {"type": "keyword"},
                        "question": {
                            "type": "text",
                            "analyzer": "ik_max_word_analyzer",
                            "search_analyzer": "ik_smart_analyzer",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "answer": {
                            "type": "text",
                            "analyzer": "ik_max_word_analyzer"
                        },
                        "keywords": {"type": "keyword"},
                        "category": {"type": "keyword"},
                        "kb_id": {"type": "keyword"},
                        "status": {"type": "keyword"},
                        "priority": {"type": "integer"},
                        "created_at": {"type": "date"},
                        "updated_at": {"type": "date"},
                        "boost_score": {"type": "float"}
                    }
                }
            }
            
            # Create FAQ index if not exists
            if not await self.client.indices.exists(index=self.faq_index):
                await self.client.indices.create(
                    index=self.faq_index,
                    body=faq_mapping
                )
                logger.info(f"Created FAQ index: index={self.faq_index}")
            
            # Document index mapping
            doc_mapping = {
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "ik_max_word_analyzer": {"type": "standard"},
                            "ik_smart_analyzer": {"type": "standard"}
                        }
                    },
                    "number_of_shards": 3,
                    "number_of_replicas": 1
                },
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "kb_id": {"type": "keyword"},
                        "content": {
                            "type": "text",
                            "analyzer": "ik_max_word_analyzer",
                            "search_analyzer": "ik_smart_analyzer"
                        },
                        "summary": {"type": "text"},
                        "chunk_index": {"type": "integer"},
                        "created_at": {"type": "date"}
                    }
                }
            }
            
            # Create document index if not exists
            if not await self.client.indices.exists(index=self.doc_index):
                await self.client.indices.create(
                    index=self.doc_index,
                    body=doc_mapping
                )
                logger.info(f"Created document index: index={self.doc_index}")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: error={str(e)}")
            raise
    
    async def index_document(
        self,
        index: str,
        doc_id: str,
        document: Dict[str, Any]
    ) -> None:
        """
        Index a document.
        
        Args:
            index: Index name (faq/doc)
            doc_id: Document ID
            document: Document data
        """
        try:
            index_name = self.faq_index if index == "faq" else self.doc_index
            await self.client.index(
                index=index_name,
                id=doc_id,
                document=document
            )
            logger.info(f"Indexed document: index={index_name}, doc_id={doc_id}")
        except Exception as e:
            logger.error(f"Failed to index document: index={index}, doc_id={doc_id}, error={str(e)}")

            raise
    
    async def search(
        self,
        index: str,
        query: Dict[str, Any],
        size: int = 10
    ) -> Dict[str, Any]:
        """
        Search documents.
        
        Args:
            index: Index name (faq/doc)
            query: Search query
            size: Number of results
            
        Returns:
            Search results
        """
        try:
            index_name = self.faq_index if index == "faq" else self.doc_index
            results = await self.client.search(
                index=index_name,
                body=query,
                size=size
            )
            logger.info(f"Searched documents: index={index_name}, hits={results['hits']['total']['value']}")
            return results
        except Exception as e:
            logger.error(f"Failed to search documents: index={index}, error={str(e)}")
            raise
    
    async def delete_document(
        self,
        index: str,
        doc_id: str
    ) -> None:
        """
        Delete a document.

        Args:
            index: Index name (faq/doc)
            doc_id: Document ID
        """
        try:
            index_name = self.faq_index if index == "faq" else self.doc_index
            await self.client.delete(
                index=index_name,
                id=doc_id
            )
            logger.info(f"Deleted document: index={index_name}, doc_id={doc_id}")
        except Exception as e:
            logger.error(f"Failed to delete document: index={index}, doc_id={doc_id}, error={str(e)}")

            raise

    async def delete_by_query(
        self,
        index: str,
        body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Delete documents by query.

        Args:
            index: Index name (faq/doc)
            body: Delete query body

        Returns:
            Delete result
        """
        try:
            index_name = self.faq_index if index == "faq" else self.doc_index
            result = await self.client.delete_by_query(
                index=index_name,
                body=body
            )
            logger.info(f"Deleted by query: index={index_name}, deleted={result.get('deleted', 0)}")
            return result
        except Exception as e:
            logger.error(f"Failed to delete by query: index={index}, error={str(e)}")
            raise


# Global ElasticSearch instance
es_db = ElasticSearchDB()
