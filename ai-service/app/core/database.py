"""
MongoDB database connection and operations.
"""
from typing import Optional, List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import IndexModel, ASCENDING, DESCENDING
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class MongoDB:
    """MongoDB connection manager."""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
    
    async def connect(self) -> None:
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(
                settings.mongodb_url,
                maxPoolSize=settings.mongodb_max_pool_size,
                minPoolSize=settings.mongodb_min_pool_size,
            )
            self.db = self.client[settings.mongodb_db_name]
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB: database={settings.mongodb_db_name}")
            
            # Create indexes
            await self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: error={str(e)}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def _create_indexes(self) -> None:
        """Create database indexes."""
        try:
            # Conversations collection indexes
            await self.db.conversations.create_indexes([
                IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)]),
                IndexModel([("employee_id", ASCENDING), ("created_at", DESCENDING)]),
                IndexModel([("session_id", ASCENDING), ("created_at", DESCENDING)]),
                IndexModel([("intent", ASCENDING), ("created_at", DESCENDING)]),
                IndexModel([("is_realtime_query", ASCENDING), ("created_at", DESCENDING)]),
                IndexModel([("created_at", DESCENDING)]),
            ])
            
            # Sessions collection indexes
            await self.db.sessions.create_indexes([
                IndexModel([("session_id", ASCENDING)], unique=True),
                IndexModel([("user_id", ASCENDING), ("last_activity", DESCENDING)]),
                IndexModel([("employee_id", ASCENDING), ("status", ASCENDING)]),
            ])
            
            # Employee configs collection indexes
            await self.db.employee_configs.create_indexes([
                IndexModel([("employee_id", ASCENDING)], unique=True),
                IndexModel([("domain", ASCENDING)]),
                IndexModel([("status", ASCENDING)]),
            ])
            
            # User profiles collection indexes
            await self.db.user_profiles.create_indexes([
                IndexModel([("user_id", ASCENDING)], unique=True),
            ])
            
            # Documents collection indexes
            await self.db.documents.create_indexes([
                IndexModel([("doc_id", ASCENDING)], unique=True),
                IndexModel([("kb_id", ASCENDING), ("status", ASCENDING)]),
                IndexModel([("uploaded_at", DESCENDING)]),
            ])
            
            # Document chunks collection indexes
            await self.db.document_chunks.create_indexes([
                IndexModel([("chunk_id", ASCENDING)], unique=True),
                IndexModel([("doc_id", ASCENDING), ("chunk_index", ASCENDING)]),
                IndexModel([("kb_id", ASCENDING)]),
            ])
            
            # Intent logs collection indexes
            await self.db.intent_logs.create_indexes([
                IndexModel([("conversation_id", ASCENDING)]),
                IndexModel([("employee_id", ASCENDING), ("created_at", DESCENDING)]),
                IndexModel([("recognized_intent", ASCENDING), ("created_at", DESCENDING)]),
            ])
            
            # KB quality metrics collection indexes
            await self.db.kb_quality_metrics.create_indexes([
                IndexModel([("kb_id", ASCENDING), ("date", DESCENDING)]),
                IndexModel([("employee_id", ASCENDING), ("date", DESCENDING)]),
                IndexModel([("date", DESCENDING)]),
            ])
            
            # Stream chunks collection indexes
            await self.db.stream_chunks.create_indexes([
                IndexModel([("chunk_id", ASCENDING)], unique=True),
                IndexModel([("session_id", ASCENDING), ("sequence", ASCENDING)]),
                IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)]),
                IndexModel([("employee_id", ASCENDING), ("created_at", DESCENDING)]),
                IndexModel([("chat_id", ASCENDING), ("sequence", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)]),
            ])
            
            logger.info("Created MongoDB indexes")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: error={str(e)}")
            raise
    
    def get_collection(self, name: str):
        """Get a collection by name."""
        if self.db is None:
            raise RuntimeError("Database not connected")
        return self.db[name]


# Global MongoDB instance
mongodb = MongoDB()


async def get_database() -> AsyncIOMotorDatabase:
    """
    Get database instance.
    
    Returns:
        Database instance
    """
    if mongodb.db is None:
        raise RuntimeError("Database not connected")
    return mongodb.db
