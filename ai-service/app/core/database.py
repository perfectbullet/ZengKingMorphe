"""
MongoDB database connection and operations.
"""
import asyncio
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import IndexModel, ASCENDING, DESCENDING
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# 最大重试次数
MAX_CONNECT_RETRIES = 3
# 重试延迟（秒）
RETRY_DELAY_SECONDS = 2


class MongoDB:
    """MongoDB connection manager."""

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None

    async def connect(self) -> None:
        """
        Connect to MongoDB with retry logic.

        最多重试 3 次，每次失败后等待 2 秒再重试。
        """
        masked_url = settings.mongodb_url
        if '@' in masked_url and '://' in masked_url:
            # Mask password in URL: mongodb://user:password@host -> mongodb://user:***@host
            parts = masked_url.split('://')
            if len(parts) == 2 and '@' in parts[1]:
                auth_and_host = parts[1].split('@')
                if ':' in auth_and_host[0]:
                    user = auth_and_host[0].split(':')[0]
                    masked_url = f"{parts[0]}://{user}:***@{auth_and_host[1]}"

        for attempt in range(1, MAX_CONNECT_RETRIES + 1):
            try:
                self.client = AsyncIOMotorClient(
                    settings.mongodb_url,
                    maxPoolSize=settings.mongodb_max_pool_size,
                    minPoolSize=settings.mongodb_min_pool_size,
                    serverSelectionTimeoutMS=5000,  # 5秒超时
                )
                self.db = self.client[settings.mongodb_db_name]

                # Test connection
                await self.client.admin.command('ping')
                logger.info(
                    f"Connected to MongoDB: database={settings.mongodb_db_name}, "
                    f"at {masked_url}, attempt={attempt}/{MAX_CONNECT_RETRIES}"
                )

                # Create indexes
                await self._create_indexes()

                return  # 连接成功，退出

            except Exception as e:
                if attempt < MAX_CONNECT_RETRIES:
                    logger.warning(
                        f"MongoDB connection failed (attempt {attempt}/{MAX_CONNECT_RETRIES}), "
                        f"retrying in {RETRY_DELAY_SECONDS}s... | error={str(e)}",
                        mongodb_url=masked_url,
                        database=settings.mongodb_db_name,
                    )
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                else:
                    # 最后一次尝试也失败了
                    logger.error(
                        f"MongoDB connection failed after {MAX_CONNECT_RETRIES} attempts | "
                        f"error={str(e)}",
                        mongodb_url=masked_url,
                        database=settings.mongodb_db_name,
                        max_pool_size=settings.mongodb_max_pool_size,
                        min_pool_size=settings.mongodb_min_pool_size,
                        exc_info=True
                    )
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
            
            # Employee configs collection indexes (legacy, keeping for backward compatibility)
            await self.db.digital_employee_configs.create_indexes([
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

            # Knowledge bases collection indexes
            await self.db.knowledge_bases.create_indexes([
                IndexModel([("kb_id", ASCENDING)], unique=True),
                IndexModel([("name", ASCENDING)]),  # 按名称检索
                IndexModel([("employee_id", ASCENDING)]),
                IndexModel([("status", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)]),
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

            # Raw stream tokens collection indexes
            await self.db.raw_stream_tokens.create_indexes([
                IndexModel([("token_id", ASCENDING)], unique=True),
                IndexModel([("chat_id", ASCENDING), ("token_index", ASCENDING)]),
                IndexModel([("session_id", ASCENDING), ("created_at", DESCENDING)]),
                IndexModel([("streaming_source", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)]),
            ])
            
            # FAQs collection indexes
            await self.db.faqs.create_indexes([
                IndexModel([("faq_id", ASCENDING)], unique=True),
                IndexModel([("employee_id", ASCENDING), ("is_enable", ASCENDING)]),
                IndexModel([("employee_id", ASCENDING), ("update_time", DESCENDING)]),
                IndexModel([("external_faq_id", ASCENDING)]),
                IndexModel([("vector_id", ASCENDING)]),
                IndexModel([("synced_at", DESCENDING)]),
            ])
            
            # Digital employee configs collection indexes
            await self.db.digital_employee_configs.create_indexes([
                IndexModel([("employee_id", ASCENDING)], unique=True),
                IndexModel([("external_employee_id", ASCENDING)]),
                IndexModel([("team_id", ASCENDING)]),
                IndexModel([("onduty_status", ASCENDING)]),
            ])
            
            # Document tasks collection indexes
            await self.db.document_tasks.create_indexes([
                IndexModel([("task_id", ASCENDING)], unique=True),
                IndexModel([("kb_id", ASCENDING), ("status", ASCENDING)]),
                IndexModel([("status", ASCENDING), ("created_at", DESCENDING)]),
            ])

            # Major collection indexes
            await self.db.thesaurus_major.create_indexes([
                IndexModel([("thesaurus_id", ASCENDING)], unique=True),
                IndexModel([("employee_id", ASCENDING), ("is_enable", ASCENDING)]),
                IndexModel([("employee_id", ASCENDING), ("update_time", DESCENDING)]),
                IndexModel([("external_thesaurus_id", ASCENDING)]),
                IndexModel([("vector_id", ASCENDING)]),
                IndexModel([("synced_at", DESCENDING)]),
            ])

            # Sensitive collection indexes
            await self.db.thesaurus_sensitive.create_indexes([
                IndexModel([("thesaurus_id", ASCENDING)], unique=True),
                IndexModel([("employee_id", ASCENDING), ("is_enable", ASCENDING)]),
                IndexModel([("employee_id", ASCENDING), ("update_time", DESCENDING)]),
                IndexModel([("external_thesaurus_id", ASCENDING)]),
                IndexModel([("vector_id", ASCENDING)]),
                IndexModel([("synced_at", DESCENDING)]),
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

    如果数据库未连接，自动尝试连接。

    Returns:
        Database instance
    """
    if mongodb.db is None:
        logger.info("Database not connected, attempting to connect...")
        await mongodb.connect()
    return mongodb.db
