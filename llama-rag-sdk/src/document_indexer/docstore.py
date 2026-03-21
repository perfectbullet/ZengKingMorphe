"""
DocStore - MongoDB 文档存储模块

用于存储和检索文档的完整内容和元数据，与向量存储（Vector Store）配合使用。

作用：
1. 按 ID 快速获取文档
2. 支持复杂的关联查询（父节点、子节点）
3. 存储完整的内容和元数据
"""

from typing import Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class DocStoreDocument:
    """DocStore 文档数据模型"""
    id: str                            # 文档 ID（chunk_id）
    text: str                          # 文档文本内容
    metadata: dict[str, Any]           # 元数据
    parent_id: Optional[str] = None    # 父节点 ID
    child_ids: list[str] = None        # 子节点 ID 列表
    prev_id: Optional[str] = None      # 前一个节点 ID
    next_id: Optional[str] = None      # 后一个节点 ID
    level: Optional[str] = None        # 层级（chapter/section/leaf）

    def __post_init__(self):
        if self.child_ids is None:
            self.child_ids = []

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocStoreDocument":
        """从字典创建实例"""
        return cls(**data)


class MongoDBDocStore:
    """
    MongoDB DocStore 实现

    用于存储文档完整内容和元数据，支持按 ID 快速查询。
    """

    def __init__(
        self,
        uri: str,
        db_name: str,
        collection_name: str = "docstore"
    ):
        """
        初始化 MongoDB DocStore

        Args:
            uri: MongoDB 连接 URI
            db_name: 数据库名称
            collection_name: 集合名称
        """
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    async def _get_collection(self):
        """获取 MongoDB 集合"""
        if self._collection is None:
            try:
                from motor.motor_asyncio import AsyncIOMotorClient
                self._client = AsyncIOMotorClient(self.uri)
                db = self._client[self.db_name]
                self._collection = db[self.collection_name]

                # 创建索引
                await self._collection.create_index([("id", 1)], unique=True)
                await self._collection.create_index([("parent_id", 1)])
                await self._collection.create_index([("prev_id", 1)])
                await self._collection.create_index([("next_id", 1)])
                await self._collection.create_index([("level", 1)])

                logger.info(f"MongoDB DocStore 连接成功: {self.db_name}.{self.collection_name}")
            except ImportError:
                logger.error("motor 包未安装，请运行: pip install motor")
                raise
            except Exception as e:
                logger.error(f"MongoDB DocStore 连接失败: {e}")
                raise

        return self._collection

    def _doc_to_dict(self, doc: DocStoreDocument) -> dict[str, Any]:
        """转换文档为字典"""
        data = doc.to_dict()
        # 确保 child_ids 是列表
        if data.get("child_ids") is None:
            data["child_ids"] = []
        return data

    async def add(self, doc: DocStoreDocument) -> bool:
        """添加文档"""
        collection = await self._get_collection()
        try:
            await collection.insert_one(self._doc_to_dict(doc))
            return True
        except Exception as e:
            logger.warning(f"添加文档失败（可能已存在）: {doc.id}, {e}")
            return False

    async def add_many(self, docs: list[DocStoreDocument]) -> int:
        """批量添加文档，返回添加数量"""
        if not docs:
            return 0

        collection = await self._get_collection()
        try:
            # 使用 bulk_write 提高性能
            from motor.motor_asyncio import UpdateOne

            operations = [
                UpdateOne(
                    {"id": doc.id},
                    {"$set": self._doc_to_dict(doc)},
                    upsert=True
                )
                for doc in docs
            ]
            result = await collection.bulk_write(operations, ordered=False)
            return result.upserted_count + result.modified_count
        except Exception as e:
            logger.error(f"批量添加文档失败: {e}")
            # 回退到逐个添加
            count = 0
            for doc in docs:
                if await self.add(doc):
                    count += 1
            return count

    async def get(self, doc_id: str) -> Optional[DocStoreDocument]:
        """根据 ID 获取文档"""
        collection = await self._get_collection()
        doc = await collection.find_one({"id": doc_id})
        if doc:
            doc.pop("_id", None)  # 移除 MongoDB 的 _id
            return DocStoreDocument.from_dict(doc)
        return None

    async def get_many(self, doc_ids: list[str]) -> list[DocStoreDocument]:
        """根据 ID 列表批量获取文档"""
        if not doc_ids:
            return []

        collection = await self._get_collection()
        cursor = collection.find({"id": {"$in": doc_ids}})
        docs = []
        async for doc in cursor:
            doc.pop("_id", None)
            docs.append(DocStoreDocument.from_dict(doc))
        return docs

    async def update(self, doc_id: str, updates: dict[str, Any]) -> bool:
        """更新文档"""
        collection = await self._get_collection()
        result = await collection.update_one(
            {"id": doc_id},
            {"$set": updates}
        )
        return result.modified_count > 0

    async def delete(self, doc_id: str) -> bool:
        """删除文档"""
        collection = await self._get_collection()
        result = await collection.delete_one({"id": doc_id})
        return result.deleted_count > 0

    async def delete_by_collection(self, collection_name: str) -> int:
        """根据 collection_name 元数据删除文档"""
        collection = await self._get_collection()
        result = await collection.delete_many({
            "metadata.collection_name": collection_name
        })
        return result.deleted_count

    async def delete_all(self) -> int:
        """删除所有文档"""
        collection = await self._get_collection()
        result = await collection.delete_many({})
        return result.deleted_count

    async def get_children(self, parent_id: str) -> list[DocStoreDocument]:
        """获取父节点的所有子节点"""
        collection = await self._get_collection()
        cursor = collection.find({"parent_id": parent_id}).sort("metadata.page_idx", 1)
        docs = []
        async for doc in cursor:
            doc.pop("_id", None)
            docs.append(DocStoreDocument.from_dict(doc))
        return docs

    async def count(self) -> int:
        """获取文档总数"""
        collection = await self._get_collection()
        return await collection.count_documents({})

    async def close(self) -> None:
        """关闭连接"""
        if self._client:
            self._client.close()
            self._client = None
            self._collection = None
            logger.info("MongoDB DocStore 连接已关闭")


def create_docstore(
    uri: str,
    db_name: str,
    collection_name: str = "docstore"
) -> MongoDBDocStore:
    """
    创建 MongoDB DocStore 实例

    Args:
        uri: MongoDB URI
        db_name: 数据库名称
        collection_name: 集合名称

    Returns:
        MongoDBDocStore 实例
    """
    return MongoDBDocStore(uri, db_name, collection_name)
