"""
DocStore - 文档存储模块

用于存储和检索文档的完整内容和元数据，与向量存储（Vector Store）配合使用。

作用：
1. 按 ID 快速获取文档
2. 支持复杂的关联查询（父节点、子节点）
3. 存储完整的内容和元数据
"""

from abc import ABC, abstractmethod
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


class DocStore(ABC):
    """DocStore 抽象基类"""

    @abstractmethod
    async def add(self, doc: DocStoreDocument) -> bool:
        """添加文档"""
        pass

    @abstractmethod
    async def add_many(self, docs: list[DocStoreDocument]) -> int:
        """批量添加文档，返回添加数量"""
        pass

    @abstractmethod
    async def get(self, doc_id: str) -> Optional[DocStoreDocument]:
        """根据 ID 获取文档"""
        pass

    @abstractmethod
    async def get_many(self, doc_ids: list[str]) -> list[DocStoreDocument]:
        """根据 ID 列表批量获取文档"""
        pass

    @abstractmethod
    async def update(self, doc_id: str, updates: dict[str, Any]) -> bool:
        """更新文档"""
        pass

    @abstractmethod
    async def delete(self, doc_id: str) -> bool:
        """删除文档"""
        pass

    @abstractmethod
    async def delete_by_collection(self, collection_name: str) -> int:
        """根据集合名称删除所有文档，返回删除数量"""
        pass

    @abstractmethod
    async def get_children(self, parent_id: str) -> list[DocStoreDocument]:
        """获取父节点的所有子节点"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭连接"""
        pass


class MemoryDocStore(DocStore):
    """内存 DocStore 实现（用于开发测试）"""

    def __init__(self):
        self._documents: dict[str, DocStoreDocument] = {}

    async def add(self, doc: DocStoreDocument) -> bool:
        self._documents[doc.id] = doc
        return True

    async def add_many(self, docs: list[DocStoreDocument]) -> int:
        for doc in docs:
            self._documents[doc.id] = doc
        return len(docs)

    async def get(self, doc_id: str) -> Optional[DocStoreDocument]:
        return self._documents.get(doc_id)

    async def get_many(self, doc_ids: list[str]) -> list[DocStoreDocument]:
        return [self._documents.get(doc_id) for doc_id in doc_ids if doc_id in self._documents]

    async def update(self, doc_id: str, updates: dict[str, Any]) -> bool:
        if doc_id not in self._documents:
            return False
        doc = self._documents[doc_id]
        for key, value in updates.items():
            if hasattr(doc, key):
                setattr(doc, key, value)
        return True

    async def delete(self, doc_id: str) -> bool:
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False

    async def delete_by_collection(self, collection_name: str) -> int:
        count = 0
        to_delete = []
        for doc_id, doc in self._documents.items():
            if doc.metadata.get("collection_name") == collection_name:
                to_delete.append(doc_id)
        for doc_id in to_delete:
            del self._documents[doc_id]
            count += 1
        return count

    async def get_children(self, parent_id: str) -> list[DocStoreDocument]:
        return [doc for doc in self._documents.values() if doc.parent_id == parent_id]

    async def close(self) -> None:
        self._documents.clear()


class MongoDBDocStore(DocStore):
    """MongoDB DocStore 实现"""

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
        collection = await self._get_collection()
        try:
            await collection.insert_one(self._doc_to_dict(doc))
            return True
        except Exception as e:
            logger.warning(f"添加文档失败（可能已存在）: {doc.id}, {e}")
            return False

    async def add_many(self, docs: list[DocStoreDocument]) -> int:
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
        collection = await self._get_collection()
        doc = await collection.find_one({"id": doc_id})
        if doc:
            doc.pop("_id", None)  # 移除 MongoDB 的 _id
            return DocStoreDocument.from_dict(doc)
        return None

    async def get_many(self, doc_ids: list[str]) -> list[DocStoreDocument]:
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
        collection = await self._get_collection()
        result = await collection.update_one(
            {"id": doc_id},
            {"$set": updates}
        )
        return result.modified_count > 0

    async def delete(self, doc_id: str) -> bool:
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

    async def get_children(self, parent_id: str) -> list[DocStoreDocument]:
        collection = await self._get_collection()
        cursor = collection.find({"parent_id": parent_id}).sort("metadata.page_idx", 1)
        docs = []
        async for doc in cursor:
            doc.pop("_id", None)
            docs.append(DocStoreDocument.from_dict(doc))
        return docs

    async def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
            self._collection = None
            logger.info("MongoDB DocStore 连接已关闭")


def create_docstore(
    store_type: str = "memory",
    uri: Optional[str] = None,
    db_name: Optional[str] = None,
    collection_name: str = "docstore"
) -> DocStore:
    """
    创建 DocStore 实例

    Args:
        store_type: 存储类型（memory/mongodb）
        uri: MongoDB URI（仅当 store_type=mongodb 时需要）
        db_name: 数据库名称（仅当 store_type=mongodb 时需要）
        collection_name: 集合名称

    Returns:
        DocStore 实例
    """
    if store_type == "mongodb":
        if not uri or not db_name:
            raise ValueError("MongoDB DocStore 需要 uri 和 db_name 参数")
        return MongoDBDocStore(uri, db_name, collection_name)
    else:
        return MemoryDocStore()
