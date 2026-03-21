"""
MongoDB DocStore 和上下文扩展测试
"""

import pytest

from src.document_indexer.docstore import (
    DocStoreDocument,
    MongoDBDocStore,
    create_docstore
)
from src.retrieval.context_expander import ContextExpander, AutoMergingRetriever
from src.retrieval.base import RetrievedDocument


class TestDocStoreDocument:
    """DocStoreDocument 数据模型测试"""

    def test_create_document(self):
        """测试创建文档"""
        doc = DocStoreDocument(
            id="test_id",
            text="测试内容",
            metadata={"page_idx": 1}
        )
        assert doc.id == "test_id"
        assert doc.text == "测试内容"
        assert doc.metadata["page_idx"] == 1
        assert doc.parent_id is None
        assert doc.child_ids == []

    def test_to_dict(self):
        """测试转换为字典"""
        doc = DocStoreDocument(
            id="test_id",
            text="测试内容",
            metadata={"page_idx": 1},
            prev_id="prev_1",
            next_id="next_1"
        )
        data = doc.to_dict()
        assert data["id"] == "test_id"
        assert data["prev_id"] == "prev_1"
        assert data["next_id"] == "next_1"

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "id": "test_id",
            "text": "测试内容",
            "metadata": {"page_idx": 1},
            "parent_id": None,
            "child_ids": [],
            "prev_id": None,
            "next_id": None,
            "level": None
        }
        doc = DocStoreDocument.from_dict(data)
        assert doc.id == "test_id"
        assert doc.text == "测试内容"


@pytest.mark.mongodb
class TestMongoDBDocStore:
    """MongoDBDocStore 测试"""

    @pytest.fixture
    async def store(self):
        """创建 DocStore 实例"""
        # 使用测试数据库
        store = create_docstore(
            uri="mongodb://funasr:funasr2026@192.168.8.233:27017/funasr?authSource=admin",
            db_name="funasr",
            collection_name="docstore_test"
        )
        yield store
        # 清理测试数据
        await store.delete_all()

    @pytest.mark.asyncio
    async def test_add_and_get(self, store):
        """测试添加和获取文档"""
        doc = DocStoreDocument(
            id="test_1",
            text="测试内容1",
            metadata={"page_idx": 1}
        )

        await store.add(doc)
        retrieved = await store.get("test_1")

        assert retrieved is not None
        assert retrieved.id == "test_1"
        assert retrieved.text == "测试内容1"

    @pytest.mark.asyncio
    async def test_add_many(self, store):
        """测试批量添加"""
        docs = [
            DocStoreDocument(id=f"test_{i}", text=f"内容{i}", metadata={})
            for i in range(5)
        ]

        count = await store.add_many(docs)
        assert count == 5

    @pytest.mark.asyncio
    async def test_get_many(self, store):
        """测试批量获取"""
        docs = [
            DocStoreDocument(id=f"test_{i}", text=f"内容{i}", metadata={})
            for i in range(3)
        ]
        await store.add_many(docs)

        retrieved = await store.get_many(["test_0", "test_1", "test_9"])
        assert len(retrieved) == 2  # test_9 不存在

    @pytest.mark.asyncio
    async def test_update(self, store):
        """测试更新文档"""
        doc = DocStoreDocument(
            id="test_1",
            text="原始内容",
            metadata={"page_idx": 1}
        )
        await store.add(doc)

        updated = await store.update("test_1", {"text": "更新内容"})
        assert updated is True

        retrieved = await store.get("test_1")
        assert retrieved.text == "更新内容"

    @pytest.mark.asyncio
    async def test_delete(self, store):
        """测试删除文档"""
        doc = DocStoreDocument(id="test_1", text="内容", metadata={})
        await store.add(doc)

        deleted = await store.delete("test_1")
        assert deleted is True

        retrieved = await store.get("test_1")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_children(self, store):
        """测试获取子节点"""
        parent_id = "parent_1"
        child_docs = [
            DocStoreDocument(
                id=f"child_{i}",
                text=f"子节点{i}",
                metadata={},
                parent_id=parent_id
            )
            for i in range(3)
        ]
        await store.add_many(child_docs)

        children = await store.get_children(parent_id)
        assert len(children) == 3
        assert all(c.parent_id == parent_id for c in children)

    @pytest.mark.asyncio
    async def test_count(self, store):
        """测试统计文档数量"""
        # 先清空
        await store.delete_all()

        docs = [
            DocStoreDocument(id=f"test_{i}", text=f"内容{i}", metadata={})
            for i in range(3)
        ]
        await store.add_many(docs)

        count = await store.count()
        assert count == 3


class TestContextExpander:
    """ContextExpander 测试"""

    @pytest.fixture
    async def setup_store(self):
        """设置测试用的 DocStore"""
        from src.document_indexer.docstore import MongoDBDocStore
        store = MongoDBDocStore(
            uri="mongodb://funasr:funasr2026@192.168.8.233:27017/funasr?authSource=admin",
            db_name="funasr",
            collection_name="docstore_test"
        )

        # 添加测试文档（带顺序关系）
        docs = [
            DocStoreDocument(
                id="chunk_1",
                text="第一个内容块",
                metadata={"page_idx": 1, "title_path": ["第一章"]},
                prev_id=None,
                next_id="chunk_2"
            ),
            DocStoreDocument(
                id="chunk_2",
                text="第二个内容块",
                metadata={"page_idx": 2, "title_path": ["第一章"]},
                prev_id="chunk_1",
                next_id="chunk_3"
            ),
            DocStoreDocument(
                id="chunk_3",
                text="第三个内容块",
                metadata={"page_idx": 3, "title_path": ["第一章"]},
                prev_id="chunk_2",
                next_id="chunk_4"
            ),
            DocStoreDocument(
                id="chunk_4",
                text="第四个内容块",
                metadata={"page_idx": 4, "title_path": ["第一章"]},
                prev_id="chunk_3",
                next_id=None
            ),
        ]
        await store.add_many(docs)
        return store

    @pytest.mark.asyncio
    async def test_expand_forward(self, setup_store):
        """测试向前扩展"""
        expander = ContextExpander(setup_store, window=1)

        results = [
            RetrievedDocument(
                text="第三个内容块",
                metadata={"chunk_id": "chunk_3", "prev_chunk_id": "chunk_2", "next_chunk_id": "chunk_4"},
                score=0.8,
                chunk_id="chunk_3",
                source="test.pdf"
            )
        ]

        expanded = await expander.expand(results)

        # 应该包含 chunk_2（前一个）
        ids = [r.chunk_id for r in expanded]
        assert "chunk_2" in ids

    @pytest.mark.asyncio
    async def test_expand_both_directions(self, setup_store):
        """测试双向扩展"""
        expander = ContextExpander(setup_store, window=1)

        results = [
            RetrievedDocument(
                text="第三个内容块",
                metadata={"chunk_id": "chunk_3", "prev_chunk_id": "chunk_2", "next_chunk_id": "chunk_4"},
                score=0.8,
                chunk_id="chunk_3",
                source="test.pdf"
            )
        ]

        expanded = await expander.expand(results, window=1)

        # 应该包含 chunk_2（前一个）和 chunk_4（后一个）
        ids = [r.chunk_id for r in expanded]
        assert "chunk_2" in ids
        assert "chunk_4" in ids

    @pytest.mark.asyncio
    async def test_expand_deduplication(self, setup_store):
        """测试去重"""
        expander = ContextExpander(setup_store, window=1)

        results = [
            RetrievedDocument(
                text="第三个内容块",
                metadata={"chunk_id": "chunk_3", "prev_chunk_id": "chunk_2", "next_chunk_id": "chunk_4"},
                score=0.8,
                chunk_id="chunk_3",
                source="test.pdf"
            )
        ]

        expanded = await expander.expand(results)

        # 检查没有重复
        ids = [r.chunk_id for r in expanded]
        assert len(ids) == len(set(ids))


class TestAutoMergingRetriever:
    """AutoMergingRetriever 测试"""

    @pytest.fixture
    async def setup_store(self):
        """设置测试用的 DocStore"""
        from src.document_indexer.docstore import MongoDBDocStore
        store = MongoDBDocStore(
            uri="mongodb://funasr:funasr2026@192.168.8.233:27017/funasr?authSource=admin",
            db_name="funasr",
            collection_name="docstore_test"
        )

        # 父节点
        parent_doc = DocStoreDocument(
            id="parent_1",
            text="父节点完整内容\n这是第一章的完整内容...",
            metadata={"title_path": ["第一章"]},
            child_ids=["child_1", "child_2", "child_3"]
        )

        # 子节点
        child_docs = [
            DocStoreDocument(
                id="child_1",
                text="子节点1内容",
                metadata={"title_path": ["第一章"], "parent_id": "parent_1"},
                parent_id="parent_1"
            ),
            DocStoreDocument(
                id="child_2",
                text="子节点2内容",
                metadata={"title_path": ["第一章"], "parent_id": "parent_1"},
                parent_id="parent_1"
            ),
            DocStoreDocument(
                id="child_3",
                text="子节点3内容",
                metadata={"title_path": ["第一章"], "parent_id": "parent_1"},
                parent_id="parent_1"
            ),
        ]

        await store.add(parent_doc)
        await store.add_many(child_docs)
        return store

    @pytest.mark.asyncio
    async def test_merge_when_threshold_met(self, setup_store):
        """测试达到阈值时合并"""
        merger = AutoMergingRetriever(setup_store, merge_threshold=0.5)

        # 检索到 2 个子节点（共 3 个，比例 2/3 > 0.5）
        results = [
            RetrievedDocument(
                text="子节点1内容",
                metadata={"chunk_id": "child_1", "parent_id": "parent_1"},
                score=0.8,
                chunk_id="child_1",
                source="test.pdf"
            ),
            RetrievedDocument(
                text="子节点2内容",
                metadata={"chunk_id": "child_2", "parent_id": "parent_1"},
                score=0.7,
                chunk_id="child_2",
                source="test.pdf"
            ),
        ]

        merged = await merger.merge(results)

        # 应该合并为父节点
        ids = [r.chunk_id for r in merged]
        assert "parent_1" in ids
        # 子节点应该被移除
        assert "child_1" not in ids
        assert "child_2" not in ids

    @pytest.mark.asyncio
    async def test_no_merge_when_threshold_not_met(self, setup_store):
        """测试未达到阈值时不合并"""
        merger = AutoMergingRetriever(setup_store, merge_threshold=0.8)

        # 检索到 1 个子节点（共 3 个，比例 1/3 < 0.8）
        results = [
            RetrievedDocument(
                text="子节点1内容",
                metadata={"chunk_id": "child_1", "parent_id": "parent_1"},
                score=0.8,
                chunk_id="child_1",
                source="test.pdf"
            ),
        ]

        merged = await merger.merge(results)

        # 不应该合并
        assert len(merged) == 1
        assert merged[0].chunk_id == "child_1"
