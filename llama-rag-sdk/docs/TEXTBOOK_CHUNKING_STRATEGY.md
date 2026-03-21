# 教材 RAG 分块与关联优化方案（P1 优先级）

> 针对几百页数学教材的分块策略
> 优化目标：保持结构完整性 + 解决边界截断问题

---

## 现状分析

### 当前实现（你已有的代码）

```
MinerU 解析 → 结构感知分块 → ChromaDB 存储 → 检索
     ↓              ↓              ↓
 content_list    TextChunk      向量索引
```

**已实现的优势**：
- ✅ 按标题层级分块（`title_path`）
- ✅ 保留页码、块类型等元数据
- ✅ 图片与上下文关联

**存在的问题**：
- ❌ 分块之间没有关联关系
- ❌ 检索结果可能在边界处截断
- ❌ 缺少层次化索引

---

## 优化方案 1：MinerU 结构感知 + 层次化结合

### 1.1 设计思路

```
教材文档
    ↓
┌─────────────────────────────────────────┐
│  第一层：MinerU 结构分块（已有）        │
│  - 按 title_path 分块                   │
│  - 保持章节完整性                       │
│  - 保留结构信息                         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  第二层：建立父子关系（新增）          │
│  - 章节作为父节点                       │
│  - 段落作为子节点                       │
│  - 只对子节点做 embedding               │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  第三层：动态合并（检索时）            │
│  - 检索到多个子节点                     │
│  - 自动合并到父节点                     │
│  - 返回完整上下文                       │
└─────────────────────────────────────────┘
```

### 1.2 数据结构扩展

```python
# 扩展现有的 TextChunk metadata

# 现有 metadata（已有）
metadata = {
    "chunk_id": "xxx",
    "page_idx": 10,
    "title_path": ["第1章", "导数", "1.1 导数的定义"],
    "structure_level": 3,
    # ... 其他现有字段
}

# 新增层次关系字段
metadata.update({
    # 层次关系
    "parent_id": None,              # 父节点 ID
    "child_ids": [],                # 子节点 ID 列表（JSON 字符串）
    "is_leaf": True,                # 是否为叶子节点

    # 顺序关系
    "prev_id": "chunk_5",           # 前一个节点 ID
    "next_id": "chunk_7",           # 后一个节点 ID

    # 章节信息（从 title_path 提取）
    "chapter": "第1章",
    "section": "1.1 导数的定义",
})
```

### 1.3 实现代码

```python
# src/document_indexer/hierarchical_builder.py

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from src.document_parser.base import TextChunk


@dataclass
class HierarchicalNode:
    """层次化节点"""
    chunk_id: str
    text: str
    level: str                      # "chapter", "section", "leaf"
    parent_id: Optional[str] = None
    child_ids: List[str] = None
    prev_id: Optional[str] = None
    next_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.child_ids is None:
            self.child_ids = []
        if self.metadata is None:
            self.metadata = {}


class HierarchicalBuilder:
    """
    层次化关系构建器

    为 MinerU 分块结果添加父子关系和前后关系
    """

    def build_hierarchy(
        self,
        chunks: List[TextChunk]
    ) -> List[HierarchicalNode]:
        """
        为分块建立层次化关系

        Args:
            chunks: MinerUStructureAwareChunker 生成的分块

        Returns:
            带有关系信息的层次化节点列表
        """
        if not chunks:
            return []

        # 1. 按 title_path 分组
        sections = self._group_by_section(chunks)

        # 2. 为每个 section 创建父节点
        parent_nodes = self._create_parent_nodes(sections)

        # 3. 为每个 chunk 创建子节点并建立关系
        leaf_nodes = self._create_leaf_nodes(chunks, parent_nodes)

        # 4. 建立前后关系
        all_nodes = parent_nodes + leaf_nodes
        self._link_sequential(all_nodes)

        return all_nodes

    def _group_by_section(
        self,
        chunks: List[TextChunk]
    ) -> Dict[str, List[TextChunk]]:
        """
        按 section 分组

        使用 title_path 的最后一层作为 section 标识
        """
        sections = {}

        for chunk in chunks:
            title_path = chunk.metadata.get("title_path", [])
            if not title_path:
                # 没有 title_path，使用 chunk_id 作为独立 section
                section_key = chunk.metadata["chunk_id"]
            else:
                # 使用完整路径作为 section key
                section_key = "|".join(title_path)

            if section_key not in sections:
                sections[section_key] = []
            sections[section_key].append(chunk)

        logger.info(f"按 section 分组：{len(sections)} 个 section")
        return sections

    def _create_parent_nodes(
        self,
        sections: Dict[str, List[TextChunk]]
    ) -> List[HierarchicalNode]:
        """
        为每个 section 创建父节点

        父节点包含该 section 所有子节点的合并文本
        """
        parent_nodes = []

        for section_key, chunks in sections.items():
            # 获取 section 信息
            first_chunk = chunks[0]
            title_path = first_chunk.metadata.get("title_path", [])

            # 计算父节点文本（合并所有子节点）
            # 可选：使用 LLM 生成摘要而不是简单合并
            parent_text = self._create_parent_text(chunks)

            # 生成父节点 ID
            parent_id = f"parent_{section_key.replace('|', '_')}"

            # 提取章节信息
            chapter = title_path[0] if len(title_path) > 0 else None
            section = title_path[-1] if title_path else None

            parent_node = HierarchicalNode(
                chunk_id=parent_id,
                text=parent_text,
                level="section",
                child_ids=[c.metadata["chunk_id"] for c in chunks],
                metadata={
                    "title_path": title_path,
                    "chapter": chapter,
                    "section": section,
                    "structure_level": len(title_path),
                    "child_count": len(chunks),
                    "page_indices": list(set([
                        p for c in chunks
                        for p in c.metadata.get("page_indices", [c.metadata.get("page_idx", 0)])
                    ])),
                }
            )
            parent_nodes.append(parent_node)

        logger.info(f"创建 {len(parent_nodes)} 个父节点")
        return parent_nodes

    def _create_parent_text(
        self,
        chunks: List[TextChunk],
        max_length: int = 2000
    ) -> str:
        """
        创建父节点文本

        策略：
        1. 合并所有子节点文本
        2. 如果超过长度限制，使用摘要或截断
        """
        combined = "\n\n".join([c.text for c in chunks])

        if len(combined) <= max_length:
            return combined

        # 截断并添加提示
        return combined[:max_length] + "\n\n...(内容过长，已截断)"

    def _create_leaf_nodes(
        self,
        chunks: List[TextChunk],
        parent_nodes: List[HierarchicalNode]
    ) -> List[HierarchicalNode]:
        """
        创建叶子节点并建立父子关系
        """
        # 创建 parent_id 映射
        parent_map = {
            "|".join(p.metadata.get("title_path", [])): p.chunk_id
            for p in parent_nodes
        }

        leaf_nodes = []

        for chunk in chunks:
            title_path = chunk.metadata.get("title_path", [])
            section_key = "|".join(title_path)
            parent_id = parent_map.get(section_key)

            leaf_node = HierarchicalNode(
                chunk_id=chunk.metadata["chunk_id"],
                text=chunk.text,
                level="leaf",
                parent_id=parent_id,
                metadata={
                    **chunk.metadata,
                    "is_leaf": True,
                }
            )
            leaf_nodes.append(leaf_node)

        logger.info(f"创建 {len(leaf_nodes)} 个叶子节点")
        return leaf_nodes

    def _link_sequential(
        self,
        nodes: List[HierarchicalNode]
    ) -> None:
        """
        建立前后关系（按页码顺序）
        """
        # 按页码排序
        sorted_nodes = sorted(
            nodes,
            key=lambda n: n.metadata.get("page_indices", [0])[0]
        )

        for i, node in enumerate(sorted_nodes):
            if i > 0:
                node.prev_id = sorted_nodes[i - 1].chunk_id
            if i < len(sorted_nodes) - 1:
                node.next_id = sorted_nodes[i + 1].chunk_id

        logger.debug("建立前后关系完成")

    def get_leaf_nodes(
        self,
        nodes: List[HierarchicalNode]
    ) -> List[HierarchicalNode]:
        """获取所有叶子节点"""
        return [n for n in nodes if n.level == "leaf"]

    def get_parent_nodes(
        self,
        nodes: List[HierarchicalNode]
    ) -> List[HierarchicalNode]:
        """获取所有父节点"""
        return [n for n in nodes if n.level == "section"]
```

### 1.4 存储策略

```python
# src/document_indexer/hierarchical_storage.py

from typing import List, Dict, Any
from loguru import logger

from src.document_indexer.storage import VectorStore
from src.document_indexer.hierarchical_builder import (
    HierarchicalBuilder,
    HierarchicalNode
)


class HierarchicalStorage:
    """
    层次化存储

    策略：
    1. 叶子节点：向量库 + docstore
    2. 父节点：仅 docstore
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: Any,
        docstore: Optional[Any] = None
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.docstore = docstore  # 可选：MongoDB 或内存存储

    def store(
        self,
        nodes: List[HierarchicalNode]
    ) -> Dict[str, int]:
        """
        存储层次化节点

        Returns:
            存储统计
        """
        leaf_nodes = [n for n in nodes if n.level == "leaf"]
        parent_nodes = [n for n in nodes if n.level == "section"]

        logger.info(
            f"存储层次化节点：叶子 {len(leaf_nodes)}，父节点 {len(parent_nodes)}"
        )

        # 1. 只对叶子节点做 embedding
        leaf_embeddings = self.embedding_model.embed_documents([
            n.text for n in leaf_nodes
        ])

        # 2. 叶子节点存入向量库
        self.vector_store.add(
            ids=[n.chunk_id for n in leaf_nodes],
            embeddings=leaf_embeddings,
            documents=[n.text for n in leaf_nodes],
            metadatas=[self._prepare_metadata(n) for n in leaf_nodes]
        )

        # 3. 所有节点存入 docstore（如果有）
        if self.docstore:
            for node in nodes:
                self.docstore.add(
                    id=node.chunk_id,
                    document={
                        "text": node.text,
                        "level": node.level,
                        "parent_id": node.parent_id,
                        "child_ids": node.child_ids,
                        "prev_id": node.prev_id,
                        "next_id": node.next_id,
                        "metadata": node.metadata
                    }
                )

        return {
            "total_nodes": len(nodes),
            "leaf_nodes": len(leaf_nodes),
            "parent_nodes": len(parent_nodes),
            "embedded_nodes": len(leaf_nodes)
        }

    def _prepare_metadata(
        self,
        node: HierarchicalNode
    ) -> Dict[str, Any]:
        """准备向量库的 metadata"""
        metadata = node.metadata.copy() if node.metadata else {}

        # 添加关系字段（ChromaDB 支持）
        metadata.update({
            "parent_id": node.parent_id or "",
            "prev_id": node.prev_id or "",
            "next_id": node.next_id or "",
            "is_leaf": node.level == "leaf",
            "level": node.level,
        })

        return metadata
```

### 1.5 检索时动态合并

```python
# src/retrieval/auto_merging_retriever.py

from typing import List, Dict, Optional
from loguru import logger

from src.retrieval.base import RetrievedDocument


class AutoMergingRetriever:
    """
    自动合并检索器

    当检索到多个同一父节点的子节点时，自动合并为父节点
    """

    def __init__(
        self,
        docstore,                  # 用于获取父节点
        merge_threshold: float = 0.5,  # 合并阈值
    ):
        self.docstore = docstore
        self.merge_threshold = merge_threshold

    async def merge(
        self,
        results: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        自动合并检索结果

        Args:
            results: 初始检索结果

        Returns:
            合并后的结果
        """
        # 按 parent_id 分组
        parent_groups = self._group_by_parent(results)

        nodes_to_remove = set()
        nodes_to_add = []

        # 对每个父节点组判断是否合并
        for parent_id, children in parent_groups.items():
            if not parent_id:
                continue

            # 获取父节点
            parent = self.docstore.get(parent_id)
            if not parent:
                continue

            # 计算合并比例
            total_children = len(parent.get("child_ids", []))
            matched_children = len(children)
            ratio = matched_children / total_children if total_children > 0 else 0

            logger.debug(
                f"父节点 {parent_id}: 匹配 {matched_children}/{total_children} = {ratio:.2f}"
            )

            # 如果超过阈值，进行合并
            if ratio >= self.merge_threshold:
                # 标记子节点为待删除
                for child in children:
                    nodes_to_remove.add(child.chunk_id)

                # 创建父节点结果
                avg_score = sum(c.score for c in children) / len(children)

                parent_result = RetrievedDocument(
                    text=parent["text"],
                    metadata=parent.get("metadata", {}),
                    score=avg_score,
                    chunk_id=parent_id,
                    source=parent.get("metadata", {}).get("source", "")
                )
                nodes_to_add.append(parent_result)

                logger.info(
                    f"合并 {matched_children} 个子节点到父节点 {parent_id}"
                )

        # 构建新结果列表
        merged_results = [
            r for r in results
            if r.chunk_id not in nodes_to_remove
        ]
        merged_results.extend(nodes_to_add)

        # 按分数排序
        merged_results.sort(key=lambda x: x.score, reverse=True)

        return merged_results

    def _group_by_parent(
        self,
        results: List[RetrievedDocument]
    ) -> Dict[str, List[RetrievedDocument]]:
        """按父节点分组"""
        groups = {}
        for result in results:
            parent_id = result.metadata.get("parent_id", "")
            if parent_id:
                if parent_id not in groups:
                    groups[parent_id] = []
                groups[parent_id].append(result)
        return groups
```

---

## 优化方案 2：前后节点关系 + 上下文扩展

### 2.1 设计思路

```
检索到 chunk_3
    ↓
┌─────────────────────────────────────────┐
│  自动获取上下文                         │
│  - prev_id → chunk_2, chunk_1          │
│  - next_id → chunk_4, chunk_5          │
│  - parent_id → 完整章节                │
└─────────────────────────────────────────┘
    ↓
返回扩展后的完整上下文
```

### 2.2 实现代码

```python
# src/retrieval/context_expander.py

from typing import List, Optional, Set
from loguru import logger

from src.retrieval.base import RetrievedDocument


class ContextExpander:
    """
    上下文扩展器

    基于前后节点关系自动扩展检索结果的上下文
    """

    def __init__(
        self,
        vector_store,              # 用于获取相邻节点
        docstore,                  # 用于获取父节点
        default_window: int = 1,   # 默认扩展窗口
    ):
        self.vector_store = vector_store
        self.docstore = docstore
        self.default_window = default_window

    async def expand(
        self,
        results: List[RetrievedDocument],
        window: Optional[int] = None
    ) -> List[RetrievedDocument]:
        """
        扩展检索结果的上下文

        Args:
            results: 初始检索结果
            window: 扩展窗口大小（前后各获取几个节点）

        Returns:
            扩展后的结果
        """
        window = window or self.default_window

        # 使用集合去重
        expanded_ids: Set[str] = set()
        expanded_results: List[RetrievedDocument] = []

        for result in results:
            # 添加当前节点
            if result.chunk_id not in expanded_ids:
                expanded_results.append(result)
                expanded_ids.add(result.chunk_id)

            # 获取前后节点
            context_nodes = await self._get_context_nodes(
                result,
                window
            )

            # 添加到结果
            for node in context_nodes:
                if node.chunk_id not in expanded_ids:
                    # 降低上下文节点的分数
                    node.score = result.score * 0.8
                    expanded_results.append(node)
                    expanded_ids.add(node.chunk_id)

        logger.info(
            f"上下文扩展：{len(results)} → {len(expanded_results)} "
            f"(窗口={window})"
        )

        # 保持原有排序
        expanded_results.sort(key=lambda x: x.score, reverse=True)
        return expanded_results

    async def _get_context_nodes(
        self,
        result: RetrievedDocument,
        window: int
    ) -> List[RetrievedDocument]:
        """
        获取前后节点
        """
        context_nodes = []

        # 获取 metadata
        metadata = result.metadata
        prev_id = metadata.get("prev_id")
        next_id = metadata.get("next_id")
        parent_id = metadata.get("parent_id")

        # 向前扩展
        current_prev_id = prev_id
        for _ in range(window):
            if current_prev_id:
                node = await self._get_node_by_id(current_prev_id)
                if node:
                    context_nodes.append(node)
                    # 继续向前
                    current_prev_id = node.metadata.get("prev_id")
                else:
                    break

        # 向后扩展
        current_next_id = next_id
        for _ in range(window):
            if current_next_id:
                node = await self._get_node_by_id(current_next_id)
                if node:
                    context_nodes.append(node)
                    # 继续向后
                    current_next_id = node.metadata.get("next_id")
                else:
                    break

        # 可选：添加父节点摘要
        if parent_id:
            parent = self.docstore.get(parent_id)
            if parent:
                # 创建一个摘要节点（分数较低）
                summary_node = RetrievedDocument(
                    text=parent.get("text", "")[:500] + "...",
                    metadata=parent.get("metadata", {}),
                    score=result.score * 0.5,
                    chunk_id=f"{parent_id}_summary",
                    source=""
                )
                context_nodes.append(summary_node)

        return context_nodes

    async def _get_node_by_id(
        self,
        chunk_id: str
    ) -> Optional[RetrievedDocument]:
        """
        根据 chunk_id 获取节点
        """
        # 优先从 docstore 获取
        if self.docstore:
            doc = self.docstore.get(chunk_id)
            if doc:
                return RetrievedDocument(
                    text=doc.get("text", ""),
                    metadata=doc.get("metadata", {}),
                    score=0.0,
                    chunk_id=chunk_id,
                    source=""
                )

        # 从向量库获取（通过 ID 过滤）
        # 注意：ChromaDB 不支持直接通过 ID 获取文档，
        # 需要使用 docstore 或其他存储
        logger.warning(f"无法从向量库直接获取节点 {chunk_id}")
        return None
```

### 2.3 简化实现（使用 ChromaDB 元数据过滤）

如果没有 docstore，可以使用向量库的元数据查询：

```python
# src/retrieval/simple_context_expander.py

from typing import List
from loguru import logger

from src.document_indexer.storage import VectorStore
from src.retrieval.base import RetrievedDocument


class SimpleContextExpander:
    """
    简化的上下文扩展器（仅使用向量库）
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: Any,
        window: int = 1,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.window = window

    async def expand(
        self,
        results: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        扩展检索结果的上下文

        使用向量库的元数据过滤获取相邻节点
        """
        expanded = results.copy()

        for result in results:
            metadata = result.metadata

            # 获取 prev_id 和 next_id
            prev_id = metadata.get("prev_id")
            next_id = metadata.get("next_id")

            # 使用虚拟查询获取相邻节点
            #（ChromaDB 支持通过 where 过滤获取特定文档）
            if prev_id:
                prev_node = await self._get_by_id(prev_id)
                if prev_node:
                    expanded.append(prev_node)

            if next_id:
                next_node = await self._get_by_id(next_id)
                if next_node:
                    expanded.append(next_node)

        # 去重
        seen_ids = set()
        unique_results = []
        for r in expanded:
            if r.chunk_id not in seen_ids:
                unique_results.append(r)
                seen_ids.add(r.chunk_id)

        return unique_results

    async def _get_by_id(
        self,
        chunk_id: str
    ) -> Optional[RetrievedDocument]:
        """
        通过 ID 从向量库获取文档

        注意：这需要向量库支持通过 ID 查询
        """
        try:
            # 使用虚拟 embedding 进行查询
            dummy_embedding = self.embedding_model.get_text_embedding("查询")

            # ChromaDB 不支持直接通过 ID 查询
            # 需要存储时映射 ID 到文档
            # 这里简化处理：返回 None
            logger.warning(
                f"向量库不支持直接通过 ID 获取 {chunk_id}，"
                f"请使用 docstore 或扩展 metadata"
            )
            return None
        except Exception as e:
            logger.error(f"获取节点 {chunk_id} 失败: {e}")
            return None
```

---

## 完整使用流程

```python
# examples/textbook_hierarchical_retrieval.py

import asyncio
from src.document_parser.mineru_client import MinerUParser
from src.document_parser.mineru_structure_aware_chunker import MinerUStructureAwareChunker
from src.document_indexer.hierarchical_builder import HierarchicalBuilder
from src.document_indexer.hierarchical_storage import HierarchicalStorage
from src.document_indexer.storage import VectorStore
from src.retrieval.auto_merging_retriever import AutoMergingRetriever
from src.retrieval.context_expander import ContextExpander
from src.config import settings


async def main():
    # 1. 解析 PDF
    parser = MinerUParser()
    content_list = await parser.parse_pdf("textbook.pdf")

    # 2. 结构感知分块（已有）
    chunker = MinerUStructureAwareChunker()
    chunks = chunker.chunk_content_list(content_list, "textbook.pdf")

    # 3. 构建层次关系（新增）
    builder = HierarchicalBuilder()
    hierarchical_nodes = builder.build_hierarchy(chunks)

    # 4. 存储（只对叶子节点做 embedding）
    vector_store = VectorStore()
    storage = HierarchicalStorage(vector_store, embedding_model)
    stats = storage.store(hierarchical_nodes)
    print(f"存储统计: {stats}")

    # 5. 检索（带自动合并和上下文扩展）
    # ... 检索代码


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 实施检查清单

### 阶段 1：数据结构准备
- [ ] 扩展 `TextChunk` metadata 添加关系字段
- [ ] 创建 `HierarchicalNode` 数据类
- [ ] 创建 `HierarchicalBuilder` 构建层次关系

### 阶段 2：存储改造
- [ ] 修改 `VectorStore.add()` 支持关系 metadata
- [ ] 创建 `HierarchicalStorage` 实现分层存储
- [ ] 验证：只对叶子节点做 embedding

### 阶段 3：检索增强
- [ ] 实现 `AutoMergingRetriever` 自动合并
- [ ] 实现 `ContextExpander` 上下文扩展
- [ ] 集成到现有检索流程

### 阶段 4：测试验证
- [ ] 单元测试：关系构建正确性
- [ ] 集成测试：检索结果完整性
- [ ] 性能测试：embedding 成本降低

---

## 预期效果

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| **Embedding 成本** | 所有分块 | 只叶子节点（降低 ~50%） |
| **边界截断问题** | 频繁 | 通过上下文扩展解决 |
| **上下文完整性** | 差 | 自动合并保证 |
| **检索精度** | 中 | 更好（层次化 + 扩展） |
| **实现复杂度** | 低 | 中 |

---

## 关键文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `mineru_structure_aware_chunker.py` | ✅ 已有 | MinerU 结构分块 |
| `hierarchical_builder.py` | 🔨 新增 | 构建层次关系 |
| `hierarchical_storage.py` | 🔨 新增 | 分层存储 |
| `auto_merging_retriever.py` | 🔨 新增 | 自动合并检索 |
| `context_expander.py` | 🔨 新增 | 上下文扩展 |
