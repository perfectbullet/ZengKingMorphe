"""
上下文扩展模块

基于 DocStore 和顺序关系自动扩展检索结果的上下文，
解决边界截断问题，提供更完整的上下文信息。
"""

from typing import Any, Optional

from loguru import logger

from src.document_indexer.docstore import DocStore
from src.retrieval.base import RetrievedDocument


class ContextExpander:
    """
    上下文扩展器

    基于前后节点关系自动扩展检索结果的上下文。

    功能：
    1. 获取前 N 个节点
    2. 获取后 N 个节点
    3. 可选包含父节点摘要

    使用示例:
        >>> expander = ContextExpander(docstore, window=1)
        >>> expanded = await expander.expand(results)
    """

    def __init__(
        self,
        docstore: DocStore,
        window: int = 1,
        include_parent: bool = False,
        context_score_multiplier: float = 0.8
    ):
        """
        初始化上下文扩展器

        Args:
            docstore: DocStore 实例
            window: 扩展窗口大小（前后各获取几个节点）
            include_parent: 是否包含父节点摘要
            context_score_multiplier: 上下文节点的分数乘数
        """
        self.docstore = docstore
        self.window = window
        self.include_parent = include_parent
        self.context_score_multiplier = context_score_multiplier

    async def expand(
        self,
        results: list[RetrievedDocument],
        window: Optional[int] = None
    ) -> list[RetrievedDocument]:
        """
        扩展检索结果的上下文

        Args:
            results: 初始检索结果
            window: 扩展窗口大小（覆盖初始化时的值）

        Returns:
            扩展后的结果列表
        """
        if not results:
            return results

        window = window or self.window

        # 去重集合
        seen_ids: set[str] = {r.chunk_id for r in results}
        expanded_results: list[RetrievedDocument] = list(results)

        for result in results:
            # 获取上下文节点
            context_nodes = await self._get_context_nodes(result, window)

            # 添加到结果
            for node in context_nodes:
                if node.chunk_id not in seen_ids:
                    # 降低上下文节点的分数
                    node.score = result.score * self.context_score_multiplier
                    expanded_results.append(node)
                    seen_ids.add(node.chunk_id)

        logger.info(
            f"上下文扩展: {len(results)} → {len(expanded_results)} "
            f"(窗口={window})"
        )

        # 按分数排序
        expanded_results.sort(key=lambda x: x.score, reverse=True)
        return expanded_results

    async def _get_context_nodes(
        self,
        result: RetrievedDocument,
        window: int
    ) -> list[RetrievedDocument]:
        """
        获取上下文节点

        Args:
            result: 检索结果
            window: 扩展窗口大小

        Returns:
            上下文节点列表
        """
        context_nodes = []
        metadata = result.metadata

        # 获取顺序关系
        prev_id = metadata.get("prev_chunk_id")
        next_id = metadata.get("next_chunk_id")
        parent_id = metadata.get("parent_id")

        # 向前扩展
        if prev_id:
            prev_nodes = await self._get_prev_nodes(prev_id, window)
            context_nodes.extend(prev_nodes)

        # 向后扩展
        if next_id:
            next_nodes = await self._get_next_nodes(next_id, window)
            context_nodes.extend(next_nodes)

        # 可选：添加父节点摘要
        if self.include_parent and parent_id:
            parent_node = await self._get_parent_summary(parent_id, result.score)
            if parent_node:
                context_nodes.append(parent_node)

        return context_nodes

    async def _get_prev_nodes(
        self,
        start_id: str,
        window: int
    ) -> list[RetrievedDocument]:
        """获取前 N 个节点"""
        nodes = []
        current_id = start_id

        for _ in range(window):
            if not current_id:
                break

            doc = await self.docstore.get(current_id)
            if not doc:
                break

            nodes.append(self._doc_to_retrieved(doc))
            current_id = doc.prev_id

        return nodes

    async def _get_next_nodes(
        self,
        start_id: str,
        window: int
    ) -> list[RetrievedDocument]:
        """获取后 N 个节点"""
        nodes = []
        current_id = start_id

        for _ in range(window):
            if not current_id:
                break

            doc = await self.docstore.get(current_id)
            if not doc:
                break

            nodes.append(self._doc_to_retrieved(doc))
            current_id = doc.next_id

        return nodes

    async def _get_parent_summary(
        self,
        parent_id: str,
        base_score: float
    ) -> Optional[RetrievedDocument]:
        """获取父节点摘要"""
        doc = await self.docstore.get(parent_id)
        if not doc:
            return None

        # 截取前 500 字符作为摘要
        summary_text = doc.text[:500]
        if len(doc.text) > 500:
            summary_text += "..."

        return RetrievedDocument(
            text=summary_text,
            metadata=doc.metadata,
            score=base_score * 0.5,  # 父节点摘要分数更低
            chunk_id=f"{parent_id}_summary",
            source=doc.metadata.get("source", "")
        )

    def _doc_to_retrieved(self, doc: Any) -> RetrievedDocument:
        """将 DocStoreDocument 转换为 RetrievedDocument"""
        return RetrievedDocument(
            text=doc.text,
            metadata=doc.metadata,
            score=0.0,
            chunk_id=doc.id,
            source=doc.metadata.get("source", "")
        )


class AutoMergingRetriever:
    """
    自动合并检索器

    当检索到多个同一父节点的子节点时，自动合并为父节点。

    使用示例:
        >>> merger = AutoMergingRetriever(docstore, threshold=0.5)
        >>> merged = await merger.merge(results)
    """

    def __init__(
        self,
        docstore: DocStore,
        merge_threshold: float = 0.5
    ):
        """
        初始化自动合并检索器

        Args:
            docstore: DocStore 实例
            merge_threshold: 合并阈值（匹配比例超过此值时合并）
        """
        self.docstore = docstore
        self.merge_threshold = merge_threshold

    async def merge(
        self,
        results: list[RetrievedDocument]
    ) -> list[RetrievedDocument]:
        """
        自动合并检索结果

        Args:
            results: 初始检索结果

        Returns:
            合并后的结果
        """
        if not results:
            return results

        # 按 parent_id 分组
        parent_groups = self._group_by_parent(results)

        nodes_to_remove: set[str] = set()
        nodes_to_add: list[RetrievedDocument] = []

        # 对每个父节点组判断是否合并
        for parent_id, children in parent_groups.items():
            if not parent_id:
                continue

            # 获取父节点
            parent_doc = await self.docstore.get(parent_id)
            if not parent_doc:
                continue

            # 计算合并比例
            total_children = len(parent_doc.child_ids) if parent_doc.child_ids else 1
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
                    text=parent_doc.text,
                    metadata=parent_doc.metadata,
                    score=avg_score,
                    chunk_id=parent_id,
                    source=parent_doc.metadata.get("source", "")
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
        results: list[RetrievedDocument]
    ) -> dict[str, list[RetrievedDocument]]:
        """按父节点分组"""
        groups: dict[str, list[RetrievedDocument]] = {}
        for result in results:
            parent_id = result.metadata.get("parent_id", "")
            if parent_id:
                if parent_id not in groups:
                    groups[parent_id] = []
                groups[parent_id].append(result)
        return groups
