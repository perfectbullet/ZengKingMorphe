"""
查询扩展模块

仅使用 LLM 语义扩展策略，通过精心设计的提示词实现：
1. 查询分解 - 将复合查询分解为多个子查询
2. 术语同义词扩展 - 教育领域专业术语和同义词扩展
3. 语义扩展 - 生成语义相关的查询

所有扩展逻辑由 LLM 通过 Few-shot 示例学习。
"""

from loguru import logger


class QueryExpander:
    """查询扩展器（仅使用 LLM 语义扩展）"""

    def __init__(
        self,
        llm_client,
        max_total_expansions: int = 5,
    ):
        """
        初始化查询扩展器

        Args:
            llm_client: LLM 客户端（必需）
            max_total_expansions: 总扩展数量限制（包括原始查询）
        """
        if llm_client is None:
            raise ValueError("llm_client 必须提供")
        self.llm_client = llm_client
        self.max_total_expansions = max_total_expansions

    async def expand(
        self,
        query: str,
        max_expansions: int = 3
    ) -> list[str]:
        """
        扩展查询（仅使用 LLM 语义扩展）

        Args:
            query: 原始查询
            max_expansions: 最大扩展数量（不包括原始查询）

        Returns:
            扩展后的查询列表（原始查询在最后）
        """
        expansions = await self._llm_semantic_expand(query, max_expansions)

        # 添加原始查询到最后
        expansions.append(query)

        # 去重并限制数量
        unique_expansions = self._unique(expansions)

        logger.debug(
            f"查询扩展: 原始='{query[:50]}..., "
            f"扩展后数量={len(unique_expansions)}"
        )

        return unique_expansions[:self.max_total_expansions]

    async def _llm_semantic_expand(
        self,
        query: str,
        max_expansions: int
    ) -> list[str]:
        """使用 LLM 进行语义扩展（包含查询分解和术语扩展）"""
        prompt = self._build_prompt(query, max_expansions)

        try:
            response = await self.llm_client.ainvoke(prompt)
            expansions = self._parse_llm_response(response)
            return expansions[:max_expansions]
        except Exception as e:
            logger.error(f"LLM 语义扩展失败: {e}")
            return []

    def _build_prompt(self, query: str, max_expansions: int) -> str:
        """构建 LLM 提示词（包含 Few-shot 示例和术语对照表）"""
        return f"""你是一个查询优化助手。请为以下查询生成 {max_expansions} 个相关的扩展查询，用于提高教育领域（特别是数学教材）的检索召回率。

原查询: {query}

请参考以下示例来生成扩展查询：

=== 示例 1：复合查询需要分解 ===
输入: 导数和积分的关系
输出:
- 导数的定义
- 积分的定义
- 导数和积分的联系和区别

=== 示例 2：单概念查询不需要分解 ===
输入: 什么是导数？
输出:
- 导数的定义是什么
- 导数的概念和含义
- 微商的解释

=== 示例 3：术语同义词扩展 ===
输入: 如何求微商？
输出:
- 如何求导数？
- 导数的计算方法
- 变化率的求法

=== 示例 4：不需要过度分解的简单查询 ===
输入: 极限的定义
输出:
- 极限的含义
- 极限的概念

**重要术语同义词对照表**（用于生成同义词扩展）：
- 导数 ↔ 微商、变化率、斜率、导函数
- 积分 ↔ 原函数、不定积分、积分运算
- 极限 ↔ 极限值
- 微分 ↔ 微分元、微分运算
- 函数 ↔ 映射、对应关系
- 方程 ↔ 等式、方程式
- 几何 ↔ 图形、几何图形
- 概率 ↔ 几率、可能性
- 统计 ↔ 数据分析

**扩展规则**：
1. 复合查询（包含"和"、"与"、"及"、"区别"、"关系"等连接词）应分解为多个子查询
2. 单概念查询使用同义词、近义词扩展
3. 不要过度分解简单查询
4. 保持原意不变
5. 使查询更具体、更完整
6. 每行一个扩展查询，不要编号

请生成扩展查询："""

    def _parse_llm_response(self, response: str) -> list[str]:
        """解析 LLM 响应"""
        lines = response.strip().split("\n")
        expansions = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 移除编号和列表符号
            if line[0].isdigit() and "." in line:
                line = line.split(".", 1)[1].strip()
            elif line.startswith(("-", "*", "•")):
                line = line[1:].strip()

            if line:
                expansions.append(line)

        return expansions

    def _unique(self, items: list[str]) -> list[str]:
        """去重并保持顺序"""
        seen = set()
        unique = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                unique.append(item)
        return unique
