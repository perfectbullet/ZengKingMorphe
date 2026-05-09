"""
查询扩展功能测试

测试 QueryExpander 的 LLM 语义扩展功能
"""

import pytest
import asyncio
from unittest.mock import AsyncMock
from llama_rag_sdk.retrieval.query_expansion import QueryExpander
from llama_rag_sdk.retrieval.llm_client import OllamaLLMClient


class TestQueryExpander:
    """查询扩展器测试（仅 LLM 语义扩展）"""

    def test_initialization(self):
        """测试初始化"""
        llm_client = OllamaLLMClient(
            base_url="http://localhost:11434",
            model="qwen2.5:14b"
        )
        expander = QueryExpander(llm_client=llm_client)

        assert expander.llm_client is not None
        assert expander.max_total_expansions == 5

    def test_initialization_with_custom_limit(self):
        """测试自定义初始化"""
        llm_client = OllamaLLMClient(
            base_url="http://localhost:11434",
            model="qwen2.5:14b"
        )
        expander = QueryExpander(
            llm_client=llm_client,
            max_total_expansions=10
        )

        assert expander.max_total_expansions == 10

    def test_initialization_requires_llm(self):
        """测试初始化必须提供 LLM 客户端"""
        with pytest.raises(ValueError, match="llm_client 必须提供"):
            QueryExpander(llm_client=None)

    @pytest.mark.asyncio
    async def test_expand_order(self):
        """测试扩展结果顺序（原始查询在最后）"""
        llm_client = OllamaLLMClient(
            base_url="http://localhost:11434",
            model="qwen2.5:14b"
        )
        expander = QueryExpander(llm_client=llm_client)

        # Mock LLM 响应
        async def mock_ainvoke(prompt):
            return "扩展查询1\n扩展查询2\n扩展查询3"

        expander.llm_client.ainvoke = mock_ainvoke

        expansions = await expander.expand("测试查询")

        # 原始查询应该在最后
        assert expansions[-1] == "测试查询"

    @pytest.mark.asyncio
    async def test_expand_limit(self):
        """测试扩展数量限制"""
        llm_client = OllamaLLMClient(
            base_url="http://localhost:11434",
            model="qwen2.5:14b"
        )
        expander = QueryExpander(
            llm_client=llm_client,
            max_total_expansions=5
        )

        # Mock LLM 响应（返回 10 个扩展）
        async def mock_ainvoke(prompt):
            return "\n".join([f"扩展查询{i}" for i in range(1, 11)])

        expander.llm_client.ainvoke = mock_ainvoke

        expansions = await expander.expand("测试查询", max_expansions=10)

        # 结果数量应该受限
        assert len(expansions) <= 5

    def test_parse_llm_response(self):
        """测试 LLM 响应解析"""
        llm_client = OllamaLLMClient(
            base_url="http://localhost:11434",
            model="qwen2.5:14b"
        )
        expander = QueryExpander(llm_client=llm_client)

        # 测试编号列表
        response = "1. 扩展查询1\n2. 扩展查询2\n3. 扩展查询3"
        parsed = expander._parse_llm_response(response)
        assert len(parsed) == 3
        assert parsed == ["扩展查询1", "扩展查询2", "扩展查询3"]

        # 测试无编号列表
        response = "扩展查询1\n扩展查询2\n扩展查询3"
        parsed = expander._parse_llm_response(response)
        assert len(parsed) == 3

        # 测试带符号的列表
        response = "- 扩展查询1\n- 扩展查询2\n* 扩展查询3"
        parsed = expander._parse_llm_response(response)
        assert len(parsed) == 3

        # 测试混合格式
        response = "1. 扩展查询1\n- 扩展查询2\n扩展查询3"
        parsed = expander._parse_llm_response(response)
        assert len(parsed) == 3

    @pytest.mark.asyncio
    async def test_expand_deduplication(self):
        """测试扩展去重"""
        llm_client = OllamaLLMClient(
            base_url="http://localhost:11434",
            model="qwen2.5:14b"
        )
        expander = QueryExpander(llm_client=llm_client)

        # Mock LLM 响应（包含重复）
        async def mock_ainvoke(prompt):
            return "扩展查询1\n扩展查询2\n扩展查询1\n扩展查询3"

        expander.llm_client.ainvoke = mock_ainvoke

        expansions = await expander.expand("测试查询")

        # 检查没有重复
        assert len(expansions) == len(set(expansions))

    @pytest.mark.asyncio
    async def test_expand_empty_response(self):
        """测试 LLM 返回空响应"""
        llm_client = OllamaLLMClient(
            base_url="http://localhost:11434",
            model="qwen2.5:14b"
        )
        expander = QueryExpander(llm_client=llm_client)

        # Mock LLM 响应（空）
        async def mock_ainvoke(prompt):
            return ""

        expander.llm_client.ainvoke = mock_ainvoke

        expansions = await expander.expand("测试查询")

        # 应该只包含原始查询
        assert expansions == ["测试查询"]

    @pytest.mark.asyncio
    async def test_expand_error_handling(self):
        """测试 LLM 调用失败时的错误处理"""
        llm_client = OllamaLLMClient(
            base_url="http://localhost:11434",
            model="qwen2.5:14b"
        )
        expander = QueryExpander(llm_client=llm_client)

        # Mock LLM 调用失败
        async def mock_ainvoke(prompt):
            raise Exception("LLM 服务不可用")

        expander.llm_client.ainvoke = mock_ainvoke

        expansions = await expander.expand("测试查询")

        # 应该只包含原始查询
        assert expansions == ["测试查询"]


class TestQueryExpanderIntegration:
    """查询扩展集成测试（需要 LLM 服务）"""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="需要 LLM 服务运行")
    async def test_llm_semantic_expand_with_decomposition(self):
        """测试复合查询分解"""
        llm_client = OllamaLLMClient(
            base_url="http://localhost:11434",
            model="qwen2.5:14b"
        )

        expander = QueryExpander(llm_client=llm_client)
        expansions = await expander.expand("导数和积分的关系")

        # 应该包含分解后的查询
        assert any("导数" in e for e in expansions)
        assert any("积分" in e for e in expansions)
        assert "导数和积分的关系" in expansions
        assert expansions[-1] == "导数和积分的关系"

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="需要 LLM 服务运行")
    async def test_llm_semantic_expand_with_synonyms(self):
        """测试同义词扩展"""
        llm_client = OllamaLLMClient(
            base_url="http://localhost:11434",
            model="qwen2.5:14b"
        )

        expander = QueryExpander(llm_client=llm_client)
        expansions = await expander.expand("如何求微商？")

        # 应该包含"导数"等同义词
        assert any("导数" in e for e in expansions)
        assert "如何求微商？" in expansions

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="需要 LLM 服务运行")
    async def test_llm_semantic_expand_simple_query(self):
        """测试简单查询不过度分解"""
        llm_client = OllamaLLMClient(
            base_url="http://localhost:11434",
            model="qwen2.5:14b"
        )

        expander = QueryExpander(llm_client=llm_client)
        expansions = await expander.expand("极限的定义")

        # 简单查询应该有语义扩展，但不过度分解
        assert len(expansions) >= 2
        assert "极限的定义" in expansions

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="需要 LLM 服务运行")
    async def test_full_expansion_with_llm(self):
        """测试完整扩展流程（包含 LLM）"""
        llm_client = OllamaLLMClient(
            base_url="http://localhost:11434",
            model="qwen2.5:14b"
        )

        expander = QueryExpander(llm_client=llm_client)
        expansions = await expander.expand("导数和积分的关系")

        # 应该有多种扩展
        assert len(expansions) >= 3
        assert "导数和积分的关系" in expansions
        # 原始查询应该在最后
        assert expansions[-1] == "导数和积分的关系"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
