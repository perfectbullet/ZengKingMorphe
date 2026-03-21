"""
文档摘要生成器

为 chunk/section/document 三个层级生成中文摘要
"""

from typing import Optional, List
import asyncio
from loguru import logger

from src.retrieval.llm_client import create_llm_client
from src.config import settings


class DocumentSummarizer:
    """
    文档摘要生成器

    使用 LLM 为文本生成简洁的中文摘要
    """

    # 摘要提示词模板（中文输出）
    CHUNK_SUMMARY_PROMPT = """请用简洁的中文总结以下文本的核心内容，不超过50字。

文本：
{text}

摘要："""

    SECTION_SUMMARY_PROMPT = """请用简洁的中文总结以下内容的主要观点，不超过100字。

文本：
{text}

摘要："""

    DOCUMENT_SUMMARY_PROMPT = """请用简洁的中文总结以下文档的整体内容和主要要点，不超过200字。

文本：
{text}

摘要："""

    def __init__(self):
        self._llm_client = None

    @property
    def llm_client(self):
        """延迟初始化 LLM 客户端"""
        if self._llm_client is None:
            self._llm_client = create_llm_client(
                provider=settings.llm_provider,
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                timeout=settings.llm_timeout
            )
        return self._llm_client

    async def summarize_chunk(
        self,
        text: str,
        max_length: int = 50
    ) -> Optional[str]:
        """
        为单个 chunk 生成摘要

        Args:
            text: 文本内容
            max_length: 摘要最大长度

        Returns:
            摘要文本，失败返回 None
        """
        if not text or len(text.strip()) < 10:
            return None

        # 截取前 1000 字符用于生成摘要
        prompt_text = text[:1000]
        prompt = self.CHUNK_SUMMARY_PROMPT.format(text=prompt_text)

        try:
            summary = await self.llm_client.ainvoke(
                prompt,
                max_tokens=100,
                temperature=0.3
            )
            # 清理并截断
            summary = summary.strip()
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            return summary
        except Exception as e:
            logger.warning(f"Chunk 摘要生成失败: {e}")
            return None

    async def summarize_section(
        self,
        chunks: List[str],
        max_length: int = 100
    ) -> Optional[str]:
        """
        为 section 生成摘要（合并多个 chunks）

        Args:
            chunks: 文本块列表
            max_length: 摘要最大长度

        Returns:
            摘要文本，失败返回 None
        """
        if not chunks:
            return None

        # 只取前 3 个 chunk，控制输入长度
        combined = "\n\n".join(chunks[:3])[:2000]
        prompt = self.SECTION_SUMMARY_PROMPT.format(text=combined)

        try:
            summary = await self.llm_client.ainvoke(
                prompt,
                max_tokens=150,
                temperature=0.3
            )
            summary = summary.strip()
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            return summary
        except Exception as e:
            logger.warning(f"Section 摘要生成失败: {e}")
            return None

    async def summarize_document(
        self,
        chunks: List[str],
        max_length: int = 200
    ) -> Optional[str]:
        """
        为整个文档生成摘要

        Args:
            chunks: 文本块列表
            max_length: 摘要最大长度

        Returns:
            摘要文本，失败返回 None
        """
        if not chunks:
            return None

        # 采样策略：取开头、中间、结尾的 chunk
        sample_chunks = []
        n = len(chunks)

        if n <= 3:
            sample_chunks = chunks
        else:
            sample_chunks = [
                chunks[0],           # 开头
                chunks[n // 2],      # 中间
                chunks[-1]           # 结尾
            ]

        combined = "\n\n".join(sample_chunks)[:3000]
        prompt = self.DOCUMENT_SUMMARY_PROMPT.format(text=combined)

        try:
            summary = await self.llm_client.ainvoke(
                prompt,
                max_tokens=250,
                temperature=0.3
            )
            summary = summary.strip()
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            return summary
        except Exception as e:
            logger.warning(f"Document 摘要生成失败: {e}")
            return None

    async def summarize_batch(
        self,
        texts: List[str],
        summary_type: str = "chunk",
        concurrent: int = 5
    ) -> List[Optional[str]]:
        """
        批量生成摘要（并行处理）

        Args:
            texts: 文本列表
            summary_type: 摘要类型 (chunk/section/document)
            concurrent: 并发数量

        Returns:
            摘要列表
        """
        async def summarize_one(text: str) -> Optional[str]:
            if summary_type == "chunk":
                return await self.summarize_chunk(text)
            elif summary_type == "section":
                return await self.summarize_section([text])
            else:
                return await self.summarize_chunk(text)

        # 分批并行处理，避免过载
        results = []
        for i in range(0, len(texts), concurrent):
            batch = texts[i:i + concurrent]
            batch_results = await asyncio.gather(
                *[summarize_one(text) for text in batch],
                return_exceptions=True
            )
            # 处理异常结果
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.warning(f"摘要生成异常: {result}")
                    results.append(None)
                else:
                    results.append(result)

        return results
