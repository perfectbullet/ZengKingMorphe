"""
测试文档摘要生成器
"""

import pytest

from src.document_indexer.summarizer import DocumentSummarizer


@pytest.mark.asyncio
async def test_summarize_chunk():
    """测试 chunk 摘要生成"""
    summarizer = DocumentSummarizer()

    text = """
    人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
    这包括学习、推理、问题解决、感知和语言理解等能力。
    """

    summary = await summarizer.summarize_chunk(text)

    assert summary is not None
    assert len(summary) <= 53  # 50 + "..."
    assert len(summary) > 0


@pytest.mark.asyncio
async def test_summarize_short_text():
    """测试短文本摘要生成（应返回 None）"""
    summarizer = DocumentSummarizer()

    text = "短文本"

    summary = await summarizer.summarize_chunk(text)

    # 短文本（<10 字符）返回 None
    assert summary is None


@pytest.mark.asyncio
async def test_summarize_batch():
    """测试批量摘要生成"""
    summarizer = DocumentSummarizer()

    texts = [
        "机器学习是人工智能的一个子领域，专注于算法和统计模型。",
        "深度学习使用神经网络来模拟人脑的学习过程。"
    ]

    summaries = await summarizer.summarize_batch(texts)

    assert len(summaries) == 2
    assert all(s is not None for s in summaries)


@pytest.mark.asyncio
async def test_summarize_section():
    """测试 section 摘要生成"""
    summarizer = DocumentSummarizer()

    chunks = [
        "第一章介绍机器学习的基本概念。",
        "第二章介绍监督学习和无监督学习的区别。",
        "第三章介绍常用的机器学习算法。"
    ]

    summary = await summarizer.summarize_section(chunks)

    assert summary is not None
    assert len(summary) <= 103  # 100 + "..."
    assert len(summary) > 0


@pytest.mark.asyncio
async def test_summarize_document():
    """测试文档摘要生成"""
    summarizer = DocumentSummarizer()

    chunks = [
        "第一章介绍机器学习的基本概念。",
        "第二章介绍监督学习和无监督学习的区别。",
        "第三章介绍常用的机器学习算法。",
        "第四章介绍模型评估和选择方法。",
        "第五章介绍实际应用案例。"
    ]

    summary = await summarizer.summarize_document(chunks)

    assert summary is not None
    assert len(summary) <= 203  # 200 + "..."
    assert len(summary) > 0
