"""
工具函数模块
"""

import logging
from pathlib import Path
from typing import Any, List
from loguru import logger


def setup_logger(log_level: str = "INFO", log_file: str = "./logs/rag_sdk.log") -> logging.Logger:
    """
    配置日志系统

    Args:
        log_level: 日志级别
        log_file: 日志文件路径

    Returns:
        配置好的 logger 实例
    """
    # 确保日志目录存在
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 配置 loguru
    logger.remove()  # 移除默认 handler
    logger.add(
        log_file,
        rotation="10 MB",
        retention="7 days",
        level=log_level,
        encoding="utf-8"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level
    )

    return logger


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[str]:
    """
    将文本分块

    Args:
        text: 输入文本
        chunk_size: 每块大小（字符数）
        overlap: 重叠大小（字符数）

    Returns:
        分块后的文本列表
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # 计算下一块的起始位置
        start = end - overlap

        # 防止无限循环
        if start >= len(text):
            break

    return chunks


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除不安全字符

    Args:
        filename: 原始文件名

    Returns:
        清理后的文件名
    """
    invalid_chars = '<>:"/\\|?*\0'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    截断文本到指定长度

    Args:
        text: 输入文本
        max_length: 最大长度

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def merge_metadata(base: dict, override: dict) -> dict:
    """
    合并元数据字典

    Args:
        base: 基础元数据
        override: 覆盖的元数据

    Returns:
        合并后的元数据
    """
    merged = base.copy()
    merged.update(override)
    return merged
