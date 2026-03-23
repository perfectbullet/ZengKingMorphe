"""
文档解析模块

提供文档解析、图片处理等功能
"""

from llama_rag_sdk.document_parser.base import DocumentParser, ParsedDocument, TextChunk, ImageInfo
from llama_rag_sdk.document_parser.mineru_client import (
    MinerUParser,
    ParseOptions,
    ReturnOptions,
    STAGE_NAMES,
    MinerUError,
    FileNotFoundError as MinerUFileNotFoundError,
    FileUploadError,
    ParseError as MinerUParseError,
    DownloadError,
    ExtractionError,
    TimeoutError as MinerUTimeoutError,
)
from llama_rag_sdk.document_parser.image_processor import ImageDescriptor

__all__ = [
    # 基类和模型
    "DocumentParser",
    "ParsedDocument",
    "TextChunk",
    "ImageInfo",
    # MinerU 解析器
    "MinerUParser",
    "ParseOptions",
    "ReturnOptions",
    "STAGE_NAMES",
    # MinerU 异常
    "MinerUError",
    "MinerUFileNotFoundError",
    "FileUploadError",
    "MinerUParseError",
    "DownloadError",
    "ExtractionError",
    "MinerUTimeoutError",
    # 图片处理器
    "ImageDescriptor",
]
