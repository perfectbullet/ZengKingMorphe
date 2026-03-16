"""
文档解析模块

提供文档解析、图片处理等功能
"""

from src.document_parser.base import DocumentParser, ParsedDocument, TextChunk, ImageInfo
from src.document_parser.mineru_client import MinerUParser
from src.document_parser.image_processor import ImageDescriptor

__all__ = [
    "DocumentParser",
    "ParsedDocument",
    "TextChunk",
    "ImageInfo",
    "MinerUParser",
    "ImageDescriptor",
]
