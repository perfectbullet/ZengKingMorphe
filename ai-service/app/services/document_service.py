"""
Document service utilities.

文档处理相关工具函数和常量。

注意：DocumentProcessor 已被移除，所有文档处理功能已迁移到：
- llama-rag-sdk RAGSystem（核心功能）
- app.services.task_processor（MongoDB 文档记录管理）
"""

import os
import hashlib
from datetime import datetime
from pathlib import Path

import aiohttp
from fastapi import HTTPException, status

from app.core.logging import setup_logging, get_logger
from app.models.schemas import ResponseResult

logger = get_logger(__name__)

# 临时文件下载目录
UPLOAD_DIR = Path(__file__).parent.parent.parent.parent / "upload_docs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def download_file(
        file_name: str,
        resource_id: int,
        resource_url: str) -> str:
    try:
        logger.info(f"download_file resource_id={resource_id} resource_url={resource_url}")

        # Generate temporary file path (using pathlib)
        file_ext = os.path.splitext(file_name)[1] or ".txt"
        temp_filename = f"java_upload_{resource_id}_{datetime.utcnow().timestamp()}{file_ext}"
        file_path = UPLOAD_DIR / temp_filename

        # Download file with timeout
        async with aiohttp.ClientSession() as session:
            async with session.get(
                    resource_url, timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status != 200:
                    return ResponseResult.error(status.HTTP_400_BAD_REQUEST,
                                                f"download_file from URL={resource_url}", None)

                # Save file to disk
                with open(file_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)

        logger.info(f"download_file resource_id={resource_id} file_path={file_path}")

        return str(file_path)

    except aiohttp.ClientError as e:
        logger.error(f"download_file exception resource_id={resource_id}, error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"download_file error",
        )
        
        
        
def generate_doc_id(filename: str, kb_id: str) -> str:
    """
        Generate unique document ID.

        Args:
            filename: Document filename
            kb_id: Knowledge base ID

        Returns:
            Unique document ID
    """
    timestamp = datetime.utcnow().timestamp()
    content = f"{filename}_{kb_id}_{timestamp}"
    hash_obj = hashlib.md5(content.encode())
    return f"doc_{hash_obj.hexdigest()[:12]}"
