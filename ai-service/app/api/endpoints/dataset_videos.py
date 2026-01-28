import hashlib
import os
from datetime import datetime

import aiohttp
from fastapi import (
    APIRouter, Depends, HTTPException
)
from starlette import status

from app.api.endpoints.documents import UPLOAD_DIR
from app.api.middleware.auth import get_api_key
from app.core.database import get_database
from app.core.logging import get_logger
from app.models.schemas import CreateRagDocumentResponse, CreateDatasetVideoRequest, ResponseResult
from app.services.task_processor import task_processor

logger = get_logger(__name__)

router = APIRouter()


@router.post("/create", response_model=CreateRagDocumentResponse)
async def create_dataset_video(
    request: CreateDatasetVideoRequest,
    api_key: str = Depends(get_api_key)
):
    """
        创建知识库视频资源

        Args:
        - request: create_dataset_video request
        - api_key: API key from auth
        - db: Database instance

    Returns:
        - Created dataset video data
    """
    try:
        logger.info(
            "create_dataset_video request",
            kb_id=request.kb_id
        )
        hash_obj = hashlib.md5(f"{request.document_name}_{request.kb_id}_{datetime.utcnow().timestamp()}".encode())
        doc_id = f"video_{hash_obj.hexdigest()[:12]}"

        file_path = None
        try:
            # Generate temporary file path (using pathlib)
            file_ext = os.path.splitext(request.document_name)[1]
            temp_filename = f"java_upload_{request.resource_id}_{datetime.utcnow().timestamp()}{file_ext}"
            file_path = UPLOAD_DIR / temp_filename

            # Download file with timeout
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        request.resource_url, timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status != 200:
                        return ResponseResult.error(status.HTTP_400_BAD_REQUEST,
                                                    "Failed to download file from URL", None)

                    # Save file to disk
                    with open(file_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)

            logger.info("Downloaded file from URL", url=request.resource_url, file_path=str(file_path))
        except aiohttp.ClientError as e:
            logger.error("Failed to download file", url=request.resource_url, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download file: {str(e)}",
            )

        # 创建任务
        task_id = await task_processor.submit_task(
                kb_id=request.kb_id,
                filename=request.document_name,
                file_path=str(file_path),
                category=None,
                chunk_config=None,
                doc_id=doc_id,
                resource_id=request.resource_id
            )

        data = {
            "doc_id": doc_id,
            "task_id": task_id,
            "status": "processing",
            "resource_id": request.resource_id,
            "document_name": request.document_name,
            "kb_id": request.kb_id
        }
        return ResponseResult.success(data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "create_dataset_video exception",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="create_dataset_video error",
        )


@router.delete("/delete/{doc_id}")
async def delete_documents(
    doc_id: str,
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
        删除文档或视频资源

        nArgs:
            - doc_id: document ID
            - api_key: API key from auth
            - db: Database instance

        Returns:
            - result
        """
    try:
        logger.info("delete_documents request", doc_id=doc_id)

        # 删除数据
        result = await db.documents.delete_one({'doc_id': doc_id})

        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="delete_documents cannot be deleted (not found or already completed)"
            )

        if result.deleted_count == 1:
            # 删除任务
            await db.document_tasks.delete_one({'doc_id': doc_id})

            return ResponseResult.success(None)
        else:
            logger.error(f"delete_documents {doc_id} not found")
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"delete_documents {doc_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "delete_documents exception",
            doc_id=doc_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="delete_documents error"
        )


@router.post("/restart_task/{task_id}")
async def restart_task(
    task_id: str,
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
        重新启动任务：文档或视频资源

        nArgs:
            - task_id: task ID
            - api_key: API key from auth
            - db: Database instance

        Returns:
            - result
        """
    try:
        logger.info("restart_task request", task_id=task_id)

        task = await db.document_tasks.find_one({"task_id": task_id})

        if not task:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"restart_task {task_id} not found")

        await task_processor.restart_task(
            kb_id=task["kb_id"],
            filename=task["filename"],
            file_path=task["file_path"],
            category=task["category"],
            chunk_config=task["metadata"]["chunk_config"],
            doc_id=task["doc_id"],
            resource_id=task["metadata"]["resource_id"],
        )

        return ResponseResult.success(None)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "restart_task exception",
            task_id=task_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="restart_task error"
        )


@router.post("set_enhance/{video_id}")
async def set_enhance(
    doc_id: str,
    enhance: int,
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
        设置文档或视频资源是否知识增强

        nArgs:
            - doc_id: 文档或视频资源 ID
            - enhance: 0=不增强，1=增强
            - api_key: API key from auth
            - db: Database instance

        Returns:
            - result
        """
    try:
        logger.info("set_enhance request", doc_id=doc_id)

        doc = await db.documents.find_one({"doc_id": doc_id})

        if not doc:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"set_enhance {doc_id} not found")

        result = await db.documents.update_one(
            {'doc_id': doc_id},
            {"$set": {"enhance": enhance}}
        )

        if result and result.modified_count == 1:
            return ResponseResult.success(None)
        else:
            return ResponseResult.error(status.HTTP_400_BAD_REQUEST, "error", f"set_enhance {doc_id} failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "set_enhance exception",
            doc_id=doc_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="set_enhance error"
        )
