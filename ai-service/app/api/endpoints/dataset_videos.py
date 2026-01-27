import hashlib
import os
from datetime import datetime
from pathlib import Path

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
from app.services.document_service import generate_doc_id
from app.services.task_processor import task_processor

logger = get_logger(__name__)

router = APIRouter()


@router.post("/create", response_model=CreateRagDocumentResponse)
async def create_dataset_video(
    request: CreateDatasetVideoRequest,
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
        创建知识库视频资源
    """
    try:
        # kb = await db.knowledge_bases.find_one({"kb_id": request.kb_id})
        # if not kb:
        #     return ResponseResult.error(status.HTTP_404_NOT_FOUND, f"Knowledge base {request.kb_id} not found", None)

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
                        return ResponseResult.error(status.HTTP_400_BAD_REQUEST, "Failed to download file from URL", None)

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

        task_id = await task_processor.submit_task(
                kb_id=request.kb_id,
                filename=request.document_name,
                file_path=str(file_path),
                category=None,
                chunk_config=None,
                doc_id=doc_id,
                resource_id=request.resource_id
            )

        return CreateRagDocumentResponse(
            code=200,
            message="success",
            data={
                "doc_id": doc_id,
                "task_id": task_id,
                "status": "processing",
                "resource_id": request.resource_id,
                "document_name": request.document_name,
                "kb_id": request.kb_id
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Create RAG document error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create RAG document",
        )


@router.post("/delete/{video_id}")
async def delete_dataset_video(
    video_id: str,
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
        删除知识库视频资源

        nArgs:
            - video_id: video ID

        Returns:
            - result
        """
    try:
        logger.info("delete_dataset_video request", video_id=video_id)

        # 查询条件
        query = {
            'doc_id': video_id,
            'flag': 1
        }

        # Update knowledge base
        result = await db.documents.update_one(
            query,
            {"$set": {"flag": -1}}
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="delete_dataset_video (not found or already completed)"
            )

        if result.modified_count == 1:
            return ResponseResult.success()
        else:
            return ResponseResult.error(status.HTTP_400_BAD_REQUEST, "error", f"delete_dataset_video {video_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "delete_dataset_video exception",
            video_id=video_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="delete_dataset_video failed"
        )
