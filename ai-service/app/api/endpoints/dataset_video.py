from fastapi import (
    APIRouter, Depends, HTTPException
)
from starlette import status

from app.api.middleware.auth import get_api_key
from app.core.database import get_database
from app.core.logging import get_logger
from app.models.schemas import CreateRagDocumentResponse, CreateDatasetVideoRequest, ResponseResult
from app.services.document_service import generate_video_id, download_file
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
    kb_id = request.kb_id
    try:
        logger.info(
            f"create_dataset_video request: kb_id={request.kb_id}, "
            f"enhance={request.enhance}, document_name={request.document_name}, "
            f"resource_id={request.resource_id}, resource_url={request.resource_url}"
        )

        # 下载远程服务器上的文档文件
        file_path = await download_file(request.document_name, request.resource_id, request.resource_url)

        # 生成唯一文档id
        doc_id = generate_video_id(request.document_name, kb_id)

        # 添加到文档任务列表中
        task_id = await task_processor.submit_task(
            kb_id=request.kb_id,
            enhance=request.enhance,
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
            f"create_dataset_video exception: error={str(e)}",
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
    db=Depends(get_database)
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
        logger.info(f"delete_documents request: doc_id={doc_id}")

        # 删除数据
        result = await db.documents.delete_one({'doc_id': doc_id})

        if not result:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"delete_documents not found doc_id={doc_id}")

        if result.deleted_count == 1:
            # 删除任务
            await db.document_tasks.delete_one({'doc_id': doc_id})

            return ResponseResult.success(None)
        else:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"delete_documents not found doc_id={doc_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"delete_documents exception: doc_id={doc_id}, error={str(e)}",
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
    db=Depends(get_database)
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
        logger.info(f"restart_task request: task_id={task_id}")

        task = await db.document_tasks.find_one({"task_id": task_id})

        if not task:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"restart_task task not found task_id={task_id}")

        await task_processor.re_submit_task(
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
            f"restart_task exception: task_id={task_id}, error={str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="restart_task error"
        )


@router.post("/set_enhance")
async def set_enhance(
    doc_id: str,
    enhance: int,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
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
        logger.info(f"set_enhance request: doc_id={doc_id}")

        doc = await db.documents.find_one({"doc_id": doc_id})

        if not doc:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"set_enhance doc not found doc_id={doc_id}")

        result = await db.documents.update_one(
            {'doc_id': doc_id},
            {"$set": {"enhance": enhance}}
        )

        if result and result.modified_count == 1:
            logger.info(f"set_enhance success doc_id={doc_id}")

            task = await db.document_tasks.find_one({"doc_id": doc_id})

            if not task:
                return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                            f"set_enhance task not found doc_id={doc_id}")

            await task_processor.re_submit_task(
                kb_id=task["kb_id"],
                filename=task["filename"],
                file_path=task["file_path"],
                category=task["category"],
                chunk_config=task["metadata"]["chunk_config"],
                doc_id=task["doc_id"],
                resource_id=task["metadata"]["resource_id"],
            )

            return ResponseResult.success(None)
        else:
            return ResponseResult.error(status.HTTP_400_BAD_REQUEST, "error", f"set_enhance failed doc_id={doc_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"set_enhance exception: doc_id={doc_id}, error={str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="set_enhance error"
        )
