"""
Knowledge base management API endpoints.
"""
import hashlib
import os
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse
from datetime import datetime
from pathlib import Path

from app.api.middleware.auth import get_api_key
from app.core.logging import get_logger
from app.core.database import get_database
from app.core.config import settings
from app.core.chroma import chroma_db
from app.core.elasticsearch import es_db
from app.models.schemas import (
    CreateKnowledgeBaseRequest,
    UpdateKnowledgeBaseRequest, ResponseResult,
)

logger = get_logger(__name__)

router = APIRouter()

# Temporary upload directory (using pathlib for cross-platform compatibility)
UPLOAD_DIR = Path(__file__).parent.parent.parent.parent / "../upload_docs"
UPLOAD_DIR = UPLOAD_DIR.resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Upload directory configured: {UPLOAD_DIR}")


@router.post("/create")
async def create_knowledge_base(
    request: CreateKnowledgeBaseRequest,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
):
    """
        创建知识库。

        Args:
            - request: create_knowledge_base request
            - api_key: API key from auth
            - db: Database instance

        Returns:
            - Created knowledge base data
    """
    try:
        logger.info(f"create_knowledge_base request name={request.name} category={request.category}")

        # Generate KB ID
        kb_id = f"kb_{hashlib.md5(f'{request.name}_{datetime.utcnow().timestamp()}'.encode()).hexdigest()[:12]}"

        # Create KB document
        kb_doc = {
            "kb_id": kb_id,
            "name": request.name,
            "description": request.description,
            "category": request.category,
            "priority": request.priority,
            "tags": request.tags,
            "config": request.config.model_dump(),
            "status": "active",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        # Insert into knowledge_bases collection
        result = await db.knowledge_bases.insert_one(kb_doc)

        if result and result.inserted_id:
            data = {
                "kb_id": kb_id,
                "name": request.name,
                "status": "active",
                "created_at": kb_doc["created_at"].isoformat() + "Z"
            }
            return ResponseResult.success(data)
        else:
            return ResponseResult.error(status.HTTP_400_BAD_REQUEST, "error",
                                        "create_knowledge_base failed")

    except Exception as e:
        logger.error(f"create_knowledge_base exception error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="create_knowledge_base error"
        )


@router.post("/update")
async def update_knowledge_base(
    request: UpdateKnowledgeBaseRequest,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
):
    """
        更新知识库信息。

        Args:
            - request: Update knowledge base request (includes kb_id)
            - api_key: API key from auth
            - db: Database instance

        Returns:
            - Updated knowledge base data
    """
    kb_id = request.kb_id
    try:
        logger.info(f"update_knowledge_base request kb_id={kb_id} updates={request.model_dump(exclude_none=True)}")

        # Check if knowledge base exists
        kb = await db.knowledge_bases.find_one({'kb_id': kb_id})
        if not kb:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"update_knowledge_base not found kb_id={kb_id}")

        # Build update data (only include non-None fields)
        update_data = {}
        if request.name is not None:
            update_data["name"] = request.name
        if request.description is not None:
            update_data["description"] = request.description
        if request.priority is not None:
            update_data["priority"] = request.priority
        if request.tags is not None:
            update_data["tags"] = request.tags

        # Add updated_at timestamp
        update_data["updated_at"] = datetime.utcnow()

        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="update_knowledge_base no valid fields to update"
            )

        # Update knowledge base
        result = await db.knowledge_bases.update_one(
            {'kb_id': kb_id},
            {"$set": update_data}
        )

        if result and result.modified_count == 1:
            logger.info(
                f"update_knowledge_base {kb_id} updated success",
                kb_id=kb_id,
                fields_updated=list(update_data.keys())
            )

            # Get updated document
            updated_kb = await db.knowledge_bases.find_one({"kb_id": kb_id})
            updated_kb.pop("_id", None)

            data = {
                "kb_id": updated_kb["kb_id"],
                "name": updated_kb["name"],
                "description": updated_kb["description"],
                "priority": updated_kb["priority"],
                "tags": updated_kb.get("tags", []),
                "status": updated_kb["status"],
                "created_at": updated_kb["created_at"].isoformat() + "Z",
                "updated_at": updated_kb["updated_at"].isoformat() + "Z"
            }
            return ResponseResult.success(data)
        else:
            return ResponseResult.error(status.HTTP_400_BAD_REQUEST, "error",
                                        "update_knowledge_base failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"update_knowledge_base exception kb_id={kb_id} error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="update_knowledge_base error"
        )


@router.delete("/delete/{kb_id}")
async def delete_knowledge_bases(
    kb_id: str,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
):
    """
    删除知识库及其所有关联数据。

    Args:
        - kb_id: knowledge base ID
        - api_key: API key from auth
        - db: Database instance

    Returns:
        - result with chunks_deleted count
    """
    try:
        logger.info(f"delete_knowledge_bases request kb_id={kb_id}")

        # 检查知识库是否存在
        kb = await db.knowledge_bases.find_one({'kb_id': kb_id})
        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base {kb_id} not found"
            )

        # 获取该知识库的所有 chunk_id（用于删除 ChromaDB 和 ES）
        chunks_cursor = db.document_chunks.find({"kb_id": kb_id}, {"chunk_id": 1})
        chunks = await chunks_cursor.to_list(length=None)
        chunk_ids = [c["chunk_id"] for c in chunks]

        logger.info(
            "Deleting knowledge base data",
            kb_id=kb_id,
            chunks_count=len(chunk_ids)
        )

        # 1. 删除 MongoDB document_chunks
        chunks_result = await db.document_chunks.delete_many({"kb_id": kb_id})
        logger.info(
            "Deleted document_chunks",
            kb_id=kb_id,
            count=chunks_result.deleted_count
        )

        # 2. 删除 MongoDB documents
        docs_result = await db.documents.delete_many({"kb_id": kb_id})
        logger.info(
            "Deleted documents",
            kb_id=kb_id,
            count=docs_result.deleted_count
        )

        # 3. 删除 ChromaDB 向量数据
        if chunk_ids:
            try:
                await chroma_db.delete_documents(collection_name="doc", ids=chunk_ids)
                logger.info(
                    "Deleted ChromaDB documents",
                    kb_id=kb_id,
                    count=len(chunk_ids)
                )
            except Exception as e:
                logger.error(
                    "Failed to delete ChromaDB data",
                    kb_id=kb_id,
                    error=str(e),
                    exc_info=True
                )

        # 4. 删除 ElasticSearch 索引数据
        if chunk_ids:
            deleted_es_count = 0
            for chunk_id in chunk_ids:
                try:
                    await es_db.delete_document(index=es_db.doc_index, doc_id=chunk_id)
                    deleted_es_count += 1
                except Exception as e:
                    logger.warning(
                        "Failed to delete ES document",
                        chunk_id=chunk_id,
                        error=str(e)
                    )
            logger.info(
                "Deleted ES documents",
                kb_id=kb_id,
                count=deleted_es_count
            )

        # 5. 最后删除知识库元数据
        result = await db.knowledge_bases.delete_one({'kb_id': kb_id})

        if result.deleted_count == 1:
            logger.info(
                "Knowledge base deleted successfully",
                kb_id=kb_id,
                chunks_deleted=len(chunk_ids)
            )
            return ResponseResult.success({
                "kb_id": kb_id,
                "chunks_deleted": len(chunk_ids)
            })
        else:
            logger.error(f"delete_knowledge_bases {kb_id} failed")
            return ResponseResult.error(
                status.HTTP_400_BAD_REQUEST,
                "error",
                f"delete_knowledge_bases {kb_id} failed"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"delete_knowledge_bases exception kb_id={kb_id} error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="delete_knowledge_bases error"
        )


@router.get("/list")
async def list_knowledge_bases(
    category: str = Query(None, description="Filter by category"),
    status_filter: str = Query(None, alias="status", description="Filter by status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    获取知识库列表。

    \nArgs:
        \n- category: Filter by category
        \n- status_filter: Filter by status
        \n- page: Page number
        \n- page_size: Page size
        \n- api_key: API key from auth
        \n- db: Database instance

    \nReturns:
        \n- List of knowledge bases
    """
    try:
        # Build query
        query = {}
        if category:
            query["category"] = category
        if status_filter:
            query["status"] = status_filter

        # Get total count
        total = await db.knowledge_bases.count_documents(query)

        # Get paginated results
        cursor = db.knowledge_bases.find(query).skip((page - 1) * page_size).limit(page_size)
        kbs = await cursor.to_list(length=page_size)

        # Format results
        items = []
        for kb in kbs:
            kb.pop("_id", None)
            items.append({
                "kb_id": kb["kb_id"],
                "name": kb["name"],
                "category": kb["category"],
                "status": kb["status"],
                "created_at": kb["created_at"].isoformat() + "Z"
            })

        return {
            "code": 200,
            "message": "success",
            "data": {
                "total": total,
                "page": page,
                "page_size": page_size,
                "items": items
            }
        }

    except Exception as e:
        logger.error(
            "List knowledge bases error",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list knowledge bases"
        )


@router.get("/test-files")
async def list_test_files(
    api_key: str = Depends(get_api_key)
):
    """
    列出测试文件夹中的所有文件（仅用于测试）。

    \nReturns:
        \n- List of available test files with download URLs
    """
    try:
        logger.info("List test files request")

        # Ensure test files directory exists
        test_files_dir = settings.test_files_dir
        os.makedirs(test_files_dir, exist_ok=True)

        # Get all files in test files directory
        test_files_path = Path(test_files_dir)
        if not test_files_path.exists():
            return {
                "code": 200,
                "message": "success",
                "data": {
                    "files": [],
                    "test_files_dir": str(test_files_path),
                    "note": "Test files directory does not exist yet"
                }
            }

        files = []
        for file_path in test_files_path.iterdir():
            if file_path.is_file():
                # Get file info
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "download_url": f"/api/knowledge-base/test-files/download/{file_path.name}",
                    "full_download_url": f"http://localhost:8000/api/knowledge-base/test-files/download/{file_path.name}"
                })

        return {
            "code": 200,
            "message": "success",
            "data": {
                "test_files_dir": str(test_files_path),
                "total_files": len(files),
                "files": files
            }
        }

    except Exception as e:
        logger.error(
            "List test files error",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list test files"
        )


@router.get("/test-files/download/{filename}")
async def download_test_file(
    filename: str,
    api_key: str = Depends(get_api_key)
):
    """
    下载测试文件（仅用于测试）。

    \nArgs:
        \n- filename: Name of the file to download

    \nReturns:
        \n- File download response
    """
    try:
        logger.info("Download test file request", filename=filename)

        # Security check: prevent path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename"
            )

        # Build file path
        test_files_dir = settings.test_files_dir
        file_path = Path(test_files_dir) / filename

        # Check if file exists
        if not file_path.exists() or not file_path.is_file():
            logger.warning(
                "Test file not found",
                filename=filename,
                requested_path=str(file_path)
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File '{filename}' not found in test files directory"
            )

        # Return file
        logger.info(
            "Serving test file",
            filename=filename,
            file_path=str(file_path)
        )
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Download test file error",
            filename=filename,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download test file"
        )


@router.get("/{kb_id}")
async def get_knowledge_base_detail(
    kb_id: str,
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    获取知识库详情及前10个文档块。

    \nArgs:
        \n- kb_id: Knowledge base ID
        \n- api_key: API key from auth
        \n- db: Database instance

    \nReturns:
        \n- Knowledge base detail information with top 10 document chunks
    """
    try:
        logger.info("Get knowledge base detail request", kb_id=kb_id)

        # Query knowledge base by kb_id
        kb = await db.knowledge_bases.find_one({"kb_id": kb_id})

        if not kb:
            logger.warning("Knowledge base not found", kb_id=kb_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base {kb_id} not found"
            )

        # Remove MongoDB _id field
        kb.pop("_id", None)

        # Format datetime fields
        kb["created_at"] = kb["created_at"].isoformat() + "Z"
        if kb.get("updated_at"):
            kb["updated_at"] = kb["updated_at"].isoformat() + "Z"

        # Query top 10 document chunks for this knowledge base
        chunks_cursor = db.document_chunks.find({"kb_id": kb_id}).limit(10).sort("created_at", -1)
        chunks = await chunks_cursor.to_list(length=10)

        # Format chunks for response
        formatted_chunks = []
        for chunk in chunks:
            chunk.pop("_id", None)
            # Format datetime if exists
            if "created_at" in chunk and chunk["created_at"]:
                chunk["created_at"] = chunk["created_at"].isoformat() + "Z"
            formatted_chunks.append(chunk)

        # Add chunks to knowledge base data
        kb["chunks"] = formatted_chunks
        kb["total_chunks"] = len(formatted_chunks)

        logger.info(
            "Knowledge base detail retrieved",
            kb_id=kb_id,
            name=kb.get("name"),
            chunks_count=len(formatted_chunks)
        )

        return {
            "code": 200,
            "message": "success",
            "data": kb
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Get knowledge base detail error",
            kb_id=kb_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get knowledge base detail"
        )
