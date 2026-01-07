"""
Knowledge base management API endpoints.
"""
import os
import shutil
import aiohttp
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form, Query
from fastapi.responses import FileResponse
from datetime import datetime
from pathlib import Path

from app.api.middleware.auth import get_api_key
from app.core.logging import get_logger
from app.core.database import get_database
from app.core.chroma import chroma_db
from app.core.config import settings
from app.services.task_processor import task_processor
from app.models.schemas import (
    CreateKnowledgeBaseRequest,
    UpdateKnowledgeBaseRequest,
    CreateRagDocumentRequest,
    CreateRagDocumentResponse
)


logger = get_logger(__name__)

router = APIRouter()

# Temporary upload directory
UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/create")
async def create_knowledge_base(
    request: CreateKnowledgeBaseRequest,
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    创建知识库。
    
    \nArgs:
        \n- request: Create knowledge base request
        \n- api_key: API key from auth
        \n- db: Database instance

    \nReturns:
        \n- Created knowledge base data
    """
    try:
        logger.info(f"Create knowledge base request: name={request.name}, category={request.category}")
        
        # Generate KB ID
        import hashlib
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
            #"doc_count": 0,
            #"chunk_count": 0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Insert into knowledge_bases collection
        await db.knowledge_bases.insert_one(kb_doc)
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "kb_id": kb_id,
                "name": request.name,
                "status": "active",
                "created_at": kb_doc["created_at"].isoformat() + "Z"
            }
        }
        
    except Exception as e:
        logger.error(f"Create knowledge base error: error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create knowledge base"
        )


@router.post("/update")
async def update_knowledge_base(
    request: UpdateKnowledgeBaseRequest,
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
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
    try:
        kb_id = request.kb_id
        logger.info(f"Update knowledge base request: kb_id={kb_id}, updates={request.model_dump(exclude_none=True)}")
        
        # Check if knowledge base exists
        kb = await db.knowledge_bases.find_one({"kb_id": kb_id})
        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base {kb_id} not found"
            )
        
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
                detail="No valid fields to update"
            )
        
        # Update knowledge base
        await db.knowledge_bases.update_one(
            {"kb_id": kb_id},
            {"$set": update_data}
        )
        
        # Get updated document
        updated_kb = await db.knowledge_bases.find_one({"kb_id": kb_id})
        updated_kb.pop("_id", None)
        
        logger.info(f"Knowledge base updated: kb_id={kb_id}, fields_updated={list(update_data.keys())}")
        
        return {
            "code": 200,
            "message": "Knowledge base updated successfully",
            "data": {
                "kb_id": updated_kb["kb_id"],
                "name": updated_kb["name"],
                "description": updated_kb["description"],
                "priority": updated_kb["priority"],
                "tags": updated_kb.get("tags", []),
                "status": updated_kb["status"],
                "created_at": updated_kb["created_at"].isoformat() + "Z",
                "updated_at": updated_kb["updated_at"].isoformat() + "Z"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update knowledge base error: kb_id={kb_id}, error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update knowledge base"
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
                #"doc_count": kb.get("doc_count", 0),
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
        logger.error(f"List knowledge bases error: error={str(e)}", exc_info=True)
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
        logger.error(f"List test files error: error={str(e)}", exc_info=True)
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
            logger.warning("Test file not found", filename=filename, requested_path=str(file_path))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File '{filename}' not found in test files directory"
            )

        # Return file
        logger.info("Serving test file", filename=filename, file_path=str(file_path))
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download test file error: filename={filename}, error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download test file"
        )


@router.get("/{kb_id}")
async def get_knowledge_base_detail(
    kb_id: str = 'kb_316a7dbc75d0',
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

        logger.info("Knowledge base detail retrieved", kb_id=kb_id, name=kb.get("name"), chunks_count=len(formatted_chunks))

        return {
            "code": 200,
            "message": "success",
            "data": kb
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get knowledge base detail error: kb_id={kb_id}, error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get knowledge base detail"
        )


@router.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(..., description="Files to upload (max 50)"),
    kb_id: str = Form('kb_316a7dbc75d0', description="Knowledge base ID"),
    category: str = Form('首饰雕蜡工艺课程', description="Document category"),
    api_key: str = Depends(get_api_key)
):
    """
    上传并处理文档（支持同步/异步模式）。
    
    \nArgs:
        \n- files: Files to upload
        \n- kb_id: Knowledge base ID
        \n- category: Document category
        \n- api_key: API key from auth

    \nReturns:
        \n- Upload results (sync mode: doc_id list; async mode: task_id list)
    """
    try:
        if len(files) > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 50 files allowed per upload"
            )
        logger.info(f"Upload documents request: kb_id={kb_id}, file_count={len(files)}")
        
        # Async mode: submit tasks and return immediately
        task_ids = []
        for file in files:
            # Save uploaded file
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            # Submit task
            task_id = await task_processor.submit_task(
                kb_id=kb_id,
                filename=file.filename,
                file_path=file_path,
                category=category
            )
            task_ids.append(task_id)
        return {
            "status": "success",
            "message": f"Submitted {len(task_ids)} tasks for processing",
            "tasks": task_ids
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload documents error: error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload documents"
        )


@router.get("/documents/list")
async def list_documents(
    kb_id: str = Query('kb_316a7dbc75d0', description="Knowledge base ID"),
    category: str = Query(None, description="Document category"),
    status_filter: str = Query(None, alias="status", description="Processing status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    获取文档列表。
    
    \nArgs:
        \n- kb_id: Knowledge base ID
        \n- category: Document category
        \n- status_filter: Processing status
        \n- page: Page number
        \n- page_size: Page size
        \n- api_key: API key from auth
        \n- db: Database instance

    \nReturns:
        \n- List of documents
    """
    try:
        # Build query
        query = {}
        if kb_id:
            query["kb_id"] = kb_id
        if category:
            query["category"] = category
        if status_filter:
            query["status"] = status_filter
        
        # Get total count
        total = await db.documents.count_documents(query)
        
        # Get paginated results
        cursor = db.documents.find(query).sort("uploaded_at", -1).skip((page - 1) * page_size).limit(page_size)
        docs = await cursor.to_list(length=page_size)
        
        # Format results
        items = []
        for doc in docs:
            doc.pop("_id", None)
            items.append({
                "doc_id": doc["doc_id"],
                "filename": doc["filename"],
                "kb_id": doc["kb_id"],
                "category": doc.get("category"),
                "size": doc["size"],
                "chunks_count": doc.get("chunks_count", 0),
                "status": doc["status"],
                "uploaded_at": doc["uploaded_at"].isoformat() + "Z"
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
        logger.error(f"List documents error: error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list documents"
        )


@router.get("/documents/{doc_id}")
async def get_document_detail(
    doc_id: str,
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    获取文档详情（包含切片统计和所属知识库信息）。
    
    \nArgs:
        \n- doc_id: Document ID
        \n- api_key: API key from auth
        \n- db: Database instance

    \nReturns:
        \n- Document details with chunks summary and knowledge base info
    """
    try:
        logger.info(f"Get document detail request: doc_id={doc_id}")
        
        # Get document from MongoDB
        doc = await db.documents.find_one({"doc_id": doc_id})
        
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Get knowledge base info
        kb_info = None
        if doc.get("kb_id"):
            kb = await db.knowledge_bases.find_one({"kb_id": doc["kb_id"]})
            if kb:
                kb_info = {
                    "kb_id": kb["kb_id"],
                    "name": kb["name"],
                    "category": kb["category"],
                    "description": kb.get("description", "")
                }
        
        # Get chunks statistics and content from Chroma
        chunks_stats = {
            "total_chunks": doc.get("chunks_count", 0),
            "avg_chunk_size": 0,
            "total_characters": 0
        }
        full_content = ""
        
        try:
            # Query chunks from Chroma to get statistics and content
            if chroma_db.client and doc.get("chunks_count", 0) > 0:
                results = chroma_db.doc_collection.get(
                    where={"doc_id": doc_id},
                    limit=10000  # Get all chunks
                )
                
                if results and results.get("documents"):
                    documents = results["documents"]
                    metadatas = results.get("metadatas", [])
                    
                    # Build chunk list with indices for proper ordering
                    chunks_with_index = []
                    for i, text in enumerate(documents):
                        metadata = metadatas[i] if i < len(metadatas) else {}
                        chunk_index = metadata.get("chunk_index", i)
                        chunks_with_index.append((chunk_index, text))
                    
                    # Sort by chunk_index
                    chunks_with_index.sort(key=lambda x: x[0])
                    
                    # Calculate statistics
                    chunks = [text for _, text in chunks_with_index]
                    total_chars = sum(len(chunk) for chunk in chunks)
                    chunks_stats["total_characters"] = total_chars
                    chunks_stats["avg_chunk_size"] = total_chars // len(chunks) if chunks else 0
                    
                    # Concatenate all chunks to form complete content
                    full_content = "".join(chunks)
        except Exception as e:
            logger.warning(f"Failed to get chunks statistics and content: doc_id={doc_id}, error={str(e)}")
        
        # Format response
        doc.pop("_id", None)
        doc_data = {
            "doc_id": doc["doc_id"],
            "filename": doc["filename"],
            "kb_id": doc.get("kb_id"),
            "category": doc.get("category"),
            "size": doc["size"],
            "status": doc["status"],
            "uploaded_at": doc["uploaded_at"].isoformat() + "Z",
            "processed_at": doc.get("processed_at").isoformat() + "Z" if doc.get("processed_at") else None,
            "chunks_stats": chunks_stats,
            "content": full_content,
            "knowledge_base": kb_info
        }
        
        return {
            "code": 200,
            "message": "success",
            "data": doc_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document detail error: doc_id={doc_id}, error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get document detail"
        )


@router.get("/documents/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    获取文档切片列表（包含切片文本、长度等详细信息）。
    
    \nArgs:
        \n- doc_id: Document ID
        \n- page: Page number
        \n- page_size: Page size
        \n- api_key: API key from auth
        \n- db: Database instance

    \nReturns:
        \n- List of document chunks with text content and metadata
    """
    try:
        logger.info(f"Get document chunks request: doc_id={doc_id}, page={page}, page_size={page_size}")
        
        # Verify document exists
        doc = await db.documents.find_one({"doc_id": doc_id})
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        kb_id = doc.get("kb_id")
        
        # Get chunks from Chroma
        chunks_data = []
        total_chunks = 0
        
        if chroma_db.client:
            # Get all chunks for this document
            results = chroma_db.doc_collection.get(
                where={"doc_id": doc_id},
                limit=10000  # Get all chunks
            )
            
            if results and results.get("documents"):
                all_chunks = []
                documents = results["documents"]
                metadatas = results.get("metadatas", [])
                ids = results.get("ids", [])
                
                # Combine data
                for i, text in enumerate(documents):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    chunk_id = ids[i] if i < len(ids) else f"chunk_{i}"
                    
                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "chunk_index": metadata.get("chunk_index", i),
                        "content": text,
                        "length": len(text),
                        "kb_id": metadata.get("kb_id"),
                        "doc_id": metadata.get("doc_id")
                    })
                
                # Sort by chunk_index
                all_chunks.sort(key=lambda x: x["chunk_index"])
                
                total_chunks = len(all_chunks)
                
                # Paginate
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                paginated_chunks = all_chunks[start_idx:end_idx]
                
                return {
                    "status": "success",
                    "doc_id": doc_id,
                    "kb_id": kb_id,
                    "filename": doc.get("filename"),
                    "total_chunks": total_chunks,
                    "page": page,
                    "page_size": page_size,
                    "chunks": paginated_chunks
                }
        
        # Return empty result if no chunks found
        return {
            "code": 200,
            "status": "success",
            "doc_id": doc_id,
            "kb_id": kb_id,
            "filename": doc.get("filename"),
            "total_chunks": 0,
            "page": page,
            "page_size": page_size,
            "chunks": []
        }
        
    except Exception as e:
        logger.error(f"Get document chunks error: doc_id={doc_id}, error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get document chunks"
        )


@router.get("/documents/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    api_key: str = Depends(get_api_key)
):
    """
    查询文档处理任务状态。
    
    \nArgs:
        \n- task_id: Task ID
        \n- api_key: API key from auth

    \nReturns:
        \n- Task status information (status, progress, doc_id, error, etc.)
    """
    try:
        logger.info(f"Get task status request: task_id={task_id}")
        
        task = await task_processor.get_task_status(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        return {
            "code": 200,
            "status": "success",
            "data": task
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get task status error: task_id={task_id}, error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get task status"
        )


@router.delete("/documents/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    api_key: str = Depends(get_api_key)
):
    """
    取消文档处理任务。
    
    \nArgs:
        \n- task_id: Task ID
        \n- api_key: API key from auth

    \nReturns:
        \n- Cancellation result
    """
    try:
        logger.info(f"Cancel task request: task_id={task_id}")
        
        cancelled = await task_processor.cancel_task(task_id)
        
        if not cancelled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Task cannot be cancelled (not found or already completed)"
            )
        
        return {
            "code": 200,
            "status": "success",
            "message": f"Task {task_id} cancelled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cancel task error: task_id={task_id}, error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel task"
        )


@router.post("/documents/create_with_segment", response_model=CreateRagDocumentResponse)
async def create_rag_document_with_segment(
    request: CreateRagDocumentRequest,
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    创建RAG文档（Java平台集成接口，支持自定义分段策略）。
    
    该接口接受Java平台的文档创建请求，支持：
    - 从URL下载文档
    - 自定义文本预处理（删除空格/换行/目录）
    - 自定义分段策略（换行切分/标识符切分）
    - 分段合并与最大长度控制
    - 系统内置或自定义分隔符
    
    Args:
        request: 创建RAG文档请求（包含分段配置）
        api_key: API key from auth
        db: Database instance
    
    Returns:
        创建结果（task_id和rag_document_id）
    """
    try:
        logger.info(f"Create RAG document with segment config: kb_id={request.kb_id}, document_name={request.document_name}, resource_id={request.resource_id}, segment_flag={request.segment_flag}")
        
        # Verify knowledge base exists
        kb = await db.knowledge_bases.find_one({"kb_id": request.kb_id})
        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base {request.kb_id} not found"
            )
        
        # Download file from resource_url
        file_path = None
        try:
            # Generate temporary file path
            file_ext = os.path.splitext(request.document_name)[1] or '.txt'
            temp_filename = f"java_upload_{request.resource_id}_{datetime.utcnow().timestamp()}{file_ext}"
            file_path = os.path.join(UPLOAD_DIR, temp_filename)
            
            # Download file with timeout
            async with aiohttp.ClientSession() as session:
                async with session.get(request.resource_url, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Failed to download file from URL: HTTP {response.status}"
                        )
                    
                    # Save file to disk
                    with open(file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
            
            logger.info(f"Downloaded file from URL: url={request.resource_url}, file_path={file_path}")
        
        except aiohttp.ClientError as e:
            logger.error(f"Failed to download file: url={request.resource_url}, error={str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download file: {str(e)}"
            )
        
        # Build chunk_config from segment_vo
        chunk_config = None
        if request.segment_flag == 1 and request.segment_vo:
            segment = request.segment_vo
            chunk_config = {
                'is_space_flag': segment.is_space_flag,
                'is_menu_flag': segment.is_menu_flag,
                'segment_type': segment.segment_type,
                'is_segment_union_flag': segment.is_segment_union_flag,
                'segment_union_max_length': segment.segment_union_max_length,
                'segment_identifier_type': segment.segment_identifier_type,
                'identifier_default': segment.identifier_default,
                'identifier_customize': segment.identifier_customize,
            }
            logger.info(f"Using custom segment config: chunk_config={chunk_config}")
        
        # Submit async task
        task_id = await task_processor.submit_task(
            kb_id=request.kb_id,
            filename=request.document_name,
            file_path=file_path,
            category=None,  # Could map from Java dataset info
            chunk_config=chunk_config
        )
        
        return CreateRagDocumentResponse(
            code=200,
            message="success",
            data={
                "task_id": task_id,
                "status": "processing",
                "resource_id": request.resource_id,
                "document_name": request.document_name,
                "kb_id": request.kb_id
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create RAG document error: error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create RAG document"
        )


