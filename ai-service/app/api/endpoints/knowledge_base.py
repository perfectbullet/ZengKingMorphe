"""
Knowledge base management API endpoints.
"""
import os
import shutil
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form, Query
from datetime import datetime

from app.api.middleware.auth import get_api_key
from app.core.logging import get_logger
from app.core.database import get_database
from app.core.chroma import chroma_db
from app.services.task_processor import task_processor
from app.models.schemas import CreateKnowledgeBaseRequest


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
        logger.info("Create knowledge base request", name=request.name, category=request.category)
        
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
        logger.error("Create knowledge base error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create knowledge base"
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
        logger.error("List knowledge bases error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list knowledge bases"
        )


@router.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(..., description="Files to upload (max 50)"),
    kb_id: str = Form(..., description="Knowledge base ID"),
    category: str = Form(None, description="Document category"),
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
        logger.info(
            "Upload documents request",
            kb_id=kb_id,
            file_count=len(files)
        )
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
        logger.error("Upload documents error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload documents"
        )


@router.get("/documents/list")
async def list_documents(
    kb_id: str = Query(None, description="Knowledge base ID"),
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
        logger.error("List documents error", error=str(e), exc_info=True)
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
        logger.info("Get document detail request", doc_id=doc_id)
        
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
        
        # Get chunks statistics from Chroma
        chunks_stats = {
            "total_chunks": doc.get("chunks_count", 0),
            "avg_chunk_size": 0,
            "total_characters": 0
        }
        
        try:
            # Query chunks from Chroma to get statistics
            if chroma_db.client and doc.get("chunks_count", 0) > 0:
                results = chroma_db.doc_collection.get(
                    where={"doc_id": doc_id},
                    limit=1000  # Get all chunks for stats
                )
                
                if results and results.get("documents"):
                    chunks = results["documents"]
                    total_chars = sum(len(chunk) for chunk in chunks)
                    chunks_stats["total_characters"] = total_chars
                    chunks_stats["avg_chunk_size"] = total_chars // len(chunks) if chunks else 0
        except Exception as e:
            logger.warning("Failed to get chunks statistics", doc_id=doc_id, error=str(e))
        
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
        logger.error("Get document detail error", doc_id=doc_id, error=str(e), exc_info=True)
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
        logger.info("Get document chunks request", doc_id=doc_id, page=page, page_size=page_size)
        
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
        logger.error("Get document chunks error", doc_id=doc_id, error=str(e), exc_info=True)
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
        logger.info("Get task status request", task_id=task_id)
        
        task = await task_processor.get_task_status(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        return {
            "status": "success",
            "task": task
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get task status error", task_id=task_id, error=str(e), exc_info=True)
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
        logger.info("Cancel task request", task_id=task_id)
        
        cancelled = await task_processor.cancel_task(task_id)
        
        if not cancelled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Task cannot be cancelled (not found or already completed)"
            )
        
        return {
            "status": "success",
            "message": f"Task {task_id} cancelled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Cancel task error", task_id=task_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel task"
        )

