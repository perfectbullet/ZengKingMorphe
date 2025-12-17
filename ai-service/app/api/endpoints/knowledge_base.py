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
from app.services.document_service import document_processor
from app.services.rag_service import rag_retrieval

logger = get_logger(__name__)

router = APIRouter()

# Temporary upload directory
UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/create")
async def create_knowledge_base(
    name: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    Create a new knowledge base.
    
    Args:
        name: Knowledge base name
        description: Description
        category: Category
        current_user: Current user
        db: Database instance
        
    Returns:
        Created knowledge base data
    """
    try:
        logger.info("Create knowledge base request", name=name, category=category)
        
        # Generate KB ID
        import hashlib
        kb_id = f"kb_{hashlib.md5(f'{name}_{datetime.utcnow().timestamp()}'.encode()).hexdigest()[:12]}"
        
        # Create KB document (placeholder - full schema not in database.py yet)
        kb_doc = {
            "kb_id": kb_id,
            "name": name,
            "description": description,
            "category": category,
            "status": "active",
            "doc_count": 0,
            "chunk_count": 0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Insert into a knowledge_bases collection
        await db.knowledge_bases.insert_one(kb_doc)
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "kb_id": kb_id,
                "name": name,
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
    List knowledge bases.
    
    Args:
        category: Filter by category
        status_filter: Filter by status
        page: Page number
        page_size: Page size
        current_user: Current user
        db: Database instance
        
    Returns:
        List of knowledge bases
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
                "doc_count": kb.get("doc_count", 0),
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
    Upload and process documents.
    
    Args:
        files: Files to upload
        kb_id: Knowledge base ID
        category: Document category
        current_user: Current user
        
    Returns:
        Upload results
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
        
        results = []
        failed_count = 0
        
        for file in files:
            try:
                # Save uploaded file
                file_path = os.path.join(UPLOAD_DIR, file.filename)
                
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Process document
                doc_id = await document_processor.process_document(
                    file_path=file_path,
                    filename=file.filename,
                    kb_id=kb_id,
                    category=category
                )
                
                # Clean up temp file
                os.remove(file_path)
                
                results.append({
                    "doc_id": doc_id,
                    "filename": file.filename,
                    "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                    "status": "processing"
                })
                
            except Exception as e:
                logger.error(
                    "Failed to process file",
                    filename=file.filename,
                    error=str(e)
                )
                failed_count += 1
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": str(e)
                })
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "uploaded_count": len(files) - failed_count,
                "failed_count": failed_count,
                "documents": results
            }
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
    List documents.
    
    Args:
        kb_id: Knowledge base ID
        category: Document category
        status_filter: Processing status
        page: Page number
        page_size: Page size
        current_user: Current user
        db: Database instance
        
    Returns:
        List of documents
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
