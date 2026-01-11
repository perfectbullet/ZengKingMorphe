"""
Document management API endpoints.
"""
import os
import shutil
import aiohttp
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form, Query
from datetime import datetime
from pathlib import Path

from app.api.middleware.auth import get_api_key
from app.core.logging import get_logger
from app.core.database import get_database
from app.core.chroma import chroma_db
from app.core.config import settings
from app.services.task_processor import task_processor
from app.models.schemas import (
    CreateRagDocumentRequest,
    CreateRagDocumentResponse
)

logger = get_logger(__name__)

router = APIRouter()

# Temporary upload directory (using pathlib for cross-platform compatibility)
UPLOAD_DIR = Path(__file__).parent.parent.parent.parent / "../upload_docs"
UPLOAD_DIR = UPLOAD_DIR.resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Upload directory configured: {UPLOAD_DIR}")


@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(..., description="Files to upload (max 50)"),
    kb_id: str = Form(..., description="Knowledge base ID"),
    category: str = Form(..., description="Document category"),
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
            # Save uploaded file (using pathlib for cross-platform compatibility)
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            # Submit task
            task_id = await task_processor.submit_task(
                kb_id=kb_id,
                filename=file.filename,
                file_path=str(file_path),
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
        logger.error(
            "Upload documents error",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload documents"
        )


@router.get("/list")
async def list_documents(
    kb_id: str = Query(..., description="Knowledge base ID"),
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
        logger.error(
            "List documents error",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list documents"
        )


@router.get("/{doc_id}")
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
            logger.warning(
                "Failed to get chunks statistics and content",
                doc_id=doc_id,
                error=str(e)
            )

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
        logger.error(
            "Get document detail error",
            doc_id=doc_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get document detail"
        )


@router.get("/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    获取文档切片列表（包含切片文本、长度、分层摘要等详细信息）。

    \nArgs:
        \n- doc_id: Document ID
        \n- page: Page number
        \n- page_size: Page size
        \n- api_key: API key from auth
        \n- db: Database instance

    \nReturns:
        \n- List of document chunks with text content, summaries, and hierarchical summary info
    """
    try:
        logger.info(
            "Get document chunks request",
            doc_id=doc_id,
            page=page,
            page_size=page_size
        )

        # Verify document exists and get hierarchical summary
        doc = await db.documents.find_one({"doc_id": doc_id})
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        kb_id = doc.get("kb_id")

        # Get hierarchical summary from document metadata
        hierarchical_summary = doc.get("hierarchical_summary", {})

        # Fetch all chunks from MongoDB to get summary data
        chunks_cursor = db.document_chunks.find({"doc_id": doc_id}).sort("chunk_index", 1)
        mongo_chunks = await chunks_cursor.to_list(length=None)

        total_chunks = len(mongo_chunks)

        if not mongo_chunks:
            return {
                "code": 200,
                "status": "success",
                "doc_id": doc_id,
                "kb_id": kb_id,
                "filename": doc.get("filename"),
                "total_chunks": 0,
                "page": page,
                "page_size": page_size,
                "chunks": [],
                "hierarchical_summary": hierarchical_summary
            }

        # Build chunks list with summaries
        all_chunks = []
        for chunk in mongo_chunks:
            chunk_id = chunk.get("chunk_id")
            chunk_index = chunk.get("chunk_index", 0)
            content = chunk.get("content", "")

            # Get summary from metadata
            metadata = chunk.get("metadata", {})
            summary = metadata.get("summary", "")

            all_chunks.append({
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "content": content,
                "length": len(content),
                "kb_id": chunk.get("kb_id"),
                "doc_id": chunk.get("doc_id"),
                "summary": summary  # Chunk-level summary
            })

        # Paginate
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_chunks = all_chunks[start_idx:end_idx]

        return {
            "code": 200,
            "status": "success",
            "doc_id": doc_id,
            "kb_id": kb_id,
            "filename": doc.get("filename"),
            "total_chunks": total_chunks,
            "page": page,
            "page_size": page_size,
            "chunks": paginated_chunks,
            "hierarchical_summary": {
                "document_summary": hierarchical_summary.get("document_summary", ""),
                "section_summaries": hierarchical_summary.get("section_summaries", []),
                "chunks_with_summaries": sum(1 for c in all_chunks if c.get("summary"))
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Get document chunks error",
            doc_id=doc_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get document chunks"
        )


@router.get("/tasks/{task_id}")
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
            "code": 200,
            "status": "success",
            "data": task
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Get task status error",
            task_id=task_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get task status"
        )


@router.delete("/tasks/{task_id}")
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
            "code": 200,
            "status": "success",
            "message": f"Task {task_id} cancelled"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Cancel task error",
            task_id=task_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel task"
        )


@router.post("/create_with_segment", response_model=CreateRagDocumentResponse)
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
    - 文档处理在后台异步进行，实时更新任务状态

    Args:
        request: 创建RAG文档请求（包含分段配置）
        api_key: API key from auth
        db: Database instance

    Returns:
        创建结果（立即返回doc_id和task_id，后续可通过task_id查询处理进度）
    """
    try:
        logger.info(
            "Create RAG document with segment config",
            kb_id=request.kb_id,
            document_name=request.document_name,
            resource_id=request.resource_id,
            segment_flag=request.segment_flag
        )

        # Verify knowledge base exists
        kb = await db.knowledge_bases.find_one({"kb_id": request.kb_id})
        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base {request.kb_id} not found"
            )

        # Generate doc_id immediately (before async processing)
        import hashlib
        timestamp = datetime.utcnow().timestamp()
        content = f"{request.document_name}_{request.kb_id}_{timestamp}"
        hash_obj = hashlib.md5(content.encode())
        doc_id = f"doc_{hash_obj.hexdigest()[:12]}"

        # Download file from resource_url
        file_path = None
        try:
            # Generate temporary file path (using pathlib)
            file_ext = os.path.splitext(request.document_name)[1] or '.txt'
            temp_filename = f"java_upload_{request.resource_id}_{datetime.utcnow().timestamp()}{file_ext}"
            file_path = UPLOAD_DIR / temp_filename

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

            logger.info(
                "Downloaded file from URL",
                url=request.resource_url,
                file_path=str(file_path)
            )

        except aiohttp.ClientError as e:
            logger.error(
                "Failed to download file",
                url=request.resource_url,
                error=str(e)
            )
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
            logger.info(
                "Using custom segment config",
                chunk_config=chunk_config
            )

        # Submit async task with pre-generated doc_id
        task_id = await task_processor.submit_task(
            kb_id=request.kb_id,
            filename=request.document_name,
            file_path=str(file_path),
            category=None,
            chunk_config=chunk_config,
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
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Create RAG document error",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create RAG document"
        )
