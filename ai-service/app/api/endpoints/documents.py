"""
Document management API endpoints.
"""

import shutil
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    File,
    UploadFile,
    Form,
    Query,
)

from app.api.middleware.auth import get_api_key
from app.api.middleware.rate_limit import ip_rate_limit_dependency
from app.core.logging import get_logger
from app.core.database import get_database
from app.core.chroma import chroma_db

from app.services.task_processor import task_processor
from app.services.document_service import generate_doc_id, download_file, UPLOAD_DIR
from app.models.schemas import CreateRagDocumentRequest, CreateRagDocumentResponse, SegmentVo, ResponseResult

logger = get_logger(__name__)

router = APIRouter()


def _format_datetime(dt: Optional[datetime]) -> Optional[str]:
    """Format datetime to ISO 8601 with UTC timezone suffix."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


async def _get_chunks_from_chroma(doc_id: str) -> Dict[str, Any]:
    """
    Fetch chunks from ChromaDB and calculate statistics.

    Returns dict with:
        - total_chunks: int
        - avg_chunk_size: int
        - total_characters: int
        - content: str (concatenated chunks)
    """
    chunks_stats = {"total_chunks": 0, "avg_chunk_size": 0, "total_characters": 0}
    full_content = ""

    if not chroma_db.client:
        return chunks_stats

    try:
        results = chroma_db.doc_collection.get(where={"doc_id": doc_id}, limit=10000)

        if not results or not results.get("documents"):
            return chunks_stats

        documents = results["documents"]
        metadatas = results.get("metadatas", [])

        # Build chunks list with proper ordering by chunk_index
        chunks_with_index = []
        for i, text in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            chunk_index = metadata.get("chunk_index", i)
            chunks_with_index.append((chunk_index, text))

        # Sort by chunk_index and extract content
        chunks_with_index.sort(key=lambda pair: pair[0])
        chunks = [text for _, text in chunks_with_index]

        # Calculate statistics
        total_chars = sum(len(chunk) for chunk in chunks)
        chunks_stats["total_chunks"] = len(chunks)
        chunks_stats["total_characters"] = total_chars
        chunks_stats["avg_chunk_size"] = total_chars // len(chunks) if chunks else 0

        # Concatenate all chunks to form complete content
        full_content = "".join(chunks)

    except Exception as e:
        logger.warning("Failed to get chunks from Chroma", doc_id=doc_id, error=str(e))

    return {**chunks_stats, "content": full_content}


@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(..., description="Files to upload (max 50)"),
    kb_id: str = Form(..., description="Knowledge base ID"),
    category: str = Form(..., description="Document category"),
    api_key: str = Depends(get_api_key),
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
                detail="Maximum 50 files allowed per upload",
            )
        logger.info("Upload documents request", kb_id=kb_id, file_count=len(files))

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
                category=category,
            )
            task_ids.append(task_id)
        return {
            "status": "success",
            "message": f"Submitted {len(task_ids)} tasks for processing",
            "tasks": task_ids,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload documents error error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload documents",
        )


@router.get("/list")
async def list_documents(
    kb_id: str = Query(..., description="Knowledge base ID"),
    category: str = Query(None, description="Document category"),
    status_filter: str = Query(None, alias="status", description="Processing status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    api_key: str = Depends(get_api_key),
    db=Depends(get_database),
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
        # Build query filter
        query = {"kb_id": kb_id}
        if category:
            query["category"] = category
        if status_filter:
            query["status"] = status_filter

        # Get total count
        total = await db.documents.count_documents(query)

        # Get paginated results
        cursor = (
            db.documents.find(query)
            .sort("uploaded_at", -1)
            .skip((page - 1) * page_size)
            .limit(page_size)
        )
        docs = await cursor.to_list(length=page_size)

        # Format results
        items = [
            {
                "doc_id": doc["doc_id"],
                "filename": doc["filename"],
                "kb_id": doc["kb_id"],
                "category": doc.get("category"),
                "size": doc["size"],
                "chunks_count": doc.get("chunks_count", 0),
                "status": doc["status"],
                "uploaded_at": _format_datetime(doc["uploaded_at"]),
            }
            for doc in docs
        ]

        data = {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": items,
        }
        return ResponseResult.success(data)

    except Exception as e:
        logger.error(f"List documents error error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list documents",
        )


@router.get("/{doc_id}")
async def get_document_detail(
    doc_id: str,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
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
        logger.info(f"get_document_detail request doc_id={doc_id}")

        # Get document from MongoDB
        doc = await db.documents.find_one({"doc_id": doc_id})

        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
            )

        # Get knowledge base info
        kb_info = None
        doc_kb_id = doc.get("kb_id")
        if doc_kb_id:
            kb = await db.knowledge_bases.find_one({"kb_id": doc_kb_id})
            if kb:
                kb_info = {
                    "kb_id": kb["kb_id"],
                    "name": kb["name"],
                    "category": kb["category"],
                    "description": kb.get("description", ""),
                }

        # Get chunks statistics and content from Chroma
        chroma_data = await _get_chunks_from_chroma(doc_id)

        # Format response
        doc_data = {
            "doc_id": doc["doc_id"],
            "filename": doc["filename"],
            "kb_id": doc_kb_id,
            "category": doc.get("category"),
            "size": doc["size"],
            "status": doc["status"],
            "uploaded_at": _format_datetime(doc["uploaded_at"]),
            "processed_at": _format_datetime(doc.get("processed_at")),
            "chunks_stats": chroma_data,
            "content": chroma_data.get("content", ""),
            "knowledge_base": kb_info,
        }

        return ResponseResult.success(doc_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_document_detail exception doc_id={doc_id}, error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="get_document_detail error",
        )


@router.get("/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    api_key: str = Depends(get_api_key),
    db=Depends(get_database),
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
            "Get document chunks request", doc_id=doc_id, page=page, page_size=page_size
        )

        # Verify document exists and get hierarchical summary
        doc = await db.documents.find_one({"doc_id": doc_id})
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
            )

        kb_id = doc.get("kb_id")

        # Get hierarchical summary from document metadata
        hierarchical_summary = doc.get("hierarchical_summary", {})

        # Fetch all chunks from MongoDB to get summary data
        chunks_cursor = db.document_chunks.find({"doc_id": doc_id}).sort(
            "chunk_index", 1
        )
        mongo_chunks = await chunks_cursor.to_list(length=None)

        # Build chunks list with summaries
        all_chunks = []
        for chunk in mongo_chunks:
            metadata = chunk.get("metadata", {})
            all_chunks.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "content": chunk.get("content", ""),
                    "length": len(chunk.get("content", "")),
                    "kb_id": chunk.get("kb_id"),
                    "doc_id": chunk.get("doc_id"),
                    "summary": metadata.get("summary", ""),
                }
            )

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
            "total_chunks": len(mongo_chunks),
            "page": page,
            "page_size": page_size,
            "chunks": paginated_chunks,
            "hierarchical_summary": {
                "document_summary": hierarchical_summary.get("document_summary", ""),
                "section_summaries": hierarchical_summary.get("section_summaries", []),
                "chunks_with_summaries": sum(1 for c in all_chunks if c.get("summary")),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Get document chunks error: doc_id={doc_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get document chunks",
        )


@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    api_key: str = Depends(get_api_key),
    _ip_rate_limit: None = Depends(ip_rate_limit_dependency),
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
        logger.info(f"get_task_status request task_id={task_id}")

        task = await task_processor.get_task_status(task_id)

        if not task:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"get_task_status {task_id} not found")

        ResponseResult.success(task)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_task_status exception task_id={task_id} error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="get_task_status error",
        )


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str, api_key: str = Depends(get_api_key)):
    """
        取消文档处理任务。

        \nArgs:
            \n- task_id: Task ID
            \n- api_key: API key from auth

        \nReturns:
            \n- Cancellation result
    """
    try:
        logger.info(f"cancel_task request task_id={task_id}")

        cancelled = await task_processor.cancel_task(task_id)

        if not cancelled:
            return ResponseResult.error(status.HTTP_400_BAD_REQUEST, "error", "cancel_task failed")

        return ResponseResult.success(None)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"cancel_task exception task_id={task_id} error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="cancel_task error",
        )


@router.post("/create_with_segment", response_model=CreateRagDocumentResponse)
async def create_rag_document_with_segment(
    request: CreateRagDocumentRequest,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database),
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
    kb_id = request.kb_id
    try:
        logger.info(
            f"create_rag_document_with_segment "
            f"request kb_id={kb_id} "
            f"enhance={request.enhance} "
            f"document_name={request.document_name} "
            f"resource_id={request.resource_id} "
            f"resource_url={request.resource_url} "
            f"segment_flag={request.segment_flag}"
        )

        # 校验知识库是否存在
        kb = await db.knowledge_bases.find_one({"kb_id": kb_id})
        if not kb:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"create_rag_document_with_segment knowledge base not found kb_id={kb_id}")

        # Build chunk_config from segment_vo
        chunk_config = None
        if request.segment_flag == 1 and request.segment_vo:
            segment = request.segment_vo
            chunk_config = {
                "is_space_flag": segment.is_space_flag,
                "is_menu_flag": segment.is_menu_flag,
                "segment_type": segment.segment_type,
                "is_segment_union_flag": segment.is_segment_union_flag,
                "segment_union_max_length": segment.segment_union_max_length,
                "segment_identifier_type": segment.segment_identifier_type,
                "identifier_default": segment.identifier_default,
                "identifier_customize": segment.identifier_customize,
            }
            logger.info(f"create_rag_document_with_segment chunk_config={chunk_config}")

        # 下载远程服务器上的文档文件
        file_path = await download_file(request.document_name, request.resource_id, request.resource_url)

        # 生成唯一文档id
        doc_id = generate_doc_id(request.document_name, kb_id)

        # 添加到文档任务列表中
        task_id = await task_processor.submit_task(
            kb_id=kb_id,
            enhance=request.enhance,
            filename=request.document_name,
            file_path=str(file_path),
            category=None,
            chunk_config=chunk_config,
            doc_id=doc_id,
            resource_id=request.resource_id,
        )

        data = {
            "doc_id": doc_id,
            "task_id": task_id,
            "status": "processing",
            "resource_id": request.resource_id,
            "document_name": request.document_name,
            "kb_id": request.kb_id,
        }
        return ResponseResult.success(data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"create_rag_document_with_segment exception error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="create_rag_document_with_segment error",
        )


@router.post("/set_segment")
async def set_segment(
    request: SegmentVo,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
):
    """
        设置文档或视频资源是否知识增强

        nArgs:
            - request: 文档分段设置
            - api_key: API key from auth
            - db: Database instance

        Returns:
            - result
    """
    doc_id = request.doc_id
    try:
        logger.info(f"set_segment request doc_id={doc_id}")

        doc = await db.documents.find_one({"doc_id": doc_id})

        if not doc:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error", "set_segment not found")

        chunk_config = {
            "is_space_flag": request.is_space_flag,
            "is_menu_flag": request.is_menu_flag,
            "segment_type": request.segment_type,
            "is_segment_union_flag": request.is_segment_union_flag,
            "segment_union_max_length": request.segment_union_max_length,
            "segment_identifier_type": request.segment_identifier_type,
            "identifier_default": request.identifier_default,
            "identifier_customize": request.identifier_customize,
        }
        update_data = {
            "status": "processing",
            "segment_config": chunk_config
        }

        result = await db.documents.update_one(
            {"doc_id": doc_id},
            {"$set": update_data}
        )

        if result and result.modified_count == 1:
            logger.info(f"set_segment success doc_id={doc_id}")

            task = await db.document_tasks.find_one({"doc_id": doc_id})

            if not task:
                return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error", "set_segment task not found")

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
        else:
            return ResponseResult.error(status.HTTP_400_BAD_REQUEST, "error", "set_segment failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"set_segment exception doc_id={doc_id} error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="set_segment error"
        )


@router.get("/get_study_file/{doc_id}")
async def get_study_file(
    doc_id: str,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
):
    """
        获取文件学习后的分段知识文件（返回json格式的字符串）

        nArgs:
            - doc_id: 文档id
            - api_key: API key from auth
            - db: Database instance

        Returns:
            - result
    """
    try:
        logger.info(f"get_study_file request doc_id={doc_id}")

        doc = await db.documents.find_one({"doc_id": doc_id})

        if not doc:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error", "get_study_file not found")

        # 功能待实现

        return ResponseResult.success(None)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_study_file exception doc_id={doc_id} error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="get_study_file error"
        )


@router.post("/upload_study_file")
async def upload_study_file(
    doc_id: str,
    json_content: str,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
):
    """
        上传文件学习后的分段知识文件（json格式）

        nArgs:
            - doc_id: 文档id
            - json_content: 分段知识文件
            - api_key: API key from auth
            - db: Database instance

        Returns:
            - result
    """
    try:
        logger.info(f"update_study_file request doc_id={doc_id}")

        doc = await db.documents.find_one({"doc_id": doc_id})

        if not doc:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error", "update_study_file not found")

        # 功能待实现

        return ResponseResult.success(None)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"update_study_file exception doc_id={doc_id} error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="update_study_file error"
        )
