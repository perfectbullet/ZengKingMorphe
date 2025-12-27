"""
Async document task processor with queue management.
"""
import asyncio
import random
import string
from datetime import datetime
from typing import Dict, Optional, List
from app.core.logging import get_logger
from app.core.database import get_database
from app.models.database import DocumentTaskModel, FAQModel
from app.services.document_service import document_processor as doc_processor

logger = get_logger(__name__)


class DocumentTaskProcessor:
    """Background task processor for document processing."""
    
    def __init__(self):
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.worker_task: Optional[asyncio.Task] = None
        self.active_tasks: Dict[str, bool] = {}  # task_id -> cancellation flag
    
    def generate_task_id(self) -> str:
        """Generate unique task ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M")
        random_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return f"document_task_{timestamp}_{random_chars}"
    
    async def submit_task(
        self,
        kb_id: str,
        filename: str,
        file_path: str,
        category: Optional[str] = None,
        chunk_config: Optional[Dict] = None  # Custom chunk configuration
    ) -> str:
        """
        Submit a new document processing task.
        
        Args:
            kb_id: Knowledge base ID
            filename: Original filename
            file_path: Path to uploaded file
            category: Document category
            chunk_config: Custom chunking configuration (optional)
            
        Returns:
            Task ID
        """
        task_id = self.generate_task_id()
        
        # Create task record in database
        db = await get_database()
        task_model = DocumentTaskModel(
            task_id=task_id,
            kb_id=kb_id,
            filename=filename,
            file_path=file_path,
            category=category,
            status="pending",
            metadata={"chunk_config": chunk_config} if chunk_config else {}
        )
        
        await db.document_tasks.insert_one(task_model.model_dump())
        
        # Add to queue
        await self.task_queue.put({
            "task_id": task_id,
            "kb_id": kb_id,
            "filename": filename,
            "file_path": file_path,
            "category": category,
            "chunk_config": chunk_config
        })
        
        logger.info(f"Document task submitted: task_id={task_id}, filename={filename}")
        
        return task_id
    
    async def start(self):
        """Start the background task processor."""
        if self.running:
            logger.warning("Task processor already running")
            return
        
        self.running = True
        self.worker_task = asyncio.create_task(self._process_tasks())
        logger.info("Document task processor started")
    
    async def stop(self):
        """Stop the background task processor."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all active tasks
        for task_id in list(self.active_tasks.keys()):
            await self.cancel_task(task_id)
        
        # Wait for worker to finish
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Document task processor stopped")
    
    async def _process_tasks(self):
        """Background worker that processes tasks from the queue."""
        logger.info("Task processor worker started")
        
        while self.running:
            try:
                # Get task from queue with timeout
                try:
                    task_data = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                    logger.info(f"Fetched task from queue: task_id={task_data['task_id']}, task_type={task_data.get('task_type', 'document')}")
                except asyncio.TimeoutError:
                    continue
                
                task_id = task_data["task_id"]
                task_type = task_data.get("task_type", "document")
                
                # Mark task as active
                self.active_tasks[task_id] = False  # cancellation flag
                
                # Dispatch task based on type
                if task_type == "faq_vectorization":
                    asyncio.create_task(self._execute_faq_vectorization(task_data))
                else:
                    # Default to document processing
                    asyncio.create_task(self._execute_task(task_data))
                
            except Exception as e:
                logger.error(f"Error in task processor loop: error={str(e)}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _execute_task(self, task_data: Dict):
        """Execute a single document processing task."""
        task_id = task_data["task_id"]
        db = await get_database()
        
        try:
            # Update task status to running
            await db.document_tasks.update_one(
                {"task_id": task_id},
                {
                    "$set": {
                        "status": "running",
                        "started_at": datetime.utcnow()
                    }
                }
            )
            
            logger.info(f"Processing document task: task_id={task_id}, task_data={task_data}")
            
            # Process document with progress tracking
            doc_id = await self._process_with_progress(task_data)
            
            # Check if task was cancelled
            if self.active_tasks.get(task_id, False):
                await db.document_tasks.update_one(
                    {"task_id": task_id},
                    {
                        "$set": {
                            "status": "cancelled",
                            "completed_at": datetime.utcnow()
                        }
                    }
                )
                logger.info(f"Document task cancelled: task_id={task_id}")
                return
            
            # Update task status to completed
            await db.document_tasks.update_one(
                {"task_id": task_id},
                {
                    "$set": {
                        "status": "completed",
                        "doc_id": doc_id,
                        "progress": 100.0,
                        "completed_at": datetime.utcnow()
                    }
                }
            )
            
            logger.info(f"Document task completed: task_id={task_id}, doc_id={doc_id}")
            
        except Exception as e:
            logger.error(f"Document task failed: task_id={task_id}, error={str(e)}", exc_info=True)
            
            # Update task status to failed
            await db.document_tasks.update_one(
                {"task_id": task_id},
                {
                    "$set": {
                        "status": "failed",
                        "error_message": str(e),
                        "completed_at": datetime.utcnow()
                    }
                }
            )
        
        finally:
            # Remove from active tasks
            self.active_tasks.pop(task_id, None)
    
    async def _process_with_progress(self, task_data: Dict) -> str:
        """Process document with progress updates."""
        task_id = task_data["task_id"]
        
        # Call original document processor
        # We'll wrap it to track progress
        doc_id = await doc_processor.process_document(
            file_path=task_data["file_path"],
            filename=task_data["filename"],
            kb_id=task_data["kb_id"],
            category=task_data.get("category"),
            task_id=task_id,  # Pass task_id for progress tracking
            chunk_config=task_data.get("chunk_config")  # Pass chunk_config
        )
        
        return doc_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """
        Get task status.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status or None if not found
        """
        db = await get_database()
        task = await db.document_tasks.find_one({"task_id": task_id})
        
        if not task:
            return None
        
        task.pop("_id", None)
        
        # Format timestamps
        if task.get("created_at"):
            task["created_at"] = task["created_at"].isoformat() + "Z"
        if task.get("started_at"):
            task["started_at"] = task["started_at"].isoformat() + "Z"
        if task.get("completed_at"):
            task["completed_at"] = task["completed_at"].isoformat() + "Z"
        
        return task
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if task was cancelled, False otherwise
        """
        db = await get_database()
        
        # Get current task status
        task = await db.document_tasks.find_one({"task_id": task_id})
        
        if not task:
            return False
        
        status = task["status"]
        
        # Can only cancel pending or running tasks
        if status not in ["pending", "running"]:
            return False
        
        # If pending, update status directly
        if status == "pending":
            await db.document_tasks.update_one(
                {"task_id": task_id},
                {
                    "$set": {
                        "status": "cancelled",
                        "completed_at": datetime.utcnow()
                    }
                }
            )
            logger.info(f"Cancelled pending task: task_id={task_id}")
            return True
        
        # If running, set cancellation flag
        if task_id in self.active_tasks:
            self.active_tasks[task_id] = True  # Set cancellation flag
            logger.info(f"Cancellation requested for running task: task_id={task_id}")
            return True
        
        return False
    
    async def submit_faq_vectorization_task(
        self,
        employee_id: str,
        faqs: List[Dict]
    ) -> str:
        """
        提交FAQ向量化任务（异步后台处理）。
        
        Args:
            employee_id: 员工ID
            faqs: FAQ列表（外部API格式）
            
        Returns:
            Task ID
        """
        task_id = f"faq_task_{employee_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # Add to queue for background processing
        await self.task_queue.put({
            "task_id": task_id,
            "task_type": "faq_vectorization",
            "employee_id": employee_id,
            "faqs": faqs
        })
        
        logger.info(f"FAQ vectorization task submitted: task_id={task_id}, employee_id={employee_id}, faq_count={len(faqs)}")
        
        return task_id
    
    async def _execute_faq_vectorization(self, task_data: Dict):
        """
        执行FAQ向量化任务（包含增量更新逻辑）。
        
        处理流程：
        1. 从外部API FAQ数据提取字段
        2. 检查数据库中是否已存在FAQ（基于updateTime判断）
        3. 生成combined_text（questionName + similarQuestions）
        4. 调用Embedding API生成向量
        5. 写入ChromaDB（向量存储）
        6. 写入ElasticSearch（关键词索引）
        7. 写入MongoDB（元数据）
        """
        task_id = task_data["task_id"]
        employee_id = task_data["employee_id"]
        faqs = task_data["faqs"]
        
        db = await get_database()
        
        try:
            logger.info(f"Starting FAQ vectorization: task_id={task_id}, employee_id={employee_id}, total_faqs={len(faqs)}")
            from app.utils.embeddings import get_embedding
            from app.core.chroma import chroma_db
            from app.core.elasticsearch import es_db
            
            vectorized_count = 0
            skipped_count = 0
            updated_count = 0
            
            for idx, faq_data in enumerate(faqs):
                try:
                    # Generate FAQ ID first (needed for both enabled and disabled FAQs)
                    external_faq_id = faq_data["faq_id"]  # Use new field name (str type)
                    faq_id = f"faq_{employee_id}_{external_faq_id}"
                    
                    # Handle disabled FAQs - delete from vector stores
                    if faq_data.get("isEnable", 0) != 1:
                        existing_faq = await db.faqs.find_one({"faq_id": faq_id})
                        
                        if existing_faq:
                            logger.info(f"FAQ {faq_id} is disabled (isEnable=0), cleaning up vector stores")
                            deleted_count = 0
                            
                            # Delete from ChromaDB
                            try:
                                chroma_collection = chroma_db.get_collection("faqs")
                                chroma_collection.delete(ids=[faq_id])
                                deleted_count += 1
                                logger.debug(f"Deleted FAQ {faq_id} from ChromaDB")
                            except Exception as e:
                                logger.warning(f"ChromaDB delete failed for {faq_id}: {e}")
                            
                            # Delete from ElasticSearch
                            try:
                                await es_db.client.delete(
                                    index="digital_employee_faqs",
                                    id=faq_id,
                                    ignore=[404]  # Ignore if not found
                                )
                                deleted_count += 1
                                logger.debug(f"Deleted FAQ {faq_id} from ElasticSearch")
                            except Exception as e:
                                logger.warning(f"ES delete failed for {faq_id}: {e}")
                            
                            # Update MongoDB status (soft delete - keep record)
                            await db.faqs.update_one(
                                {"faq_id": faq_id},
                                {"$set": {
                                    "is_enable": 0,
                                    "es_indexed": False,
                                    "vector_id": None,  # Mark vector as removed
                                    "synced_at": datetime.utcnow()
                                }}
                            )
                            
                            logger.info(f"FAQ {faq_id} cleanup completed: {deleted_count}/2 vector stores cleaned")
                        
                        skipped_count += 1
                        continue  # Skip to next FAQ
                    
                    # Check if FAQ exists in database (for enabled FAQs)
                    existing_faq = await db.faqs.find_one({"faq_id": faq_id})
                    
                    # Compare updateTime for incremental update
                    if existing_faq:
                        existing_update_time = existing_faq.get("update_time", "")
                        new_update_time = faq_data.get("updateTime", "")
                        
                        if new_update_time <= existing_update_time:
                            # Skip if not newer
                            skipped_count += 1
                            logger.debug(f"FAQ {faq_id} not updated (existing: {existing_update_time}, new: {new_update_time})")
                            continue
                        else:
                            logger.info(f"FAQ {faq_id} needs update (existing: {existing_update_time}, new: {new_update_time})")
                            updated_count += 1
                    
                    # Extract FAQ fields
                    question_name = faq_data.get("questionName", "")
                    similar_questions = faq_data.get("similarQuestions", [])
                    answers = faq_data.get("answers", [])
                    
                    # Combine text for embedding (questionName + all similarQuestions)
                    combined_text = question_name + " " + " ".join(similar_questions)
                    combined_text = combined_text.strip()
                    
                    if not combined_text:
                        skipped_count += 1
                        logger.warning(f"FAQ {faq_id} has empty combined_text, skipping")
                        continue
                    
                    # Generate embedding vector
                    embedding = await get_embedding(combined_text)
                    
                    # Prepare FAQ model
                    faq_model = FAQModel(
                        faq_id=faq_id,
                        employee_id=employee_id,
                        external_faq_id=external_faq_id,
                        team_id=faq_data.get("teamId", 0),
                        question_name=question_name,
                        similar_questions=similar_questions,
                        answers=answers,
                        is_enable=faq_data.get("isEnable", 1),
                        is_clear=faq_data.get("isClear", 0),
                        start_time=faq_data.get("startTime"),
                        end_time=faq_data.get("endTime"),
                        update_time=faq_data.get("updateTime", ""),
                        create_time=faq_data.get("createTime", ""),
                        combined_text=combined_text,
                        keywords=similar_questions,  # Use similar questions as keywords
                        vector_id=faq_id,  # Use faq_id as vector_id
                        es_indexed=False  # Will be set to True after ES indexing
                    )
                    
                    # Step 1: Write to ChromaDB (vector storage)
                    try:
                        chroma_collection = chroma_db.get_collection("faqs")
                        
                        # Delete existing vector if updating
                        if existing_faq:
                            try:
                                chroma_collection.delete(ids=[faq_id])
                            except Exception as e:
                                logger.warning(f"Failed to delete existing FAQ vector {faq_id}: {e}")
                        
                        # Add new vector
                        chroma_collection.add(
                            ids=[faq_id],
                            embeddings=[embedding],
                            metadatas=[{
                                "employee_id": employee_id,
                                "faq_id": faq_id,
                                "question_name": question_name,
                                "combined_text": combined_text[:500]  # Truncate for metadata
                            }],
                            documents=[combined_text]
                        )
                        logger.debug(f"FAQ {faq_id} vectorized in ChromaDB")
                    except Exception as e:
                        logger.error(f"Failed to write FAQ {faq_id} to ChromaDB: {e}", exc_info=True)
                        raise
                    
                    # Step 2: Write to ElasticSearch (keyword index)
                    try:
                        es_doc = {
                            "faq_id": faq_id,
                            "employee_id": employee_id,
                            "question_name": question_name,
                            "similar_questions": similar_questions,
                            "combined_text": combined_text,
                            "answers": answers,
                            "is_enable": faq_data.get("isEnable", 1),
                            "update_time": faq_data.get("updateTime", "")
                        }
                        
                        await es_db.client.index(
                            index="digital_employee_faqs",
                            id=faq_id,
                            document=es_doc
                        )
                        
                        faq_model.es_indexed = True
                        logger.debug(f"FAQ {faq_id} indexed in ElasticSearch")
                    except Exception as e:
                        logger.error(f"Failed to write FAQ {faq_id} to ElasticSearch: {e}", exc_info=True)
                        # Continue even if ES fails (can retry later)
                    
                    # Step 3: Write to MongoDB (metadata storage)
                    await db.faqs.update_one(
                        {"faq_id": faq_id},
                        {"$set": faq_model.model_dump()},
                        upsert=True
                    )
                    
                    vectorized_count += 1
                    logger.debug(f"FAQ {faq_id} saved to MongoDB ({vectorized_count}/{len(faqs)})")
                    
                except Exception as e:
                    logger.error(f"Failed to process FAQ {idx}: {e}", exc_info=True)
                    # Continue processing other FAQs
            
            logger.info(
                f"FAQ vectorization completed: task_id={task_id}, "
                f"employee_id={employee_id}, total={len(faqs)}, "
                f"vectorized={vectorized_count}, updated={updated_count}, skipped={skipped_count}"
            )
            
        except Exception as e:
            logger.error(f"FAQ vectorization task failed: task_id={task_id}, error={str(e)}", exc_info=True)
            raise


# Global task processor instance
task_processor = DocumentTaskProcessor()
