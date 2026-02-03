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
from app.services.dataset_faq_service import faq_processor

logger = get_logger(__name__)


def generate_task_id() -> str:
    """Generate unique task ID."""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M")
    random_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"document_task_{timestamp}_{random_chars}"


class DocumentTaskProcessor:
    """Background task processor for document processing."""
    
    def __init__(self):
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.worker_task: Optional[asyncio.Task] = None
        self.active_tasks: Dict[str, bool] = {}  # task_id -> cancellation flag

    async def submit_task(
        self,
        kb_id: str,
        enhance: int,
        filename: str,
        file_path: str,
        category: Optional[str] = None,
        chunk_config: Optional[Dict] = None,  # Custom chunk configuration
        doc_id: Optional[str] = None,  # Pre-generated doc_id (for immediate return)
        resource_id: Optional[int] = None  # External system resource ID
    ) -> str:
        """
        Submit a new document processing task.

        Args:
            kb_id: Knowledge base ID
            enhance: 设置文档或视频资源是否知识增强：0=不增强，1=增强
            filename: Original filename
            file_path: Path to uploaded file
            category: Document category
            chunk_config: Custom chunking configuration (optional)
            doc_id: Pre-generated document ID (optional, for immediate return)
            resource_id: External system resource ID (optional)

        Returns:
            Task ID
        """
        task_id = generate_task_id()

        # Create task record in database
        db = await get_database()
        task_model = DocumentTaskModel(
            task_id=task_id,
            kb_id=kb_id,
            enhance=enhance,
            filename=filename,
            file_path=file_path,
            category=category,
            status="pending",
            metadata={
                "chunk_config": chunk_config,
                "resource_id": resource_id
            } if chunk_config or resource_id else {}
        )

        task_dict = task_model.model_dump()

        # Add doc_id if provided (pre-generated)
        if doc_id:
            task_dict["doc_id"] = doc_id
        if resource_id:
            task_dict["resource_id"] = resource_id

        await db.document_tasks.insert_one(task_dict)

        # Add to queue
        await self.task_queue.put({
            "task_id": task_id,
            "kb_id": kb_id,
            "enhance": enhance,
            "filename": filename,
            "file_path": file_path,
            "category": category,
            "chunk_config": chunk_config,
            "doc_id": doc_id,  # Pass doc_id to processing
            "resource_id": resource_id
        })

        logger.info(f"submit_task put task_queue task_id={task_id}")

        return task_id

    async def re_submit_task(
        self,
        task_id: str,
        kb_id: str,
        enhance: str,
        filename: str,
        file_path: str,
        category: Optional[str] = None,
        chunk_config: Optional[Dict] = None,  # Custom chunk configuration
        doc_id: Optional[str] = None,  # Pre-generated doc_id (for immediate return)
        resource_id: Optional[int] = None  # External system resource ID):
    ):
        # Add to queue
        await self.task_queue.put({
            "task_id": task_id,
            "kb_id": kb_id,
            "enhance": enhance,
            "filename": filename,
            "file_path": file_path,
            "category": category,
            "chunk_config": chunk_config,
            "doc_id": doc_id,  # Pass doc_id to processing
            "resource_id": resource_id
        })

        logger.info(f"re_submit_task task_id={task_id} doc_id={doc_id}")

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
                    await asyncio.create_task(self._execute_faq_vectorization(task_data))
                elif task_type == "thesaurus_major_vectorization":
                    await asyncio.create_task(self._execute_thesaurus_major_vectorization(task_data))
                elif task_type == "thesaurus_sensitive_vectorization":
                    await asyncio.create_task(self._execute_thesaurus_sensitive_vectorization(task_data))
                else:
                    # Default to document processing
                    await asyncio.create_task(self._execute_document_vectorization(task_data))
                
            except Exception as e:
                logger.error(f"Error in task processor loop: error={str(e)}", exc_info=True)
                await asyncio.sleep(1)

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

    async def submit_faq_vectorization_task_by_employee_id(self, employee_id: str):
        """
            提交FAQ向量化任务（异步后台处理）。
            从MongoDB的faqs集合中读取该员工的FAQ数据进行向量化。
        """
        task_id = f"faq_task_{employee_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        # Add to queue for background processing
        await self.task_queue.put({
            "task_id": task_id,
            "task_type": "faq_vectorization",
            "employee_id": employee_id
        })

        logger.info(f"submit_faq_vectorization_task_by_employee_id task_id={task_id} employee_id={employee_id}")

    async def submit_faq_vectorization_task(self, faq_id: str):
        """
            提交FAQ向量化任务（异步后台处理）。
            从MongoDB的faqs集合中读取该员工的FAQ数据进行向量化。
        """
        task_id = f"faq_task_{faq_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        # Add to queue for background processing
        await self.task_queue.put({
            "task_id": task_id,
            "task_type": "faq_vectorization",
            "faq_id": faq_id
        })

        logger.info(f"submit_faq_vectorization_task task_id={task_id} faq_id={faq_id}")

    async def _execute_document_vectorization(self, task_data: Dict):
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
            doc_id = await self._exec_process_document(task_data)

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

    async def _exec_process_document(self, task_data: Dict) -> str:
        """Process document with progress updates.
            保存文档
        """
        task_id = task_data["task_id"]

        # Call original document processor
        # We'll wrap it to track progress
        doc_id = await doc_processor.process_document(
            file_path=task_data["file_path"],
            filename=task_data["filename"],
            kb_id=task_data["kb_id"],
            enhance=task_data["enhance"],
            category=task_data.get("category"),
            task_id=task_id,  # Pass task_id for progress tracking
            chunk_config=task_data.get("chunk_config"),  # Pass chunk_config
            doc_id=task_data.get("doc_id"),  # Pass pre-generated doc_id
            resource_id=task_data.get("resource_id")  # Pass resource_id
        )

        return doc_id

    async def _execute_faq_vectorization(self, task_data: Dict):
        """ 执行FAQ向量化任务 """
        if task_data["employee_id"]:
            return await faq_processor.vectorization_faq_by_employee_id(
                task_id=task_data["task_id"],
                employee_id=task_data["employee_id"]
            )
        elif task_data["faq_id"]:
            return await faq_processor.vectorization_faq(
                task_id=task_data["task_id"],
                faq_id=task_data["faq_id"]
            )

    async def _execute_thesaurus_major_vectorization(self, task_data: Dict):
        """ 执行专业词库向量化任务 """
        task_id = task_data["task_id"]
        db = await get_database()

    async def _execute_thesaurus_sensitive_vectorization(self, task_data: Dict):
        """ 执行敏感词库向量化任务 """
        db = await get_database()


# Global task processor instance
task_processor = DocumentTaskProcessor()
