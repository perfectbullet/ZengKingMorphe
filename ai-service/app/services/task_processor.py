"""
Async document task processor with queue management.
"""
import asyncio
import random
import string
from datetime import datetime
from typing import Dict, Optional
from app.core.logging import get_logger
from app.core.database import get_database
from app.models.database import DocumentTaskModel
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
                    logger.info(f"Fetched task from queue: task_id={task_data['task_id']}")
                except asyncio.TimeoutError:
                    continue
                
                task_id = task_data["task_id"]
                
                # Mark task as active
                self.active_tasks[task_id] = False  # cancellation flag
                
                # Process task in background
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


# Global task processor instance
task_processor = DocumentTaskProcessor()
