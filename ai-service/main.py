"""
Main FastAPI application.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.core.database import mongodb
from app.core.elasticsearch import es_db

from app.api.middleware.error_handler import (
    http_exception_handler,
    validation_exception_handler,
    general_exception_handler
)
from app.api.endpoints import chat, session, employee, conversation, mineru, thesaurus_sensitive, thesaurus_major
from app.api.endpoints import knowledge_base_kb, documents, metrics, websocket, websocket_view

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Uvicorn reload configuration
UVICORN_RELOAD_EXCLUDES = [
    "*.log",
    "*.db",
    "logs/*",
    "data/*",
    ".venv/*",
    "__pycache__/*",
    "*.pyc",
    ".git/*",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Digital Employee AI Service...")
    
    try:
        # Connect to databases
        await mongodb.connect()
        await es_db.connect()
        
        # Start task processor
        # await task_processor.start()

        logger.info("All databases connected and task processor started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: error={str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Digital Employee AI Service...")

    try:
        # Stop task processor first (停止任务处理器)
        # await task_processor.stop()
        logger.debug("Task processor stopped")

        # Disconnect databases in reverse order (按相反顺序断开数据库连接)
        await mongodb.disconnect()
        logger.debug("MongoDB disconnected")

        await es_db.disconnect()
        logger.debug("ElasticSearch disconnected")

        logger.info("All services stopped successfully")

    except Exception as e:
        logger.error("Error during shutdown", error=str(e), exc_info=True)


# Create FastAPI application
app = FastAPI(
    title="Digital Employee AI Service",
    description="AI驱动的数字员工服务，集成RAG、网络搜索和对话管理功能。",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register exception handlers
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat 数字员工对话"])
app.include_router(session.router, prefix="/api/chat/session", tags=["session 数字员工会话"])
app.include_router(employee.router, prefix="/api/employee", tags=["employee 数字员工"])
# Knowledge base endpoints (split into two files)
app.include_router(knowledge_base_kb.router, prefix="/api/knowledge_base", tags=["knowledge_base RAG文档库"])
app.include_router(documents.router, prefix="/api/knowledge_base/documents", tags=["documents RAG文档"])
# app.include_router(dataset_video.router, prefix="/api/knowledge_base/video", tags=["video 视频资源"])
app.include_router(conversation.router, prefix="/api/conversation", tags=["Conversation 数字员工对话记录"])
app.include_router(mineru.router, prefix="/api/mineru", tags=["MinerU"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["Metrics"])
app.include_router(websocket.router, prefix="/api/chat", tags=["WebSocket 数字员工对话"])
app.include_router(websocket_view.router, prefix="/api/chat", tags=["WebSocket 数字员工对话v2"])
# app.include_router(dataset_faq.router, prefix="/api/dataset_faq", tags=["dataset_faq FAQ问答库"])
app.include_router(thesaurus_major.router, prefix="/api/thesaurus_major", tags=["thesaurus_major 专业词库"])
app.include_router(thesaurus_sensitive.router, prefix="/api/thesaurus_sensitive", tags=["thesaurus_sensitive 敏感词库"])


@app.get("/")
async def root():
    """重定向到文档页面。"""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mongodb": "connected" if mongodb.db is not None else "disconnected",
        "chroma": "connected" if chroma_db.client is not None else "disconnected",
        "elasticsearch": "connected" if es_db.client is not None else "disconnected"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
        reload_excludes=UVICORN_RELOAD_EXCLUDES,
        log_level="warning"
    )
