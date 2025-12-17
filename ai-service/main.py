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
from app.core.chroma import chroma_db
from app.core.elasticsearch import es_db
from app.services.task_processor import task_processor
from app.api.middleware.error_handler import (
    http_exception_handler,
    validation_exception_handler,
    general_exception_handler
)
from app.api.endpoints import chat, session, employee, knowledge_base, conversation, webhook

# Setup logging
setup_logging()
logger = get_logger(__name__)


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
        chroma_db.connect()
        await es_db.connect()
        
        # Start task processor
        await task_processor.start()
        
        logger.info("All databases connected and task processor started successfully")
        
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Digital Employee AI Service...")
    
    try:
        # Stop task processor
        await task_processor.stop()
        
        await mongodb.disconnect()
        chroma_db.disconnect()
        await es_db.disconnect()
        
        logger.info("Task processor stopped and all databases disconnected successfully")
        
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


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
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(session.router, prefix="/api/chat/session", tags=["Session"])
app.include_router(employee.router, prefix="/api/ai/digital-employee", tags=["Digital Employee"])
app.include_router(knowledge_base.router, prefix="/api/knowledge-base", tags=["Knowledge Base"])
app.include_router(conversation.router, prefix="/api/conversation", tags=["Conversation"])
app.include_router(webhook.router, prefix="/api/ai", tags=["Webhook"])


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
        reload=settings.debug
    )
