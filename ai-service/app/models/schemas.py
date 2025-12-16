"""
API request and response schemas.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# Chat API Schemas
class ChatRequest(BaseModel):
    """Chat request schema."""
    user_id: str = Field(..., description="User ID")
    employee_id: str = Field(..., description="Digital employee ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    query: str = Field(..., description="User query", max_length=1000)
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class ChatResponse(BaseModel):
    """Chat response schema."""
    code: int = 200
    message: str = "success"
    data: Dict[str, Any]


# Session API Schemas
class SessionResponse(BaseModel):
    """Session response schema."""
    code: int = 200
    message: str = "success"
    data: Dict[str, Any]


# Digital Employee API Schemas
class EmployeePersonality(BaseModel):
    """Employee personality configuration."""
    tone: str = Field(default="professional")
    style: str = Field(default="friendly")
    language: str = Field(default="zh-CN")
    formality: str = Field(default="moderate")


class EmployeeCapabilities(BaseModel):
    """Employee capabilities configuration."""
    kb_ids: List[str] = Field(default_factory=list)
    web_search_enabled: bool = True
    max_context_turns: int = 10
    multimodal_enabled: bool = False


class EmployeePersonalization(BaseModel):
    """Employee personalization configuration."""
    user_profiling_enabled: bool = True
    personalized_recommendations: bool = True
    adaptive_tone: bool = True


class CreateEmployeeRequest(BaseModel):
    """Create employee request schema."""
    employee_id: str
    name: str
    domain: str
    role: str
    description: str
    personality: EmployeePersonality = Field(default_factory=EmployeePersonality)
    capabilities: EmployeeCapabilities = Field(default_factory=EmployeeCapabilities)
    greeting: str = ""
    hot_questions: List[str] = Field(default_factory=list)
    personalization: EmployeePersonalization = Field(default_factory=EmployeePersonalization)


class UpdateEmployeeRequest(BaseModel):
    """Update employee request schema."""
    name: Optional[str] = None
    domain: Optional[str] = None
    role: Optional[str] = None
    description: Optional[str] = None
    personality: Optional[EmployeePersonality] = None
    capabilities: Optional[EmployeeCapabilities] = None
    greeting: Optional[str] = None
    hot_questions: Optional[List[str]] = None
    personalization: Optional[EmployeePersonalization] = None


# Knowledge Base API Schemas
class KnowledgeBaseConfig(BaseModel):
    """Knowledge base configuration."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "text-embedding-3-small"


class CreateKnowledgeBaseRequest(BaseModel):
    """Create knowledge base request schema."""
    name: str
    description: str
    category: str
    priority: str = "medium"
    tags: List[str] = Field(default_factory=list)
    config: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)


class UpdateKnowledgeBaseRequest(BaseModel):
    """Update knowledge base request schema."""
    name: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[str] = None
    tags: Optional[List[str]] = None


# Document API Schemas
class DocumentUploadResponse(BaseModel):
    """Document upload response schema."""
    code: int = 200
    message: str = "success"
    data: Dict[str, Any]


# Webhook Schemas
class SyncNotifyRequest(BaseModel):
    """Sync notify webhook request schema."""
    event_type: str = Field(..., description="Event type: create/update/delete")
    word_ids: List[str] = Field(..., description="Word IDs")
    timestamp: str = Field(..., description="Timestamp")


# Conversation Record Schemas
class ConversationRecordQuery(BaseModel):
    """Conversation record query parameters."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    user_id: Optional[str] = None
    employee_id: Optional[str] = None
    session_id: Optional[str] = None
    keyword: Optional[str] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class ConversationStatisticsQuery(BaseModel):
    """Conversation statistics query parameters."""
    start_date: str = Field(..., description="Start date")
    end_date: str = Field(..., description="End date")
    dimension: Optional[str] = Field(None, description="Dimension: time/user/employee/intent/kb")
    employee_id: Optional[str] = None


# Error Response Schema
class ErrorResponse(BaseModel):
    """Error response schema."""
    code: int
    message: str
    error: Dict[str, Any] = Field(default_factory=dict)
