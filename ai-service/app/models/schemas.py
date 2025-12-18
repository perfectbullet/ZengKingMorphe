"""
API request and response schemas.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# Chat API Schemas
class ChatRequest(BaseModel):
    """Chat request schema."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user_123456",
                "employee_id": "hutao",
                "session_id": "sess_20251218_abc123",
                "query": "分别介绍雕蜡与铸造工艺基本原理",
                "context": {"platform": "web", "version": "1.0"}
            }
        }
    )
    
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


# OpenAI-style API Schemas
class OpenAIMessage(BaseModel):
    """OpenAI message format."""
    role: str = Field(..., description="Message role: system/user/assistant")
    content: str = Field(..., description="Message content")


class OpenAIChatRequest(BaseModel):
    """OpenAI-style chat completion request."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": "今天天气怎么样？"}
                ],
                "stream": True,
                "employee_id": "hutao",
                "user_id": "user_123456"
            }
        }
    )
    
    model: str = Field(default="gpt-3.5-turbo", description="Model name")
    messages: List[OpenAIMessage] = Field(..., description="Conversation messages")
    stream: bool = Field(default=False, description="Enable streaming")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    # Custom fields for our system
    employee_id: str = Field(default="default", description="Digital employee ID")
    user_id: str = Field(..., description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")


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
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "employee_id": "emp_customer_service",
                "name": "客服小助手",
                "domain": "客户服务",
                "role": "客服专员",
                "description": "专业的客户服务数字员工，擅长解答产品问题和售后咨询",
                "personality": {
                    "tone": "professional",
                    "style": "friendly",
                    "language": "zh-CN",
                    "formality": "moderate"
                },
                "capabilities": {
                    "kb_ids": ["kb_product_manual", "kb_faq"],
                    "web_search_enabled": True,
                    "max_context_turns": 10,
                    "multimodal_enabled": False
                },
                "greeting": "您好！我是客服小助手，很高兴为您服务。有什么我可以帮到您的吗？",
                "hot_questions": [
                    "如何申请退款？",
                    "产品保修期是多久？",
                    "如何联系人工客服？"
                ],
                "personalization": {
                    "user_profiling_enabled": True,
                    "personalized_recommendations": True,
                    "adaptive_tone": True
                }
            }
        }
    )
    
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
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "产品使用手册",
                "description": "包含所有产品的详细使用说明和常见问题解答",
                "category": "产品文档",
                "priority": "high",
                "tags": ["产品", "教程", "FAQ"],
                "config": {
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "embedding_model": "text-embedding-3-small"
                }
            }
        }
    )
    
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
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start_date": "2025-12-01",
                "end_date": "2025-12-18",
                "user_id": "user_123456",
                "employee_id": "emp_customer_service",
                "keyword": "退款",
                "page": 1,
                "page_size": 20
            }
        }
    )
    
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
