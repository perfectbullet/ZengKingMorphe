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


class RAGSource(BaseModel):
    """RAG document source reference."""
    rank: int = Field(..., description="Ranking position")
    doc_id: str = Field(..., description="Document ID")
    kb_id: str = Field(..., description="Knowledge base ID")
    content_snippet: str = Field(..., description="Content snippet (max 200 chars)")
    score: float = Field(..., description="Relevance score (RRF or similarity)")
    chunk_index: Optional[int] = Field(None, description="Chunk index within document")


class WebSource(BaseModel):
    """Web search source reference."""
    rank: int = Field(..., description="Ranking position")
    title: str = Field(..., description="Page title")
    url: str = Field(..., description="Source URL")
    score: float = Field(..., description="Relevance score from search API")


class SourceAttribution(BaseModel):
    """Source attribution for answer generation."""
    rag_sources: List[RAGSource] = Field(default_factory=list, description="RAG document sources (top 3)")
    web_sources: List[WebSource] = Field(default_factory=list, description="Web search sources (top 5)")


class ChatResponseData(BaseModel):
    """Chat response data schema."""
    conversation_id: str = Field(..., description="Conversation ID")
    session_id: str = Field(..., description="Session ID")
    answer: str = Field(..., description="AI-generated answer")
    intent: str = Field(default="general_query", description="Detected user intent")
    confidence: float = Field(..., description="Answer confidence score")
    kb_used: List[str] = Field(default_factory=list, description="Knowledge base IDs used")
    web_search_used: bool = Field(default=False, description="Whether web search was triggered")
    sources: SourceAttribution = Field(..., description="Source attribution for answer")
    timestamp: str = Field(..., description="Response timestamp (ISO 8601)")


class ChatResponse(BaseModel):
    """Chat response schema."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": 200,
                "message": "success",
                "data": {
                    "conversation_id": "conv_abc123def456",
                    "session_id": "sess_xyz789",
                    "answer": "根据知识库资料，雕蜡工艺是通过雕刻蜡模来制作首饰原型...",
                    "intent": "knowledge_query",
                    "confidence": 0.92,
                    "kb_used": ["kb_jewelry_tech"],
                    "web_search_used": False,
                    "sources": {
                        "rag_sources": [
                            {
                                "rank": 1,
                                "doc_id": "doc_12345",
                                "kb_id": "kb_jewelry_tech",
                                "content_snippet": "雕蜡工艺是传统首饰制作中的重要环节，通过精密雕刻蜡材形成首饰雏形...",
                                "score": 0.8756,
                                "chunk_index": 3
                            }
                        ],
                        "web_sources": []
                    },
                    "timestamp": "2025-12-19T10:30:00Z"
                }
            }
        }
    )
    
    code: int = 200
    message: str = "success"
    data: Dict[str, Any]  # Flexible dict to support both structured and legacy formats


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
                "model": "qwen3:32b",
                "messages": [
                    {"role": "user", "content": "今天天气怎么样？"}
                ],
                "stream": True,
                "employee_id": "hutao",
                "user_id": "user_123456",
                "session_id": "sess_20251218_abc123"
            }
        }
    )
    
    model: str = Field(default="qwen3:32b", description="Model name")
    messages: List[OpenAIMessage] = Field(..., description="Conversation messages")
    stream: bool = Field(default=False, description="Enable streaming")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    # Custom fields for our system
    employee_id: str = Field(default="hutao", description="Digital employee ID")
    user_id: str = Field('user_123456', description="User ID")
    session_id: Optional[str] = Field('sess_20251218_abc123', description="Session ID")
    session_id2: Optional[str] = Field('session_id2_default', description="Secondary Session ID")

# Session API Schemas
class CreateSessionRequest(BaseModel):
    """Create session request schema."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user_123456",
                "employee_id": "hutao",
                "metadata": {
                    "platform": "web",
                    "device": "desktop"
                }
            }
        }
    )
    
    user_id: str = Field(..., description="User ID")
    employee_id: str = Field(..., description="Digital employee ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


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
    kb_id: str = Field(..., description="Knowledge base ID")
    name: Optional[str] = Field(..., description="Knowledge base ID")
    description: Optional[str] = ""
    priority: Optional[str] = ""
    tags: Optional[List[str]] = []


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


class StreamChunkQuery(BaseModel):
    """Stream chunk query parameters."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user_123456",
                "employee_id": "hutao",
                "session_id": "sess_20251218_abc123",
                "chat_id": "chatcmpl-abc123",
                "start_date": "2025-12-01",
                "end_date": "2025-12-22",
                "page": 1,
                "page_size": 50
            }
        }
    )
    
    user_id: Optional[str] = Field(None, description="User ID")
    employee_id: Optional[str] = Field(None, description="Employee ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    chat_id: Optional[str] = Field(None, description="Chat completion ID")
    chunk_type: Optional[str] = Field(None, description="Chunk type filter")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=50, ge=1, le=200, description="Page size")


class StreamChunkResponse(BaseModel):
    """Stream chunk response schema."""
    code: int = 200
    message: str = "success"
    data: Dict[str, Any]


# Error Response Schema
class ErrorResponse(BaseModel):
    """Error response schema."""
    code: int
    message: str
    error: Dict[str, Any] = Field(default_factory=dict)


# RAG Document Segment Configuration Schemas
class SegmentVo(BaseModel):
    """RAG文档分段设置。"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "team_id": 40,
                "dataset_document_id": 16,
                "rag_document_id": "doc_abc123",
                "is_space_flag": 1,
                "is_menu_flag": 1,
                "segment_type": 1,
                "is_segment_union_flag": 1,
                "segment_union_max_length": 700,
                "segment_identifier_type": 0,
                "identifier_default": "1111111",
                "identifier_customize": ""
            }
        }
    )
    
    team_id: int = Field(..., description="团队ID")
    dataset_document_id: int = Field(..., description="RAG文档ID")
    rag_document_id: Optional[str] = Field(None, description="RAG系统文档ID")
    is_space_flag: int = Field(0, description="文本预处理：删除连续空格、换行、制表符：0=不启用，1=启用")
    is_menu_flag: int = Field(0, description="文本预处理：删除目录、页眉、页脚：0=不启用，1=启用")
    segment_type: int = Field(0, description="分段方式：0=换行切分，1=分段标识符切分")
    is_segment_union_flag: int = Field(0, description="分段方式-换行切分-是否分段合并：0=否，1=是")
    segment_union_max_length: int = Field(512, ge=128, le=2048, description="分段方式-换行切分-分段最大长度")
    segment_identifier_type: int = Field(0, description="分段方式-分段标识符切分-分段标识符：0=系统内置，1=自定义")
    identifier_default: str = Field("", description="分段标识符-系统内置：[省略号......],[中文句号。],[英文句号.],[中文感叹号！],[英文感叹号!],[中文问号？],[英文问号?]")
    identifier_customize: str = Field("", description="分段标识符-自定义：逗号分隔的标识符列表")


class CreateRagDocumentRequest(BaseModel):
    """创建RAG文档请求（Java平台集成）。"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "team_id": 40,
                "dataset_id": 3,
                "resource_id": 48907,
                "document_name": "首饰雕蜡工艺-全本.txt",
                "segment_flag": 1,
                "segment_vo": {
                    "team_id": 40,
                    "dataset_document_id": 16,
                    "is_space_flag": 1,
                    "is_menu_flag": 0,
                    "segment_type": 1,
                    "is_segment_union_flag": 1,
                    "segment_union_max_length": 700,
                    "segment_identifier_type": 0,
                    "identifier_default": "1111111",
                    "identifier_customize": ""
                },
                "rag_data_set_id": "kb_c8507c336f48",
                "resource_url": "https://example.com/file.txt"
            }
        }
    )
    
    team_id: int = Field(..., description="团队ID")
    dataset_id: int = Field(..., description="知识库ID")
    resource_id: int = Field(..., description="系统资源ID")
    document_name: str = Field(..., description="文档名称")
    start_time: Optional[str] = Field(None, description="生效开始时间")
    end_time: Optional[str] = Field(None, description="生效结束时间")
    segment_flag: int = Field(0, description="分段策略：0=自动分段，1=自定义文档分段")
    segment_vo: Optional[SegmentVo] = Field(None, description="RAG文档分段设置")
    rag_data_set_id: str = Field(..., description="RAG知识库ID (kb_id)")
    resource_url: str = Field(..., description="系统资源URL")


class CreateRagDocumentResponse(BaseModel):
    """创建RAG文档响应。"""
    code: int = 200
    message: str = "success"
    data: Dict[str, Any] = Field(
        default_factory=lambda: {
            "task_id": "",
            "rag_document_id": "",
            "status": "processing"
        }
    )
