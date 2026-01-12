"""
API request and response schemas.
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator


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
                "session_id": "sess_hutao_abc123",
                "metadata": {
                    "platform": "web",
                    "device": "desktop",
                    "source": "metahuman_app",
                }
            }
        }
    )
    
    user_id: str = Field(..., description="User ID")
    employee_id: str = Field(..., description="Digital employee ID")
    session_id: Optional[str] = Field(
        None,
        description="Custom session ID for idempotent creation (format: sess_{12 hex digits})"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional session metadata (e.g., platform, device, source)"
    )


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
                "tags": ["产品", "教程"],
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
                "kb_id": "kb_id_abc123",
                "resource_id": 48907,
                "document_name": "首饰雕蜡工艺-全本.txt",
                "segment_flag": 1,
                "segment_vo": {
                    "kb_id": "kb_id_abc123",
                    "is_space_flag": 1,
                    "is_menu_flag": 0,
                    "segment_type": 1,
                    "is_segment_union_flag": 1,
                    "segment_union_max_length": 700,
                    "segment_identifier_type": 0,
                    "identifier_default": "1111111",
                    "identifier_customize": ""
                },
                "resource_url": "https://example.com/file.txt"
            }
        }
    )

    kb_id: str = Field(..., description="知识库ID")
    resource_id: int = Field(..., description="系统资源ID")
    document_name: str = Field(..., description="文档名称")
    start_time: Optional[str] = Field(None, description="生效开始时间")
    end_time: Optional[str] = Field(None, description="生效结束时间")
    segment_flag: int = Field(0, description="分段策略：0=自动分段，1=自定义文档分段")
    segment_vo: Optional[SegmentVo] = Field(None, description="RAG文档分段设置")
    resource_url: str = Field(..., description="系统资源URL")


class CreateRagDocumentResponse(BaseModel):
    """创建RAG文档响应。"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": 200,
                "message": "success",
                "data": {
                    "task_id": "document_task_202512301645_abc12345",
                    "status": "processing",
                    "resource_id": 48907,
                    "document_name": "首饰雕蜡工艺-全本.txt",
                    "kb_id": "kb_id_abc123"
                }
            }
        }
    )
    
    code: int = 200
    message: str = "success"
    data: Dict[str, Any] = Field(
        ...,
        description="响应数据，包含task_id（通过查询任务接口获取document_id）、status、resource_id、document_name、kb_id"
    )


# External API Data Schemas (Java Platform Integration)
class ExternalFAQItem(BaseModel):
    """外部API FAQ项（来自Java平台）。"""
    faq_id: Union[str, int] = Field(..., alias="id", description="FAQ ID")
    team_id: Optional[int] = Field(None, alias="teamId", description="Team ID")
    question_name: str = Field(..., alias="questionName", description="主问题")
    start_time: Optional[str] = Field(None, alias="startTime", description="生效开始时间")
    end_time: Optional[str] = Field(None, alias="endTime", description="生效结束时间")
    is_enable: Optional[int] = Field(None, alias="isEnable", description="是否启用：0=禁用，1=启用")
    is_clear: Optional[int] = Field(None, alias="isClear", description="是否清除：0=不清除，1=清除")
    create_user_id: Optional[int] = Field(None, alias="createUserId", description="创建用户ID")
    update_time: Optional[str] = Field(None, alias="updateTime", description="更新时间")
    create_time: Optional[str] = Field(None, alias="createTime", description="创建时间")
    similar_questions: List[str] = Field(default_factory=list, alias="similarQuestions", description="相似问题列表")
    answers: List[str] = Field(default_factory=list, description="答案列表")

    model_config = ConfigDict(populate_by_name=True)

    # Convert faq_id to string for consistency
    @model_validator(mode='before')
    @classmethod
    def convert_faq_id_to_str(cls, data):
        if isinstance(data, dict):
            faq_id = data.get('id')
            if isinstance(faq_id, int):
                data['id'] = str(faq_id)
        return data


class ExternalRAGDataset(BaseModel):
    """外部API RAG数据集（知识库）。"""
    kb_id: Union[str, int] = Field(..., alias="id", description="Dataset ID")
    team_id: Optional[int] = Field(None, alias="teamId", description="Team ID")
    rag_dataset_id: Optional[str] = Field(None, alias="ragDatasetId", description="RAG数据集ID（映射为kb_id）")
    name: Optional[str] = Field(None, description="数据集名称")
    is_enable: Optional[int] = Field(None, alias="isEnable", description="是否启用：0=禁用，1=启用")
    is_publish: Optional[int] = Field(None, alias="isPublish", description="是否发布：0=未发布，1=已发布")
    create_user_id: Optional[int] = Field(None, alias="createUserId", description="创建用户ID")
    flag: Optional[int] = Field(None, description="标志位")
    create_time: Optional[str] = Field(None, alias="createTime", description="创建时间")

    model_config = ConfigDict(populate_by_name=True)

    # Convert kb_id to string for consistency
    @model_validator(mode='before')
    @classmethod
    def convert_kb_id_to_str(cls, data):
        if isinstance(data, dict):
            kb_id = data.get('id')
            if isinstance(kb_id, int):
                data['id'] = str(kb_id)
        return data


class ExternalEmployeeInfo(BaseModel):
    """外部API员工信息。"""
    employee_id: Union[str, int] = Field(..., alias="id", description="员工ID")
    team_id: Optional[int] = Field(None, alias="teamId", description="Team ID")
    name: Optional[str] = Field(None, description="员工名称")
    position: Optional[str] = Field(None, description="职位")
    type: Optional[str] = Field(None, description="类型：AVATAR等")
    tone: Optional[str] = Field(None, description="语气风格")
    language: Optional[str] = Field(None, description="语言")
    create_user_id: Optional[int] = Field(None, alias="createUserId", description="创建用户ID")
    onduty_status: Optional[int] = Field(None, alias="ondutyStatus", description="在岗状态")
    update_time: Optional[str] = Field(None, alias="updateTime", description="更新时间")
    create_time: Optional[str] = Field(None, alias="createTime", description="创建时间")
    gender: Optional[int] = Field(None, description="性别：0=女，1=男")
    intro: Optional[str] = Field(None, description="简介")
    portrait: Optional[str] = Field(None, description="头像URL")
    model_image: Optional[str] = Field(None, alias="modelImage", description="模型图片URL")
    digital_code: Optional[str] = Field(None, alias="digitalCode", description="数字代码")

    model_config = ConfigDict(populate_by_name=True)

    # Convert employee_id to string for consistency
    @model_validator(mode='before')
    @classmethod
    def convert_employee_id_to_str(cls, data):
        if isinstance(data, dict):
            employee_id = data.get('id')
            if isinstance(employee_id, int):
                data['id'] = str(employee_id)
        return data


class ExternalKnowledgeConfig(BaseModel):
    """外部API知识配置。"""
    rag_datasets: List[ExternalRAGDataset] = Field(default_factory=list, alias="ragDatasets", description="RAG数据集列表")
    faqs: List[ExternalFAQItem] = Field(default_factory=list, description="FAQ列表")
    
    model_config = ConfigDict(populate_by_name=True)


class ExternalPrologueConfig(BaseModel):
    """外部API开场白配置。"""
    prologue: Optional[str] = Field(None, description="开场白文本")
    is_opening_questions: bool = Field(False, alias="isOpeningQuestions", description="是否开启问题")
    question_type: int = Field(0, alias="questionType", description="问题类型")
    faqs: List[ExternalFAQItem] = Field(default_factory=list, description="开场推荐FAQ列表")
    my_questions: List[str] = Field(default_factory=list, alias="myQuestions", description="自定义问题列表")
    
    model_config = ConfigDict(populate_by_name=True)


class ExternalChatRule(BaseModel):
    """外部API对话规则配置。"""
    is_multimodal: bool = Field(False, alias="isMultimodal", description="是否多模态")
    faq_sim_threshold: float = Field(0.0, alias="faqSimThreshold", description="FAQ相似度阈值")
    faq_top_k: int = Field(1, alias="faqTopK", description="FAQ返回Top-K数量")
    
    model_config = ConfigDict(populate_by_name=True)


class ExternalLLMReply(BaseModel):
    """外部API LLM回复配置。"""
    is_web_search: bool = Field(False, alias="isWebSearch", description="是否启用网络搜索")
    is_show_sign: bool = Field(False, alias="isShowSign", description="是否显示标识")
    is_my_prompt: bool = Field(False, alias="isMyPrompt", description="是否自定义Prompt")
    my_prompt: Optional[str] = Field(None, alias="myPrompt", description="自定义Prompt内容")
    
    model_config = ConfigDict(populate_by_name=True)


class ExternalUnusualRule(BaseModel):
    """外部API异常规则配置。"""
    excepition_reply: Optional[str] = Field(None, alias="excepitonReply", description="异常回复")
    not_match_reply_type: int = Field(0, alias="notMatchReplyType", description="未匹配回复类型")
    fixed_replys: List[str] = Field(default_factory=list, alias="fixedReplys", description="固定回复列表")
    llm_reply: Optional[ExternalLLMReply] = Field(None, alias="llmReply", description="LLM回复配置")
    
    model_config = ConfigDict(populate_by_name=True)


class ExternalRuleConfig(BaseModel):
    """外部API规则配置。"""
    rule_config_id: Optional[Union[str, int]] = Field(None, alias="id", description="规则ID")
    chat_rule: Optional[ExternalChatRule] = Field(None, alias="chatRule", description="对话规则")
    unusual_rule: Optional[ExternalUnusualRule] = Field(None, alias="unusualRule", description="异常规则")

    model_config = ConfigDict(populate_by_name=True)

    # Convert rule_config_id to string for consistency
    @model_validator(mode='before')
    @classmethod
    def convert_rule_id_to_str(cls, data):
        if isinstance(data, dict):
            rule_id = data.get('id')
            if isinstance(rule_id, int):
                data['id'] = str(rule_id)
        return data


class ExternalRoleConfig(BaseModel):
    """外部API角色配置。"""
    persona: Optional[str] = Field(None, description="人设")
    style: Optional[str] = Field(None, description="风格")
    style_desc: Optional[str] = Field(None, alias="styleDesc", description="风格描述")
    
    model_config = ConfigDict(populate_by_name=True)


class ExternalSettingConfig(BaseModel):
    """外部API设置配置。"""
    knowledge: ExternalKnowledgeConfig = Field(..., description="知识配置")
    prologue: ExternalPrologueConfig = Field(..., description="开场白配置")
    rule: ExternalRuleConfig = Field(..., description="规则配置")
    role: ExternalRoleConfig = Field(..., description="角色配置")
    employee_id: str = Field('', description="员工ID")
    model_config = ConfigDict(populate_by_name=True)


class ExternalEmployeeAPIData(BaseModel):
    """外部API完整数据结构（data字段）。"""
    employee: ExternalEmployeeInfo = Field(..., description="员工信息")
    setting: ExternalSettingConfig = Field(..., description="设置配置")
    
    model_config = ConfigDict(populate_by_name=True)


class ExternalEmployeeAPIResponse(BaseModel):
    """外部API响应结构。"""
    status: int = Field(..., description="状态码")
    message: str = Field(..., description="消息")
    data: ExternalEmployeeAPIData = Field(..., description="数据内容")
    success: bool = Field(..., description="是否成功")
    error: Optional[str] = Field(None, description="错误信息")
    
    model_config = ConfigDict(populate_by_name=True)
