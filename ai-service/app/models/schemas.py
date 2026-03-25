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


class OpenAITool(BaseModel):
    """OpenAI tool/function calling format."""
    type: str = Field(default="function", description="Tool type: function")
    function: Optional[Dict[str, Any]] = Field(None, description="Function definition")


class OpenAIChatRequest(BaseModel):
    """OpenAI-style chat completion request."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "qwen2.5:7b",
                "messages": [
                    {"role": "system", "content": "你是一个专业的客服助手。"},
                    {"role": "user", "content": "胡桃是谁？"}
                ],
                "stream": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "seed": 42,
                "n": 1,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "获取天气信息",
                            "parameters": {"type": "object", "properties": {}}
                        }
                    }
                ],
                "employee_id": "hutao",
                "user_id": "user_123456",
                "session_id": "sess_20251218_abc123",
                "channel_name": "web",
                "team_id": "team_001",
                "user_name": "苏文心",
                "head_url": "/edu-api/fileserver/default/image/2025/5/14/e6d399d9-3785-4d08-bbab-213a6df390c4.jpeg"
            }
        }
    )

    # OpenAI standard parameters
    model: str = Field(default="qwen3:32b", description="Model name")
    messages: List[OpenAIMessage] = Field(..., description="Conversation messages")
    stream: bool = Field(default=False, description="Enable streaming")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens to generate")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    seed: Optional[int] = Field(default=None, ge=0, description="Random seed for reproducibility")
    n: Optional[int] = Field(default=1, ge=1, description="Number of completions to generate")
    tools: Optional[List[OpenAITool]] = Field(default=None, description="List of tools/functions available")

    # Custom fields for our system
    employee_id: str = Field(default="29", description="Digital employee ID")
    user_id: str = Field(default="user_20260122", description="User ID")
    user_name: str = Field(default="用户名称", description="用户登录后的名字")
    head_url: str = Field(default="用户头像", description="用户头像")
    session_id: Optional[str] = Field(default="sess_4_42478261_29", description="Session ID")
    channel_name: Optional[str] = Field(default=None, description="Channel name (web, mobile, etc.)")
    team_id: Optional[str] = Field(default=None, description="Team ID")

    # Extra body for non-OpenAI standard parameters (passed via OpenAI SDK's extra_body)
    extra_body: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extra body parameters for non-OpenAI standard fields (channel_name, team_id, user_id, employee_id, etc.)"
    )


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
                    "device_id": "CJQX-YJO1",
                    "device_name": "数字人全息舱 DSee型号",
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
        description="Additional session metadata (e.g., platform, device, source, device_id, device_name)"
    )


class SessionResponse(BaseModel):
    """Session response schema."""
    code: int = 200
    message: str = "success"
    data: Dict[str, Any]


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
    name: str = Field(..., description="Knowledge base Name")
    description: Optional[str] = ""
    priority: Optional[str] = "medium"
    tags: List[str] = Field(default_factory=list)


class SearchDocumentsRequest(BaseModel):
    """Search documents in knowledge base request schema."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kb_id": "kb_12ad113e4c80",
                "query": "失蜡铸造的原理",
                "top_k": 5,
                "use_hybrid": True
            }
        }
    )

    kb_id: str = Field(..., description="Knowledge base ID")
    query: str = Field(..., description="Search query", max_length=500)
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    use_hybrid: bool = Field(default=True, description="Use hybrid search (vector + rerank)")
    enable_rerank: Optional[bool] = Field(default=None, description="Enable reranking (default: True)")


class SearchResultItem(BaseModel):
    """Single search result item."""
    rank: int = Field(..., description="Result ranking")
    doc_id: str = Field(..., description="Document ID")
    chunk_id: Optional[str] = Field(None, description="Chunk ID")
    content: str = Field(..., description="Document content snippet")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchDocumentsResponse(BaseModel):
    """Search documents response schema."""
    query: str = Field(..., description="Original search query")
    kb_id: str = Field(..., description="Knowledge base ID")
    total: int = Field(..., description="Total number of results")
    results: List[SearchResultItem] = Field(default_factory=list, description="Search results")


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
    id: int = Field(..., description="主键id")
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
                "doc_id": "doc_994051b1c13e",
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

    doc_id: str = Field(..., description="文档ID")
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
                "enhance": 1,
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
    enhance: int = Field(1, description="设置文档或视频资源是否知识增强：0=不增强，1=增强")


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
    prologue: Optional[str] = Field(None, description="开场白内容")
    is_opening_questions: bool = Field(False, description="是否开启开场热门问题")
    question_type: int = Field(0, description="热门问题类型: 1-自动推荐 2-FAQ 3-自定义")
    faqs: List[ExternalFAQItem] = Field(default_factory=list, description="开场推荐FAQ列表")
    my_questions: List[str] = Field(default_factory=list, description="自定义问题列表")
    
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


class EmployeeSettingConfig(BaseModel):
    """外部API设置配置。"""
    employee_id: str = Field('', description="数字员工id")
    knowledge: ExternalKnowledgeConfig = Field(..., description="对话准备--知识库配置")
    prologue: ExternalPrologueConfig = Field(..., description="对话开始--开场白配置")
    rule: ExternalRuleConfig = Field(..., description="对话中--规则规则、异常或未匹配规则、安全规则配置")
    role: ExternalRoleConfig = Field(..., description="角色--人设配置")
    plugins: str = Field(..., description="高级设置--插件配置")
    major_word: str = Field(..., description="高级设置--专业词库配置")

    model_config = ConfigDict(populate_by_name=True)


class ExternalEmployeeAPIData(BaseModel):
    """外部API完整数据结构（data字段）。"""
    employee: ExternalEmployeeInfo = Field(..., description="员工信息")
    setting: EmployeeSettingConfig = Field(..., description="对话设定")
    
    model_config = ConfigDict(populate_by_name=True)


class ExternalEmployeeAPIResponse(BaseModel):
    """ 数字员工请求参数对象 """
    status: int = Field(..., description="状态码")
    message: str = Field(..., description="消息")
    data: ExternalEmployeeAPIData = Field(..., description="数据内容")
    success: bool = Field(..., description="是否成功")
    error: Optional[str] = Field(None, description="错误信息")
    
    model_config = ConfigDict(populate_by_name=True)


class ResponseResult:
    """
        返回结果对象
    """
    @staticmethod
    def success(data: Any):
        return {
                "code": 200,
                "message": "success",
                "data": data
        }

    @staticmethod
    def error(code: int, message: str, data: Any):
        return {
                "code": code,
                "message": message,
                "data": data
        }


class CreateDatasetVideoRequest(BaseModel):
    """创建视频资源请求"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kb_id": "team_28",
                "enhance": 1,
                "resource_id": 49392,
                "document_name": "测试视频.mp4",
                "resource_url": "https://education-test-private.oss-cn-beijing.aliyuncs.com/text/04057b3a-aa93-4848-b60d-9aad3282d714.md?Expires=1769436115&OSSAccessKeyId=LTAI4FctZ3DLxBxVPrqD4sCo&Signature=bvtIhWU%2Ff6dBYodO2vgkfVHqAvo%3D"
            }
        }
    )

    kb_id: str = Field(..., description="团队ID")
    resource_id: int = Field(..., description="系统资源ID")
    document_name: str = Field(..., description="文档名称")
    start_time: Optional[str] = Field(None, description="生效开始时间")
    end_time: Optional[str] = Field(None, description="生效结束时间")
    resource_url: str = Field(..., description="系统资源URL")
    enhance: int = Field(..., description="设置文档或视频资源是否知识增强：0=不增强，1=增强")


class DatasetFaqRequest(BaseModel):
    """ FAQ请求参数对象 """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "faq_id": 8,
                "employee_ids": [
                    1,
                    2
                ],
                "question_name": "问题101",
                "end_time": "2025-12-16 18:09:45",
                "end_time": "2099-12-31 23:23:59",
                "is_enable": 1,
                "is_clear": 0,
                "update_time": "2025-12-17 16:49:29",
                "similar_questions": [
                    "相似问题101",
                    "相似问题102",
                    "相似问题103"
                ],
                "answers": [
                    "答案101",
                    "答案102"
                ]
            }
        }
    )

    faq_id: int = Field(..., description="FAQ问答id")
    employee_ids: List[int] = Field(default_factory=list, description="数字员工id列表")
    question_name: str = Field(..., description="标准问题")
    start_time: Optional[str] = Field(None, description="生效开始时间")
    end_time: Optional[str] = Field(None, description="生效结束时间")
    is_enable: int = Field(1, description="是否启用：0=不启用，1=启用")
    is_clear: int = Field(0, description="是否澄清：0=不澄清，1=澄清")
    update_time: Optional[str] = Field(None, description="更新时间")
    similar_questions: List[str] = Field(default_factory=list)
    answers: List[str] = Field(default_factory=list)


class ThesaurusWord(BaseModel):
    word_id: int = Field(..., description="词条id")
    word_name: str = Field(..., description="词条名称")
    similar_words: List[str] = Field(default_factory=list, description="相似词条名称列表")


class ThesaurusRequest(BaseModel):
    """ 词库请求参数对象 """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "thesaurus_id": 8,
                "employee_ids": [
                    1,
                    2
                ],
                "thesaurus_name": "古典名著",
                "is_enable": 1,
                "update_time": "2025-12-17 16:49:29",
                "thesaurus_words": [
                    {
                        "word_id": 1,
                        "word_name": "红楼梦",
                        "similar_words": [
                            "石头记",
                            "小红书"
                        ]
                    }
                ]
            }
        }
    )

    thesaurus_id: int = Field(..., description="词库id")
    employee_ids: List[int] = Field(default_factory=list, description="数字员工id列表")
    thesaurus_name: str = Field(..., description="词库名称")
    is_enable: int = Field(1, description="是否启用：0=不启用，1=启用")
    update_time: Optional[str] = Field(None, description="更新时间")
    thesaurus_words: List[ThesaurusWord] = Field(default_factory=list, description="词条名称列表")


class CreateEmployeeRequest(BaseModel):
    """ 创建数字员工请求参数对象 """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "employee_id": "23",
                "team_id": 28,
                "name": "陈晓燕",
                "position": "校园助教",
                "employee_type": "FULL",
                "tone": "intellectual",
                "language": "mandarin",
                "gender": 2,
                "intro": "心理关怀/n辅导员",
                "portrait": "/edu-api/fileserver/default/image/2026/2/5/371587a2-ef39-4c04-83ea-991e15e33801.png",
                "model_image": "/edu-api/fileserver/default/image/2026/2/5/371587a2-ef39-4c04-83ea-991e15e33801.png",
                "digital_code": "suwenxin",
                "onduty_status": 1,
                "create_time": "2025-12-17 16:49:29",
                "update_time": "2025-12-17 16:49:29"
            }
        }
    )

    employee_id: str = Field(..., description="数字员工id")
    team_id: int = Field(..., description="团队id")
    name: str = Field(..., description="员工名称")
    position: str = Field(None, description="职位")
    employee_type: str = Field(..., description="数字员工类型: AVATAR=头像, HALF=半身, FULL=全身")
    tone: str = Field(..., description="音色")
    language: str = Field(..., description="语言")
    gender: int = Field(0, description="数字员工性别: 1-男 2-女, 0-未知")
    intro: str = Field(..., description="数字员工简介")
    portrait: str = Field(..., description="头像地址")
    model_image: str = Field(..., description="模型图片或模型地址")
    digital_code: str = Field(..., description="数字人在AI平台的唯一编码")
    onduty_status: int = Field(0, description="值班状态: 0-休息中，1-值班中")
    create_time: str = Field(None, description="创建时间")
    update_time: str = Field(None, description="更新时间")


class UpdateEmployeeRequest(BaseModel):
    """ 修改数字员工请求参数对象 """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "employee_id": "23",
                "team_id": 28,
                "name": "陈晓燕",
                "position": "校园助教",
                "employee_type": "FULL",
                "tone": "intellectual",
                "language": "mandarin",
                "update_time": "2025-12-17 16:49:29"
            }
        }
    )

    employee_id: str = Field(..., description="数字员工id")
    team_id: int = Field(..., description="团队id")
    name: str = Field(..., description="员工名称")
    position: str = Field(None, description="职位")
    employee_type: str = Field(..., description="数字员工类型: AVATAR=头像, HALF=半身, FULL=全身")
    tone: str = Field(..., description="音色")
    language: str = Field(..., description="语言")
    update_time: Optional[str] = Field(None, description="更新时间")


class EmployeeSettingKnowledge(BaseModel):
    kb_ids: List[str] = Field(default_factory=list, description="知识库id列表")
    faqs: List[DatasetFaqRequest] = Field(default_factory=list, description="FAQ问答列表")
    video_ids: List[str] = Field(default_factory=list, description="视频资源id列表")


class EmployeeSettingPrologue(BaseModel):
    prologue: Optional[str] = Field(None, description="开场白内容")
    is_opening_questions: bool = Field(False, description="是否开启开场热门问题")
    prologue_question_type: int = Field(None, description="热门问题类型: 1-自动推荐 2-FAQ 3-自定义")
    prologue_faqs: List[DatasetFaqRequest] = Field(default_factory=list, description="开场热门问题关联FAQ列表")
    hot_questions: Optional[List[str]] = Field(default_factory=list, description="开场热门问题自定义列表")


class EmployeeSettingChatRule(BaseModel):
    is_multimodal: bool = Field(False, description="是否支持多模态")
    fixed_answer: Optional[str] = Field(None, description="关闭图片理解后返回的固定回复话术")
    faq_sim_threshold: Optional[float] = Field(None, description="FAQ相似度阈值")
    faq_top_k: Optional[int] = Field(None, description="FAQ最多推荐数量")


class EmployeeSettingUnusualRule(BaseModel):
    exception_reply: Optional[str] = Field(None, description="系统异常时的回复话术")
    not_match_reply_type: Optional[int] = Field(None, description="未匹配时的回复类型：0：固定话术，1：模型闲聊回复")
    fixed_replys: Optional[List[str]] = Field(None, description="not_match_reply_type=0时的固定回复列表")
    is_web_search: bool = Field(False, description="not_match_reply_type=1时的模型闲聊回复：是否联网搜索")
    is_show_sign: bool = Field(False, description="not_match_reply_type=1时的模型闲聊回复：是否显示标识")
    is_my_prompt: bool = Field(False, description="not_match_reply_type=1时的模型闲聊回复：是否启用自定义提示词")
    my_prompt: Optional[str] = Field(None, description="not_match_reply_type=1时的模型闲聊回复：自定义提示词内容")


class EmployeeSettingSafeRule(BaseModel):
    is_reject_answer: bool = Field(False, description="命中敏感词时是否拒绝回答")
    reject_answer: Optional[str] = Field(None, description="拒绝回答时的回复内容")
    thesaurus_sensitive: List[ThesaurusRequest] = Field(default_factory=list, description="敏感词库列表")


class EmployeeSettingRole(BaseModel):
    persona: Optional[str] = Field(None, description="人设")
    style: Optional[str] = Field(None, description="风格名称")
    style_desc: Optional[str] = Field(None, description="风格描述")


class EmployeeSettingPlugin(BaseModel):
    plugin_id: int = Field(..., description="插件id")
    plugin_name: str = Field(None, description="插件名称")
    plugin_code: Optional[str] = Field(None, description="插件编码")
    plugin_intro: Optional[str] = Field(None, description="插件简介")
    plugin_icon: Optional[str] = Field(None, description="插件图标")
    plugin_params: Optional[str] = Field(None, description="插件参数")


class EmployeeSettingThesaurusMajor(BaseModel):
    is_synonym_rewrite: bool = Field(False, description="是否同义词重写")
    thesaurus_major: List[ThesaurusRequest] = Field(default_factory=list, description="专业词库列表")


class EmployeePersonality(BaseModel):
    """Employee personality configuration."""
    tone: str = Field(default="professional")
    style: str = Field(default="friendly")
    language: str = Field(default="zh-CN")


class EmployeeCapabilities(BaseModel):
    """Employee capabilities configuration."""
    kb_ids: List[str] = Field(default_factory=list)
    is_web_search: bool = True
    max_context_turns: int = 10
    multimodal_enabled: bool = False


class EmployeePersonalization(BaseModel):
    """Employee personalization configuration."""
    user_profiling_enabled: bool = True
    personalized_recommendations: bool = True
    adaptive_tone: bool = True


class UpdateEmployeeSettingRequest(BaseModel):
    """ 数字员工对话设定请求参数对象 """
    employee_id: str = Field(..., description="数字员工id")
    update_time: str = Field(None, description="更新时间")
    update_type: str = Field(..., description="更新类型：knowledge、prologue、rule、role、plugins、thesaurus_major")
    knowledge: EmployeeSettingKnowledge = Field(None, description="对话准备--知识库配置")
    prologue: EmployeeSettingPrologue = Field(None, description="对话开始--开场白、开场热门问题")
    chat_rule: EmployeeSettingChatRule = Field(None, description="对话中--对话规则")
    unusual_rule: EmployeeSettingUnusualRule = Field(None, description="对话中--异常或未匹配规则")
    safe_rule: EmployeeSettingSafeRule = Field(None, description="对话中--安全规则配置")
    role: EmployeeSettingRole = Field(None, description="角色--人设")
    plugins: List[EmployeeSettingPlugin] = Field(None, description="高级设置--插件")
    thesaurus_major: EmployeeSettingThesaurusMajor = Field(None, description="高级设置--专业词库配置")
    # Nested structure (for new API) 具体作用？
    # personality: Optional[EmployeePersonality] = None
    # capabilities: Optional[EmployeeCapabilities] = None
    # greeting: Optional[str] = None
    # personalization: Optional[EmployeePersonalization] = None
    # metadata: Optional[Dict[str, Any]] = None

