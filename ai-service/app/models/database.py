"""
MongoDB database models.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ConversationModel(BaseModel):
    """Conversation record model."""
    conversation_id: str
    session_id: str
    user_id: str
    employee_id: str
    employee_name: str
    user_query: str
    ai_response: str
    is_realtime_query: bool = False
    realtime_category: Optional[str] = None
    intent: Optional[str] = None
    entities: Dict[str, Any] = Field(default_factory=dict)
    kb_used: List[str] = Field(default_factory=list)
    web_search_used: bool = False
    web_search_results: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)
    relevance_score: float = 0.0
    confidence: float = 0.0
    response_time_ms: int = 0
    satisfaction: Optional[str] = None
    has_sensitive: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class SessionModel(BaseModel):
    """Session model."""
    session_id: str
    user_id: str
    employee_id: str
    status: str = "active"  # active/ended/timeout
    message_count: int = 0
    context_messages: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmployeeConfigModel(BaseModel):
    """Digital employee configuration model."""
    employee_id: str
    name: str
    domain: str
    role: str
    description: str
    personality: Dict[str, str] = Field(default_factory=dict)
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    greeting: str = ""
    hot_questions: List[str] = Field(default_factory=list)
    personalization: Dict[str, bool] = Field(default_factory=dict)
    status: str = "active"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    synced_at: Optional[datetime] = None


class UserProfileModel(BaseModel):
    """User profile model."""
    user_id: str
    nickname: Optional[str] = None
    preferences: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    conversation_count: int = 0
    satisfaction_avg: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class KnowledgeBaseModel(BaseModel):
    """Knowledge base model."""
    kb_id: str
    name: str                                    # 知识库名称，如 "首饰设计"
    description: Optional[str] = None             # 描述
    category: Optional[str] = None               # 分类
    employee_id: Optional[str] = None            # 关联的数字员工
    document_count: int = 0                      # 文档数量
    chunk_count: int = 0                         # 分块数量
    status: str = "active"                       # active/inactive
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentModel(BaseModel):
    """Document model."""
    doc_id: str
    filename: str
    kb_id: str
    enhance: int = 1  # 0=不增强，1=增强
    category: Optional[str] = None
    size: int = 0
    format: str  # PDF/DOCX/TXT/MD/HTML/MP4
    chunks_count: int = 0
    vectors_count: int = 0
    status: str = "processing"  # processing/completed/failed
    error_message: Optional[str] = None
    segment_config: Optional[Dict[str, Any]] = None  # Custom segment configuration
    metadata: Dict[str, Any] = Field(default_factory=dict)
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None


class DocumentChunkModel(BaseModel):
    """Document chunk model."""
    chunk_id: str
    doc_id: str
    kb_id: str
    content: str
    chunk_index: int
    summary: Optional[Dict[str, Any]] = None
    vector_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    # MinerU结构化字段（新增）
    page_idx: Optional[int] = None                      # 起始页码
    page_indices: List[int] = Field(default_factory=list)  # 包含的所有页码
    block_types: List[str] = Field(default_factory=list)   # 包含的块类型 ['text', 'title']
    image_references: List[str] = Field(default_factory=list)  # 图片URL列表
    image_captions: List[str] = Field(default_factory=list)    # 图片描述列表
    title_path: List[str] = Field(default_factory=list)  # 标题路径（面包屑）
    structure_level: int = 0            # 在文档结构中的层级


class IntentLogModel(BaseModel):
    """Intent recognition log model."""
    log_id: str
    conversation_id: str
    employee_id: str
    user_query: str
    recognized_intent: str
    confidence: float
    entities: Dict[str, Any] = Field(default_factory=dict)
    method: str  # rule-based/llm-based
    processing_time_ms: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class KBQualityMetricModel(BaseModel):
    """Knowledge base quality metrics model."""
    metric_id: str
    kb_id: str
    kb_name: str
    employee_id: Optional[str] = None
    date: str  # YYYY-MM-DD
    hit_rate: float = 0.0
    recall_rate: float = 0.0
    precision_rate: float = 0.0
    avg_response_time: float = 0.0
    avg_confidence: float = 0.0
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class DocumentTaskModel(BaseModel):
    """Document processing task model."""
    task_id: str
    kb_id: str
    enhance: int
    filename: str
    file_path: str
    category: Optional[str] = None
    status: str = "pending"  # pending/running/completed/failed/cancelled
    doc_id: Optional[str] = None
    total_chunks: int = 0
    processed_chunks: int = 0
    progress: float = 0.0  # 0-100
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StreamChunkModel(BaseModel):
    """Streaming chunk model for OpenAI-style chat completions."""
    chunk_id: str
    conversation_id: Optional[str] = None
    session_id: str
    user_id: str
    employee_id: str
    chat_id: str  # OpenAI-style chat completion ID
    chunk_type: str  # start/role/token/progress/done/error
    chunk_data: Dict[str, Any] = Field(default_factory=dict)  # Full chunk payload
    sequence: int = 0  # Chunk sequence number
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FAQModel(BaseModel):
    """FAQ model for vector and keyword retrieval."""
    faq_id: str  # Format: faq_{employee_id}_{external_faq_id}
    employee_id: str
    external_faq_id: str  # Original FAQ ID from Java platform
    team_id: int
    question_name: str  # Main question
    similar_questions: List[str] = Field(default_factory=list)
    answers: List[str] = Field(default_factory=list)
    is_enable: int = 0  # 0=disabled, 1=enabled
    is_clear: int = 0  # 0=not cleared, 1=cleared
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    update_time: str  # Used for incremental update detection
    create_time: str
    # Vector and keyword indexing metadata
    vector_id: Optional[str] = None  # ChromaDB vector ID
    es_indexed: bool = False  # ElasticSearch indexing status
    combined_text: str = ""  # questionName + similarQuestions (for embedding)
    keywords: List[str] = Field(default_factory=list)  # Extracted keywords
    created_at: datetime = Field(default_factory=datetime.utcnow)
    synced_at: datetime = Field(default_factory=datetime.utcnow)


class DigitalEmployeeConfigModel(BaseModel):
    """Digital employee configuration from external API (Java platform)."""
    employee_id: str  # Converted from external employee.id
    external_employee_id: str  # Original employee.id from Java platform
    team_id: int
    name: str
    position: str
    employee_type: str  # AVATAR, etc.
    tone: str
    language: str
    gender: int  # 0=female, 1=male
    intro: Optional[str] = None
    portrait: Optional[str] = None
    model_image: Optional[str] = None
    digital_code: Optional[str] = None
    onduty_status: int
    # Knowledge configuration
    kb_ids: List[str] = Field(default_factory=list)  # Converted from ragDatasets[].ragDatasetId
    faq_count: int = 0  # Total FAQ count
    # Prologue configuration
    prologue: Optional[str] = None
    is_opening_questions: bool = False
    hot_questions: List[str] = Field(default_factory=list)
    # Chat rules
    is_multimodal: bool = False
    faq_sim_threshold: float = 0.0  # FAQ similarity threshold for direct return
    faq_top_k: int = 1  # FAQ top-K retrieval
    # LLM configuration
    web_search_enabled: bool = False
    is_show_sign: bool = False
    is_my_prompt: bool = False
    my_prompt: Optional[str] = None
    # Role configuration
    persona: Optional[str] = None
    style: Optional[str] = None
    style_desc: Optional[str] = None
    # Status and timestamps
    status: str = "active"
    external_update_time: Optional[str] = None  # From external employee.updateTime
    external_create_time: Optional[str] = None  # From external employee.createTime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    synced_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MinerUImageCaptionModel(BaseModel):
    """MinerU image caption cache model for VLM-generated descriptions."""
    image_url: str  # Original image URL (used as lookup key)
    caption: str  # VLM-generated image description
    context: str = ""  # Surrounding text context when generating caption
    model_used: str = ""  # VLM model used (e.g., qwen-vl-max)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MinerUStructuredModel(BaseModel):
    """MinerU structured document data model."""
    doc_id: str  # Document ID
    json_file_path: str  # Path to original MinerU JSON file
    backend: str = ""  # MinerU backend (vlm)
    version: str = ""  # MinerU version
    total_pages: int = 0
    titles: List[Dict[str, Any]] = Field(default_factory=list)  # All titles with positions
    images: List[Dict[str, Any]] = Field(default_factory=list)  # All images with URLs
    title_hierarchy: List[Dict[str, Any]] = Field(default_factory=list)  # Title hierarchy
    processed_at: datetime = Field(default_factory=datetime.utcnow)


class ThesaurusMajorModel(BaseModel):
    """ MongoDB专业词库 """
    thesaurus_major_id: str  # 专业词库唯一id: thesaurus_major_{employee_id}_{external_thesaurus_major_id}
    employee_id: str  # 数字员工id（mysql库）
    external_thesaurus_major_id: int  # 专业词库id（mysql库）
    thesaurus_name: str  # 专业词库名称
    is_enable: int = 1  # 是否启用：0=不启用，1=启用
    update_time: str  # 更新时间（mysql库）
    thesaurus_major_words: List[str] = Field(default_factory=list)
    # Vector and keyword indexing metadata
    vector_id: Optional[str] = None  # ChromaDB vector ID
    es_indexed: bool = False  # ElasticSearch indexing status
    combined_text: str = ""  # thesaurus_name + thesaurus_major_words (for embedding)
    keywords: List[str] = Field(default_factory=list)  # Extracted keywords
    created_at: datetime = Field(default_factory=datetime.utcnow)  # 创建时间
    synced_at: datetime = Field(default_factory=datetime.utcnow)  # 同步时间
