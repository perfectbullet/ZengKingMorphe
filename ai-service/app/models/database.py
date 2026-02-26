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
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class SessionModel(BaseModel):
    """Session model."""
    session_id: str
    user_id: str
    user_name: Optional[str] = None
    head_url: Optional[str] = None
    employee_id: str
    status: str = "active"  # active/ended/timeout
    message_count: int = 0
    context_messages: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
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
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    synced_at: Optional[datetime] = None


class UserProfileModel(BaseModel):
    """User profile model."""
    user_id: str
    nickname: Optional[str] = None
    preferences: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    conversation_count: int = 0
    satisfaction_avg: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


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
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


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
    uploaded_at: datetime = Field(default_factory=datetime.now)
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
    created_at: datetime = Field(default_factory=datetime.now)
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
    created_at: datetime = Field(default_factory=datetime.now)


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
    created_at: datetime = Field(default_factory=datetime.now)
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
    created_at: datetime = Field(default_factory=datetime.now)
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
    timestamp: datetime = Field(default_factory=datetime.now)
    created_at: datetime = Field(default_factory=datetime.now)


class FAQModel(BaseModel):
    """FAQ model for vector and keyword retrieval."""
    faq_id: str  # FAQ问答唯一id: faq_{employee_id}_{external_faq_id}
    employee_id: str  # 数字员工id（mysql库）
    external_faq_id: int  # FAQ问答id（mysql库）
    question_name: str  # 问题名称
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    is_enable: int = 0  # 是否启用：0=不启用，1=启用
    is_clear: int = 0  # 0=not cleared, 1=cleared
    similar_questions: List[str] = Field(default_factory=list)
    answers: List[str] = Field(default_factory=list)
    update_time: str  # 更新时间（mysql库）
    # Vector and keyword indexing metadata
    combined_text: str = ""  # questionName + similarQuestions (for embedding)
    keywords: List[str] = Field(default_factory=list)  # Extracted keywords
    vector_id: Optional[str] = None  # ChromaDB vector ID
    es_indexed: bool = False  # ElasticSearch indexing status
    created_at: datetime = Field(default_factory=datetime.now)
    synced_at: datetime = Field(default_factory=datetime.now)


class EmployeeSettingKnowledge(BaseModel):
    kb_ids: List[str] = Field(default_factory=list, description="知识库id列表")
    faq_ids: List[str] = Field(default_factory=list, description="FAQ问答id列表")
    video_ids: List[str] = Field(default_factory=list, description="视频资源id列表")


class EmployeeSettingPrologue(BaseModel):
    prologue: Optional[str] = Field(None, description="开场白内容")
    is_opening_questions: bool = Field(False, description="是否开启开场热门问题")
    prologue_question_type: int = Field(1, description="热门问题类型: 1-自动推荐 2-FAQ 3-自定义")
    prologue_faqs: Optional[List[str]] = Field(default_factory=list, description="开场热门问题关联FAQ列表")
    hot_questions: Optional[List[str]] = Field(default_factory=list, description="开场热门问题自定义列表")


class EmployeeSettingChatRule(BaseModel):
    is_multimodal: bool = Field(False, description="是否支持多模态")
    fixed_answer: Optional[str] = Field(None, description="关闭图片理解后返回的固定回复话术")
    faq_sim_threshold: Optional[float] = Field(0.0, description="FAQ相似度阈值")
    faq_top_k: Optional[int] = Field(1, description="FAQ最多推荐数量")


class EmployeeSettingUnusualRule(BaseModel):
    exception_reply: Optional[str] = Field(None, description="系统异常时的回复话术")
    not_match_reply_type: Optional[int] = Field(None, description="未匹配时的回复类型：0：固定话术，1：模型闲聊回复")
    fixed_replys: Optional[List[str]] = Field(default_factory=list, description="not_match_reply_type=0时的固定回复列表")
    is_web_search: bool = Field(False, description="not_match_reply_type=1时的模型闲聊回复：是否联网搜索")
    is_show_sign: bool = Field(False, description="not_match_reply_type=1时的模型闲聊回复：是否显示标识")
    is_my_prompt: bool = Field(False, description="not_match_reply_type=1时的模型闲聊回复：是否启用自定义提示词")
    my_prompt: Optional[str] = Field(None, description="not_match_reply_type=1时的模型闲聊回复：自定义提示词内容")


class ThesaurusWord(BaseModel):
    word_id: int = Field(..., description="词条id")
    word_name: str = Field(..., description="词条名称")
    similar_words: List[str] = Field(default_factory=list, description="相似词条名称列表")


class EmployeeSettingSafeRule(BaseModel):
    is_reject_answer: bool = Field(False, description="命中敏感词时是否拒绝回答")
    reject_answer: Optional[str] = Field(None, description="拒绝回答时的回复内容")
    sensitive_ids: List[str] = Field(default_factory=list, description="敏感词库的词条id列表：sensitive_41_67_118")


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
    major_ids: List[str] = Field(default_factory=list, description="专业词库的词条id列表：major_41_67_118")


class DigitalEmployeeConfigModel(BaseModel):
    """ MongoDB数字员工 """
    # 基本信息
    employee_id: str = Field(..., description="数字员工id")
    external_employee_id: str = Field(..., description="数字员工id（mysql库）")
    team_id: int = Field(..., description="团队id")
    name: str = Field(..., description="员工姓名")
    position: str = Field(..., description="职位")
    employee_type: str = Field(..., description="数字员工类型: AVATAR=头像, HALF=半身, FULL=全身")
    tone: str = Field(..., description="音色")
    language: str = Field(..., description="语言")
    gender: int = Field(..., description="数字员工性别: 1-男 2-女, 0-未知")
    intro: Optional[str] = Field(None, description="数字员工简介")
    portrait: Optional[str] = Field(None, description="头像地址")
    model_image: Optional[str] = Field(None, description="模型图片或模型地址")
    digital_code: Optional[str] = Field(None, description="数字人在AI平台的唯一编码")
    onduty_status: int = Field(None, description="值班状态: 0-休息中，1-值班中")
    create_time: str = Field(None, description="创建时间（mysql库）")
    update_time: str = Field(None, description="更新时间（mysql库）")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    synced_at: datetime = Field(default_factory=datetime.now, description="同步时间")
    knowledge: EmployeeSettingKnowledge = Field(None, description="对话准备--知识库配置")
    prologue: EmployeeSettingPrologue = Field(None, description="对话开始--开场白、开场热门问题")
    chat_rule: EmployeeSettingChatRule = Field(None, description="对话中--对话规则")
    unusual_rule: EmployeeSettingUnusualRule = Field(None, description="对话中--异常或未匹配规则")
    safe_rule: EmployeeSettingSafeRule = Field(None, description="对话中--安全规则配置")
    role: EmployeeSettingRole = Field(None, description="角色--人设")
    plugins: List[EmployeeSettingPlugin] = Field(default_factory=list, description="高级设置--插件")
    thesaurus_major: EmployeeSettingThesaurusMajor = Field(None, description="高级设置--专业词库配置")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DigitalEmployeeConfigSettingModel(BaseModel):
    """ MongoDB数字员工对话设定 """
    employee_id: str = Field(..., description="数字员工id")
    update_time: str = Field(None, description="更新时间（mysql库）")
    knowledge: EmployeeSettingKnowledge = Field(None, description="对话准备--知识库配置")
    prologue: EmployeeSettingPrologue = Field(None, description="对话开始--开场白、开场热门问题")
    chat_rule: EmployeeSettingChatRule = Field(None, description="对话中--对话规则")
    unusual_rule: EmployeeSettingUnusualRule = Field(None, description="对话中--异常或未匹配规则")
    safe_rule: EmployeeSettingSafeRule = Field(None, description="对话中--安全规则配置")
    role: EmployeeSettingRole = Field(..., description="角色--人设")
    plugins: List[EmployeeSettingPlugin] = Field(default_factory=list, description="高级设置--插件")
    thesaurus_major: EmployeeSettingThesaurusMajor = Field(None, description="高级设置--专业词库配置")
    updated_at: datetime = Field(default_factory=datetime.now)


class MinerUImageCaptionModel(BaseModel):
    """MinerU image caption cache model for VLM-generated descriptions."""
    image_url: str  # Original image URL (used as lookup key)
    caption: str  # VLM-generated image description
    context: str = ""  # Surrounding text context when generating caption
    model_used: str = ""  # VLM model used (e.g., qwen-vl-max)
    created_at: datetime = Field(default_factory=datetime.now)


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
    processed_at: datetime = Field(default_factory=datetime.now)


class ThesaurusMajorModel(BaseModel):
    """ MongoDB专业词库 """
    thesaurus_id: str  # 专业词库唯一id: major_{employee_id}_{external_thesaurus_id}_{external_word_id}
    employee_id: str  # 数字员工id（mysql库）
    external_thesaurus_id: int  # 专业词库id（mysql库）
    external_word_id: int  # 专业词条id（mysql库）
    thesaurus_name: str  # 专业词库名称
    is_enable: int = 1  # 是否启用：0=不启用，1=启用
    update_time: str  # 更新时间（mysql库）
    word_name: str  # 词条名称
    similar_words: List[str] = Field(default_factory=list)  # 相似词条名称列表
    # Vector and keyword indexing metadata
    combined_text: str = ""  # thesaurus_name + word_name + similar_words (for embedding)
    keywords: List[str] = Field(default_factory=list)  # Extracted keywords
    vector_id: Optional[str] = None  # ChromaDB vector ID
    es_indexed: bool = False  # ElasticSearch indexing status
    created_at: datetime = Field(default_factory=datetime.now)  # 创建时间
    synced_at: datetime = Field(default_factory=datetime.now)  # 同步时间


class ThesaurusSensitiveModel(BaseModel):
    """ MongoDB敏感词库 """
    thesaurus_id: str  # 敏感词库唯一id: major_{employee_id}_{external_thesaurus_id}_{external_word_id}
    employee_id: str  # 数字员工id（mysql库）
    external_thesaurus_id: int  # 敏感词库id（mysql库）
    external_word_id: int  # 敏感词条id（mysql库）
    thesaurus_name: str  # 敏感词库名称
    is_enable: int = 1  # 是否启用：0=不启用，1=启用
    update_time: str  # 更新时间（mysql库）
    word_name: str  # 词条名称
    # Vector and keyword indexing metadata
    combined_text: str = ""  # thesaurus_name + word_name (for embedding)
    keywords: List[str] = Field(default_factory=list)  # Extracted keywords
    vector_id: Optional[str] = None  # ChromaDB vector ID
    es_indexed: bool = False  # ElasticSearch indexing status
    created_at: datetime = Field(default_factory=datetime.now)  # 创建时间
    synced_at: datetime = Field(default_factory=datetime.now)  # 同步时间
