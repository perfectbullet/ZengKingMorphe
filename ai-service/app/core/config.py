"""
Configuration settings for the Digital Employee AI Service.
"""

import os

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# Load environment variables before importing settings
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
def _load_env_file():
    """加载项目根目录的 .env 文件"""
    load_dotenv(
        dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        override=True  # 覆盖已存在的环境变量
    )

# 加载环境变量
_load_env_file()


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",  # ⚠️ 允许未定义的字段（不推荐）
    )

    # LLM Configuration
    use_ollama: bool = Field(
        default=False, description="Use Ollama instead of OpenAI-style API (deprecated, use llm_routing_mode)"
    )

    # LLM Routing Mode (推荐使用此配置控制模型选择)
    # local_only - 仅使用本地 Ollama 模型
    # remote_only - 仅使用外部 API 模型
    # hybrid - 根据问题复杂度自动选择模型
    llm_routing_mode: str = Field(
        default="local_only",
        description="LLM routing mode: local_only, remote_only, or hybrid"
    )

    # ===== 统一 LLM 配置（单一数据源） =====
    # 所有 LLM 调用默认使用此配置，切换模型只需改这里
    llm_base_url: str = Field(
        default="http://localhost:11434",
        description="统一 LLM base URL（单一数据源）"
    )
    llm_model: str = Field(
        default="qwen2.5:14b",
        description="统一 LLM 模型名（单一数据源）"
    )
    llm_api_key: str = Field(
        default="",
        description="统一 LLM API Key（单一数据源）"
    )

    # --- 以下为旧配置，保留作为 fallback 兼容，新部署无需设置 ---
    # Ollama Configuration (本地模型配置)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5:7b")
    ollama_grader_model: str = Field(default="qwen2.5:7b")
    ollama_keep_alive_interval: int = Field(
        default=180,
        description="Interval in seconds between Ollama keep-alive requests (0 to disable)"
    )

    # OpenAI-style API Configuration (外部模型配置)
    openai_api_key: str = Field(
        default="", description="OpenAI-style API key (e.g., SiliconFlow)"
    )
    openai_api_base: str = Field(default="https://api.siliconflow.cn/v1")
    openai_model: str = Field(default="deepseek-ai/DeepSeek-V3.1-Terminus")
    openai_grader_model: str = Field(default="deepseek-ai/DeepSeek-V3")
    openai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # Math Model Configuration 已迁移到环境变量（MATH_LLM_BASE_URL / MATH_MODEL_NAME / MATH_LLM_ENABLED），
    # 在 conversation_service.py 的 get_math_streaming_llm() 中直接读取。

    # 混合模式复杂度阈值配置 (hybrid 模式下生效)
    # 复杂度评估使用本地 LLM 快速判断问题复杂度 (0-10分)
    # 0-6分: 使用本地 Ollama (简单到中等复杂)
    # 7-10分: 使用外部 API (高复杂度)
    complexity_threshold: float = Field(
        default=7.0,
        ge=0.0,
        le=10.0,
        description="Complexity score threshold (0-10) above which to use remote LLM"
    )

    # Embedding Configuration
    embedding_type: str = Field(
        default="openai_style",
        description="Embedding type: openai_style, siliconflow, or ollama",
    )
    embedding_model: str = Field(default="bge-large-zh-v1.5-2k:latest")
    embedding_base_url: str = Field(
        default="http://localhost:50009", description="Embedding service base URL"
    )
    embedding_api_url: str = Field(
        default="http://localhost:50009", description="Embedding API URL"
    )
    embedding_api_key: Optional[str] = Field(
        default=None, description="Embedding API key (for SiliconFlow)"
    )

    # SiliconFlow Configuration
    siliconflow_api_key: Optional[str] = Field(
        default=None, description="SiliconFlow API key"
    )
    siliconflow_embedding_api_url: str = Field(
        default="https://api.siliconflow.cn/v1/embeddings",
        description="SiliconFlow Embedding API URL",
    )
    siliconflow_embedding_model: str = Field(
        default="BAAI/bge-large-zh-v1.5", description="SiliconFlow Embedding Model"
    )

    # Web Search Configuration
    tavily_api_key: Optional[str] = Field(default=None)
    web_search_enabled: bool = Field(default=True)
    web_search_timeout: int = Field(default=5, ge=1)
    web_search_max_results: int = Field(default=5, ge=1, le=10)
    web_search_only_for_realtime: bool = Field(default=False)
    realtime_query_enabled: bool = Field(default=True)
    realtime_query_llm_fallback_enabled: bool = Field(default=True)
    realtime_query_search_rewrite_enabled: bool = Field(default=True)
    realtime_traffic_min_web_score: float = Field(default=0.5, ge=0.0, le=1.0)
    # A股复盘关键词，提升行情页命中率
    web_search_market_anchor_hint_zh: str = Field(
        default=" 上证指数收盘 深证成指收盘 创业板指收盘点位 涨跌幅",
        description="Merged into Tavily anchored query text when realtime_category=market",
    )
    # 二次检索语句，空则关闭
    web_search_market_secondary_query_zh: str = Field(
        default="{date_iso} A股上证指数收盘 深证成指收盘 创业板指涨跌 复盘",
        description="Optional second Tavily query for market realtime; empty disables",
    )
    # 使用Tavily高级检索（付费）
    web_search_market_tavily_advanced: bool = Field(
        default=True,
        description="Use Tavily search_depth=advanced when realtime_category=market",
    )

    # MongoDB Configuration
    mongodb_uri: str = Field(default="mongodb://localhost:27017/digital_employee")
    mongodb_db_name: str = Field(default="digital_employee")
    mongodb_max_pool_size: int = Field(default=100, ge=1)
    mongodb_min_pool_size: int = Field(default=10, ge=1)


    # ElasticSearch Configuration
    es_host: str = Field(default="localhost")
    es_port: int = Field(default=9200)
    es_index_prefix: str = Field(default="digital_employee")
    es_username: Optional[str] = Field(default=None)
    es_password: Optional[str] = Field(default=None)

    # Java Platform API Configuration
    java_api_base_url: str = Field(default="http://java-platform:8080")
    java_api_key: Optional[str] = Field(default=None)
    java_api_timeout: int = Field(default=10, ge=1)

    # Service Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=4, ge=1)
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = Field(
        default=None,
        description="Optional log file path for file logging with rotation"
    )
    debug: bool = Field(default=False)

    # Rate Limiting Configuration
    rate_limit_per_minute: int = Field(default=60, ge=1)
    rate_limit_per_session: int = Field(default=10, ge=1)
    max_concurrent_sessions: int = Field(default=100, ge=1)
    rate_limit_per_ip_per_minute: int = Field(
        default=30,
        ge=1,
        description="Rate limit per IP address per minute for polling endpoints"
    )

    # Conversation Configuration
    max_context_turns: int = Field(default=10, ge=1)
    session_timeout_minutes: int = Field(default=30, ge=1)
    relevance_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # 动态上下文记忆：判定本轮问题是否依赖历史对话
    # - True  → 走 LLM 判定，无关历史问题不注入上下文，相关问题保留完整上下文
    # - False → 关闭判定，保持旧行为（始终注入上下文）
    dynamic_context_memory_enabled: bool = Field(
        default=True,
        description="Enable dynamic context memory: drop history when current query is unrelated"
    )

    # ---- 噪声预设话术拦截（"抱歉，我没有听清您的问题"路径）----
    # 背景：分类器把用户输入归为 noise 时，原实现会**完全跳过 LLM**直接返回预设话术，
    # 用户在数字人侧听到这句话极易误判为"麦克风/ASR 故障"。
    # 这一组开关让该拦截可以快速回滚或收紧到只在"高度确信噪声"时才触发。
    #
    # noise_preset_response_enabled
    # - True  → 启用预设话术拦截（命中 noise 且通过下方闸门时直出文案）；
    # - False → **整条路径回滚**，noise 一律走通用 LLM 兜底（紧急回滚开关）。
    noise_preset_response_enabled: bool = Field(
        default=True,
        description="Enable preset response for queries classified as noise. "
                    "Set False to fully roll back the noise→preset pathway and let GENERAL_LLM handle it."
    )
    # noise_preset_min_confidence
    # 仅当 LLM 分类置信度 >= 此等级时才允许判 noise；其余降级为 GENERAL_LLM。
    # qwen2.5:7b 这类小分类器对短/口语化输入误判率较高，默认要求 ``high``
    # 才能触发预设话术，把 medium/low 的噪声判定一律放行给 LLM。
    noise_preset_min_confidence: str = Field(
        default="high",
        description="Minimum classifier confidence to trigger noise preset response: high|medium|low"
    )
    # noise_preset_max_query_length
    # 仅当 query 长度（按 strip 后字符数）<= 此值时，才允许触发噪声预设。
    # 含义：超过这个长度的输入即便分类器判 noise 也按"内容足够丰富"放行，
    # 避免吞掉"那个数列怎么算？""嗯，刚刚那道函数题再讲一遍"这类合法长问。
    noise_preset_max_query_length: int = Field(
        default=12,
        ge=1,
        description="Max query length (chars) eligible for noise preset response; "
                    "longer queries bypass preset and go to GENERAL_LLM."
    )

    # Document Processing Configuration
    chunk_size: int = Field(
        default=256,
        ge=128,
        description="Chunk size in characters (use 256 for 512-token models)",
    )
    chunk_overlap: int = Field(default=50, ge=0)
    enable_chunk_summary: bool = Field(default=True)
    summary_model: str = Field(default="gpt-4o-mini")
    summary_max_tokens: int = Field(default=200, ge=1)
    summary_batch_size: int = Field(default=10, ge=1)

    # Semantic Chunking Configuration
    enable_semantic_chunking: bool = Field(
        default=True,
        description="Enable semantic chunking using embeddings for better context preservation"
    )
    semantic_chunk_similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for semantic chunk boundaries (lower = more chunks)"
    )
    semantic_chunk_min_size: int = Field(
        default=100,
        ge=50,
        description="Minimum chunk size in characters for semantic chunking"
    )
    semantic_chunk_max_size: int = Field(
        default=500,
        ge=100,
        description="Maximum chunk size in characters for semantic chunking"
    )
    semantic_chunk_window_size: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of sentences to consider for similarity calculation"
    )

    # Hierarchical Summarization Configuration
    enable_hierarchical_summary: bool = Field(
        default=True,
        description="Enable hierarchical summarization (chunk -> section -> document levels)"
    )
    summary_chunk_level: bool = Field(
        default=True,
        description="Generate summaries for each chunk"
    )
    summary_section_level: bool = Field(
        default=True,
        description="Generate summaries for sections (groups of related chunks)"
    )
    summary_document_level: bool = Field(
        default=True,
        description="Generate document-level summary"
    )
    summary_section_threshold: int = Field(
        default=5,
        ge=2,
        description="Number of chunks to trigger section-level summary"
    )
    summary_document_threshold: int = Field(
        default=50,
        ge=10,
        description="Number of chunks to trigger document-level summary"
    )

    # MinerU PDF Parsing Configuration
    mineru_api_url: str = Field(
        default="http://192.168.8.233:8000/file_parse",
        description="MinerU API endpoint URL for enhanced PDF parsing"
    )
    mineru_enabled: bool = Field(
        default=False,
        description="Enable MinerU API for PDF parsing (better table/formula extraction)"
    )
    mineru_pages_per_chunk: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Number of pages to process per chunk when using MinerU"
    )
    mineru_timeout: int = Field(
        default=600,
        ge=60,
        description="API request timeout in seconds for MinerU calls"
    )
    mineru_cache_ttl_days: int = Field(
        default=30,
        ge=1,
        description="Cache time-to-live in days for MinerU results"
    )

    # MinerU JSON Structure Parsing Configuration
    mineru_json_enabled: bool = Field(
        default=True,
        description="Enable JSON structure parsing from MinerU for better chunking"
    )
    mineru_structure_aware_chunking: bool = Field(
        default=True,
        description="Use structure-aware chunking for MinerU JSON (by title/section)"
    )
    mineru_chunking_strategy: str = Field(
        default="hybrid",
        description="Chunking strategy: by_title, by_page, or hybrid"
    )
    mineru_include_images: bool = Field(
        default=True,
        description="Include image references in RAG chunks"
    )
    mineru_image_captioning: bool = Field(
        default=False,
        description="Enable VLM-based image captioning for multi-modal RAG"
    )
    mineru_vlm_backend: str = Field(
        default="qwen_vl",
        description="VLM backend for image captioning: openai, qwen_vl, or custom"
    )
    mineru_vlm_api_key: Optional[str] = Field(
        default=None,
        description="VLM API key for image captioning"
    )
    mineru_vlm_base_url: Optional[str] = Field(
        default=None,
        description="VLM API base URL for image captioning"
    )
    mineru_vlm_model: str = Field(
        default="qwen-vl-max",
        description="VLM model name for image captioning"
    )
    mineru_vlm_timeout: int = Field(
        default=30,
        ge=5,
        description="VLM API request timeout in seconds"
    )

    # RAG Optimization Configuration
    query_rewrite_enabled: bool = Field(
        default=False,
        description="Enable query rewriting using LLM for better retrieval"
    )
    rerank_enabled: bool = Field(
        default=True,
        description="Enable document reranking using BGE Reranker"
    )
    reranker_type: str = Field(
        default="bge_api",
        description="Reranker type: bge_api, ollama (legacy), bge, noop, hybrid"
    )
    ollama_reranker_model: str = Field(
        default="qllama/bge-reranker-v2-m3:latest",
        description="Ollama reranker model name"
    )

    # BGE Reranker API Configuration
    bge_reranker_api_url: str = Field(
        default="http://192.168.8.233:8091",
        description="BGE Reranker API base URL"
    )
    bge_reranker_api_key: str = Field(
        default="sk-aaabbbcccdddeeefffggghhhiiijjjkkk",
        description="BGE Reranker API key"
    )
    bge_reranker_model: str = Field(
        default="/model",
        description="BGE Reranker model name (vLLM OpenAI-compatible API uses '/model')"
    )
    reranker_max_text_length: int = Field(
        default=200,
        ge=50,
        le=512,
        description="Maximum text length in characters for reranker input (truncated if exceeded)"
    )
    rerank_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of documents to return after reranking"
    )
    context_compression_enabled: bool = Field(
        default=False,
        description="Enable context compression to reduce token usage"
    )

    # RAGAnything Configuration (替代 llama-rag-sdk)
    raganything_working_dir: str = Field(
        default="./rag_storage_db",
        description="Working directory for RAGAnything storage"
    )
    raganything_enabled: bool = Field(
        default=True,
        description="Enable RAGAnything for RAG queries"
    )

    # Milvus Configuration (向量数据库)
    milvus_uri: str = Field(
        default="http://192.168.8.233:19530",
        description="Milvus service URI"
    )
    milvus_user: str = Field(default="root", description="Milvus username")
    milvus_password: str = Field(default="", description="Milvus password")
    milvus_db_name: str = Field(default="rag_db", description="Milvus database name")

    # Neo4j Configuration (图数据库)
    neo4j_uri: str = Field(
        default="bolt://192.168.8.233:7687",
        description="Neo4j connection URI"
    )
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="", description="Neo4j password")

    # VLLM Embedding Configuration (RAGAnything 使用)
    vllm_embedding_base_url: str = Field(
        default="http://192.168.8.233:8092",
        description="VLLM embedding service base URL"
    )
    vllm_embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="VLLM embedding model name"
    )
    vllm_embedding_dim: int = Field(
        default=1024,
        description="VLLM embedding dimension"
    )

    # JWT Configuration
    jwt_secret_key: str = Field(..., description="JWT secret key for token signing")
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_expiration_minutes: int = Field(
        default=60, ge=1, description="JWT token expiration time in minutes"
    )

    # API Key Configuration
    api_key: str = Field(..., description="Primary API key for authentication")
    api_keys: List[str] = Field(
        default=[], description="List of valid API keys for authentication"
    )

    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"]
    )
    cors_allow_credentials: bool = Field(default=True)

    # Test Files Configuration
    test_files_dir: str = Field(
        default="./test_files",
        description="Directory for test files used in file download endpoints"
    )

    @property
    def mongodb_url(self) -> str:
        """Get MongoDB connection URL."""
        return self.mongodb_uri

    @property
    def es_url(self) -> str:
        """Get ElasticSearch URL."""
        return f"http://{self.es_host}:{self.es_port}"


# Global settings instance
settings = Settings()
