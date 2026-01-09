"""
Configuration settings for the Digital Employee AI Service.
"""

import os
import sys
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# Load environment variables before importing settings
from dotenv import load_dotenv

# 智能环境变量加载：根据平台和运行环境选择合适的 .env 文件
def _load_env_file():
    """
    根据运行环境自动加载对应的环境变量文件。

    优先级：
    1. 如果显式指定了 ENV_FILE 环境变量，使用该文件
    2. Windows 本地开发：尝试加载 .env-win
    3. Docker/容器环境：使用 .env
    4. 其他平台：使用 .env
    """
    # 检查是否显式指定了环境文件
    env_file = os.getenv("ENV_FILE")
    if env_file and os.path.exists(env_file):
        load_dotenv(env_file, override=True)
        print(f"[OK] Loaded environment from: {env_file}")
        return

    # Windows 平台特殊处理
    if sys.platform == "win32":
        # 优先尝试 .env-win (Windows本地开发配置)
        # 从 app/core/config.py 向上两级到 ai-service 目录
        env_win_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env-win")
        env_win_path = os.path.abspath(env_win_path)
        if os.path.exists(env_win_path):
            load_dotenv(env_win_path, override=True)
            print(f"[OK] Windows detected: Loaded environment from .env-win")
            return

    # 默认加载 .env (Docker/容器环境)
    load_dotenv()
    print(f"[OK] Loaded default .env file")

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
        default=False, description="Use Ollama instead of OpenAI-style API"
    )

    # Ollama Configuration (used when use_ollama=True)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5:7b")
    ollama_grader_model: str = Field(default="qwen2.5:7b")
    ollama_keep_alive_interval: int = Field(
        default=180,
        description="Interval in seconds between Ollama keep-alive requests (0 to disable)"
    )

    # OpenAI-style API Configuration (used when use_ollama=False)
    openai_api_key: str = Field(
        default="", description="OpenAI-style API key (e.g., SiliconFlow)"
    )
    openai_api_base: str = Field(default="https://api.siliconflow.cn/v1")
    openai_model: str = Field(default="deepseek-ai/DeepSeek-V3.1-Terminus")
    openai_grader_model: str = Field(default="deepseek-ai/DeepSeek-V3")
    openai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=2000, ge=1)

    # Embedding Configuration
    embedding_type: str = Field(
        default="openai_style",
        description="Embedding type: openai_style, siliconflow, or ollama",
    )
    embedding_model: str = Field(default="BAAI/bge-large-zh-v1.5")
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

    # Ollama Embedding Configuration
    embedding_ollama_model: str = Field(
        default="smartcreation/bge-large-zh-v1.5:latest",
        description="Ollama embedding model name",
    )

    # Web Search Configuration
    tavily_api_key: Optional[str] = Field(default=None)
    web_search_enabled: bool = Field(default=True)
    web_search_timeout: int = Field(default=5, ge=1)
    web_search_max_results: int = Field(default=5, ge=1, le=10)
    web_search_only_for_realtime: bool = Field(default=False)
    realtime_query_enabled: bool = Field(default=True)
    realtime_query_llm_fallback_enabled: bool = Field(default=False)

    # MongoDB Configuration
    mongodb_uri: str = Field(default="mongodb://localhost:27017/digital_employee")
    mongodb_db_name: str = Field(default="digital_employee")
    mongodb_max_pool_size: int = Field(default=100, ge=1)
    mongodb_min_pool_size: int = Field(default=10, ge=1)

    # Chroma Configuration
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8001)
    chroma_persist_dir: str = Field(default="./chroma_db")

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
    debug: bool = Field(default=False)

    # Rate Limiting Configuration
    rate_limit_per_minute: int = Field(default=60, ge=1)
    rate_limit_per_session: int = Field(default=10, ge=1)
    max_concurrent_sessions: int = Field(default=100, ge=1)

    # Conversation Configuration
    max_context_turns: int = Field(default=10, ge=1)
    session_timeout_minutes: int = Field(default=30, ge=1)
    relevance_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

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

    # MinerU PDF Parsing Configuration
    mineru_api_url: str = Field(
        default="http://192.168.8.231:8000/file_parse",
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

    # RAG Optimization Configuration
    query_rewrite_enabled: bool = Field(
        default=False,
        description="Enable query rewriting using LLM for better retrieval"
    )
    rerank_enabled: bool = Field(
        default=False,
        description="Enable document reranking using Grader LLM"
    )
    context_compression_enabled: bool = Field(
        default=False,
        description="Enable context compression to reduce token usage"
    )
    answer_verification_enabled: bool = Field(
        default=False,
        description="Enable answer consistency checking with source documents"
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
