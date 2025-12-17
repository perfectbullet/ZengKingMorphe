"""
Configuration settings for the Digital Employee AI Service.
"""
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # LLM Configuration
    use_ollama: bool = Field(default=False, description="Use Ollama instead of OpenAI-style API")
    
    # Ollama Configuration (used when use_ollama=True)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5:7b")
    ollama_grader_model: str = Field(default="qwen2.5:7b")
    
    # OpenAI-style API Configuration (used when use_ollama=False)
    openai_api_key: str = Field(default="", description="OpenAI-style API key (e.g., SiliconFlow)")
    openai_api_base: str = Field(default="https://api.siliconflow.cn/v1")
    openai_model: str = Field(default="deepseek-ai/DeepSeek-V3.1-Terminus")
    openai_grader_model: str = Field(default="deepseek-ai/DeepSeek-V3")
    openai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=2000, ge=1)
    
    # Embedding Configuration
    embedding_type: str = Field(default="openai_style", description="Embedding type: openai_style or siliconflow")
    embedding_model: str = Field(default="BAAI/bge-large-zh-v1.5")
    embedding_base_url: str = Field(..., description="Embedding service base URL")
    embedding_api_url: str = Field(..., description="Embedding API URL")
    embedding_api_key: Optional[str] = Field(default=None, description="Embedding API key (for SiliconFlow)")
    siliconflow_api_key: Optional[str] = Field(default=None, description="SiliconFlow API key")
    
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
    chunk_size: int = Field(default=512, ge=128)
    chunk_overlap: int = Field(default=50, ge=0)
    enable_chunk_summary: bool = Field(default=True)
    summary_model: str = Field(default="gpt-4o-mini")
    summary_max_tokens: int = Field(default=200, ge=1)
    summary_batch_size: int = Field(default=10, ge=1)
    
    # JWT Configuration
    jwt_secret_key: str = Field(..., description="JWT secret key for token signing")
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_expiration_minutes: int = Field(default=60, ge=1, description="JWT token expiration time in minutes")
    
    # API Key Configuration
    api_key: str = Field(..., description="Primary API key for authentication")
    api_keys: List[str] = Field(
        default=[],
        description="List of valid API keys for authentication"
    )
    
    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"]
    )
    cors_allow_credentials: bool = Field(default=True)
    
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
