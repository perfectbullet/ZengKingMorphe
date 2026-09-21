"""应用的集中配置。

专项功能（数学模型、LightRAG、概念检索等）直接读取各自的环境变量；本模块只
维护被通用运行链路共享的配置。服务始终从 ``ai-service/.env`` 加载一次配置。
"""

from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_FILE, override=True)


class Settings(BaseSettings):
    """运行时共享配置。"""

    # 已由上方 load_dotenv 显式加载，避免依赖进程工作目录再次读取 .env。
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    # 对话模型与路由
    llm_routing_mode: str = Field(default="local_only")
    llm_model: str = Field(default="qwen2.5:14b")
    openai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    complexity_threshold: float = Field(default=7.0, ge=0.0, le=10.0)

    # 向量模型
    embedding_model: str = Field(default="bge-large-zh-v1.5-2k:latest")
    embedding_base_url: str = Field(default="http://localhost:50009")
    embedding_api_key: Optional[str] = Field(default=None)

    # 联网检索
    tavily_api_key: Optional[str] = Field(default=None)
    web_search_enabled: bool = Field(default=True)
    web_search_max_results: int = Field(default=5, ge=1, le=10)
    realtime_query_llm_fallback_enabled: bool = Field(default=True)
    realtime_query_search_rewrite_enabled: bool = Field(default=True)
    realtime_traffic_min_web_score: float = Field(default=0.5, ge=0.0, le=1.0)
    web_search_market_anchor_hint_zh: str = Field(
        default=" 上证指数收盘 深证成指收盘 创业板指收盘点位 涨跌幅"
    )
    web_search_market_secondary_query_zh: str = Field(
        default="{date_iso} A股上证指数收盘 深证成指收盘 创业板指涨跌 复盘"
    )
    web_search_market_tavily_advanced: bool = Field(default=True)

    # MongoDB
    mongodb_uri: str = Field(default="mongodb://localhost:27017/digital_employee")
    mongodb_db_name: str = Field(default="digital_employee")
    mongodb_max_pool_size: int = Field(default=100, ge=1)
    mongodb_min_pool_size: int = Field(default=10, ge=1)

    # API 服务与日志
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=4, ge=1)
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = Field(default=None)
    debug: bool = Field(default=False)

    # 限流与会话质量指标
    rate_limit_per_minute: int = Field(default=60, ge=1)
    rate_limit_per_session: int = Field(default=10, ge=1)
    rate_limit_per_ip_per_minute: int = Field(default=30, ge=1)
    relevance_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # 对话路由
    dynamic_context_memory_enabled: bool = Field(default=True)
    noise_preset_response_enabled: bool = Field(default=True)
    noise_preset_min_confidence: str = Field(default="high")
    noise_preset_max_query_length: int = Field(default=12, ge=1)

    # API 鉴权与跨域
    api_key: str = Field(..., min_length=1)
    api_keys: List[str] = Field(default_factory=list)
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"]
    )
    cors_allow_credentials: bool = Field(default=True)

    # 知识库测试文件目录
    test_files_dir: str = Field(default="./test_files")

    @model_validator(mode="after")
    def include_primary_api_key(self) -> "Settings":
        """兼容单个 API_KEY，同时允许 API_KEYS 扩展多个密钥。"""
        self.api_keys = list(dict.fromkeys([self.api_key, *self.api_keys]))
        return self

    @property
    def mongodb_url(self) -> str:
        """兼容数据库客户端的连接字符串属性。"""
        return self.mongodb_uri


settings = Settings()
