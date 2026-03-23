"""
配置管理模块

使用 Pydantic Settings 管理所有环境变量配置
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """RAG SDK 配置类"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # ========== MinerU API 配置 ==========
    mineru_api_url: str = Field(
        ...,
        description="MinerU API 服务器地址（需在环境变量中配置）"
    )
    mineru_output_dir: str = Field(
        default="./data/output",
        description="MinerU 输出目录"
    )
    mineru_backend: str = Field(
        default="pipeline",
        description="MinerU 解析后端 (pipeline/vlm-auto-engine/hybrid-auto-engine)"
    )
    mineru_parse_method: str = Field(
        default="auto",
        description="MinerU 解析方法 (auto/txt/ocr)"
    )
    mineru_lang: str = Field(
        default="ch",
        description="MinerU 语言代码 (ch=中文, en=英文)"
    )
    mineru_timeout: int = Field(
        default=1800,
        description="MinerU API 请求超时时间（秒）"
    )
    mineru_formula_enable: bool = Field(
        default=True,
        description="是否启用公式解析"
    )
    mineru_table_enable: bool = Field(
        default=True,
        description="是否启用表格解析"
    )
    mineru_structure_aware_chunking: bool = Field(
        default=True,
        description="是否启用 MinerU 结构感知分块（按章节分块，保持语义完整性）"
    )

    # ========== vLLM Embedding 配置（保留向后兼容）==========
    vllm_embedding_base_url: str = Field(
        ...,
        description="vLLM Embedding 服务地址（需在环境变量中配置）"
    )
    vllm_embedding_model: str = Field(
        ...,
        description="Embedding 模型名称（需在环境变量中配置）"
    )
    vllm_api_key: str = Field(
        default="not-needed",
        description="vLLM API Key（不需要真实 key）"
    )

    # ========== 多模态模型配置（Qwen2-VL）==========

    # ========== 多模态模型配置（Qwen2-VL）==========
    qwen_vl_model: str = Field(
        default="qwen2-vl:latest",
        description="Qwen2-VL 模型名称"
    )
    qwen_vl_base_url: str = Field(
        default="http://192.168.8.233:11434",
        description="Qwen2-VL 服务地址（使用 Ollama）"
    )
    enable_image_description: bool = Field(
        default=True,
        description="是否启用图片描述生成"
    )

    # ========== ChromaDB 配置 ==========
    chroma_host: str = Field(
        ...,
        description="ChromaDB 主机地址（需在环境变量中配置）"
    )
    chroma_port: int = Field(
        ...,
        description="ChromaDB 端口（需在环境变量中配置）"
    )
    chroma_persist_dir: str = Field(
        ...,
        description="ChromaDB 持久化目录（需在环境变量中配置）"
    )
    chroma_collection_name: str = Field(
        ...,
        description="ChromaDB 集合名称（需在环境变量中配置）"
    )
    chroma_use_remote: bool = Field(
        ...,
        description="是否使用远程 ChromaDB（需在环境变量中配置）"
    )

    # ========== 索引配置 ==========
    chunk_size: int = Field(
        ...,
        description="文本分块大小（需在环境变量中配置）"
    )
    chunk_overlap: int = Field(
        ...,
        description="文本分块重叠大小（需在环境变量中配置）"
    )
    top_k: int = Field(
        ...,
        description="检索返回的文档数量（需在环境变量中配置）"
    )
    similarity_threshold: float = Field(
        ...,
        description="相似度阈值（需在环境变量中配置）"
    )

    # ========== 目录过滤配置 ==========
    enable_toc_filter: bool = Field(
        default=True,
        description="是否启用目录内容过滤（过滤文档开头的章节目录）"
    )
    max_toc_pages: int = Field(
        default=4,
        description="目录内容最大页码范围（前N页，超过此页码的内容不会被识别为目录）"
    )
    toc_min_page_numbers: int = Field(
        default=3,
        description="判断为目录的最小页码数量（内容中包含的页码引用数量）"
    )
    toc_short_line_ratio: float = Field(
        default=0.7,
        description="目录短行占比阈值（短行占比超过此值且在前N页，会被识别为目录）"
    )
    toc_short_line_length: int = Field(
        default=50,
        description="短行最大长度（用于计算短行占比）"
    )

    # ========== DocStore 配置 ==========
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017/llamarag",
        description="MongoDB 连接 URI"
    )
    mongodb_db_name: str = Field(
        default="llamarag",
        description="MongoDB 数据库名称"
    )
    docstore_collection: str = Field(
        default="docstore",
        description="DocStore 集合名称"
    )

    # ========== 上下文扩展配置 ==========
    context_expansion_enabled: bool = Field(
        default=True,
        description="是否启用上下文扩展"
    )
    context_expansion_window: int = Field(
        default=1,
        description="上下文扩展窗口大小（前后各获取几个节点）"
    )
    context_expansion_include_parent: bool = Field(
        default=False,
        description="是否包含父节点摘要"
    )

    # ========== 文档摘要配置 ==========
    enable_summarization: bool = Field(
        default=True,
        description="是否启用文档摘要生成"
    )
    summarization_min_length: int = Field(
        default=512,
        description="chunk 最小长度，低于此长度不生成摘要"
    )
    summarization_concurrent: int = Field(
        default=5,
        description="批量生成摘要时的并发数"
    )

    # ========== 嵌入向量配置常量 ==========
    # BGE Reranker 分数归一化常量
    rerank_score_min: float = Field(
        default=-10.0,
        description="BGE Reranker 最小分数（用于归一化）"
    )
    rerank_score_max: float = Field(
        default=10.0,
        description="BGE Reranker 最大分数（用于归一化）"
    )

    # ========== 检索配置（包含 Reranker 配置）==========
    rerank_model: str = Field(
        ...,
        description="重排序模型名称（需在环境变量中配置）"
    )
    rerank_base_url: str = Field(
        ...,
        description="BGE Reranker 服务地址（需在环境变量中配置）"
    )
    rerank_timeout: int = Field(
        default=120,
        description="Rerank 请求超时时间（秒）"
    )
    rerank_candidate_multiplier: int = Field(
        default=4,
        description="候选数量倍数（候选数量 = top_k * multiplier）"
    )

    # ========== 查询扩展配置 ==========
    query_expansion_enabled: bool = Field(
        default=False,
        description="是否启用查询扩展（需要 LLM 服务）"
    )
    query_expansion_max_expansions: int = Field(
        default=3,
        description="最大扩展查询数量"
    )

    # ========== LLM 配置（用于查询扩展）==========
    llm_provider: str = Field(
        default="ollama",
        description="LLM 提供商 (ollama/openai_compatible/siliconflow)"
    )
    llm_base_url: str = Field(
        default="http://192.168.8.233:11434",
        description="LLM 服务地址"
    )
    llm_model: str = Field(
        default="qwen2.5:14b",
        description="LLM 模型名称"
    )
    llm_api_key: str = Field(
        default="not-needed",
        description="LLM API Key"
    )
    llm_timeout: int = Field(
        default=60,
        description="LLM 请求超时时间（秒）"
    )

    # ========== 日志配置 ==========
    log_level: str = Field(
        default="INFO",
        description="日志级别"
    )
    log_file: str = Field(
        default="./logs/rag_sdk.log",
        description="日志文件路径"
    )

    @property
    def chroma_client_settings(self) -> dict:
        """获取 ChromaDB 客户端配置"""
        if self.chroma_use_remote:
            return {
                "host": self.chroma_host,
                "port": self.chroma_port,
            }
        else:
            return {
                "persist_directory": self.chroma_persist_dir,
            }

    @property
    def mineru_output_path(self) -> Path:
        """获取 MinerU 输出目录路径对象"""
        return Path(self.mineru_output_dir)

    @property
    def log_file_path(self) -> Path:
        """获取日志文件路径对象"""
        return Path(self.log_file).parent

    def ensure_directories(self) -> None:
        """确保必要的目录存在"""
        self.mineru_output_path.mkdir(parents=True, exist_ok=True)
        self.log_file_path.mkdir(parents=True, exist_ok=True)


# 延迟初始化 settings 实例
_settings: Settings | None = None


def get_settings() -> Settings:
    """获取全局配置实例（延迟初始化）"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# 向后兼容：访问 settings 属性时自动初始化
class _SettingsProxy:
    """Settings 代理类，支持延迟初始化"""

    def __getattr__(self, name: str):
        return getattr(get_settings(), name)

    def __setattr__(self, name: str, value):
        if name == "_settings":
            super().__setattr__(name, value)
        else:
            setattr(get_settings(), name, value)


settings = _SettingsProxy()
