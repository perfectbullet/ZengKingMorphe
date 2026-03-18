"""
配置管理模块

使用 Pydantic Settings 管理所有环境变量配置
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


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
        default="http://192.168.8.231:8000",
        description="MinerU API 服务器地址"
    )
    mineru_mcp_url: str = Field(
        default="http://192.168.8.231:8000",
        description="MinerU API 服务地址（向后兼容）"
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

    # ========== vLLM Embedding 配置 ==========
    vllm_embedding_base_url: str = Field(
        default="http://192.168.8.233:8092",
        description="vLLM Embedding 服务地址"
    )
    vllm_embedding_api_base: str = Field(
        default="http://192.168.8.233:8092/v1",
        description="vLLM Embedding API Base 地址"
    )
    vllm_embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="Embedding 模型名称"
    )
    vllm_api_key: str = Field(
        default="not-needed",
        description="vLLM API Key（不需要真实 key）"
    )

    # ========== vLLM Chat 配置 ==========
    vllm_chat_base_url: str = Field(
        default="http://192.168.8.233:8092",
        description="vLLM Chat 服务地址"
    )
    vllm_chat_api_base: str = Field(
        default="http://192.168.8.233:8092/v1",
        description="vLLM Chat API Base 地址"
    )
    vllm_chat_model: str = Field(
        default="deepseek-coder:instruct",
        description="问答模型名称"
    )

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
        default="192.168.8.233",
        description="ChromaDB 主机地址"
    )
    chroma_port: int = Field(
        default=8200,
        description="ChromaDB 端口"
    )
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="ChromaDB 持久化目录"
    )
    chroma_collection_name: str = Field(
        default="documents",
        description="ChromaDB 集合名称"
    )
    chroma_use_remote: bool = Field(
        default=True,
        description="是否使用远程 ChromaDB"
    )

    # ========== 索引配置 ==========
    chunk_size: int = Field(
        default=512,
        description="文本分块大小"
    )
    chunk_overlap: int = Field(
        default=50,
        description="文本分块重叠大小"
    )
    top_k: int = Field(
        default=5,
        description="检索返回的文档数量"
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="相似度阈值"
    )

    # ========== 检索配置 ==========
    use_hybrid_retrieval: bool = Field(
        default=False,
        description="是否使用混合检索（向量 + 关键词）"
    )
    use_rerank: bool = Field(
        default=True,
        description="是否使用重排序"
    )
    rerank_model: str = Field(
        default="bge-reranker-v2-m3",
        description="重排序模型名称"
    )

    # ========== BGE Reranker 配置 ==========
    rerank_base_url: str = Field(
        default="http://192.168.8.233:8091",
        description="BGE Reranker 服务地址"
    )
    rerank_api_base: str = Field(
        default="http://192.168.8.233:8091/v1",
        description="BGE Reranker API Base 地址"
    )
    rerank_timeout: int = Field(
        default=120,
        description="Rerank 请求超时时间（秒）"
    )
    rerank_candidate_multiplier: int = Field(
        default=4,
        description="候选数量倍数（候选数量 = top_k * multiplier）"
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


# 全局配置实例
settings = Settings()
