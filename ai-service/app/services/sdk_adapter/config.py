"""
SDK 配置适配器

将 ai-service settings 映射到 llama-rag-sdk 环境变量
"""
import os
from app.core.config import settings


def setup_sdk_env() -> None:
    """
    将 ai-service settings 映射到 SDK 环境变量

    llama-rag-sdk 使用 Pydantic Settings 读取环境变量，
    我们需要在导入 SDK 之前设置这些变量。
    """
    # Embedding 配置
    if not os.environ.get("VLLM_EMBEDDING_BASE_URL"):
        os.environ["VLLM_EMBEDDING_BASE_URL"] = settings.embedding_base_url or settings.embedding_api_url
    if not os.environ.get("VLLM_EMBEDDING_MODEL"):
        os.environ["VLLM_EMBEDDING_MODEL"] = settings.embedding_model
    os.environ["VLLM_API_KEY"] = settings.embedding_api_key or "not-needed"

    # ChromaDB 配置
    if not os.environ.get("CHROMA_HOST"):
        os.environ["CHROMA_HOST"] = settings.chroma_host
    if not os.environ.get("CHROMA_PORT"):
        os.environ["CHROMA_PORT"] = str(settings.chroma_port)
    os.environ["CHROMA_USE_REMOTE"] = "true"
    os.environ["CHROMA_COLLECTION_NAME"] = "rag_documents"
    if settings.chroma_persist_dir:
        os.environ["CHROMA_PERSIST_DIR"] = settings.chroma_persist_dir

    # MongoDB 配置
    os.environ["MONGODB_URI"] = settings.mongodb_uri
    os.environ["MONGODB_DB_NAME"] = settings.mongodb_db_name
    os.environ["DOCSTORE_COLLECTION"] = "llamarag_docstore"

    # Reranker 配置
    os.environ["RERANK_BASE_URL"] = settings.bge_reranker_api_url or "http://localhost:8091"
    os.environ["RERANK_MODEL"] = settings.bge_reranker_model or "/model"

    # 索引配置
    os.environ["CHUNK_SIZE"] = "256"
    os.environ["CHUNK_OVERLAP"] = "50"
    os.environ["TOP_K"] = "5"
    os.environ["SIMILARITY_THRESHOLD"] = str(settings.relevance_threshold or 0.6)

    # 功能开关
    enable_image_desc = getattr(settings, 'enable_image_description', False)
    os.environ["ENABLE_IMAGE_DESCRIPTION"] = str(enable_image_desc).lower()
    os.environ["ENABLE_SUMMARIZATION"] = "true"
    os.environ["QUERY_EXPANSION_ENABLED"] = "false"

    # 图片描述配置（可选）
    qwen_vl_model = getattr(settings, 'qwen_vl_model', None)
    if qwen_vl_model:
        os.environ["QWEN_VL_MODEL"] = qwen_vl_model
    qwen_vl_base_url = getattr(settings, 'qwen_vl_base_url', None)
    if qwen_vl_base_url:
        os.environ["QWEN_VL_BASE_URL"] = qwen_vl_base_url

    # MinerU 配置
    os.environ["MINERU_API_URL"] = settings.mineru_api_url


# 模块导入时自动设置环境变量
setup_sdk_env()
