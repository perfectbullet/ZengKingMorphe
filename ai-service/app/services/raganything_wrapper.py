"""
RAGAnything 包装器 - 提供流式查询接口

集成 RAGAnything 到 ai-service，提供统一的 RAG 查询接口。
"""

import os
from typing import AsyncIterator, Dict, Any
from app.core.logging import get_logger

import contextlib
import io
from typing import List

import aiohttp
import numpy as np
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache
from raganything.utils import (
    validate_required_env_vars,
    get_required_env,
    load_content_list_v2,
    ContentProcessingProgressTracker,
    RetryConfig,
    ProgressMessage,
    get_chinese_query_prompt,
)

logger = get_logger(__name__)
_raganything_instance = None


# =============================================================================
# 配置验证
# =============================================================================
REQUIRED_ENV_VARS = {
    # OpenAI API
    "RAG_Anything_OPENAI_API_BASE": "OpenAI API base URL",
    "OPENAI_API_KEY": "OpenAI API key",
    "RAG_Anything_OPENAI_MODEL": "OpenAI model name",
    "VISION_MODEL": "Vision model name",
    # VLLM Embedding
    "VLLM_EMBED_URL": "VLLM embedding service URL",
    "VLLM_EMBED_MODEL": "VLLM embedding model name",
    "VLLM_EMBED_DIM": "VLLM embedding dimension",
    # Reranker
    "VLLM_RERANK_URL": "VLLM reranker service URL",
    "VLLM_RERANK_MODEL": "VLLM reranker model name",
    # Databases
    "MONGO_URI": "MongoDB connection URI",
    "MONGO_DATABASE": "MongoDB database name",
    "NEO4J_URI": "Neo4j connection URI",
    "NEO4J_USERNAME": "Neo4j username",
    "NEO4J_PASSWORD": "Neo4j password",
    "MILVUS_URI": "Milvus service URI",
    "MILVUS_USER": "Milvus username (default: root)",
    "MILVUS_PASSWORD": "Milvus password",
    "MILVUS_DB_NAME": "Milvus database name",
}


def validate_required_env():
    """验证所有必需的环境变量是否已配置"""
    missing_vars = []
    for var_name, description in REQUIRED_ENV_VARS.items():
        value = os.getenv(var_name)
        if value is None or value.strip() == "":
            missing_vars.append(f"  - {var_name}: {description}")

    if missing_vars:
        error_msg = "❌ 缺少必需的环境变量配置:\n\n" + "\n".join(missing_vars)
        error_msg += f"\n\n请在 .env 文件中配置以上变量后再运行。"
        raise ValueError(error_msg)

    logger.info("✅ 环境变量配置验证通过")


def get_required_env(var_name: str) -> str:
    """获取必需的环境变量，如果不存在则报错"""
    value = os.getenv(var_name)
    if value is None or value.strip() == "":
        raise ValueError(f"缺少必需的环境变量: {var_name}")
    return value.strip()


# =============================================================================
# 模型函数
# =============================================================================
async def llm_model_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[Dict] = None,
    **kwargs,
) -> str:
    """OpenAI兼容API的LLM函数"""
    return await openai_complete_if_cache(
        get_required_env("OLLAMA_MODEL"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=get_required_env("RAG_Anything_OLLAMA_BASE_URL"),
        api_key="no-api-key",
        **kwargs,
    )


async def vision_model_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[Dict] = None,
    image_data: str = None,
    messages: List[Dict] = None,
    **kwargs,
) -> str:
    """
    视觉模型函数，用于图像分析和VLM增强查询

    支持两种调用模式：
    1. messages格式：多模态VLM增强查询（包含文本和图像的混合消息）
    2. image_data格式：图像处理（base64编码的图像数据）
    """
    # 从 kwargs 中移除 image_data 和 messages，避免传递给 openai_complete_if_cache
    kwargs.pop("image_data", None)
    kwargs.pop("messages", None)

    # 抑制 stdout（避免打印 image_data）
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        if messages:
            result = await openai_complete_if_cache(
                get_required_env("VISION_MODEL"),
                "",
                system_prompt=system_prompt,
                messages=messages,
                base_url=get_required_env("RAG_Anything_OPENAI_API_BASE"),
                api_key=get_required_env("OPENAI_API_KEY"),
                **kwargs,
            )
        elif image_data:
            # 将 image_data 转换为标准的 OpenAI messages 格式
            image_messages = [
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                },
            ]
            # 过滤掉 None 值
            image_messages = [m for m in image_messages if m is not None]
            result = await openai_complete_if_cache(
                get_required_env("VISION_MODEL"),
                "",
                system_prompt=None,  # 已在 messages 中
                messages=image_messages,
                base_url=get_required_env("RAG_Anything_OPENAI_API_BASE"),
                api_key=get_required_env("OPENAI_API_KEY"),
                **kwargs,
            )
        else:
            result = await openai_complete_if_cache(
                get_required_env("VISION_MODEL"),
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=get_required_env("RAG_Anything_OPENAI_API_BASE"),
                api_key=get_required_env("OPENAI_API_KEY"),
                **kwargs,
            )

    return result


async def vllm_embedding_func(texts: List[str]) -> np.ndarray:
    """VLLM向量嵌入函数"""
    embed_url = get_required_env("VLLM_EMBED_URL")
    embed_model = get_required_env("VLLM_EMBED_MODEL")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{embed_url}/embeddings",
            json={"input": texts, "model": embed_model},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            result = await response.json()
            return np.array(
                [item["embedding"] for item in result["data"]], dtype=np.float32
            )


async def vllm_reranker_func(
    query: str, documents: List[str], top_k: int = None, **kwargs
) -> List[Dict]:
    """VLLM Reranker函数"""
    rerank_url = get_required_env("VLLM_RERANK_URL")
    rerank_model = get_required_env("VLLM_RERANK_MODEL")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{rerank_url}/rerank",
            json={"model": rerank_model, "query": query, "documents": documents},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            result = await response.json()
            reranked = []
            for item in result.get("results", []):
                idx = item["index"]
                reranked.append(
                    {
                        "doc_id": idx,
                        "index": idx,
                        "relevance_score": item["relevance_score"],
                        "text": item.get("document", {}).get("text", documents[idx]),
                    }
                )
            return reranked[:top_k] if top_k else reranked


def get_embedding_func():
    """创建 EmbeddingFunc"""
    embed_dim = int(get_required_env("VLLM_EMBED_DIM"))
    return EmbeddingFunc(
        embedding_dim=embed_dim, max_token_size=8192, func=vllm_embedding_func
    )


async def get_raganything_instance():
    """获取 RAGAnything 单例实例"""
    global _raganything_instance
    if _raganything_instance is None:
        config = RAGAnythingConfig(
            working_dir=get_required_env("RAG_ANYTHING_WORKING_DIR"),
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            display_content_stats=True,
        )

        # 初始化 RAGAnything（带数据库后端）
        # 隐藏密码显示 MongoDB URI
        mongo_display = get_required_env("MONGO_URI")
        if "@" in mongo_display:
            # 隐藏密码部分
            parts = mongo_display.split("@")
            auth = parts[0].split("://")[-1]
            if ":" in auth:
                username = auth.split(":")[0]
                mongo_display = f"{parts[0].split('://')[0]}//{username}:***@{parts[1]}"

        logger.info("    初始化 RAGAnything（数据库后端）...")
        logger.info("    存储配置:")
        logger.info(f"   文档状态存储 (MongoDB): {mongo_display}")
        logger.info(f"   向量存储 (Milvus): {get_required_env('MILVUS_URI')}")
        logger.info(
            f"   图存储 (Neo4j): http://{get_required_env('NEO4J_URI').replace('bolt://', '').replace(':7687', '')}:7474"
        )
        logger.info(f"   Reranker (VLLM): {get_required_env('VLLM_RERANK_URL')}")
        logger.info(f"   OPENAI_API_BASE (VLLM): {get_required_env('OPENAI_API_BASE')}")
        logger.info(f"   OPENAI_MODEL (VLLM): {get_required_env('OPENAI_MODEL')}")


        milvus_config = {
            "uri": get_required_env("MILVUS_URI"),
            "user": get_required_env("MILVUS_USER"),
            "password": get_required_env("MILVUS_PASSWORD"),
            "db_name": get_required_env("MILVUS_DB_NAME"),
        }

        _raganything_instance = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=get_embedding_func(),
            lightrag_kwargs={
                "kv_storage": "MongoKVStorage",
                "vector_storage": "MilvusVectorDBStorage",
                "doc_status_storage": "MongoDocStatusStorage",
                "graph_storage": "Neo4JStorage",
                "rerank_model_func": vllm_reranker_func,
                "vector_db_storage_cls_kwargs": milvus_config,
                # LightRAG 配置 (注意: 使用 cosine_better_than_threshold 而不是 cosine_threshold)
                "cosine_better_than_threshold": 0.5,  # 向量相似度阈值
                "min_rerank_score": 0.3,  # 过滤 rerank 分数低于 0.2 的 chunks
                # 缓存开关
                "enable_llm_cache": False,
                # 语言配置
                "addon_params": {
                    "language": "Chinese",  # 知识图谱构建和查询的语言
                    "entity_types": [
                        "organization",
                        "person",
                        "location",
                        "event",
                        "concept",
                        "method",
                    ],
                },
            },
        )
        await _raganything_instance._ensure_lightrag_initialized()
        await _raganything_instance.lightrag.initialize_storages()
        logger.info("RAGAnything instance initialized successfully")

    return _raganything_instance


async def get_raganything_stream(
    query: str, mode: str = "hybrid"
) -> AsyncIterator[Dict[str, Any]]:
    """
    流式查询接口

    Args:
        query: 查询文本
        mode: 检索模式 ("hybrid", "local", "global", "naive")

    Yields:
        Dict[str, Any]: 流式数据块，包含以下类型:
            - {"type": "sources_info", "content": {...}}: 召回统计信息
            - {"type": "chunk", "content": str}: 答案文本片段
            - {"type": "sources", "content": {...}}: 完整召回数据
            - {"type": "error", "content": str}: 错误信息
    """
    rag = await get_raganything_instance()
    async for chunk in rag.aquery_stream_with_sources(
        query,
        mode="hybrid",
        system_prompt=get_chinese_query_prompt(),  # 中文系统提示词（无 References）
        top_k=20,  # 召回实体/关系数量
        chunk_top_k=10,  # (默认10) - 召回文档块数量
        enable_rerank=True,  # (默认True) - 是否启用重排序
    ):
        yield chunk


async def reset_raganything_instance():
    """重置 RAGAnything 实例（用于配置更新或错误恢复）"""
    global _raganything_instance
    if _raganything_instance is not None:
        try:
            await _raganything_instance.finalize_storages()
        except Exception as e:
            logger.warning(f"Error finalizing RAGAnything instance: {e}")
    _raganything_instance = None
    logger.info("RAGAnything instance reset")
