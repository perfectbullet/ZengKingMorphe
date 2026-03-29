"""
RAGAnything 包装器 - 提供流式查询接口

集成 RAGAnything 到 ai-service，提供统一的 RAG 查询接口。
"""
from typing import AsyncIterator, Dict, Any
from app.core.logging import get_logger

logger = get_logger(__name__)
_raganything_instance = None


async def _llm_model_func(prompt: str, system_prompt: str = None, **kwargs) -> str:
    """OpenAI兼容API的LLM函数"""
    from lightrag.llm.openai import openai_complete_if_cache
    from app.core.config import settings
    return await openai_complete_if_cache(
        settings.openai_model, prompt, system_prompt=system_prompt,
        base_url=settings.openai_api_base, api_key=settings.siliconflow_api_key, **kwargs
    )


async def _embedding_func(texts):
    """VLLM向量嵌入函数"""
    import aiohttp
    import numpy as np
    from app.core.config import settings

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{settings.vllm_embedding_base_url}/embeddings",
            json={"input": texts, "model": settings.vllm_embedding_model},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            result = await response.json()
            return np.array([item["embedding"] for item in result["data"]], dtype=np.float32)


async def get_raganything_instance():
    """获取 RAGAnything 单例实例"""
    global _raganything_instance
    if _raganything_instance is None:
        from raganything import RAGAnything, RAGAnythingConfig
        from lightrag.utils import EmbeddingFunc
        from app.core.config import settings

        config = RAGAnythingConfig(working_dir=settings.raganything_working_dir)

        # 构建存储配置
        lightrag_kwargs = {
            "kv_storage": "MongoKVStorage",
            "vector_storage": "MilvusVectorDBStorage",
            "doc_status_storage": "MongoDocStatusStorage",
            "graph_storage": "Neo4JStorage",
            "vector_db_storage_cls_kwargs": {
                "uri": settings.milvus_uri,
                "user": settings.milvus_user,
                "password": settings.milvus_password,
                "db_name": settings.milvus_db_name,
            },
        }

        # 添加 Neo4j 配置
        if settings.neo4j_uri:
            lightrag_kwargs["graph_db_storage_cls_kwargs"] = {
                "uri": settings.neo4j_uri,
                "username": settings.neo4j_username,
                "password": settings.neo4j_password,
            }

        logger.info(f"Initializing RAGAnything with working_dir={settings.raganything_working_dir}")

        _raganything_instance = RAGAnything(
            config=config,
            llm_model_func=_llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=int(settings.vllm_embedding_dim),
                max_token_size=8192,
                func=_embedding_func
            ),
            lightrag_kwargs=lightrag_kwargs,
        )
        await _raganything_instance._ensure_lightrag_initialized()
        logger.info("RAGAnything instance initialized successfully")
    return _raganything_instance


async def get_raganything_stream(query: str, mode: str = "hybrid") -> AsyncIterator[Dict[str, Any]]:
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
    async for chunk in rag.aquery_stream_with_sources(query, mode=mode):
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
