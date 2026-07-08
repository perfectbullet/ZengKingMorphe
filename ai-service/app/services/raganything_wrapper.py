"""
RAGAnything 包装器 - 提供流式查询接口

集成 RAGAnything 到 ai-service，提供统一的 RAG 查询接口。
"""

import asyncio
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
)
from raganything.query_prompts import get_chinese_query_prompt, get_english_query_prompt

logger = get_logger(__name__)
_raganything_instance = None


# =============================================================================
# 配置验证
# =============================================================================
REQUIRED_ENV_VARS = {
    # Vision model (独立于统一 LLM 配置)
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


# =============================================================================
# 模型函数
# =============================================================================
async def llm_model_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[Dict] = None,
    **kwargs,
) -> str:
    """OpenAI兼容API的LLM函数，使用统一 LLM 配置"""
    model = os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("RAG_Anything_OLLAMA_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("LLM_API_KEY") or "no-api-key"
    try:
        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )
    except Exception as e:
        logger.error(
            f"❌ RAGAnything LLM 调用失败 | "
            f"model={model} | "
            f"base_url={base_url} | "
            f"error_type={type(e).__name__} | "
            f"error={e}"
        )
        raise


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
    model = get_required_env("VISION_MODEL")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("RAG_Anything_OPENAI_API_BASE", "http://localhost:11434/v1")
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "no-key")

    # 从 kwargs 中移除 image_data 和 messages，避免传递给 openai_complete_if_cache
    kwargs.pop("image_data", None)
    kwargs.pop("messages", None)

    # 抑制 stdout（避免打印 image_data）
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        try:
            if messages:
                result = await openai_complete_if_cache(
                    model,
                    "",
                    system_prompt=system_prompt,
                    messages=messages,
                    base_url=base_url,
                    api_key=api_key,
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
                    model,
                    "",
                    system_prompt=None,  # 已在 messages 中
                    messages=image_messages,
                    base_url=base_url,
                    api_key=api_key,
                    **kwargs,
                )
            else:
                result = await openai_complete_if_cache(
                    model,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    base_url=base_url,
                    api_key=api_key,
                    **kwargs,
                )
        except Exception as e:
            logger.error(
                f"❌ RAGAnything Vision Model 调用失败 | "
                f"model={model} | "
                f"base_url={base_url} | "
                f"error_type={type(e).__name__} | "
                f"error={e}"
            )
            raise

    return result


async def vllm_embedding_func(texts: List[str]) -> np.ndarray:
    """VLLM向量嵌入函数"""
    embed_url = get_required_env("VLLM_EMBED_URL")
    embed_model = get_required_env("VLLM_EMBED_MODEL")

    try:
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
    except Exception as e:
        logger.error(
            f"❌ RAGAnything Embedding 调用失败 | "
            f"model={embed_model} | "
            f"url={embed_url}/embeddings | "
            f"error_type={type(e).__name__} | "
            f"error={e}"
        )
        raise


async def vllm_reranker_func(
    query: str, documents: List[str], top_k: int = None, **kwargs
) -> List[Dict]:
    """VLLM Reranker函数"""
    rerank_url = get_required_env("VLLM_RERANK_URL")
    rerank_model = get_required_env("VLLM_RERANK_MODEL")

    try:
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
    except Exception as e:
        logger.error(
            f"❌ RAGAnything Reranker 调用失败 | "
            f"model={rerank_model} | "
            f"url={rerank_url}/rerank | "
            f"error_type={type(e).__name__} | "
            f"error={e}"
        )
        raise


def get_embedding_func():
    """创建 EmbeddingFunc"""
    embed_dim = int(get_required_env("VLLM_EMBED_DIM"))
    return EmbeddingFunc(
        embedding_dim=embed_dim, max_token_size=8192, func=vllm_embedding_func
    )


# =============================================================================
# 服务健康检查
# =============================================================================
async def _check_http_service(name: str, url: str, timeout: int = 5) -> bool:
    """检查 HTTP 服务健康状态"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status < 500:
                    logger.info(f"  ✅ {name}: {url}")
                    return True
                else:
                    logger.warning(f"  ⚠️ {name}: {url} - HTTP {resp.status}")
                    return False
    except Exception as e:
        logger.error(
            f"  ❌ {name}: {url} - 不可用 ({type(e).__name__}: {e})"
        )
        return False


async def check_raganything_services_health() -> Dict[str, bool]:
    """
    检查 RAGAnything 依赖的所有服务健康状态。

    性能：4 个 HTTP 探针完全独立，使用 ``asyncio.gather`` 并行化（耗时 ~max
    单点 RTT 而非 4 倍累加），把首次实例初始化的健康检查阶段从 ~1s 压到 ~0.3s。
    任何单个探针抛异常时只影响自己那一项的状态，不再传染整个健康检查。

    Returns:
        Dict[str, bool]: 服务名到健康状态的映射
    """
    logger.info("🔍 检查 RAGAnything 服务健康状态...")

    # 各探针的 (服务名, 环境变量名, URL 拼接后缀, 缺失提示) 表，集中维护新增 / 调整。
    probes: list[tuple[str, str, str, str]] = [
        ("llm", "LLM_BASE_URL", "/api/tags", "LLM Server"),
        ("vllm_embed", "VLLM_EMBED_URL", "/health", "VLLM Embedding"),
        ("vllm_rerank", "VLLM_RERANK_URL", "/health", "VLLM Reranker"),
    ]

    async def _probe(key: str, env_var: str, suffix: str, label: str) -> tuple[str, bool]:
        base = os.getenv(env_var, "")
        if not base:
            logger.warning(f"  ⚠️ {label}: 未配置 {env_var}")
            return key, False
        try:
            ok = await _check_http_service(label, f"{base}{suffix}")
            return key, ok
        except Exception as e:
            logger.error(f"  ❌ {label} 检查异常: {e}")
            return key, False

    completed = await asyncio.gather(
        *[_probe(*probe) for probe in probes],
        return_exceptions=False,
    )
    results = dict(completed)

    healthy_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    if healthy_count == total_count:
        logger.info(f"✅ 所有服务健康 ({healthy_count}/{total_count})")
    else:
        logger.warning(f"⚠️ 部分服务不可用 ({healthy_count}/{total_count})")

    return results


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
        logger.info(f"   LLM_BASE_URL (VLLM): {get_required_env('LLM_BASE_URL')}")
        logger.info(f"   LLM_MODEL (VLLM): {get_required_env('LLM_MODEL')}")

        # 健康检查
        await check_raganything_services_health()

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
                "cosine_better_than_threshold": 0.6,  # 向量相似度阈值
                "min_rerank_score": 0.8,  # 过滤 rerank 分数低于 0.2 的 chunks
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


async def get_legacy_raganything_stream(
    query: str, mode: str = "hybrid", prefer_zh_output: bool = True
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

    # 中英文提示词适配：使用 raganything 自带的中英 RAG 模板，保证 system prompt 与
    # 本轮目标语言完全一致（包括 ---Role--- / ---Context--- 段落标题），避免英文
    # 方向用一段过短的英文 system prompt 时 RAG 召回的中文 context_data 反过来主导
    # 模型的输出语言。
    if prefer_zh_output:
        system_prompt = get_chinese_query_prompt()
    else:
        # 在英文 RAG 模板上补一行强约束：召回内容里若出现中文，仅作信息参考，
        # 最终回答必须 100% 英文。这条约束直接拼到 {user_prompt} 占位符位置。
        base_en_prompt = get_english_query_prompt()
        en_lang_constraint = (
            "Answer strictly in English. If the retrieved context contains "
            "Chinese or other CJK text, treat it only as reference — translate "
            "any key facts you cite into English. Do not output any Chinese "
            "characters or mixed Chinese-English sentences. Do not include a "
            "'References' section."
        )
        system_prompt = base_en_prompt.replace("{user_prompt}", en_lang_constraint)

    async for chunk in rag.aquery_stream_with_sources(
        query,
        mode=mode,
        system_prompt=system_prompt,
        top_k=2,  # 召回实体/关系数量
        chunk_top_k=3,  # (默认10) - 召回文档块数量
        enable_rerank=True,  # (默认True) - 是否启用重排序
    ):
        yield chunk


async def get_raganything_stream(
    query: str,
    mode: str = "hybrid",
    prefer_zh_output: bool = True,
    debug_meta: dict | None = None,
) -> AsyncIterator[Dict[str, Any]]:
    logger.warning(
        "get_raganything_stream is deprecated; use get_rag_stream instead."
    )
    from app.services.rag_stream_wrapper import get_rag_stream

    async for item in get_rag_stream(
        query=query,
        mode=mode,
        prefer_zh_output=prefer_zh_output,
        debug_meta=debug_meta,
    ):
        yield item


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
