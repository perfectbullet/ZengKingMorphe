"""
Embedding 模型工厂

统一管理 OpenAIEmbedding 的创建，避免重复初始化代码
"""

from typing import Optional
from loguru import logger

from llama_rag_sdk.config import settings

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
except ImportError:
    logger.warning("llama-index-embeddings-openai 未安装")
    OpenAIEmbedding = None


class EmbeddingFactory:
    """Embedding 模型工厂类

    提供统一的 Embedding 模型创建接口，避免重复代码
    """

    _embedding_model_cache: Optional["OpenAIEmbedding"] = None

    @classmethod
    def create_embedding_model(
        cls,
        model_name: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        embed_batch_size: int = 32,
        timeout: int = 300
    ) -> Optional["OpenAIEmbedding"]:
        """
        创建 Embedding 模型实例

        Args:
            model_name: 模型名称
            api_base: API 基础 URL
            api_key: API 密钥
            embed_batch_size: 批量大小
            timeout: 超时时间

        Returns:
            OpenAIEmbedding 实例，如果不可用则返回 None
        """
        if OpenAIEmbedding is None:
            logger.error("OpenAIEmbedding 不可用")
            return None

        # 使用配置默认值
        model_name = model_name or settings.vllm_embedding_model
        api_base = api_base or settings.vllm_embedding_base_url
        api_key = api_key or settings.vllm_api_key

        try:
            embed_model = OpenAIEmbedding(
                model_name=model_name,
                api_base=api_base,
                api_key=api_key,
                embed_batch_size=embed_batch_size,
                timeout=timeout,
            )
            logger.info(f"Embedding 模型初始化成功: {model_name} @ {api_base}")
            return embed_model

        except Exception as e:
            logger.error(f"初始化 Embedding 模型失败: {e}")
            return None

    @classmethod
    def get_or_create_model(
        cls,
        model_name: Optional[str] = None,
        force_recreate: bool = False
    ) -> Optional["OpenAIEmbedding"]:
        """
        获取或创建 Embedding 模型（单例模式）

        Args:
            model_name: 模型名称
            force_recreate: 是否强制重新创建

        Returns:
            OpenAIEmbedding 实例
        """
        if force_recreate or cls._embedding_model_cache is None:
            cls._embedding_model_cache = cls.create_embedding_model(
                model_name=model_name
            )

        return cls._embedding_model_cache

    @classmethod
    def clear_cache(cls):
        """清除缓存的模型实例"""
        cls._embedding_model_cache = None
        logger.debug("Embedding 模型缓存已清除")


# 便捷函数
def create_embedding_model(**kwargs) -> Optional["OpenAIEmbedding"]:
    """创建 Embedding 模型的便捷函数"""
    return EmbeddingFactory.create_embedding_model(**kwargs)
