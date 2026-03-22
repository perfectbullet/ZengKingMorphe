"""
llama-rag-sdk 适配器

提供配置映射和 SDK 导入功能
"""
from app.services.sdk_adapter.config import setup_sdk_env

__all__ = ["setup_sdk_env"]

# 确保在导入时设置环境变量
setup_sdk_env()
