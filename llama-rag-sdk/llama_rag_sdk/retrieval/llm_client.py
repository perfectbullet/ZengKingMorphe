"""
LLM 客户端模块

支持多种 LLM 后端：
- Ollama (本地)
- OpenAI 兼容 API (vLLM)
- SiliconFlow
"""

from typing import Optional
from enum import Enum
import httpx
from loguru import logger


class LLMProvider(str, Enum):
    """LLM 提供商"""
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"
    SILICONFLOW = "siliconflow"


class LLMClient:
    """LLM 客户端基类"""

    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """异步调用 LLM"""
        raise NotImplementedError


class OllamaLLMClient(LLMClient):
    """Ollama LLM 客户端"""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 60
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """调用 Ollama API"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url.rstrip('/')}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
        except httpx.HTTPError as e:
            logger.error(f"Ollama API 调用失败: {e}")
            raise
        except Exception as e:
            logger.error(f"Ollama 调用异常: {e}")
            raise


class OpenAICompatibleLLMClient(LLMClient):
    """OpenAI 兼容 API 客户端（vLLM、SiliconFlow 等）"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        timeout: int = 60
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """调用 OpenAI 兼容 API"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            if self.api_key != "not-needed":
                headers["Authorization"] = f"Bearer {self.api_key}"

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": kwargs.get("temperature", 0.7),
                        "max_tokens": kwargs.get("max_tokens", 500)
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            logger.error(f"OpenAI 兼容 API 调用失败: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI 兼容 API 调用异常: {e}")
            raise


def create_llm_client(
    provider: str = "ollama",
    base_url: str = "http://localhost:11434",
    model: str = "qwen2.5:14b",
    api_key: str = "not-needed",
    timeout: int = 60
) -> LLMClient:
    """
    创建 LLM 客户端工厂函数

    Args:
        provider: LLM 提供商 (ollama/openai_compatible/siliconflow)
        base_url: LLM 服务地址
        model: LLM 模型名称
        api_key: API Key（可选）
        timeout: 请求超时时间（秒）

    Returns:
        LLM 客户端实例

    Raises:
        ValueError: 不支持的提供商时抛出
    """
    provider = LLMProvider(provider)

    if provider == LLMProvider.OLLAMA:
        return OllamaLLMClient(base_url=base_url, model=model, timeout=timeout)
    elif provider in (LLMProvider.OPENAI_COMPATIBLE, LLMProvider.SILICONFLOW):
        return OpenAICompatibleLLMClient(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout
        )
    else:
        raise ValueError(f"不支持的 LLM 提供商: {provider}")
