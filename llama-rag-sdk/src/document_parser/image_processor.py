"""
图片描述处理器

使用多模态模型（Qwen2-VL）为图片生成描述
通过 vLLM OpenAI 兼容 API 调用
"""

import asyncio
import base64
from pathlib import Path
from typing import List, Optional
import aiohttp
from loguru import logger

from src.document_parser.base import ImageInfo
from src.config import settings


class ImageDescriptor:
    """图片描述生成器"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        初始化图片描述生成器

        Args:
            model_name: 模型名称（如 qwen2-vl:latest）
            api_base: vLLM 服务地址
            api_key: API Key（不需要真实 key）
        """
        self.model_name = model_name or settings.qwen_vl_model
        self.api_base = api_base or settings.qwen_vl_base_url
        self.api_key = api_key or settings.vllm_api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.enabled = settings.enable_image_description
        self.use_ollama = True  # 当前仍使用 Ollama 运行 Qwen2-VL

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP 会话"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def close(self) -> None:
        """关闭 HTTP 会话"""
        if self.session and not self.session.closed:
            await self.session.close()

    def _encode_image(self, image_path: str) -> str:
        """
        将图片编码为 base64

        Args:
            image_path: 图片文件路径

        Returns:
            base64 编码的图片字符串
        """
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    async def _describe_image_vllm(
        self,
        image_path: str,
        prompt: str = "请详细描述这张图片的内容，包括其中的文字、公式、图表等所有可见元素。"
    ) -> str:
        """
        使用 vLLM OpenAI 兼容 API 生成图片描述

        Args:
            image_path: 图片文件路径
            prompt: 提示词

        Returns:
            图片描述文本
        """
        if not self.api_base:
            raise ValueError("未配置 Qwen2-VL 服务地址")

        session = await self._get_session()

        # 编码图片
        image_base64 = self._encode_image(image_path)

        # 准备请求（OpenAI Chat Completions 格式）
        url = f"{self.api_base.rstrip('/')}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key and self.api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500,
            "stream": False
        }

        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"vLLM API 调用失败: {response.status} - {error_text}"
                    )

                result = await response.json()
                return result['choices'][0]['message']['content']

        except aiohttp.ClientError as e:
            logger.error(f"vLLM API 请求错误: {e}")
            raise
        except Exception as e:
            logger.error(f"图片描述生成错误: {e}")
            raise

    async def _describe_image_ollama(
        self,
        image_path: str,
        prompt: str = "请详细描述这张图片的内容，包括其中的文字、公式、图表等所有可见元素。"
    ) -> str:
        """
        使用 Ollama API 生成图片描述

        Args:
            image_path: 图片文件路径
            prompt: 提示词

        Returns:
            图片描述文本
        """
        if not self.api_base:
            raise ValueError("未配置 Qwen2-VL 服务地址")

        session = await self._get_session()

        # 读取图片文件
        with open(image_path, 'rb') as f:
            image_data = f.read()

        # 准备请求
        url = f"{self.api_base.rstrip('/')}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_data],
            "stream": False
        }

        try:
            async with session.post(url, json=payload, headers={"Content-Type": "application/json"}) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"Ollama API 调用失败: {response.status} - {error_text}"
                    )

                result = await response.json()
                return result.get('response', '')

        except aiohttp.ClientError as e:
            logger.error(f"Ollama API 请求错误: {e}")
            raise
        except Exception as e:
            logger.error(f"图片描述生成错误: {e}")
            raise

    async def describe_image(
        self,
        image_path: str,
        prompt: Optional[str] = None
    ) -> str:
        """
        为图片生成描述

        Args:
            image_path: 图片文件路径
            prompt: 自定义提示词

        Returns:
            图片描述文本
        """
        if not self.enabled:
            logger.warning("图片描述功能未启用")
            return ""

        # 检查图片是否存在
        if not Path(image_path).exists():
            logger.error(f"图片不存在: {image_path}")
            return ""

        # 使用默认提示词
        default_prompt = (
            "请详细描述这张图片的内容，包括："
            "1. 图片中的主要元素和对象"
            "2. 所有可见的文字内容"
            "3. 公式、图表等数学元素"
            "4. 图片的整体主题和用途"
        )
        final_prompt = prompt or default_prompt

        try:
            # 目前仍使用 Ollama 运行 Qwen2-VL
            # 未来可切换到 vLLM
            description = await self._describe_image_ollama(image_path, final_prompt)
            logger.info(f"图片描述生成完成: {image_path}")
            return description
        except Exception as e:
            logger.error(f"生成图片描述失败: {image_path}, 错误: {e}")
            return ""

    async def describe_images_batch(
        self,
        images: List[ImageInfo],
        prompt: Optional[str] = None,
        concurrency: int = 3
    ) -> List[ImageInfo]:
        """
        批量生成图片描述

        Args:
            images: 图片信息列表
            prompt: 自定义提示词
            concurrency: 并发数

        Returns:
            更新后的图片信息列表
        """
        if not self.enabled:
            logger.warning("图片描述功能未启用")
            return images

        semaphore = asyncio.Semaphore(concurrency)

        async def describe_with_semaphore(img: ImageInfo) -> ImageInfo:
            async with semaphore:
                description = await self.describe_image(img.path, prompt)
                img.description = description
                return img

        tasks = [describe_with_semaphore(img) for img in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"描述图片 {images[i].path} 失败: {result}")
                results[i] = images[i]  # 使用原始图片信息

        return results

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
