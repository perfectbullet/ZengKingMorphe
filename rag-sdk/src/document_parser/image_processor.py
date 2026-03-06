"""
图片描述处理器

使用多模态模型（Qwen2-VL）为图片生成描述
"""

import asyncio
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
        base_url: Optional[str] = None
    ):
        """
        初始化图片描述生成器

        Args:
            model_name: 模型名称（如 qwen2-vl:latest）
            base_url: 服务地址（如 Ollama 地址）
        """
        self.model_name = model_name or settings.qwen_vl_model
        self.base_url = base_url or settings.qwen_vl_base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.enabled = settings.enable_image_description

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
        if not self.base_url:
            raise ValueError("未配置 Qwen2-VL 服务地址")

        session = await self._get_session()

        # 读取图片文件
        with open(image_path, 'rb') as f:
            image_data = f.read()

        # 准备请求
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_data],
            "stream": False
        }

        try:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
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
