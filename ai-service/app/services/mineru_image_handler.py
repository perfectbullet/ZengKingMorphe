"""
MinerU Image Handler - 图片描述生成和缓存

该模块用于处理MinerU JSON中提取的图片信息：
1. 提取图片URL和上下文
2. 调用VLM API生成图片描述
3. 缓存图片描述到MongoDB
4. 将图片描述与上下文文本组合

支持多种VLM后端：
- OpenAI GPT-4 Vision
- 阿里云 Qwen-VL (通义千问视觉模型)
- 本地部署的视觉模型
"""
import hashlib
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum

import httpx
from app.core.logging import logger
from app.core.database import get_database
from motor.motor_asyncio import AsyncIOMotorClientSession


class VLMBackend(str, Enum):
    """视觉语言模型后端"""
    OPENAI = "openai"       # OpenAI GPT-4V
    QWEN_VL = "qwen_vl"     # 阿里云通义千问VL
    CUSTOM = "custom"       # 自定义端点


@dataclass
class ImageInfo:
    """图片信息"""
    url: str
    page_idx: int
    bbox: tuple
    context_before: str = ""
    context_after: str = ""
    caption: Optional[str] = None


@dataclass
class ImageCaption:
    """图片描述结果"""
    image_url: str
    caption: str
    context: str
    model_used: str
    created_at: datetime


class MinerUImageHandler:
    """MinerU图片处理器"""

    def __init__(
        self,
        vlm_backend: VLMBackend = VLMBackend.QWEN_VL,
        vlm_api_key: Optional[str] = None,
        vlm_base_url: Optional[str] = None,
        vlm_model: str = "qwen-vl-max",
        timeout: int = 30,
        cache_ttl_days: int = 30
    ):
        self.vlm_backend = vlm_backend
        self.vlm_api_key = vlm_api_key
        self.vlm_base_url = vlm_base_url
        self.vlm_model = vlm_model
        self.timeout = timeout
        self.cache_ttl_days = cache_ttl_days
        self._cache: Dict[str, ImageCaption] = {}

    def _get_image_hash(self, image_url: str) -> str:
        """生成图片URL的哈希值用于缓存"""
        return hashlib.md5(image_url.encode()).hexdigest()

    async def get_cached_caption(self, image_url: str) -> Optional[ImageCaption]:
        """从缓存获取图片描述

        Args:
            image_url: 图片URL

        Returns:
            缓存的图片描述，如果不存在或过期则返回None
        """
        # 先检查内存缓存
        cache_key = self._get_image_hash(image_url)
        if cache_key in self._cache:
            caption = self._cache[cache_key]
            # 检查是否过期
            if datetime.utcnow() - caption.created_at < timedelta(days=self.cache_ttl_days):
                return caption
            else:
                del self._cache[cache_key]

        # 检查MongoDB缓存
        try:
            db = get_database()
            collection = db.mineru_image_captions

            doc = await collection.find_one({"image_url": image_url})
            if doc:
                created_at = doc.get("created_at", datetime.utcnow())
                if datetime.utcnow() - created_at < timedelta(days=self.cache_ttl_days):
                    caption = ImageCaption(
                        image_url=doc["image_url"],
                        caption=doc["caption"],
                        context=doc.get("context", ""),
                        model_used=doc.get("model_used", ""),
                        created_at=created_at
                    )
                    # 更新内存缓存
                    self._cache[cache_key] = caption
                    return caption
                else:
                    # 删除过期缓存
                    await collection.delete_one({"image_url": image_url})
        except Exception as e:
            logger.warning("获取MongoDB图片缓存失败", error=str(e))

        return None

    async def save_caption_to_cache(self, caption: ImageCaption) -> None:
        """保存图片描述到缓存

        Args:
            caption: 图片描述对象
        """
        cache_key = self._get_image_hash(caption.image_url)

        # 更新内存缓存
        self._cache[cache_key] = caption

        # 更新MongoDB缓存
        try:
            db = get_database()
            collection = db.mineru_image_captions

            await collection.update_one(
                {"image_url": caption.image_url},
                {
                    "$set": {
                        "caption": caption.caption,
                        "context": caption.context,
                        "model_used": caption.model_used,
                        "created_at": caption.created_at
                    }
                },
                upsert=True
            )
        except Exception as e:
            logger.warning("保存图片缓存到MongoDB失败", error=str(e))

    async def generate_caption_openai(
        self,
        image_url: str,
        context: str = ""
    ) -> str:
        """使用OpenAI GPT-4V生成图片描述

        Args:
            image_url: 图片URL
            context: 图片上下文

        Returns:
            图片描述文本
        """
        prompt = "请详细描述这张图片的内容。"
        if context:
            prompt = f"请描述这张图片的内容。图片上下文：{context}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.vlm_api_key}"
                }

                payload = {
                    "model": self.vlm_model or "gpt-4o",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_url}}
                            ]
                        }
                    ],
                    "max_tokens": 300
                }

                base_url = self.vlm_base_url or "https://api.openai.com/v1"
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()

                result = response.json()
                caption = result["choices"][0]["message"]["content"]
                return caption.strip()

        except Exception as e:
            logger.error("OpenAI图片描述生成失败", error=str(e), image_url=image_url[:100])
            return ""

    async def generate_caption_qwen_vl(
        self,
        image_url: str,
        context: str = ""
    ) -> str:
        """使用阿里云通义千问VL生成图片描述

        Args:
            image_url: 图片URL
            context: 图片上下文

        Returns:
            图片描述文本
        """
        prompt = "请详细描述这张图片的内容。"
        if context:
            prompt = f"请描述这张图片的内容。图片上下文：{context}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.vlm_api_key}"
                }

                payload = {
                    "model": self.vlm_model or "qwen-vl-max",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"image": image_url},
                                {"text": prompt}
                            ]
                        }
                    ],
                    "max_tokens": 300
                }

                base_url = self.vlm_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()

                result = response.json()
                caption = result["choices"][0]["message"]["content"]
                return caption.strip()

        except Exception as e:
            logger.error("Qwen-VL图片描述生成失败", error=str(e), image_url=image_url[:100])
            return ""

    async def generate_caption_custom(
        self,
        image_url: str,
        context: str = ""
    ) -> str:
        """使用自定义端点生成图片描述

        Args:
            image_url: 图片URL
            context: 图片上下文

        Returns:
            图片描述文本
        """
        # 自定义端点需要用户提供具体的API格式
        # 这里提供一个通用的OpenAI兼容格式
        return await self.generate_caption_openai(image_url, context)

    async def generate_caption(
        self,
        image_url: str,
        context: str = "",
        use_cache: bool = True
    ) -> str:
        """生成图片描述（带缓存）

        Args:
            image_url: 图片URL
            context: 图片上下文
            use_cache: 是否使用缓存

        Returns:
            图片描述文本
        """
        # 检查缓存
        if use_cache:
            cached = await self.get_cached_caption(image_url)
            if cached:
                return cached.caption

        # 生成新描述
        if self.vlm_backend == VLMBackend.OPENAI:
            caption = await self.generate_caption_openai(image_url, context)
        elif self.vlm_backend == VLMBackend.QWEN_VL:
            caption = await self.generate_caption_qwen_vl(image_url, context)
        else:
            caption = await self.generate_caption_custom(image_url, context)

        # 保存到缓存
        if caption:
            await self.save_caption_to_cache(ImageCaption(
                image_url=image_url,
                caption=caption,
                context=context,
                model_used=self.vlm_model,
                created_at=datetime.utcnow()
            ))

        return caption

    async def generate_captions_batch(
        self,
        images: List[ImageInfo],
        use_cache: bool = True,
        concurrent_limit: int = 5
    ) -> Dict[str, str]:
        """批量生成图片描述

        Args:
            images: 图片信息列表
            use_cache: 是否使用缓存
            concurrent_limit: 并发限制

        Returns:
            图片URL到描述的映射
        """
        results = {}

        # 创建信号量限制并发
        semaphore = asyncio.Semaphore(concurrent_limit)

        async def process_one(img: ImageInfo) -> tuple:
            async with semaphore:
                context = f"{img.context_before}\n{img.context_after}".strip()
                caption = await self.generate_caption(img.url, context, use_cache)
                return img.url, caption

        tasks = [process_one(img) for img in images]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for item in completed:
            if isinstance(item, Exception):
                logger.error("批量生成图片描述时发生错误", error=str(item))
            else:
                url, caption = item
                results[url] = caption

        return results

    async def extract_images_from_doc(
        self,
        doc_data: Dict[str, Any],
        context_window: int = 2
    ) -> List[ImageInfo]:
        """从MinerU文档数据中提取图片信息

        Args:
            doc_data: MinerU JSON解析后的文档数据
            context_window: 上下文窗口大小（前后各取几个块）

        Returns:
            图片信息列表
        """
        from app.services.mineru_json_parser import MinerUJsonParser, BlockType

        images = []

        for page_data in doc_data.get("pdf_info", []):
            page_idx = page_data.get("page_idx", 0)
            blocks = page_data.get("para_blocks", [])

            # 获取所有文本内容用于上下文
            text_blocks = [
                b for b in blocks
                if b.get("type") in ("text", "title", "list")
            ]

            for i, block in enumerate(blocks):
                if block.get("type") == "image":
                    # 提取图片URL
                    image_url = None
                    for child in block.get("blocks", []):
                        for line in child.get("lines", []):
                            for span in line.get("spans", []):
                                if span.get("image_path"):
                                    image_url = span["image_path"]
                            if image_url:
                                break
                        if image_url:
                            break

                    if not image_url:
                        continue

                    # 提取上下文
                    context_before = ""
                    context_after = ""

                    # 在text_blocks中找到当前图片的位置
                    # 简化处理：取前后几个文本块
                    for j, text_block in enumerate(text_blocks):
                        if text_block.get("index", 0) < block.get("index", 0):
                            # 提取文本
                            for line in text_block.get("lines", []):
                                for span in line.get("spans", []):
                                    if span.get("content"):
                                        context_before += span["content"] + " "
                        elif text_block.get("index", 0) > block.get("index", 0):
                            # 提取文本
                            for line in text_block.get("lines", []):
                                for span in line.get("spans", []):
                                    if span.get("content"):
                                        context_after += span["content"] + " "

                    images.append(ImageInfo(
                        url=image_url,
                        page_idx=page_idx,
                        bbox=tuple(block.get("bbox", [0, 0, 0, 0])),
                        context_before=context_before.strip(),
                        context_after=context_after.strip()
                    ))

        return images


async def generate_image_captions(
    images: List[ImageInfo],
    vlm_backend: VLMBackend = VLMBackend.QWEN_VL,
    vlm_api_key: Optional[str] = None,
    vlm_base_url: Optional[str] = None,
    vlm_model: str = "qwen-vl-max"
) -> Dict[str, str]:
    """便捷函数：批量生成图片描述

    Args:
        images: 图片信息列表
        vlm_backend: VLM后端类型
        vlm_api_key: API密钥
        vlm_base_url: API地址
        vlm_model: 模型名称

    Returns:
        图片URL到描述的映射
    """
    handler = MinerUImageHandler(
        vlm_backend=vlm_backend,
        vlm_api_key=vlm_api_key,
        vlm_base_url=vlm_base_url,
        vlm_model=vlm_model
    )
    return await handler.generate_captions_batch(images)
