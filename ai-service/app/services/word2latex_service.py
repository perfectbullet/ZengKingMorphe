"""
ASR 结果转 LaTeX 公式服务。

调用外部 word2latex 服务将自然语言数学描述（ASR 语音识别结果）
转换为包含 LaTeX 公式的文本，供前端展示和 LLM 处理。
"""

import os
import time

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

WORD2LATEX_BASE_URL = os.getenv("WORD2LATEX_BASE_URL", "192.168.8.222")
WORD2LATEX_TIMEOUT = float(os.getenv("WORD2LATEX_TIMEOUT", "5"))


async def word_to_latex(text: str) -> str | None:
    """
    调用 word2latex 服务将自然语言数学描述转换为 LaTeX 公式。

    API 响应格式: {"input": "...", "output": "...", "success": true, "time": 0.001}

    Args:
        text: 原始自然语言数学描述

    Returns:
        转换后的文本（含 LaTeX 公式），失败返回 None
    """
    logger.info(f"word2latex converted WORD2LATEX_BASE_URL={WORD2LATEX_BASE_URL}, ")

    if not text or not text.strip():
        return None
    try:
        start = time.time()
        async with httpx.AsyncClient(timeout=WORD2LATEX_TIMEOUT) as client:
            resp = await client.post(
                f"{WORD2LATEX_BASE_URL}/word2latex",
                json={"text": text, "max_tokens": 250, "temperature": 0},
            )
            resp.raise_for_status()
            data = resp.json()
        duration = time.time() - start

        if not data.get("success"):
            logger.warning(
                f"word2latex not success: duration={duration:.2f}s, data={data}"
            )
            return None

        result = data.get("output", "").strip()
        if result:
            logger.info(
                f"word2latex converted: duration={duration:.2f}s, "
                f"original={text!r}, converted={result!r}"
            )
            return result
        logger.error("口语转公式报错了\n" * 10)
        logger.error(f"word2latex get error result: result={result}")
        
        return None
    except Exception as e:
        logger.warning(f"word2latex failed: error={e}, text={text[:60]!r}")
        return None
