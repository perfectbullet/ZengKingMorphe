"""
统一 RAG 流式入口。
"""

from __future__ import annotations

import os
from typing import AsyncIterator, Any

from app.core.logging import get_logger

logger = get_logger(__name__)


async def get_rag_stream(
    query: str,
    mode: str = "hybrid",
    prefer_zh_output: bool = True,
    debug_meta: dict | None = None,
) -> AsyncIterator[dict]:
    backend = (os.getenv("TRAINING_RAG_BACKEND", "lightrag_file").strip() or "lightrag_file")

    if backend == "lightrag_file":
        from app.services.training_lightrag_wrapper import get_training_lightrag_stream

        async for item in get_training_lightrag_stream(
            query=query,
            mode=mode,
            prefer_zh_output=prefer_zh_output,
            debug_meta=debug_meta,
        ):
            yield item
        return

    if backend == "raganything":
        from app.services.raganything_wrapper import get_legacy_raganything_stream

        async for item in get_legacy_raganything_stream(
            query=query,
            mode=mode,
            prefer_zh_output=prefer_zh_output,
        ):
            yield item
        return

    logger.error(f"Unsupported TRAINING_RAG_BACKEND: {backend}")
    yield {
        "type": "error",
        "content": f"Unsupported TRAINING_RAG_BACKEND: {backend}",
    }
