"""
统一 RAG 流式入口。
"""

from __future__ import annotations

from typing import AsyncIterator, Any


async def get_rag_stream(
    query: str,
    mode: str = "hybrid",
    prefer_zh_output: bool = True,
    debug_meta: dict | None = None,
) -> AsyncIterator[dict]:
    from app.services.training_lightrag_wrapper import get_training_lightrag_stream

    async for item in get_training_lightrag_stream(
        query=query,
        mode=mode,
        prefer_zh_output=prefer_zh_output,
        debug_meta=debug_meta,
    ):
        yield item
