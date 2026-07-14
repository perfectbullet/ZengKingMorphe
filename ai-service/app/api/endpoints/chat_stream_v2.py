"""Compatibility entry point for v2 streaming clients.

Both public chat versions intentionally use the same common LLM execution path.
"""

from typing import AsyncGenerator, Optional

from fastapi import Request

from app.api.endpoints.chat_stream_v1 import generate_openai_stream_v1
from app.models.schemas import OpenAIChatRequest


async def generate_openai_stream_v2(
    request: OpenAIChatRequest,
    http_request: Optional[Request] = None,
) -> AsyncGenerator[str, None]:
    async for event in generate_openai_stream_v1(request, http_request=http_request):
        yield event
