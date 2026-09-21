"""Compatibility entry point for the refactored v2 speech SSE stream."""

from collections.abc import AsyncGenerator

from app.models.schemas import OpenAIChatRequest
from app.services.chat.answer_events import ChatContext
from app.services.chat.chat_stream_service import ChatStreamService
from app.services.chat.request_resolver import RequestResolver


async def generate_openai_stream_v2(
    request_or_context: OpenAIChatRequest | ChatContext,
) -> AsyncGenerator[str, None]:
    """Yield the v2 TTS stream while preserving the historical import path."""

    context = (
        request_or_context
        if isinstance(request_or_context, ChatContext)
        else RequestResolver().resolve(request_or_context)
    )
    service = ChatStreamService()
    async for payload in service.stream(context):
        yield payload
