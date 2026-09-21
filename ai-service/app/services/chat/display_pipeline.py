"""Frontend display delivery through MongoDB stream chunks."""

import os
from typing import Any

from app.repositories.stream_chunk_repository import StreamChunkRepository
from app.services.chat.answer_events import AnswerEvent, ChatContext


class DisplayPipeline:
    """Persist display-only Markdown/LaTeX events for WebSocket consumers."""

    def __init__(
        self,
        context: ChatContext,
        repository: StreamChunkRepository,
    ) -> None:
        self.context = context
        self.repository = repository
        self.default_model = os.getenv("LLM_MODEL", "")

    async def start(self) -> None:
        await self.repository.save(
            chunk_type="user_query",
            chunk_data={
                "id": self.context.chat_id,
                "object": "chat.completion.chunk",
                "created": self.context.created,
                "model": self.default_model,
                "user_message": self.context.user_query,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in self.context.request.messages
                ],
            },
        )
        await self.repository.save(chunk_type="role", chunk_data=self._role_payload())

    async def handle(self, event: AnswerEvent) -> None:
        if event.event_type not in {"content", "status", "reasoning"} or not event.content:
            return
        model = event.metadata.get("model_name") or (
            "status" if event.event_type == "status" else self.default_model
        )
        payload = (
            self._reasoning_payload(event.content, model=model)
            if event.event_type == "reasoning"
            else self._content_payload(event.content, model=model)
        )
        await self.repository.save(
            chunk_type=(
                "reasoning" if event.event_type == "reasoning"
                else "status" if event.event_type == "status" else "token"
            ),
            chunk_data=payload,
            conversation_id=event.metadata.get("conversation_id"),
        )

    async def finish(self, finish_payload: dict[str, Any], conversation_id: str) -> None:
        await self.repository.save(
            chunk_type="done",
            chunk_data=finish_payload,
            conversation_id=conversation_id,
        )

    async def error(self, error_payload: dict[str, Any], conversation_id: str = "") -> None:
        await self.repository.save(
            chunk_type="error",
            chunk_data=error_payload,
            conversation_id=conversation_id,
        )

    def _role_payload(self) -> dict[str, Any]:
        return {
            "id": self.context.chat_id,
            "object": "chat.completion.chunk",
            "created": self.context.created,
            "model": self.default_model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }],
        }

    def _content_payload(self, content: str, *, model: str) -> dict[str, Any]:
        return {
            "id": self.context.chat_id,
            "object": "chat.completion.chunk",
            "created": self.context.created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }],
        }

    def _reasoning_payload(self, reasoning: str, *, model: str) -> dict[str, Any]:
        return {
            "id": self.context.chat_id,
            "object": "chat.completion.chunk",
            "created": self.context.created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"reasoning": reasoning},
                "finish_reason": None,
            }],
        }
