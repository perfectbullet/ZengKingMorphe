"""Serialize TTS speech events as OpenAI-compatible SSE payloads."""

import json
import os
from typing import Any

from app.services.chat.answer_events import ChatContext, SpeechEvent


class SSEWriter:
    """Own the HTTP wire representation for the v2 speech stream."""

    def __init__(self, context: ChatContext) -> None:
        self.context = context
        self.default_model = os.getenv("LLM_MODEL", "")

    def start(self) -> str:
        return self._encode({
            "id": self.context.chat_id,
            "object": "chat.completion.chunk",
            "created": self.context.created,
            "model": self.default_model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }],
        })

    def encode(self, event: SpeechEvent) -> str:
        model = event.metadata.get("model_name") or self.default_model
        return self._encode({
            "id": self.context.chat_id,
            "object": "chat.completion.chunk",
            "created": self.context.created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": event.content},
                "finish_reason": event.metadata.get("finish_reason"),
            }],
        })

    def finish(self, metadata: dict[str, Any]) -> str:
        return self._encode(self.finish_payload(metadata))

    def finish_payload(self, metadata: dict[str, Any]) -> dict[str, Any]:
        prompt_size = len(self.context.user_query)
        completion_size = len(metadata.get("display_text", ""))
        return {
            "id": self.context.chat_id,
            "object": "chat.completion.chunk",
            "created": self.context.created,
            "model": metadata.get("model_name") or self.default_model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_size,
                "completion_tokens": completion_size,
                "total_tokens": prompt_size + completion_size,
            },
            "metadata": {
                "conversation_id": metadata.get("conversation_id", ""),
                "confidence": metadata.get("confidence", 0.0),
                "kb_used": metadata.get("kb_used", []),
                "web_search_used": metadata.get("web_search_used", False),
                "sources": metadata.get("sources", {}),
                "intent": metadata.get("intent", ""),
                "is_realtime_query": metadata.get("is_realtime_query", False),
                "realtime_category": metadata.get("realtime_category", ""),
                "model": metadata.get("model_name") or self.default_model,
            },
        }

    @staticmethod
    def error(_exc: Exception) -> str:
        return SSEWriter._encode({
            "error": {
                "message": "Failed to process chat completion",
                "type": "server_error",
                "code": "internal_error",
            }
        })

    @staticmethod
    def _encode(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False)

