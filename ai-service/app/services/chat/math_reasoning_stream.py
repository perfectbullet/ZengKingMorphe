"""Raw vLLM SSE support for the math model's reasoning stream only."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.messages import convert_to_openai_messages

from app.core.logging import get_logger
from app.utils.get_vllm_first_model import get_vllm_first_model

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MathModelConfig:
    """The shared math-model settings used by ChatOpenAI and raw SSE clients."""

    base_url: str
    model_name: str
    api_key: str
    max_tokens: int
    temperature: float | None
    top_p: float | None

    @classmethod
    def from_env(cls) -> "MathModelConfig":
        base_url = (os.getenv("MATH_MODEL_BASE_URL") or "").rstrip("/")
        model_name = os.getenv("MATH_MODEL_NAME") or get_vllm_first_model(base_url)
        api_key = os.getenv("MATH_MODEL_API_KEY", "").strip().strip('"').strip("'") or "dummy-key"
        temperature = float(os.getenv("MATH_TEMPERATURE", 0.6))
        top_p = float(os.getenv("MATH_TOP_P", 0.95))

        # Retain the existing Qwen3-32B generation_config.json behaviour.
        if _uses_model_generation_defaults(model_name):
            temperature = None
            top_p = None

        return cls(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            max_tokens=int(os.getenv("MATH_MAX_TOKEN", 10240)),
            temperature=temperature,
            top_p=top_p,
        )


def _uses_model_generation_defaults(model_name: str | None) -> bool:
    return bool(model_name and "qwen3-32b" in model_name.lower().replace("_", "-"))


@dataclass(slots=True)
class MathStreamChunk:
    """One useful delta decoded from an OpenAI-compatible math SSE event."""

    reasoning: str = ""
    content: str = ""
    finish_reason: str | None = None
    model_name: str | None = None


class ReasoningBuffer:
    """Emit small semantic reasoning fragments without Mongo write amplification."""

    _BOUNDARY = re.compile(r"[。！？.!?\n]")

    def __init__(
        self,
        *,
        max_chars: int = 80,
        max_wait_seconds: float = 0.2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_chars = max_chars
        self.max_wait_seconds = max_wait_seconds
        self._clock = clock
        self._parts: list[str] = []
        self._started_at: float | None = None

    def add(self, text: str) -> str | None:
        if not text:
            return None
        now = self._clock()
        if self._started_at is None:
            self._started_at = now
        self._parts.append(text)
        value = "".join(self._parts)
        if (
            self._BOUNDARY.search(text)
            or len(value) >= self.max_chars
            or now - self._started_at >= self.max_wait_seconds
        ):
            return self.flush()
        return None

    def flush(self) -> str | None:
        if not self._parts:
            return None
        value = "".join(self._parts)
        self._parts.clear()
        self._started_at = None
        return value


class MathReasoningStreamAdapter:
    """Request the math vLLM endpoint and decode its raw SSE deltas."""

    def __init__(
        self,
        config: MathModelConfig,
        *,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self.config = config
        self._client_factory = client_factory or self._new_client

    def _new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0),
        )

    async def stream(self, messages: Sequence[Any]) -> AsyncIterator[MathStreamChunk]:
        """Yield reasoning/content deltas as soon as vLLM sends each SSE event."""
        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        max_tokens = self.config.max_tokens
        async with self._client_factory() as client:
            for attempt in range(2):
                body = self._request_body(messages, max_tokens)
                async with client.stream("POST", url, headers=headers, json=body) as response:
                    if response.status_code == 400 and attempt == 0:
                        error_body = await response.aread()
                        limit = self._vllm_max_total_tokens(error_body.decode(errors="replace"))
                        if limit is not None and limit < max_tokens:
                            logger.warning(
                                "Math vLLM rejected MATH_MAX_TOKEN=%s above its "
                                "max_total_tokens=%s; retrying with the server limit",
                                max_tokens,
                                limit,
                            )
                            max_tokens = limit
                            continue
                    response.raise_for_status()
                    async for raw_line in response.aiter_lines():
                        payload = self._sse_data(raw_line)
                        if payload is None or payload == "[DONE]":
                            if payload == "[DONE]":
                                return
                            continue
                        try:
                            data = json.loads(payload)
                        except json.JSONDecodeError:
                            # A malformed auxiliary SSE frame must not discard later content.
                            logger.warning("Ignoring malformed math SSE frame: %r", payload[:120])
                            continue

                        chunk = self._decode(data)
                        if chunk is not None:
                            yield chunk
                    return

    def _request_body(self, messages: Sequence[Any], max_tokens: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": self._to_openai_messages(messages),
            "stream": True,
            "max_tokens": max_tokens,
            # Qwen3.6 only exposes delta.reasoning while thinking is enabled.
            "chat_template_kwargs": {"enable_thinking": True},
        }
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            body["top_p"] = self.config.top_p
        return body

    @staticmethod
    def _to_openai_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
        converted = convert_to_openai_messages(list(messages), text_format="string")
        if isinstance(converted, dict):
            return [converted]
        return converted

    @staticmethod
    def _sse_data(raw_line: str) -> str | None:
        line = raw_line.strip()
        if not line or line.startswith(":"):
            return None
        return line[5:].strip() if line.startswith("data:") else line

    @staticmethod
    def _vllm_max_total_tokens(error_body: str) -> int | None:
        match = re.search(r"max_total_tokens=(\d+)", error_body)
        return int(match.group(1)) if match else None

    def _decode(self, data: dict[str, Any]) -> MathStreamChunk | None:
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return None
        choice = choices[0]
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            delta = {}

        # ``reasoning`` is Qwen3.6/vLLM's current field. The second spelling is
        # deliberately the only compatibility alias retained for older vLLM.
        reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
        content = delta.get("content") or ""
        finish_reason = choice.get("finish_reason")
        if not reasoning and not content and not finish_reason:
            return None
        return MathStreamChunk(
            reasoning=str(reasoning),
            content=str(content),
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            model_name=data.get("model") or self.config.model_name,
        )
