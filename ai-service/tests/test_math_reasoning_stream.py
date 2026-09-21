"""Focused tests for raw math-model reasoning SSE decoding."""

import json

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.services.chat.math_reasoning_stream import (
    MathModelConfig,
    MathReasoningStreamAdapter,
    ReasoningBuffer,
)


def _config() -> MathModelConfig:
    return MathModelConfig(
        base_url="http://math.example/v1",
        model_name="fake-math-model",
        api_key="test-key",
        max_tokens=128,
        temperature=0.6,
        top_p=0.95,
    )


@pytest.mark.asyncio
async def test_adapter_preserves_reasoning_and_content_sse_order():
    frames = [
        ": ping\n\n",
        'data: {"choices":[{"delta":{"reasoning":"先分析"}}]}\n\n',
        'data: {"choices":[{"delta":{"reasoning":"条件"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"解："}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"x=2"},"finish_reason":"stop"}]}\n\n',
        "data: [DONE]\n\n",
    ]
    received_body: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        received_body.update(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(frames).encode(),
        )

    adapter = MathReasoningStreamAdapter(
        _config(),
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    chunks = [
        chunk async for chunk in adapter.stream([
            SystemMessage(content="数学系统提示"),
            HumanMessage(content="求解"),
        ])
    ]

    assert [(chunk.reasoning, chunk.content) for chunk in chunks] == [
        ("先分析", ""),
        ("条件", ""),
        ("", "解："),
        ("", "x=2"),
    ]
    assert chunks[-1].finish_reason == "stop"
    assert received_body["stream"] is True
    assert received_body["chat_template_kwargs"] == {"enable_thinking": True}
    assert received_body["messages"] == [
        {"role": "system", "content": "数学系统提示"},
        {"role": "user", "content": "求解"},
    ]


def test_reasoning_buffer_flushes_on_boundary_size_and_explicit_finish():
    now = [0.0]
    buffer = ReasoningBuffer(max_chars=5, max_wait_seconds=1, clock=lambda: now[0])

    assert buffer.add("先") is None
    assert buffer.add("分析") is None
    assert buffer.add("条件") == "先分析条件"
    assert buffer.add("下一步") is None
    assert buffer.add("。") == "下一步。"
    assert buffer.add("收尾") is None
    assert buffer.flush() == "收尾"


@pytest.mark.asyncio
async def test_adapter_retries_only_the_vllm_declared_context_limit():
    requested_max_tokens: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_max_tokens.append(json.loads(request.content)["max_tokens"])
        if len(requested_max_tokens) == 1:
            return httpx.Response(
                400,
                json={"error": {"message": (
                    "max_completion_tokens=30720 cannot be greater than "
                    "max_model_len=max_total_tokens=20480"
                )}},
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices":[{"delta":{"content":"2"}}]}\n\ndata: [DONE]\n\n',
        )

    config = MathModelConfig(
        base_url="http://math.example/v1",
        model_name="fake-math-model",
        api_key="test-key",
        max_tokens=30720,
        temperature=0.6,
        top_p=0.95,
    )
    adapter = MathReasoningStreamAdapter(
        config,
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    chunks = [chunk async for chunk in adapter.stream([HumanMessage(content="1+1")])]

    assert requested_max_tokens == [30720, 20480]
    assert [chunk.content for chunk in chunks] == ["2"]
