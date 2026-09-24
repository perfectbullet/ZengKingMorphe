"""v1 math reasoning is delivered through MongoDB/WebSocket, not speech SSE."""

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

os.environ["DEBUG"] = "false"
os.environ.setdefault("API_KEY", "test")

from app.api.endpoints import chat_stream_v1
from app.api.endpoints.websocket_view import send_chunk
from app.models.schemas import OpenAIChatRequest
from app.services.chat.math_reasoning_stream import MathModelConfig, MathStreamChunk


class FakeCollection:
    def __init__(self):
        self.documents = []

    async def insert_one(self, document):
        self.documents.append(document)


class FakeDatabase:
    def __init__(self):
        self.stream_chunks = FakeCollection()
        self.raw_stream_tokens = FakeCollection()


class FakeMathAdapter:
    def __init__(self, config):
        assert config.model_name == "fake-math-model"

    async def stream(self, messages):
        assert messages == [{"role": "user", "content": "求解不等式"}]
        yield MathStreamChunk(reasoning="先求根")
        yield MathStreamChunk(reasoning="，再看符号")
        yield MathStreamChunk(content="2<x<3。", model_name="fake-math-model")


async def _run_math_stream(monkeypatch, tmp_path, *, reasoning_enabled):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MATH_DEBUG_DUMP_ENABLED", "false")
    monkeypatch.setenv(
        "MATH_REASONING_DISPLAY_ENABLED",
        "true" if reasoning_enabled else "false",
    )
    db = FakeDatabase()
    math_config = MathModelConfig(
        base_url="http://math.example/v1",
        model_name="fake-math-model",
        api_key="test",
        max_tokens=128,
        temperature=0.6,
        top_p=0.95,
    )

    async def workflow_stream(_initial_state, stream_mode):
        assert stream_mode == "updates"
        yield {"post_classification_preprocess": {"user_query": "求解不等式"}}
        yield {
            "generate_answer": {
                "streaming_type": "math_llm",
                "streaming_llm": None,
                "math_stream_config": math_config,
                "streaming_messages": [{"role": "user", "content": "求解不等式"}],
                "is_math_problem": True,
                "conversation_id": "conv-math-v1",
            }
        }

    saved = AsyncMock()
    workflow = SimpleNamespace(
        workflow=SimpleNamespace(astream=workflow_stream),
        save_conversation=saved,
    )

    async def passthrough_segment(segment, *args, **kwargs):
        return segment, segment

    monkeypatch.setattr(chat_stream_v1, "get_database", AsyncMock(return_value=db))
    monkeypatch.setattr(chat_stream_v1, "conversation_workflow", workflow)
    monkeypatch.setattr(chat_stream_v1, "MathReasoningStreamAdapter", FakeMathAdapter)
    monkeypatch.setattr(chat_stream_v1, "_process_segment_for_output", passthrough_segment)

    request = OpenAIChatRequest(
        model="general-model",
        messages=[{"role": "user", "content": "求解不等式"}],
        stream=True,
        user_id="user-1",
        employee_id="employee-1",
        session_id="session-1",
    )
    payloads = [
        json.loads(raw) if raw != "[DONE]" else raw
        async for raw in chat_stream_v1.generate_openai_stream_v1(request)
    ]
    return payloads, db, saved


@pytest.mark.asyncio
async def test_v1_math_reasoning_is_saved_for_websocket_only(monkeypatch, tmp_path):
    payloads, db, saved = await _run_math_stream(
        monkeypatch, tmp_path, reasoning_enabled=True
    )

    documents = db.stream_chunks.documents
    reasoning = [doc for doc in documents if doc["chunk_type"] == "reasoning"]
    assert len(reasoning) == 1
    assert reasoning[0]["chunk_data"]["choices"][0]["delta"] == {
        "reasoning": "先求根，再看符号"
    }
    assert reasoning[0]["chunk_data"]["model"] == "fake-math-model"

    answer = next(
        doc for doc in documents
        if doc["chunk_type"] == "token"
        and doc["chunk_data"]["choices"][0]["delta"].get("content") == "2<x<3。"
    )
    assert answer["chunk_data"]["model"] == "fake-math-model"
    assert reasoning[0]["sequence"] < answer["sequence"]
    assert [doc["sequence"] for doc in documents] == list(range(1, len(documents) + 1))

    class FakeWebSocket:
        async def send_json(self, payload):
            self.payload = payload

    websocket = FakeWebSocket()
    await send_chunk(websocket, reasoning[0])
    assert websocket.payload["chunk_type"] == "reasoning"
    assert websocket.payload["chunk_data"]["choices"][0]["delta"]["reasoning"].startswith(
        "先求根"
    )

    assert all(
        "reasoning" not in choice.get("delta", {})
        for payload in payloads if isinstance(payload, dict)
        for choice in payload.get("choices", [])
    )
    assert payloads[-1] == "[DONE]"
    assert payloads[-2]["metadata"]["model"] == "fake-math-model"
    assert saved.await_args.args[0]["final_answer"] == "2<x<3。"


@pytest.mark.asyncio
async def test_v1_math_reasoning_display_can_be_disabled(monkeypatch, tmp_path):
    payloads, db, saved = await _run_math_stream(
        monkeypatch, tmp_path, reasoning_enabled=False
    )

    assert not any(
        doc["chunk_type"] == "reasoning" for doc in db.stream_chunks.documents
    )
    assert payloads[-1] == "[DONE]"
    assert saved.await_args.args[0]["final_answer"] == "2<x<3。"
