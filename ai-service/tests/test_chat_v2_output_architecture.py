"""Regression tests for the v2 display/speech output boundary."""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

os.environ["DEBUG"] = "false"
os.environ.setdefault("API_KEY", "test")

from app.models.schemas import OpenAIChatRequest
from app.services.chat.answer_events import AnswerEvent, ChatContext
from app.services.chat.chat_orchestrator import ChatOrchestrator
from app.services.chat.chat_stream_service import ChatStreamService
from app.services.chat.math_reasoning_stream import MathModelConfig, MathStreamChunk
from app.services.chat.request_resolver import RequestResolver
from app.services.chat.speech_pipeline import SpeechPipeline
from app.services.revise_llm import basic_math_to_voice


class FakeCollection:
    def __init__(self):
        self.documents: list[dict] = []

    async def insert_one(self, document: dict):
        self.documents.append(document)
        return SimpleNamespace(inserted_id=document["chunk_id"])


class FakeDatabase:
    def __init__(self):
        self.stream_chunks = FakeCollection()


class FakeLLM:
    def __init__(self, chunks: list[str], delay: float = 0):
        self.chunks = chunks
        self.delay = delay
        self.emitted: list[str] = []
        self.model_name = "fake-model"

    async def astream(self, _messages):
        for content in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            self.emitted.append(content)
            yield SimpleNamespace(content=content)


class FakeMathAdapter:
    def __init__(self, chunks: list[MathStreamChunk]):
        self.chunks = chunks

    async def stream(self, _messages):
        for chunk in self.chunks:
            yield chunk


class FakeCompiledWorkflow:
    def __init__(self, state: dict | None = None, error: Exception | None = None):
        self.state = state or {}
        self.error = error

    async def astream(self, _initial_state, stream_mode):
        assert stream_mode == "updates"
        if self.error:
            raise self.error
        # The arbitrary key proves the orchestrator does not require a node name.
        yield {"renamed_generation_node": self.state}


class FakeConversationWorkflow:
    def __init__(self, state: dict | None = None, error: Exception | None = None):
        self.workflow = FakeCompiledWorkflow(state, error)
        self.saved_states: list[dict] = []

    async def save_conversation(self, state):
        self.saved_states.append(dict(state))
        return state


def make_context(query: str = "测试问题") -> ChatContext:
    request = OpenAIChatRequest(
        messages=[{"role": "user", "content": query}],
        stream=True,
        user_id="user-1",
        employee_id="employee-1",
        session_id="session-1",
    )
    return ChatContext.from_request(request, user_query=query)


def extract_content(payloads: list[str]) -> str:
    content: list[str] = []
    for payload in payloads:
        if payload == "[DONE]":
            continue
        data = json.loads(payload)
        choices = data.get("choices", [])
        if choices:
            content.append(choices[0].get("delta", {}).get("content", ""))
    return "".join(content)


def display_content(db: FakeDatabase) -> str:
    return "".join(
        document["chunk_data"]["choices"][0]["delta"].get("content", "")
        for document in db.stream_chunks.documents
        if document["chunk_type"] == "token"
    )


def assert_single_final_done(payloads: list[str], *, expect_finish: bool = True) -> None:
    assert payloads.count("[DONE]") == 1
    assert payloads.index("[DONE]") == len(payloads) - 1
    if expect_finish:
        finish = json.loads(payloads[-2])
        assert finish["choices"][0]["finish_reason"] == "stop"


async def run_state(
    state: dict,
    *,
    formula_converter=None,
    math_stream_adapter_factory=None,
) -> tuple[list[str], FakeDatabase, FakeConversationWorkflow]:
    db = FakeDatabase()
    workflow = FakeConversationWorkflow(state)
    kwargs = {}
    if math_stream_adapter_factory is not None:
        kwargs["math_stream_adapter_factory"] = math_stream_adapter_factory
    orchestrator = ChatOrchestrator(workflow, **kwargs)

    def speech_factory():
        kwargs = {}
        if formula_converter is not None:
            kwargs["formula_converter"] = formula_converter
        return SpeechPipeline(**kwargs)

    service = ChatStreamService(
        orchestrator=orchestrator,
        database_provider=lambda: asyncio.sleep(0, result=db),
        speech_pipeline_factory=speech_factory,
    )
    payloads = [payload async for payload in service.stream(make_context())]
    return payloads, db, workflow


@pytest.mark.asyncio
async def test_general_answer_has_equal_but_separate_display_and_speech_streams():
    llm = FakeLLM(["普通回答。"])
    state = {
        "confidence": 0.8,
        "conversation_id": "conversation-1",
        "streaming_type": "langchain_llm",
        "streaming_llm": llm,
        "streaming_messages": [],
    }

    payloads, db, workflow = await run_state(state)

    assert display_content(db) == "普通回答。"
    assert extract_content(payloads) == "普通回答。"
    assert_single_final_done(payloads)
    assert len(workflow.saved_states) == 1
    assert [d["chunk_type"] for d in db.stream_chunks.documents].count("done") == 1
    assert [d["sequence"] for d in db.stream_chunks.documents] == list(
        range(1, len(db.stream_chunks.documents) + 1)
    )
    assert all(d["session_id"] == "session-1" for d in db.stream_chunks.documents)


@pytest.mark.asyncio
async def test_math_llm_keeps_latex_in_mongo_and_sends_only_voice_text_to_sse():
    math_chunks = [
        MathStreamChunk(reasoning="先分析函数。", model_name="fake-math-model"),
        MathStreamChunk(reasoning="再求顶点。", model_name="fake-math-model"),
        MathStreamChunk(content="推导如下。", model_name="fake-math-model"),
        MathStreamChunk(content="$$x^2=4$$\n", model_name="fake-math-model"),
        MathStreamChunk(content="因此完成。", model_name="fake-math-model"),
    ]
    state = {
        "confidence": 0.8,
        "conversation_id": "conversation-math",
        "streaming_type": "math_llm",
        "math_stream_config": MathModelConfig(
            base_url="http://math.example/v1",
            model_name="fake-math-model",
            api_key="test",
            max_tokens=128,
            temperature=0.6,
            top_p=0.95,
        ),
        "streaming_messages": [],
    }

    async def convert_formula(text: str) -> str:
        assert "$$x^2=4$$" in text
        return text.replace("$$x^2=4$$", "x 的平方等于 4")

    payloads, db, workflow = await run_state(
        state,
        formula_converter=convert_formula,
        math_stream_adapter_factory=lambda _config: FakeMathAdapter(math_chunks),
    )
    display = display_content(db)
    speech = extract_content(payloads)
    reasoning_documents = [
        document for document in db.stream_chunks.documents
        if document["chunk_type"] == "reasoning"
    ]

    assert "$$x^2=4$$" in display
    assert "x 的平方等于 4" in speech
    assert "$$x^2=4$$" not in speech
    assert "先分析函数" not in speech
    assert "再求顶点" not in speech
    assert display != speech
    assert [
        document["chunk_data"]["choices"][0]["delta"]["reasoning"]
        for document in reasoning_documents
    ] == ["先分析函数。", "再求顶点。"]
    assert workflow.saved_states[0]["final_answer"] == "推导如下。$$x^2=4$$\n因此完成。"
    assert [document["sequence"] for document in db.stream_chunks.documents] == list(
        range(1, len(db.stream_chunks.documents) + 1)
    )
    assert_single_final_done(payloads)


@pytest.mark.asyncio
async def test_direct_match_uses_prebuilt_teaching_script_tts_without_conversion():
    conversion_calls = 0

    async def unexpected_conversion(text: str) -> str:
        nonlocal conversion_calls
        conversion_calls += 1
        return text

    state = {
        "confidence": 0.95,
        "conversation_id": "conversation-direct",
        "final_answer": "展示：$$x^2=4$$",
        "direct_match": {
            "content_type": "teaching_script",
            "teaching_script_tts": "语音：x 的平方等于 4。",
        },
    }

    payloads, db, _ = await run_state(state, formula_converter=unexpected_conversion)

    assert "$$x^2=4$$" in display_content(db)
    assert "语音：x 的平方等于 4。" in extract_content(payloads)
    assert conversion_calls == 0
    assert_single_final_done(payloads)


@pytest.mark.asyncio
async def test_direct_match_without_tts_converts_display_answer_for_speech():
    conversion_inputs: list[str] = []

    async def convert_formula(text: str) -> str:
        conversion_inputs.append(text)
        return text.replace("$$x^2=4$$", "x 的平方等于 4")

    state = {
        "confidence": 0.95,
        "conversation_id": "conversation-direct-convert",
        "final_answer": "展示：$$x^2=4$$\n",
        "direct_match": {"content_type": "teaching_script"},
    }

    payloads, db, _ = await run_state(state, formula_converter=convert_formula)

    assert "$$x^2=4$$" in display_content(db)
    assert "x 的平方等于 4" in extract_content(payloads)
    assert conversion_inputs
    assert_single_final_done(payloads)


@pytest.mark.asyncio
async def test_math_concept_answer_also_uses_two_output_channels():
    llm = FakeLLM(["平方定义为：", "$$x^2=x\\times x$$\n"])
    state = {
        "confidence": 0.9,
        "classification_label": "math_concept_explain",
        "conversation_id": "conversation-concept",
        "streaming_type": "langchain_llm",
        "streaming_llm": llm,
        "streaming_messages": [],
    }

    async def convert_formula(text: str) -> str:
        return text.replace("$$x^2=x\\times x$$", "x 的平方等于 x 乘以 x")

    payloads, db, _ = await run_state(state, formula_converter=convert_formula)

    assert "$$x^2=x\\times x$$" in display_content(db)
    assert "x 的平方等于 x 乘以 x" in extract_content(payloads)
    assert "$$" not in extract_content(payloads)
    assert_single_final_done(payloads)


@pytest.mark.asyncio
async def test_error_is_saved_and_done_is_still_unique_and_last():
    db = FakeDatabase()
    workflow = FakeConversationWorkflow(error=RuntimeError("boom"))
    service = ChatStreamService(
        orchestrator=ChatOrchestrator(workflow),
        database_provider=lambda: asyncio.sleep(0, result=db),
    )

    payloads = [payload async for payload in service.stream(make_context())]

    error_payloads = [json.loads(p) for p in payloads if p != "[DONE]" and "error" in p]
    assert error_payloads[0]["error"]["code"] == "internal_error"
    assert [d["chunk_type"] for d in db.stream_chunks.documents].count("error") == 1
    assert_single_final_done(payloads, expect_finish=False)


@pytest.mark.asyncio
async def test_streaming_emits_first_semantic_chunk_before_llm_finishes():
    db = FakeDatabase()
    first_chunk = "这是可以立即输出的第一句话。"
    second_chunk = "这是随后输出的第二句话。"
    third_chunk = "这是最后输出的第三句话。"
    llm = FakeLLM([first_chunk, second_chunk, third_chunk], delay=0.03)
    workflow = FakeConversationWorkflow({
        "confidence": 0.8,
        "conversation_id": "conversation-slow",
        "streaming_type": "langchain_llm",
        "streaming_llm": llm,
        "streaming_messages": [],
    })
    service = ChatStreamService(
        orchestrator=ChatOrchestrator(workflow),
        database_provider=lambda: asyncio.sleep(0, result=db),
    )
    stream = service.stream(make_context())

    role_payload = await anext(stream)
    first_speech_payload = await anext(stream)

    assert json.loads(role_payload)["choices"][0]["delta"]["role"] == "assistant"
    assert json.loads(first_speech_payload)["choices"][0]["delta"]["content"] == first_chunk
    assert third_chunk not in llm.emitted
    assert first_chunk in display_content(db)

    remaining = [payload async for payload in stream]
    assert_single_final_done([role_payload, first_speech_payload, *remaining])


def test_request_resolver_preserves_extra_body_priority():
    request = OpenAIChatRequest(
        messages=[{"role": "user", "content": "问题"}],
        stream=True,
        team_id="body-team",
        user_id="body-user",
        employee_id="body-employee",
        user_name="body-name",
        head_url="body-avatar",
        session_id="body-session",
        extra_body={
            "channel_name": "employee_channel-team_channel-user_channel-employee_channel-name_channel-avatar",
            "team_id": "extra-team",
            "user_id": "extra-user",
            "employee_id": "extra-employee",
            "user_name": "extra-name",
            "head_url": "extra-avatar",
            "session_id": "extra-session",
        },
    )

    context = RequestResolver().resolve(request)

    assert context.team_id == "extra-team"
    assert context.user_id == "extra-user"
    assert context.employee_id == "extra-employee"
    assert context.user_name == "extra-name"
    assert context.head_url == "extra-avatar"
    assert context.session_id == "extra-session"
    assert context.channel_name.startswith("employee_")


def test_request_resolver_uses_channel_fields_only_for_missing_values():
    request = OpenAIChatRequest(
        messages=[{"role": "user", "content": "问题"}],
        stream=True,
        team_id=None,
        user_id="",
        employee_id="",
        user_name=None,
        head_url=None,
        extra_body={
            "channel_name": "employee_team42_user42_employee42_name42_avatar42",
        },
    )

    context = RequestResolver().resolve(request)

    assert context.team_id == "team42"
    assert context.user_id == "user42"
    assert context.employee_id == "employee42"
    assert context.user_name == "name42"
    assert context.head_url == "avatar42"


def test_stream_chunk_documents_keep_existing_external_schema():
    expected = {
        "chunk_id",
        "conversation_id",
        "session_id",
        "user_id",
        "employee_id",
        "chat_id",
        "chunk_type",
        "chunk_data",
        "sequence",
        "timestamp",
        "created_at",
    }
    # Kept explicit so accidental schema expansion/removal is caught during refactors.
    from app.models.database import StreamChunkModel

    assert set(StreamChunkModel.model_fields) == expected


def test_deterministic_math_fallback_never_leaks_latex_to_tts():
    speech = basic_math_to_voice(r"$$x^2+\frac{2}{3}=4$$")

    assert "x 的平方" in speech
    assert "加" in speech
    assert "2 除以 3" in speech
    assert "等于" in speech
    assert "$" not in speech
    assert "\\" not in speech


@pytest.mark.asyncio
async def test_speech_pipeline_ignores_reasoning_events():
    speech = SpeechPipeline()
    events = [
        event async for event in speech.handle(
            AnswerEvent(event_type="reasoning", content="这是不可朗读的思考过程。")
        )
    ]

    assert events == []


@pytest.mark.asyncio
async def test_math_reasoning_display_feature_flag_keeps_answer_but_skips_mongo_reasoning(
    monkeypatch,
):
    monkeypatch.setenv("MATH_REASONING_DISPLAY_ENABLED", "false")
    state = {
        "confidence": 0.8,
        "conversation_id": "conversation-math-flag-off",
        "streaming_type": "math_llm",
        "math_stream_config": _math_test_config(),
        "streaming_messages": [],
    }
    chunks = [
        MathStreamChunk(reasoning="不应展示的思考。"),
        MathStreamChunk(content="正式答案。"),
    ]

    payloads, db, workflow = await run_state(
        state,
        math_stream_adapter_factory=lambda _config: FakeMathAdapter(chunks),
    )

    assert "正式答案。" in display_content(db)
    assert "正式答案。" in extract_content(payloads)
    assert not any(document["chunk_type"] == "reasoning" for document in db.stream_chunks.documents)
    assert workflow.saved_states[0]["final_answer"] == "正式答案。"


@pytest.mark.asyncio
async def test_math_reasoning_is_persisted_before_the_math_stream_finishes():
    reasoning_saved = asyncio.Event()
    release_content = asyncio.Event()

    class ObservingCollection(FakeCollection):
        async def insert_one(self, document: dict):
            result = await super().insert_one(document)
            if document["chunk_type"] == "reasoning":
                reasoning_saved.set()
            return result

    class PausingMathAdapter:
        async def stream(self, _messages):
            yield MathStreamChunk(reasoning="先观察函数。", model_name="fake-math-model")
            await release_content.wait()
            yield MathStreamChunk(content="正式答案。", model_name="fake-math-model")

    db = FakeDatabase()
    db.stream_chunks = ObservingCollection()
    workflow = FakeConversationWorkflow({
        "confidence": 0.8,
        "conversation_id": "conversation-live-reasoning",
        "streaming_type": "math_llm",
        "math_stream_config": _math_test_config(),
        "streaming_messages": [],
    })
    service = ChatStreamService(
        orchestrator=ChatOrchestrator(
            workflow,
            math_stream_adapter_factory=lambda _config: PausingMathAdapter(),
        ),
        database_provider=lambda: asyncio.sleep(0, result=db),
    )
    stream = service.stream(make_context())

    await anext(stream)  # role
    next_sse = asyncio.create_task(anext(stream))
    await asyncio.wait_for(reasoning_saved.wait(), timeout=1)

    assert any(item["chunk_type"] == "reasoning" for item in db.stream_chunks.documents)
    assert not any(item["chunk_type"] == "token" for item in db.stream_chunks.documents)

    release_content.set()
    assert "正式答案。" in extract_content([await next_sse])
    remaining = [payload async for payload in stream]
    assert_single_final_done(remaining)


def _math_test_config() -> MathModelConfig:
    return MathModelConfig(
        base_url="http://math.example/v1",
        model_name="fake-math-model",
        api_key="test",
        max_tokens=128,
        temperature=0.6,
        top_p=0.95,
    )
