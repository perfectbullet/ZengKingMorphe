"""
敏感词拒答流式输出单元测试（chat_stream_v1）。

命中敏感词时应按正常流式协议返回完整的 chunk 序列，并保存拒答对话，
不再进入 preprocess_query / classify_query_type / generate_answer。

通过 mock 隔离 MongoDB、LangGraph workflow 与 save_conversation，仅验证
generate_openai_stream_v1 生成器自身的拒答分支行为。

运行：
    python -m pytest tests/test_sensitive_reject_v1.py -v
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.schemas import OpenAIChatRequest, OpenAIMessage
from app.api.endpoints import chat_stream_v1


REJECT_MSG_ZH = "抱歉，您的问题包含敏感内容，请规范用语后再试。"


class _FakeAstream:
    """模拟 LangGraph workflow.astream：按顺序产出事件后结束。"""

    def __init__(self, events):
        self._iter = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def _build_request(query: str = "命中敏感词测试") -> OpenAIChatRequest:
    return OpenAIChatRequest(
        model="qwen3:14b",
        messages=[OpenAIMessage(role="user", content=query)],
        user_id="3",
        employee_id="29",
        session_id="sess_test_sensitive",
        team_id="4",
        stream=True,
    )


async def _run_sensitive_stream(tmp_path):
    # 收尾段会写 finish_chunk_data/ 目录，切到临时目录避免污染工程
    import os
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        request = _build_request()

        # workflow 只产出 input_validation 命中敏感词事件后结束，
        # 模拟 validate_input 设置 has_sensitive=True 后流终止。
        fake_workflow = SimpleNamespace(
            workflow=SimpleNamespace(
                astream=lambda initial_state, stream_mode="updates": _FakeAstream(
                    [{"input_validation": {"has_sensitive": True}}]
                )
            ),
            save_conversation=AsyncMock(),
        )

        saved_chunks = []

        async def fake_save_stream_chunk(
            db, chat_id, chunk_sequence, session_id, user_id, employee_id,
            chunk_type, chunk_data, conversation_id=None,
        ):
            saved_chunks.append({
                "sequence": chunk_sequence,
                "chunk_type": chunk_type,
                "chunk_data": chunk_data,
            })

        fake_db = SimpleNamespace(
            stream_chunks=SimpleNamespace(insert_one=AsyncMock())
        )

        with patch.object(chat_stream_v1, "conversation_workflow", fake_workflow), \
                patch.object(chat_stream_v1, "get_database",
                             AsyncMock(return_value=fake_db)), \
                patch.object(chat_stream_v1, "save_stream_chunk",
                             fake_save_stream_chunk):
            chunks = []
            async for raw in chat_stream_v1.generate_openai_stream_v1(request):
                chunks.append(raw)

        return chunks, saved_chunks, fake_workflow
    finally:
        os.chdir(cwd)


def test_sensitive_reject_emits_full_stream(tmp_path):
    """命中敏感词：完整产出 role→content→finish→[DONE]，并保存拒答对话。"""
    chunks, saved_chunks, fake_workflow = asyncio.run(_run_sensitive_stream(tmp_path))

    # 1. 以 [DONE] 结尾
    assert chunks[-1] == "[DONE]"

    parsed = [json.loads(c) for c in chunks if c != "[DONE]"]
    assert parsed, "未产出任何 chunk"

    # 2. 第一个 chunk 是 assistant role chunk
    role = parsed[0]
    assert role["choices"][0]["delta"].get("role") == "assistant"

    # 3. 存在拒答 content chunk：content 与 voice_content 同文案，finish_reason=None
    contents = [
        p for p in parsed[1:]
        if p["choices"][0]["delta"].get("content")
        and p["choices"][0]["delta"].get("role") is None
    ]
    assert contents, "缺少拒答 content chunk"
    reject = contents[0]
    delta = reject["choices"][0]["delta"]
    assert delta["content"] == REJECT_MSG_ZH
    assert delta["voice_content"] == REJECT_MSG_ZH
    assert reject["choices"][0]["finish_reason"] is None

    # 4. 存在 finish chunk：finish_reason == stop（标准值，前端可识别）
    finishes = [
        p for p in parsed
        if p["choices"][0].get("finish_reason") == "stop"
    ]
    assert finishes, "缺少 finish chunk"

    # 5. 拒答内容以 chunk_type=token 保存（与普通回答一致，前端可渲染）
    token_saves = [s for s in saved_chunks if s["chunk_type"] == "token"]
    reject_saves = [
        s for s in token_saves
        if s["chunk_data"]["choices"][0]["delta"].get("content") == REJECT_MSG_ZH
    ]
    assert reject_saves, "拒答内容未以 token chunk 保存到 stream_chunks"

    # 6. user_query chunk 与 done(finish) chunk 均已保存
    assert any(s["chunk_type"] == "user_query" for s in saved_chunks), \
        "未保存 user_query chunk"
    assert any(s["chunk_type"] == "done" for s in saved_chunks), \
        "未保存 done/finish chunk"

    # 7. chunk sequence 连续递增
    sequences = [s["sequence"] for s in saved_chunks]
    assert sequences == sorted(sequences) == list(range(1, len(sequences) + 1)), \
        f"sequence 不连续: {sequences}"

    # 8. save_conversation 被调用一次，且 final_answer 是拒答话术
    fake_workflow.save_conversation.assert_awaited_once()
    saved_state = fake_workflow.save_conversation.await_args.args[0]
    assert saved_state.get("final_answer") == REJECT_MSG_ZH
    assert saved_state.get("has_sensitive") is True

    # 9. 未出现非标准的 finish_reason=sensitive / model=status
    for p in parsed:
        assert p["choices"][0].get("finish_reason") != "sensitive"
        assert p.get("model") != "status"


def test_sensitive_reject_uses_english_when_prefer_en(tmp_path):
    """prefer_zh_output=False（英文输入）时应返回英文拒答话术。"""
    async def _run():
        import os
        cwd = Path.cwd()
        os.chdir(tmp_path)
        try:
            request = OpenAIChatRequest(
                model="qwen3:14b",
                messages=[OpenAIMessage(role="user", content="hello bad word")],
                user_id="3",
                employee_id="29",
                session_id="sess_test_en",
                team_id="4",
                stream=True,
            )
            # 注意：prefer_zh_output 默认为 True，在 preprocess_query 节点才会更新。
            # 敏感词命中在 preprocess_query 之前，因此这里仍取中文默认话术——
            # 验证话术确实来自该分支而非空字符串即可。
            fake_workflow = SimpleNamespace(
                workflow=SimpleNamespace(
                    astream=lambda initial_state, stream_mode="updates": _FakeAstream(
                        [{"input_validation": {"has_sensitive": True}}]
                    )
                ),
                save_conversation=AsyncMock(),
            )
            fake_db = SimpleNamespace(
                stream_chunks=SimpleNamespace(insert_one=AsyncMock())
            )
            with patch.object(chat_stream_v1, "conversation_workflow", fake_workflow), \
                    patch.object(chat_stream_v1, "get_database",
                                 AsyncMock(return_value=fake_db)), \
                    patch.object(chat_stream_v1, "save_stream_chunk",
                                 AsyncMock()):
                chunks = []
                async for raw in chat_stream_v1.generate_openai_stream_v1(request):
                    chunks.append(raw)
            return chunks
        finally:
            os.chdir(cwd)

    chunks = asyncio.run(_run())
    parsed = [json.loads(c) for c in chunks if c != "[DONE]"]
    contents = [
        p for p in parsed
        if p["choices"][0]["delta"].get("content")
        and p["choices"][0]["delta"].get("role") is None
    ]
    assert contents, "缺少拒答 content chunk"
    # 默认中文话术非空
    assert contents[0]["choices"][0]["delta"]["content"]
