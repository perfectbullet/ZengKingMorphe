# 数学模型 Reasoning 实时展示接入结果

## 修改文件

- `ai-service/app/services/chat/chat_orchestrator.py`：仅在 `streaming_type == "math_llm"` 时使用原始数学 SSE 流，生成 `AnswerEvent(reasoning)`；`final_answer` 仍然只累积 `content`。
- `ai-service/app/services/chat/display_pipeline.py`：将 reasoning 保存为 `chunk_type="reasoning"` 的 OpenAI-style Mongo chunk。
- `ai-service/app/services/conversation_service.py`：数学 ChatOpenAI 与 raw adapter 共用 `MathModelConfig` 解析。
- `ai-service/app/services/conversation/conversation_nodes.py`：数学分支保留原 `build_math_generation_messages()`，将共享配置放入 state，不再在此处创建会丢弃 reasoning 的 ChatOpenAI 流。
- `ai-service/app/services/conversation/conversation_state.py`：增加内部 `math_stream_config` state 字段。
- `ai-service/tests/test_chat_v2_output_architecture.py`：增加 reasoning 的 Display/Speech/feature-flag/实时持久化回归测试。

## 新增文件

- `ai-service/app/services/chat/math_reasoning_stream.py`
  - `MathModelConfig`：复用 `MATH_MODEL_BASE_URL`、`MATH_MODEL_NAME`、`MATH_MODEL_API_KEY`、`MATH_MAX_TOKEN`、`MATH_TEMPERATURE`、`MATH_TOP_P`。
  - `MathReasoningStreamAdapter`：仅负责 async raw SSE 请求、SSE 解析与 `MathStreamChunk` 产出。
  - `ReasoningBuffer`：按句末/换行、80 字符、约 200ms 进行轻量实时合并；首个 content 和流结束强制 flush。
- `ai-service/tests/test_math_reasoning_stream.py`：adapter SSE 协议与缓冲单测。

## 原始协议与真实采样

数学 vLLM 的真实 SSE 使用：

```text
delta.reasoning
delta.content
```

2026-09-21 使用 `nvidia/Qwen3.6-35B-A3B-NVFP4` 对“仅计算 1+1，并简短作答。”进行 raw SSE 探测：

- reasoning chunk：271 个、853 个字符、平均约 3.15 字符/块；首块是 `"Here"`。
- content chunk：2 个、3 个字符，顺序在 reasoning 之后。

这证明 raw chunk 非常细碎，因而采用即时小缓冲而非“一块一条 Mongo 写入”。请求显式包含：

```json
{"stream": true, "chat_template_kwargs": {"enable_thinking": true}}
```

## 数据流

```text
reasoning
  → AnswerEvent(event_type="reasoning")
  → DisplayPipeline
  → MongoDB stream_chunks (chunk_type="reasoning")
  → 既有 /api/chat/ws/view/chunks

content
  → AnswerEvent(event_type="content")
  → DisplayPipeline → MongoDB (chunk_type="token")
  → SpeechPipeline → HTTP SSE → TTS
```

`SpeechPipeline` 只处理 `content` 和 `status`，所以 reasoning 不会进入 TTS 或 HTTP SSE。`ChatStreamService` 也只把 `content` 纳入 `display_text`，保证 Conversation 的 `ai_response` 与最终答案没有 reasoning。

## Mongo 示例

```json
{
  "chunk_type": "reasoning",
  "chunk_data": {
    "id": "chatcmpl-a0fee31b5019cd74",
    "object": "chat.completion.chunk",
    "created": 1789984426,
    "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
    "choices": [{
      "index": 0,
      "delta": {"reasoning": "Here's a thinking process:"},
      "finish_reason": null
    }]
  }
}
```

未改变 `StreamChunkModel` 顶层 schema。reasoning 和 token 都经同一个 `StreamChunkRepository.save()`，共享单调递增的 `sequence`。

## WebSocket 示例

现有 endpoint 不按 `chunk_type` 过滤，保存后的 document 会原样推送：

```json
{
  "chunk_id": "chatcmpl-example_chunk_3",
  "chunk_type": "reasoning",
  "sequence": 3,
  "chunk_data": {
    "choices": [{"delta": {"reasoning": "先观察函数的开口方向。"}}]
  }
}
```

因此 `websocket_view.py` 无需修改。

## Streaming 证明

`test_math_reasoning_is_persisted_before_the_math_stream_finishes` 将 fake math stream 在第一个 reasoning 后暂停；该时刻断言 Mongo 已有 reasoning document、尚无 token document，之后才释放正式答案。生产路径同样在每个 raw chunk 到达时先执行 `DisplayPipeline.handle()`，而非等待整个模型输出。

## 测试结果

- `pytest -q tests/test_math_reasoning_stream.py tests/test_chat_v2_output_architecture.py tests/test_math_concept_intent_routing.py tests/test_context_resolution_math.py tests/test_latex_utils.py`
  - 31 passed。
- 真实 raw vLLM 探测：成功观察到 `delta.reasoning` 后跟 `delta.content`；ChatOpenAI compatibility layer 仍不保留 reasoning。
- `MATH_REASONING_DISPLAY_ENABLED=false`：有单测证明正式答案和 TTS 正常，Mongo 不保存 reasoning。
- 普通 `langchain_llm`、direct match 及 teaching-script TTS 的已有回归测试通过。

## 遗留问题 / 未完成的环境验收

已执行 `./start_ai_service.sh start`，但启动前依赖检查失败：MongoDB `192.168.100.233:27017`、Milvus `:19530`、Neo4j `:7687` 均不可达，启动脚本因此取消启动。

因此，本环境无法完成依赖 MongoDB 的真实服务端 `POST /api/chat/v2/chat/completions` + WebSocket + Mongo E2E 验收；没有将其标记为通过。恢复上述依赖后，应以同一组 `user_id`、`employee_id`、`session_id` 同时连接 WebSocket 并发送数学请求，确认 `reasoning → token → done` 的 sequence 与 HTTP SSE 不含 reasoning。
