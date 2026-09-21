# `openai_chat_completions_v2` 重构前分析

## 分支与审阅范围

- 当前分支：`math-before-industrial`
- HTTP 入口：`app/api/endpoints/chat.py`
- v2 流生成器：`app/api/endpoints/chat_stream_v2.py`
- 前端 WebSocket：`app/api/endpoints/websocket.py`、`websocket_view.py`
- 工作流：`app/services/conversation_service.py` 及 `app/services/conversation/`
- 数学/语音工具：`revise_llm.py`、`latex.py`、`sentence_buffer.py`、`tts_formatter.py`
- 数据模型及 Mongo：`app/models/`、`app/core/database.py`
- 相关测试：`tests/`

任务描述列出的 `app/db/` 在本分支不存在；实际 MongoDB 连接与 collection 索引位于
`app/core/database.py`。

## 修改前调用链

```text
POST /api/chat/v2/chat/completions
  -> chat.openai_chat_completions_v2
     -> 解析 extra_body / channel_name 并覆盖请求字段
     -> rate_limit_middleware
     -> EventSourceResponse(generate_openai_stream_v2(request))
        -> get_database
        -> 直接构造并插入 StreamChunkModel(user_query / role)
        -> conversation_workflow.workflow.astream(..., stream_mode="updates")
        -> 根据 LangGraph state_update 中的 confidence 推断“可生成”
        -> 根据 final_answer / direct_match / streaming_type 选择输出路径
        -> 直接构造 Mongo chunk
        -> 直接构造并 yield OpenAI SSE JSON
        -> conversation_workflow.save_conversation
```

Mongo 中的 `stream_chunks` 由两个 WebSocket 端点轮询：

```text
stream_chunks
  -> /api/chat/ws/chunks
  -> /api/chat/ws/view/chunks
  -> user_id + employee_id + session_id 关联
  -> 前端
```

## 当前三类生成路径

### `direct_match` / 预生成文本

`final_answer` 有值并带 `direct_match`（或 `streaming_type == "text"`）时：

1. 规范化 LaTeX；
2. 按标点拆分并只写 Mongo，供 WebSocket 展示；
3. 写 Mongo `done`；
4. **提前 yield `[DONE]`**；
5. 若有 `teaching_script_tts`，再向 HTTP 流输出它；否则对完整展示答案调用
   voice conversion LLM 并输出；
6. 再输出 finish chunk；
7. 跳出循环后公共尾部再次写 Mongo `done`、输出 finish 和 `[DONE]`。

该路径同时存在 `[DONE]` 后仍有正文、重复 `done` 和重复 `[DONE]`。

当前代码事实：`ConversationState` 仍定义 `direct_match`，v2 仍处理该字段，但当前
`conversation/` 实现中没有找到赋值者。重构会保留兼容路径，避免旧数据源或后续恢复该能力时
破坏协议。

### `math_llm`

`generate_answer` 节点把数学模型和专用 messages 写入 state，v2 端点直接调用
`streaming_llm.astream(messages)`。每个原始数学 token 同时：

- 写入 Mongo，作为前端展示；
- 原样写入 HTTP SSE，作为 TTS 输入。

因此 SSE 当前会泄漏 Markdown/LaTeX，并没有形成独立的数学语音流。

### `langchain_llm`

通用问题和人工概念命中由 `generate_answer` 配置为 `langchain_llm`。v2 同样把每个原始 token
同时写 Mongo 和 SSE；两个输出目标没有职责隔离。

## 与任务描述的代码偏差

1. `app/db/` 不存在，Mongo 实现在 `app/core/database.py`。
2. 工作流注释明确 `knowledge_retrieval` 已删除；v2 仍按该过时节点名发送状态提示。
3. 当前工作流实际分支还包含 `direct_text` 和 `rag_stream`，不只有任务中列出的三类。
4. `direct_match` 数据结构仍被 v2 消费，但当前工作流没有生产代码。
5. `SentenceBuffer` 已能保护 `$...$`、`$$...$$`、`\(...\)`、`\[...\]`、
   `\begin...\end` 和裸 `\boxed{...}`，可直接用于语义块流式处理。
6. `revise_llm.convert_formula_to_voice` 已实现“只转换完整公式、普通文本保留”的能力，
   适合作为 Speech Pipeline 的数学 renderer；无需重写整套公式口语化。
7. `generate_openai_stream_v2` 的异常路径当前不输出 `[DONE]`，正常路径则可能输出多次。

## 预计新增文件

```text
ai-service/app/services/chat/__init__.py
ai-service/app/services/chat/answer_events.py
ai-service/app/services/chat/request_resolver.py
ai-service/app/services/chat/chat_orchestrator.py
ai-service/app/services/chat/display_pipeline.py
ai-service/app/services/chat/speech_pipeline.py
ai-service/app/services/chat/sse_writer.py
ai-service/app/services/chat/chat_stream_service.py
ai-service/app/repositories/__init__.py
ai-service/app/repositories/stream_chunk_repository.py
ai-service/tests/test_chat_v2_output_architecture.py
```

## 预计修改文件

- `app/api/endpoints/chat.py`：v2 endpoint 只保留解析、限流和 EventSourceResponse。
- `app/api/endpoints/chat_stream_v2.py`：保留兼容导出，把实现收敛为新协调服务入口。
- `docs/refactor_chat_v2_result.md`：记录最终架构、兼容性、实测结果和遗留问题。

`websocket.py`、`websocket_view.py`、`StreamChunkModel` 和 collection 名不会修改。

## 实施步骤

1. 先增加当前 Mongo/WebSocket schema 兼容性及新双通道行为测试，并确认测试在生产实现前失败。
2. 提取 `RequestResolver`，锁定 body、`channel_name`、`extra_body` 的现有覆盖优先级。
3. 提取 `StreamChunkRepository`，保持 collection、document 字段、sequence 和 chunk_id 格式不变。
4. 定义 `AnswerEvent`、`SpeechEvent`、`AnswerResult`、`ChatContext`，作为生成与交付边界。
5. 建立不依赖 LangGraph 节点名称的 `ChatOrchestrator`，统一兼容预生成、数学模型、通用模型和直接文本。
6. 建立只写 Mongo 的 `DisplayPipeline`，原始 Markdown/LaTeX 实时落库。
7. 建立 `SpeechPipeline`：用 `SentenceBuffer` 形成完整语义块，普通文本直接清理，完整公式调用现有 renderer，已有 `teaching_script_tts` 则直接使用。
8. 建立 `SSEWriter` 和 `ChatStreamService`；生产与语音转换通过有界队列并行，统一控制 finish 和唯一、末尾 `[DONE]`。
9. 简化 v2 endpoint，运行新测试、数学相关定向 pytest 和可用的流式冒烟测试。
10. 输出 `docs/refactor_chat_v2_result.md`，记录真实测试结果与未覆盖的环境依赖。

## 目标调用链

```text
POST /api/chat/v2/chat/completions
  -> RequestResolver
  -> ChatStreamService
     -> ChatOrchestrator -> AsyncIterator[AnswerEvent]
        -> DisplayPipeline -> StreamChunkRepository -> MongoDB -> WebSocket -> Frontend
        -> SpeechPipeline -> SpeechEvent -> SSEWriter -> HTTP SSE -> TTS
```

Display Pipeline 不返回正文；Speech Pipeline 不写正文到 `stream_chunks`。SSE 的终止顺序统一为：

```text
all speech chunks -> finish chunk -> [DONE]
```

