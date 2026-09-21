# `openai_chat_completions_v2` 流式输出重构结果

## 1. 修改前架构

旧版 `generate_openai_stream_v2()` 同时负责：

- 构造工作流 state 并观察 LangGraph 节点；
- 选择数学模型或通用模型；
- 直接构造 `StreamChunkModel` 并写 MongoDB；
- 直接构造 OpenAI chunk 并写 HTTP SSE；
- 教材脚本的 TTS 选择和二次 LLM 转换；
- conversation 保存；
- finish 与 `[DONE]` 控制。

数学模型的同一个原始 token 会同时进入 MongoDB 和 SSE，导致 LaTeX 直接进入 TTS。
教材直命中路径还会先发送 `[DONE]`，随后继续发送语音 token，并在公共尾部再次发送 finish、
Mongo `done` 和 `[DONE]`。

## 2. 修改后架构

```text
POST /api/chat/v2/chat/completions
  -> RequestResolver
  -> ChatStreamService
     -> ChatOrchestrator
        -> ConversationWorkflow
        -> AsyncIterator[AnswerEvent]
     -> bounded async queue / Output Dispatcher
        |-> DisplayPipeline -> StreamChunkRepository -> MongoDB
        |                                      -> WebSocket -> Frontend
        `-> SpeechPipeline -> SpeechEvent -> SSEWriter -> HTTP SSE -> TTS
```

`ChatOrchestrator` 只把工作流结果适配为 `AnswerEvent`，不构造 Mongo document、SSE JSON 或
WebSocket 消息。`ChatStreamService` 通过有界异步队列让答案生成/展示落库与语音转换并行，队列
同时提供背压，避免慢速口语化造成无限内存增长。

正常终止顺序统一为：

```text
全部 SpeechEvent -> OpenAI finish chunk -> [DONE]
```

`[DONE]` 只在协调器最外层发送一次，并且是最后一个业务事件。异常路径为 error chunk 后紧跟
唯一 `[DONE]`。

## 3. 新增文件

### `app/services/chat/answer_events.py`

定义生成与交付之间的协议对象：

- `ChatContext`：解析后的请求身份及 stream 元数据；
- `AnswerEvent`：`start/status/content/done/error` 事件载体；
- `SpeechEvent`：TTS-ready 文本片段；
- `AnswerResult`：最终答案聚合对象。

### `app/services/chat/request_resolver.py`

从 endpoint 提取 `extra_body`、`channel_name`、`team_id`、`user_id`、`employee_id`、
`user_name`、`head_url`、`session_id` 优先级解析。`extra_body` 中的直接字段保持最高优先级，
并通过 request copy 避免修改输入对象。

### `app/services/chat/chat_orchestrator.py`

运行现有 `ConversationWorkflow`，根据 state 中的输出能力字段选择预生成回答、数学模型、通用
模型或兼容 fallback，并持续产生 `AnswerEvent`。它不依赖 `knowledge_retrieval`、
`generate_answer` 等具体 LangGraph 节点名。

### `app/services/chat/display_pipeline.py`

只负责展示通道：把原始 Markdown/LaTeX 的 `AnswerEvent` 写入 repository。`handle()` 返回
`None`，不会产生 HTTP 正文。

### `app/services/chat/speech_pipeline.py`

只负责语音通道：

- 使用 `SentenceBuffer` 等待完整语义块和完整公式；
- 普通文本只做 Markdown/TTS 清理，不调用二次 LLM；
- 完整 LaTeX/数学表达式调用现有 `revise_llm` renderer；
- `teaching_script_tts` 已存在时直接分段输出，不再次转换；
- 输出 `SpeechEvent`，不操作 FastAPI Response，也不写 Mongo 正文。

### `app/services/chat/sse_writer.py`

唯一负责把 `SpeechEvent` 序列化为 OpenAI-compatible role/content/finish/error JSON。

### `app/services/chat/chat_stream_service.py`

统一协调生命周期、双通道并发、错误处理、Mongo display done、SSE finish 与唯一 `[DONE]`。

### `app/repositories/stream_chunk_repository.py`

集中构造和插入 `StreamChunkModel`，保留原 collection、字段、sequence 和
`{chat_id}_chunk_{sequence}` ID 格式。

### `tests/test_chat_v2_output_architecture.py`

覆盖普通回答、数学模型、教材 TTS 命中/未命中、数学概念、异常、首包实时性、字段解析、
Mongo schema、sequence 和数学转换 fallback。

## 4. 修改文件

### `app/api/endpoints/chat.py`

`openai_chat_completions_v2()` 现只负责 RequestResolver、限流、精简日志和
`EventSourceResponse`。endpoint 不再了解数学模型、LaTeX、Mongo insert、教材脚本或 TTS 转换。

### `app/api/endpoints/chat_stream_v2.py`

原 697 行混合实现替换为兼容入口。历史 import path 和
`generate_openai_stream_v2(OpenAIChatRequest)` 调用方式仍可用，同时也接受已解析的
`ChatContext`。

### `app/services/revise_llm.py`

保留原公式转换 LLM，增加确定性的 common-LaTeX/math fallback。转换模型超时、失败或返回
残余公式符号时，仍会去除 `$`、反斜杠和常见运算符语法，避免原始 LaTeX 泄漏到 TTS。

### `app/utils/sentence_buffer.py`

修复 `_merge_punctuation_segments()` 在已有正文后遇到独立标点时未推进游标、可能无限循环的
问题。语义块和公式边界检测规则保持不变。

未修改 `websocket.py`、`websocket_view.py`、`StreamChunkModel` 或 Mongo collection schema。

## 5. Display 流程

```text
AnswerEvent(content: original Markdown/LaTeX)
  -> DisplayPipeline.handle()
  -> StreamChunkRepository.save(chunk_type="token")
  -> MongoDB stream_chunks
  -> /api/chat/ws/view/chunks
  -> Frontend
```

数学公式原文（例如 `$$x^2=4$$`）逐 token/片段实时落库。Speech Pipeline 产生的“x 的平方
等于 4”不会作为正文 token 回写 `stream_chunks`。

## 6. Speech 流程

```text
AnswerEvent
  -> SentenceBuffer / SpeechPipeline
  -> Markdown 清理或 MathSpeechRenderer
  -> SpeechEvent
  -> SSEWriter
  -> POST /api/chat/v2/chat/completions 的 HTTP SSE
  -> TTS
```

普通问题虽然通常 display 与 speech 文本相同，仍经过独立的 pipeline 和 sink。

## 7. Math Streaming

数学模型仍由 `astream()` 实时产生 token。每个 token 立即交给 Display Pipeline 写 Mongo；同一
事件也进入有界队列，由 Speech Pipeline 的 `SentenceBuffer` 累积到完整句子或完整公式后再
转换：

```text
math_llm token
  |-> original LaTeX -> Mongo display
  `-> semantic buffer -> complete formula -> voice conversion -> SSE speech
```

转换按语义块进行，不等待完整 answer。测试中的慢速 LLM 在第三个 chunk 尚未生成前，第一段
已写入 Mongo 并从 SSE 输出，防止退化成整段后处理。普通中文段落不调用公式口语化模型。

## 8. 接口兼容性

保持不变：

- `POST /api/chat/v2/chat/completions`；
- `OpenAIChatRequest` 请求结构及非流式 501 行为；
- TTS 消费的 OpenAI-compatible SSE JSON 结构；
- `/api/chat/ws/view/chunks` 和 `/api/chat/ws/chunks`；
- `user_id + employee_id + session_id` WebSocket 关联方式；
- `stream_chunks` collection 和 `StreamChunkModel` 外部字段；
- 前端继续只从 Mongo/WebSocket 获取展示内容。

语义修正：v2 SSE 的正文现在只代表 speech/TTS 文本，不再承载前端 LaTeX 展示文本。

## 9. 测试结果

### 新架构回归测试

```text
python -m pytest -q tests/test_chat_v2_output_architecture.py
11 passed in 3.22s
```

测试明确断言：

- 每个正常流的 `[DONE]` 数量为 1 且位置为最后；
- `[DONE]` 前一项为 finish chunk；
- 数学 display 保留 LaTeX，speech 不含原始 LaTeX；
- `teaching_script_tts` 命中时不调用转换器；
- 慢速 LLM 未生成末段时已经产生 Mongo 和 SSE 首段；
- error 写 Mongo，SSE error 后仍只有一个末尾 `[DONE]`。

### 数学链路与相关工具定向测试

```text
71 passed, 3 skipped in 3.30s
```

覆盖新架构、LaTeX、SentenceBuffer、上下文数学题、分类后题干转换、数学概念路由及
`revise_llm` 离线单元类。3 个 skip 为原测试内显式跳过的 provider 场景。

### 较宽离线回归集

```text
183 passed, 3 skipped, 1 failed in 3.62s
```

唯一失败为既有 `tests/test_think_tag_buffer.py::test_think_tag_sequence`：实现返回 `"\n\n"`，
旧测试期望 `None`，与本次 v2 改动无关。

### 全测试收集

`pytest --collect-only -q tests` 收集到 229 个测试，但被 4 个既有问题中止：

- `test_concept_retrieval_service.py` 第 194 行已有 SyntaxError；
- `test_greeting_import.py` 导入已不存在的 `GREETING_KEYWORDS`；
- `test_rag_baseline.py`、`test_rag_batch.py` 缺少 `llama_rag_sdk`。

`tests/test_api.py` 另有 4 个既有失败：当前 httpx 已不支持测试使用的
`AsyncClient(app=...)` 参数。

流式冒烟测试未执行：`http://localhost:8100/health` 当前连接被拒绝，没有已启动的本地服务。

静态验证：相关文件 `ruff check` 与 `git diff --check` 均通过。

## 10. 遗留问题

1. 当前分支的工作流只保留 `direct_match` state 字段，未发现实际生产该字段的节点；本次保留
   完整兼容路径和测试，但无法对真实教材数据源做在线联调。
2. `rag_stream` 不属于本次 Output Architecture 范围；v2 继续保持原行为，回退到通用流式 LLM。
   若要让 v2 直接消费 LightRAG stream，应单独增加 RAG AnswerEvent adapter。
3. 公式口语化仍依赖配置的 voice conversion LLM 处理复杂表达式；确定性 fallback 只覆盖常见
   LaTeX/运算符，作用是保证失败时不泄漏协议符号，不替代高质量复杂公式朗读。
4. 项目全量 pytest 的既有 collection、httpx 兼容和外部依赖问题没有在本次输出重构中扩大修复。
5. 本机 8100 服务未启动，因此 Mongo/WebSocket/TTS 的真实环境联调仍需在依赖服务可用后执行。
