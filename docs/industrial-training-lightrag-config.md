# industrial training LightRAG 配置梳理

本文记录当前工训 LightRAG 问答链路的入口、提示词位置，以及 LLM / RAG / rerank / threshold 相关参数。

## 1. 工训 LightRAG 链路入口

- `ai-service/app/services/rag_stream_wrapper.py`
- `ai-service/app/services/training_lightrag_wrapper.py`

当前 backend 由 `TRAINING_RAG_BACKEND` 控制，工训知识库默认走 `lightrag_file`。

当前已跑通的主链路是：

- `industrial_training_query`
- `rag_stream`
- `lightrag_file`
- `include_history=false`

`get_rag_stream()` 会统一进入 `training_lightrag_wrapper.py`。

## 2. 工训 LightRAG 提示词位置

位置：

- `ai-service/app/services/training_lightrag_wrapper.py`
- `_build_query_param()` 中 `user_prompt`
- `_build_system_prompt()` 中 `system_prompt`

当前 `user_prompt` 大意：

- 使用简体中文回答
- 只根据工训教材知识库作答
- 尽量给出章节路径或来源线索
- 资料不足时明确说明
- 当问题是“基本流程 / 制作流程”时，优先回答教材中的核心步骤
- 进阶效果 / 案例 / 延伸技法只作为补充，不列为主流程

当前 `system_prompt` 大意：

- 你是工业实训教材问答助手
- 只根据当前知识库回答
- 不要编造教材外知识

## 3. 通用 LLM 提示词位置

位置：

- `ai-service/app/services/conversation/conversation_helpers.py`
- `build_generation_messages()`

当前通用回答提示词逻辑大意：

- 有上下文时优先严格基于上下文回答
- 无上下文时允许基于模型训练知识正常作答
- 会统一追加语言约束

## 4. 分类器提示词位置

位置：

- `ai-service/app/services/query_classifier.py`
- `QueryClassifier.SYSTEM_PROMPT`
- `aclassify()` 中 user prompt

当前 `industrial_training_query` 主要不是由分类器直接输出，而是由：

- `ai-service/app/services/conversation/conversation_nodes.py`
- `finalize_classification()`

中的工训 domain gate 提升得到。

## 5. 数学提示词位置

位置：

- `ai-service/app/services/math_agent_service.py`
- `SYSTEM_PROMPTS`
- `direct / cot / tir`

## 6. 通用 LLM 参数

当前相关位置：

- `LLM_BASE_URL`
- `LLM_MODEL`
- `LLM_API_KEY`
- `settings.openai_temperature`
- `OPENAI_TEMPERATURE`
- `settings.complexity_threshold`

说明：

- `conversation_service.py` 中 `local_llm` 当前没有显式传 `temperature / top_p / max_tokens`
- `remote_llm` 当前显式传 `temperature=settings.openai_temperature`
- `OpenAIChatRequest` schema 默认 `temperature=0.7`、`top_p=0.9`、`max_tokens=None`
- 请求参数当前未必完整透传到 `ConversationWorkflow` 的 LangChain `ChatOpenAI`

TODO：

- 后续建议统一新增：
  - `GENERAL_LLM_TEMPERATURE`
  - `GENERAL_LLM_TOP_P`
  - `GENERAL_LLM_MAX_TOKENS`
- 并打通 `local_llm / remote_llm / chat_stream_v1` 的统一参数入口

本轮不改通用 LLM 默认行为。

## 7. 工训 LightRAG 参数

- `TRAINING_LIGHTRAG_WORKING_DIR`
- `TRAINING_RAG_BACKEND`
- `TRAINING_RAG_TOP_K`，默认 `12`
- `TRAINING_RAG_CHUNK_TOP_K`，默认 `4`
- `TRAINING_RAG_RESPONSE_TYPE`，默认 `Single Paragraph`
- `TRAINING_RAG_MAX_ENTITY_TOKENS`，默认 `6000`
- `TRAINING_RAG_MAX_RELATION_TOKENS`，默认 `8000`
- `TRAINING_RAG_MAX_TOTAL_TOKENS`，默认 `30000`
- `TRAINING_RAG_QUERY_MAX_TOKENS`，默认 `512`
- `TRAINING_RAG_ENABLE_RERANK`，默认 `false`
- `TRAINING_RAG_INCLUDE_REFERENCES`，默认 `true`
- `TRAINING_RAG_INCLUDE_HISTORY`，默认 `false`

## 8. Rerank 参数

推荐配置：

```env
TRAINING_RAG_ENABLE_RERANK=true
RERANK_BINDING=cohere
RERANK_MODEL=bge-reranker-m3
RERANK_BINDING_HOST=http://192.168.8.233:8091/rerank
RERANK_BASE_URL=http://192.168.8.233:8091
RERANK_BINDING_API_KEY=local-key
RERANK_ENABLE_CHUNKING=true
RERANK_MAX_TOKENS_PER_DOC=512
MAX_ASYNC_RERANK=4
RERANK_TIMEOUT=30
MIN_RERANK_SCORE=0.6
```

说明：

- `RERANK_BINDING_HOST` 优先于 `RERANK_BASE_URL`
- `RERANK_BINDING_HOST` 应配置完整 endpoint
- 不自动给 `RERANK_BASE_URL` 拼接 `/rerank`
- 用户已验证 `/rerank` 和 `/v1/rerank` 都成功，本项目建议使用 `/rerank`
- 本轮接入的是 LightRAG 内置 rerank，不在 wrapper 层自己再做一轮 rerank

## 9. Similarity threshold 参数

```env
COSINE_THRESHOLD=0.2
```

说明：

- 对应 LightRAG 的 `cosine_better_than_threshold`
- 当前本地 LightRAG 初始化签名还支持顶层 `cosine_threshold`
- 本轮显式使用默认值 `0.2`
- 后续可以手动调成 `0.25 / 0.3 / 0.35` 观察召回变化

## 10. 代理说明

建议内网 rerank 服务加入：

```env
NO_PROXY=localhost,127.0.0.1,::1,192.168.8.233,192.168.8.0/24
no_proxy=localhost,127.0.0.1,::1,192.168.8.233,192.168.8.0/24
```

当前用户本地测试经验是：访问内网模型 / rerank 服务时，最好显式去掉代理环境变量，避免反复重试或走错出口。
