# LLM Routing & Configuration

## LLM Routing Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `local_only` | Uses local Ollama for all queries | Offline, privacy, cost savings |
| `remote_only` | Uses external API (DeepSeek via SiliconFlow) | Best quality, complex tasks |
| `hybrid` | Auto-selects based on query complexity (default threshold: 7.0) | Balanced performance/cost |

## Backend Configuration

### OpenAI-style API (SiliconFlow/DeepSeek)

```bash
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.siliconflow.cn/v1
OPENAI_MODEL=deepseek-ai/DeepSeek-V3.1-Terminus
```

### Local Ollama

```bash
OLLAMA_BASE_URL=http://192.168.8.233:11434
OLLAMA_MODEL=qwen3:14b
```

### Math LLM (vLLM / Qwen / Phi)

通过环境变量直接配置（在 `conversation_service.py` 的 `get_math_streaming_llm()` 中读取）：

```bash
MATH_LLM_ENABLED=true
MATH_MODEL_BASE_URL=http://192.168.8.235:8000/v1
MATH_MODEL_NAME=                     # 留空则通过 vLLM 自动发现模型名
MATH_TEMPERATURE=0.6
MATH_TOP_P=0.95
MATH_MAX_TOKEN=10240
MATH_RUNTIME_MODE=direct             # direct / cot / tir（旧 llm 已删除，配置 llm 会报错）
```

`Qwen3-32B` 会忽略 `MATH_TEMPERATURE` / `MATH_TOP_P`，请求中不传
`temperature`、`top_p`、`top_k`、`min_p`，由模型服务端的 `generation_config.json`
决定采样参数。

**模型切换示例：**
- vLLM 自动发现（如 Phi-3）：只设 `MATH_MODEL_BASE_URL`，不设 `MATH_MODEL_NAME`
- Qwen Math 固定端点：`MATH_MODEL_BASE_URL=http://...` + `MATH_MODEL_NAME=/data/models/Qwen2.5-Math-1.5B-Instruct`

### Embedding Configuration

```bash
EMBEDDING_TYPE=openai_style
EMBEDDING_MODEL=BAAZ/bge-large-zh-v1.5
EMBEDDING_BASE_URL=http://localhost:50009
```

### Revise LLM (Voice Output)

Converts math formulas to speech-friendly text. Configured via `REVISE_PROVIDER` (siliconflow or ollama). Prompt template at `prompts/数学公式口语化讲解.txt`.

## Triple LLM Pattern in Code

Defined in `ai-service/app/services/conversation_service.py`:

- `self.local_llm` - ChatOllama for fast, simple responses (greetings, short queries)
- `self.remote_llm` - ChatOpenAI (SiliconFlow) for complex queries and RAG
- `get_math_streaming_llm()` - Dynamic LLM instance for math problems (configurable via `MATH_LLM_*` env vars)

## Query Routing Logic

The routing happens in two nodes:

1. **classify_query_type**:
   - `greeting` -> direct to generate_answer
   - `realtime` -> web_search -> generate_answer
   - `normal` -> check_math_problem

2. **check_math_problem**:
   - `math` -> skip complexity eval, direct to generate_answer with math LLM
   - `normal` -> evaluate_complexity -> generate_answer

3. **evaluate_complexity**:
   - Score >= 7.0 -> remote_llm (DeepSeek)
   - Score < 7.0 -> local_llm (Ollama)

RAG retrieval is handled by RAGAnything within the `generate_answer` node, not by separate routing.
