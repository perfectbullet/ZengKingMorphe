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
OLLAMA_MODEL=qwen2.5:7b
```

### Phi-4 Math Model (vLLM)

```bash
PHI4_ENABLED=true
PHI4_BASE_URL=http://192.168.8.235:8000/v1
PHI4_TEMPERATURE=0.0
PHI4_MAX_TOKENS=16384
```

### Embedding Configuration

```bash
EMBEDDING_TYPE=openai_style
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
EMBEDDING_BASE_URL=http://localhost:50009
```

### Revise LLM (Voice Output)

Converts math formulas to speech-friendly text. Configured via `REVISE_PROVIDER` (siliconflow or ollama). Prompt template at `prompts/数学公式口语化讲解.txt`.

## Triple LLM Pattern in Code

Defined in `ai-service/app/services/conversation_service.py`:

- `self.local_llm` - ChatOllama for fast, simple responses (greetings, short queries)
- `self.remote_llm` - ChatOpenAI (SiliconFlow) for complex queries and RAG
- `get_phi4_streaming_llm()` - Dynamic vLLM instance for math problems (configurable via `PHI4_*` env vars)

## Query Routing Logic

The routing happens in two nodes:

1. **classify_query_type**:
   - `greeting` -> direct to generate_answer
   - `realtime` -> web_search -> generate_answer
   - `normal` -> check_math_problem

2. **check_math_problem**:
   - `math` -> skip complexity eval, direct to generate_answer with Phi-4 LLM
   - `normal` -> evaluate_complexity -> generate_answer

3. **evaluate_complexity**:
   - Score >= 7.0 -> remote_llm (DeepSeek)
   - Score < 7.0 -> local_llm (Ollama)

RAG retrieval is handled by RAGAnything within the `generate_answer` node, not by separate routing.
