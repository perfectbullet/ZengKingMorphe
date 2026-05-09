# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A LangGraph-based conversational AI service providing RAG (Retrieval-Augmented Generation), multi-turn dialogue, knowledge base management, and web search for digital employee interactions. Built with FastAPI, supporting multiple LLM backends with intelligent routing.

**Tech Stack**: FastAPI + LangGraph + DeepSeek/Qwen/Phi-4 (Ollama/OpenAI/vLLM) + ChromaDB + ElasticSearch + MongoDB + Tavily + RAGAnything

**Service Architecture**:
- **ai-service** - FastAPI backend (port 8100 in Docker, 8000 local)
- **mongodb** - Document metadata and conversations (external, not in docker-compose)
- **chroma** - Vector embeddings (port 8200)
- **elasticsearch** - BM25 keyword search (port 9200)

---

## Essential Development Commands

### Python Environment (Critical!)
**ALL Python commands MUST use the project virtual environment** - dependencies are installed in conda environment `morphe`.

```bash
# Activate conda environment
conda activate morphe

# Or run directly
/home/zj/miniconda3/envs/morphe/bin/python script.py
```

### Docker Commands

```bash
# Start all services
docker-compose up -d

# Start databases only (for local AI service development)
docker-compose up -d elasticsearch chroma

# Check service status
docker-compose ps

# View ai-service logs
docker-compose logs -f ai-service

# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

### Local Development (WSL)

**Important**: This project uses `.env-local` for local development (auto-detected by `config.py`).

```bash
# Start databases only (for local AI service development)
docker-compose up -d elasticsearch chroma

# Run AI service locally
cd ai-service
conda activate morphe
uvicorn main:app --reload --port 8000

# Or use the startup script
./start_ai_service.sh start     # Start
./start_ai_service.sh status    # Status
./start_ai_service.sh logs      # View logs
./start_ai_service.sh stop      # Stop
```

Access API docs at http://localhost:8000/docs

### Testing

```bash
cd ai-service
conda activate morphe

# Run all tests (pytest)
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_web_search.py -v

# Run with coverage
pytest --cov=app --cov-report=html

# Scripts require PYTHONPATH=. when run from ai-service/
PYTHONPATH=. python scripts/test_ollama_reranker.py
PYTHONPATH=. python scripts/test_rag_e2e.py --query-only --kb-id kb_5f2a02bd5dfe --query "test query"
PYTHONPATH=. python scripts/stream_client.py --query "你好"
```

---

## Architecture Overview

### LangGraph Conversation Workflow

The core conversation engine is a **9-node StateGraph** ([ai-service/app/services/conversation_service.py](ai-service/app/services/conversation_service.py)):

**Flow**: `load_employee_config → load_session_context → input_validation → classify_query_type → [conditional branches] → check_math_problem → evaluate_complexity → web_search → generate_answer → save_conversation`

**Node count**: 9 nodes (simplified from 17). Removed: intent_recognition, knowledge_retrieval, grade_documents, compress_context, match_faq, rewrite_query. RAG retrieval is now handled by RAGAnything.

**Key Routing Logic**:
- **classify_query_type** → `greeting` (direct to generate_answer), `realtime` (→ web_search → generate_answer), `normal` (→ check_math_problem)
- **check_math_problem** → `math` (skip complexity eval, direct to generate_answer with Phi-4 LLM), `normal` (→ evaluate_complexity → generate_answer)
- RAGAnything handles all RAG retrieval within `generate_answer`

**Triple LLM Pattern**:
- `self.local_llm` - ChatOllama for fast, simple responses (greetings, short queries)
- `self.remote_llm` - ChatOpenAI (SiliconFlow) for complex queries and RAG
- `get_phi4_streaming_llm()` - Dynamic vLLM instance for math problems (configurable via `PHI4_*` env vars)

**State Management**: `ConversationState` TypedDict with ~40 fields in [conversation_state.py](ai-service/app/services/conversation/conversation_state.py), including:
- Core: `user_query`, `user_id`, `session_id`, `employee_id`
- Config: `employee_config`, `faq_matched`
- RAG: `retrieved_docs`, `relevance_score`, `kb_used`
- Math: `is_math_problem`, `direct_match`
- Web search: `web_search_results`, `is_realtime_query`
- Streaming: `streaming_type` ("langchain_llm" or "raganything_stream"), `streaming_llm`, `raganything_query`, `raganything_mode`
- Performance: `node_timings`, `ttfb_ms`, `response_time_ms`
- LLM params: `llm_temperature`, `llm_top_p`, `llm_max_tokens`, etc.

**Conversation Subdirectory**: The workflow logic is split across:
- `conversation_service.py` - Workflow class, LLM initialization, graph building
- `conversation/conversation_state.py` - State TypedDict and constants (greeting keywords, sensitive words)
- `conversation/conversation_nodes.py` - All node implementations (ConversationNodes class)
- `conversation/conversation_helpers.py` - Node timing, LLM selection (`select_llm`), message building

### RAGAnything Integration

RAG retrieval is handled by [RAGAnything](https://github.com/xxx) via `raganything_wrapper.py`, replacing the previous custom hybrid search. RAGAnything provides:
- Knowledge graph + vector retrieval with streaming
- Multiple query modes: `hybrid`, `local`, `global`, `naive`
- Configured via `RAGAnythingConfig` with OpenAI-style LLM and embedding endpoints

### Sensitive Word Filtering

Default sensitive words loaded from `DEFAULT_SENSITIVE_WORDS.txt` at ai-service root. Custom per-employee sensitive words synced from Java platform via `thesaurus_sensitive_service.py`.

### Employee Sync Service

`employee_sync_service.py` handles fetching employee config from Java API (`EXTERNAL_EMPLOYEE_API_URL`) and syncing to MongoDB. For `employee_id="hutao"`, falls back to local test data in `outer_api_docs/`.

### MinerU PDF Parsing

Enhanced PDF parsing with caching and parallel processing ([ai-service/app/services/mineru_client.py](ai-service/app/services/mineru_client.py)):
- Auto-splitting large PDFs into 8-page chunks
- MD5 caching in MongoDB `mineru_cache` collection
- Job tracking in `mineru_jobs` collection

### Ollama Keep-Alive

Background service to prevent Ollama model unloading ([ai-service/app/services/ollama_keepalive.py](ai-service/app/services/ollama_keepalive.py)):
- Periodic requests at `OLLAMA_KEEP_ALIVE_INTERVAL` (default 180s, 0 to disable)

---

## Critical Conventions

### Logging Pattern

Use f-string formatting for Loguru logging ([ai-service/app/core/logging.py](ai-service/app/core/logging.py)):

```python
from app.core.logging import logger

logger.info(f"Chat message request, user_id={user.id}, query={query[:100]}")  # Truncate PII
logger.error(f"RAG search failed, error={e}", exc_info=True)  # Always exc_info=True for errors
```

**Rules**:
- Use f-string for all log formatting
- Truncate user queries to 100 chars (PII prevention)
- Error logs must include `exc_info=True`
- Never use keyword arguments (Loguru ignores them)

### API Response Schema

All endpoints follow unified structure ([ai-service/app/models/schemas.py](ai-service/app/models/schemas.py)):

```python
class ChatResponse(BaseModel):
    code: int = 200
    message: str = "success"
    data: ChatResponseData
```

### Streaming Implementation

**Streaming responses only** - there is no non-streaming chat endpoint.

Chat streaming uses SSE via `sse-starlette` ([ai-service/app/api/endpoints/chat.py](ai-service/app/api/endpoints/chat.py)):

**API Versions**:
- `/api/chat/v1/chat/completions` - Original chat completions
- `/api/chat/v2/chat/completions` - Enhanced with math textbook direct match and TTS support

**Event Types**: `user_query`, `role`, `token`, `status`, `done`, `error`

**WebSocket**: `/api/chat/ws/chunks` and `/api/chat/ws/view/chunks` for real-time monitoring.

**Math Textbook Direct Match (v2)**: When a math textbook query directly matches a document chunk, skips LLM generation and streams pre-generated answer. Uses `teaching_script_tts` field for voice-friendly output.

### Authentication Status

**Currently DISABLED**: Middleware exists at [ai-service/app/api/middleware/auth.py](ai-service/app/api/middleware/auth.py) but `settings.api_keys` list is empty.

---

## LLM Configuration

### LLM Routing Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `local_only` | Uses local Ollama for all queries | Offline, privacy, cost savings |
| `remote_only` | Uses external API (DeepSeek via SiliconFlow) | Best quality, complex tasks |
| `hybrid` | Auto-selects based on query complexity (default threshold: 7.0) | Balanced performance/cost |

### Switching Backends

**OpenAI-style API** (SiliconFlow/DeepSeek):
```bash
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.siliconflow.cn/v1
OPENAI_MODEL=deepseek-ai/DeepSeek-V3.1-Terminus
```

**Local Ollama**:
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

### Revise LLM for Voice Output

Converts math formulas to speech-friendly text. Configured via `REVISE_PROVIDER` (siliconflow or ollama). Prompt template at `prompts/数学公式口语化讲解.txt`.

---

## Integration Points

### Java Platform Integration

AI service calls Java backend via `JAVA_API_BASE_URL`:
- Employee config validation: `POST /api/ai/employee/config`
- Employee sync via `EmployeeSyncService`

### External APIs

- **Tavily** (`tavily_api_key`) - Web search
- **MinerU** (`mineru_api_url`) - Enhanced PDF parsing
- **OpenAI/SiliconFlow** - LLM + Embeddings

---

## Key Files Reference

| Component | Path | Description |
|-----------|------|-------------|
| LangGraph workflow | [ai-service/app/services/conversation_service.py](ai-service/app/services/conversation_service.py) | 9-node StateGraph, LLM init, graph building |
| Workflow nodes | [ai-service/app/services/conversation/conversation_nodes.py](ai-service/app/services/conversation/conversation_nodes.py) | All node implementations |
| Workflow state | [ai-service/app/services/conversation/conversation_state.py](ai-service/app/services/conversation/conversation_state.py) | ConversationState TypedDict + constants |
| Workflow helpers | [ai-service/app/services/conversation/conversation_helpers.py](ai-service/app/services/conversation/conversation_helpers.py) | LLM selection, timing, message building |
| RAGAnything wrapper | [ai-service/app/services/raganything_wrapper.py](ai-service/app/services/raganything_wrapper.py) | RAGAnything integration with streaming |
| Math retrieval | [ai-service/app/services/math_textbook_retrieval.py](ai-service/app/services/math_textbook_retrieval.py) | Math textbook direct match |
| Employee sync | [ai-service/app/services/employee_sync_service.py](ai-service/app/services/employee_sync_service.py) | Java API employee data sync |
| Document service | [ai-service/app/services/document_service.py](ai-service/app/services/document_service.py) | Document CRUD and chunk management |
| MinerU client | [ai-service/app/services/mineru_client.py](ai-service/app/services/mineru_client.py) | PDF parsing with caching/chunking |
| MinerU API | [ai-service/app/api/endpoints/mineru.py](ai-service/app/api/endpoints/mineru.py) | Web interface for PDF processing |
| Chat endpoints | [ai-service/app/api/endpoints/chat.py](ai-service/app/api/endpoints/chat.py) | Streaming chat SSE (v1/v2) |
| Chat stream v1 | [ai-service/app/api/endpoints/chat_stream_v1.py](ai-service/app/api/endpoints/chat_stream_v1.py) | v1 stream generator |
| Chat stream v2 | [ai-service/app/api/endpoints/chat_stream_v2.py](ai-service/app/api/endpoints/chat_stream_v2.py) | v2 stream with math TTS support |
| WebSocket | [ai-service/app/api/endpoints/websocket.py](ai-service/app/api/endpoints/websocket.py) | Real-time stream chunks |
| WebSocket view | [ai-service/app/api/endpoints/websocket_view.py](ai-service/app/api/endpoints/websocket_view.py) | Frontend-optimized WebSocket |
| Session | [ai-service/app/api/endpoints/session.py](ai-service/app/api/endpoints/session.py) | Session CRUD |
| Employee | [ai-service/app/api/endpoints/employee.py](ai-service/app/api/endpoints/employee.py) | Digital employee management |
| Knowledge base | [ai-service/app/api/endpoints/knowledge_base_kb.py](ai-service/app/api/endpoints/knowledge_base_kb.py) | KB and document upload |
| Sensitive words | [ai-service/app/api/endpoints/thesaurus_sensitive.py](ai-service/app/api/endpoints/thesaurus_sensitive.py) | Sensitive word sync |
| Major thesaurus | [ai-service/app/api/endpoints/thesaurus_major.py](ai-service/app/api/endpoints/thesaurus_major.py) | Professional term sync |
| Configuration | [ai-service/app/core/config.py](ai-service/app/core/config.py) | Pydantic Settings with .env detection |
| Logging | [ai-service/app/core/logging.py](ai-service/app/core/logging.py) | Loguru-based structured logging |
| Database | [ai-service/app/core/database.py](ai-service/app/core/database.py) | MongoDB (Motor) connection |
| ElasticSearch | [ai-service/app/core/elasticsearch.py](ai-service/app/core/elasticsearch.py) | ES client (index prefix: `digital_employee_`) |
| Data models | [ai-service/app/models/database.py](ai-service/app/models/database.py) | MongoDB document models |
| API schemas | [ai-service/app/models/schemas.py](ai-service/app/models/schemas.py) | Pydantic request/response models |
| App entry | [ai-service/main.py](ai-service/main.py) | FastAPI app with lifespan management |
| Revise LLM | [ai-service/app/services/revise_llm.py](ai-service/app/services/revise_llm.py) | Math formula to speech text conversion |
| Ollama keep-alive | [ai-service/app/services/ollama_keepalive.py](ai-service/app/services/ollama_keepalive.py) | Prevents model unloading |
| VLLM model util | [ai-service/app/utils/get_vllm_first_model.py](ai-service/app/utils/get_vllm_first_model.py) | Dynamic vLLM model list retrieval |
| TTS formatter | [ai-service/app/utils/tts_formatter.py](ai-service/app/utils/tts_formatter.py) | TTS output formatting |
| Think tag buffer | [ai-service/app/utils/think_tag_buffer.py](ai-service/app/utils/think_tag_buffer.py) | Buffer for stripping think tags from LLM output |

---

## Common Development Patterns

### Adding New Conversation Nodes

1. Define the node method in `ConversationNodes` class ([conversation_nodes.py](ai-service/app/services/conversation/conversation_nodes.py)):
```python
async def my_new_node(self, state: ConversationState) -> ConversationState:
    async with time_node("my_new_node", state):
        state["some_field"] = "value"
        return state
```

2. Add the node to the graph in `_build_workflow()` ([conversation_service.py](ai-service/app/services/conversation_service.py)):
```python
graph.add_node("my_new_node", self.nodes.my_new_node)
```

3. Add edges to connect the node:
```python
graph.add_edge("previous_node", "my_new_node")
graph.add_edge("my_new_node", "next_node")
```

### Adding New API Endpoints

1. Create router in `app/api/endpoints/`:
```python
from fastapi import APIRouter
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.post("/my-endpoint")
async def my_endpoint():
    logger.info("My endpoint called")
    return {"code": 200, "message": "success", "data": {}}
```

2. Register router in `main.py`:
```python
from app.api.endpoints import my_feature
app.include_router(my_feature.router, prefix="/api/my-feature", tags=["My Feature"])
```

### Environment Variable Loading

The config system ([app/core/config.py](ai-service/app/core/config.py)) automatically detects:
1. `.env-local` for local development
2. `.env` for Docker/production
3. Custom path via `ENV_FILE` environment variable

**Never commit `.env-local`** - it's gitignored for local API keys.

### Debugging LangGraph Flow

The workflow can export graph structure to Mermaid format:
- Output directory: `graph_debug/` (configurable via `CRAG_GRAPH_DIR`)
- Enable/disable with: `CRAG_DUMP_GRAPH=0` or `1`

---

## Troubleshooting

### Common Issues

**Ollama responses are slow**: Model being unloaded from memory. Increase `OLLAMA_KEEP_ALIVE_INTERVAL` or verify keep-alive service is running.

**Embedding API errors**: Incorrect `EMBEDDING_BASE_URL` or `EMBEDDING_TYPE` mismatch. Verify embedding service is running.

**ChromaDB connection errors**: Chroma container not running. `docker-compose ps chroma` - should show port 8200.

**"No module named" errors**: Not using virtual environment. Always activate conda: `conda activate morphe` or use full path `/home/zj/miniconda3/envs/morphe/bin/python`.
