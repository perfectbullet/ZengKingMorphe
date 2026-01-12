# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A LangGraph-based conversational AI service providing RAG (Retrieval-Augmented Generation), multi-turn dialogue, knowledge base management, and web search for digital employee interactions. Built with FastAPI, supporting multiple LLM backends (OpenAI, Ollama), and using a three-tier storage architecture (ChromaDB for vectors, ElasticSearch for keywords, MongoDB for metadata).

**Tech Stack**: FastAPI + LangGraph + OpenAI/Ollama + ChromaDB + ElasticSearch + MongoDB + Tavily Web Search + MinerU PDF Parsing

**Service Architecture**:
- **ai-service** - FastAPI backend (port 8100 in Docker, 8000 local)
- **mongodb** - Document metadata and conversation records (port 27117)
- **chroma** - Vector embeddings (port 8101)
- **elasticsearch** - BM25 keyword search (port 9320)

---

## Essential Development Commands

### Python Environment (Critical!)
**ALL Python commands MUST use the project virtual environment** - dependencies are installed in `.venv` virtual environment.

**WSL/Ubuntu**:
```bash
# Activate virtual environment
source .venv/bin/activate

# Or run directly
/mnt/d/zenking_work/metahuman_work/ZengKingMorphe/.venv/bin/python script.py
```

### Docker Commands

```bash
# Start all services
docker-compose up -d

# Start databases only (for local AI service development)
docker-compose up -d mongodb elasticsearch chroma

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
docker-compose up -d mongodb elasticsearch chroma

# Run AI service locally
cd ai-service
source .venv/bin/activate
uvicorn main:app --reload --port 8000

# Access API docs at http://localhost:8000/docs
```

### Testing

```bash
cd ai-service
source .venv/bin/activate

# Run all tests (pytest)
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_web_search.py -v

# Run standalone test script (alternative method)
python tests/test_web_search.py

# MinerU PDF processing test
python tests/test_mineru_client.py process --file path/to/document.pdf --save-md

# RAG evaluation
python scripts/rag_evaluation_test.py
python scripts/e2e_rag_evaluation.py
```

---

## Architecture Overview

### LangGraph Conversation Workflow

The core conversation engine is a **16-node StateGraph** ([ai-service/app/services/conversation_service.py](ai-service/app/services/conversation_service.py)):

**Flow**: `load_employee_config → load_session_context → input_validation → evaluate_complexity → rewrite_query → check_realtime_query → match_faq → recognize_intent → knowledge_retrieval → grade_documents → rerank_documents → compress_context → web_search → generate_answer → verify_answer → save_conversation`

**Key Routing Logic**:
- **Realtime queries** (weather, news, stock prices) → bypass FAQ/RAG, direct to web search
- **FAQ matched** → skip RAG, generate answer directly
- **Greeting intent** → skip RAG, generate answer directly
- **Low relevance score** (< `settings.relevance_threshold`, default 0.6) → trigger web search as fallback
- **Otherwise** → RAG retrieval → document grading → reranking → context compression → LLM generation

**State Management**: `ConversationState` TypedDict with 28 fields flows through all nodes, including:
- Core: `user_query`, `user_id`, `session_id`, `employee_id`
- Config: `employee_config`, `faq_matched`
- RAG: `retrieved_docs`, `relevance_score`, `compressed_context`
- Web search: `web_search_results`, `is_realtime_query`
- Performance: `node_timings`, `ttf_ms`, `response_time_ms`

**Dual LLM Pattern**: Two separate LLM instances:
- `self.llm` - Answer generation (supports streaming)
- `self.grader_llm` - Document relevance scoring (forced JSON output mode via `format="json"` or `response_format={"type": "json_object"}`)

### Three-Tier RAG Architecture

Knowledge base documents stored across **3 databases** ([ai-service/app/services/rag_service.py](ai-service/app/services/rag_service.py)):

1. **ChromaDB** - Vector embeddings for semantic similarity search
2. **ElasticSearch** - BM25 keyword search for exact term matching
3. **MongoDB** - Document metadata, full content, and conversation records

**Hybrid Search Fusion**: Uses **RRF (Reciprocal Rank Fusion)**:
```python
rrf_score = 1 / (rank + k)  # where k=60 is the constant
```
- Combines vector search and keyword search results
- Reranks documents based on combined scores
- Default `top_k=5` for each search method

### MinerU PDF Parsing

Enhanced PDF parsing service with caching and parallel processing ([ai-service/app/services/mineru_client.py](ai-service/app/services/mineru_client.py)):

- **Auto-splitting**: Large PDFs split into 8-page chunks (configurable via `mineru_pages_per_chunk`) for processing
- **MD5 caching**: Results cached in MongoDB `mineru_cache` collection to avoid reprocessing
- **Job tracking**: Processing status tracked in `mineru_jobs` collection with states (pending/processing/completed/failed)
- **Web API**: `/api/mineru/jobs`, `/api/mineru/jobs/{id}/markdown`, `/api/mineru/view`
- **Chunk merging**: Automatically merges processed chunks back into complete markdown output

### Task Processing

Background task processor for async operations ([ai-service/app/services/task_processor.py](ai-service/app/services/task_processor.py)):
- Runs on startup via lifespan management in `main.py`
- Handles async document processing, embedding generation, etc.
- Stops gracefully on shutdown

### Ollama Keep-Alive

Background service to prevent Ollama model unloading ([ai-service/app/services/ollama_keepalive.py](ai-service/app/services/ollama_keepalive.py)):
- Sends periodic requests to Ollama API at `ollama_keep_alive_interval` (default 180s)
- Set to 0 to disable
- Critical for maintaining fast response times with local models

---

## Database Schema

### MongoDB Collections

**Core Collections**:
- `conversations` - Conversation records with metadata
- `sessions` - User session data
- `employees` - Digital employee configurations
- `knowledge_bases` - Knowledge base metadata
- `documents` - Document metadata and content
- `faq` - FAQ entries for matching
- `sensitive_words` - Sensitive word filter
- `professional_words` - Professional terminology

**MinerU Collections** (prefixed with `mineru_`):
- `mineru_cache` - MD5-based PDF parsing cache
- `mineru_jobs` - PDF processing job status tracking

**Streaming Collections**:
- `stream_chunks` - Stored streaming chunks for debugging/audit

### ChromaDB Collections

- `doc` - Document chunks with embeddings
- `faq` - FAQ embeddings for semantic matching
- Each collection has metadata filters: `kb_id`, `employee_id`, etc.

### ElasticSearch Indices

- `digital_employee_*` - BM25 keyword search index (prefix configurable via `ES_INDEX_PREFIX`)
- Supports full-text search on document content

---

## Critical Conventions

### Logging Pattern

Use structured logging via Loguru wrapper ([ai-service/app/core/logging.py](ai-service/app/core/logging.py)):

```python
from app.core.logging import logger

logger.info("Chat message request", user_id=user.id, query=query[:100])  # Truncate PII
logger.error("RAG search failed", error=str(e), exc_info=True)  # Always exc_info=True for errors
```

**Rules**:
- ✅ Use keyword arguments (not string interpolation)
- ✅ Truncate user queries to 100 chars (PII prevention)
- ✅ Error logs must include `exc_info=True`
- ❌ Never: `logger.info(f"User {user.id} asked: {query}")`

### API Response Schema

All endpoints follow unified structure ([ai-service/app/models/schemas.py](ai-service/app/models/schemas.py)):

```python
class ChatResponse(BaseModel):
    code: int = 200  # HTTP status code
    message: str = "success"  # Status message
    data: ChatResponseData  # Actual response data
```

**Key Schemas**:
- `ChatRequest`/`ChatResponse` - Chat endpoints
- `SourceAttribution` - Contains `rag_sources[]` (top 3) + `web_sources[]` (top 5)
- `StreamChunkResponse` - SSE chunks with `type` field (user_query/role/token/done/error)

### Streaming Implementation

**IMPORTANT**: This API supports **streaming responses only**. There is no non-streaming chat endpoint.

Chat streaming uses SSE (Server-Sent Events) via `sse-starlette` ([ai-service/app/api/endpoints/chat.py](ai-service/app/api/endpoints/chat.py)):

**Event Types**:
- `user_query` - Echoes the original query
- `role` - Message role (assistant)
- `token` - Individual LLM output tokens
- `done` - Stream completion (includes timing and sources)
- `error` - Error information

**TTFB Tracking**: Time To First Byte is tracked via `state["ttfb_ms"]` in the workflow.

**Chunk Storage**: Streaming chunks optionally stored in MongoDB `stream_chunks` collection for debugging.

**Non-Streaming Usage**: If you need the complete response without handling streaming, consume the stream internally:
```python
# Example for scripts/tests that need full response
response = requests.post(url, json=payload, stream=True)
full_content = ""
for line in response.iter_lines(decode_unicode=True):
    if not line or line.startswith(":"):
        continue
    if line == "data: [DONE]":
        break
    if line.startswith("data: "):
        line = line[6:]
        chunk_data = json.loads(line)
        if "choices" in chunk_data:
            delta = chunk_data["choices"][0].get("delta", {})
            if "content" in delta:
                full_content += delta["content"]
```

### Authentication Status

**Currently DISABLED**: Middleware exists at [ai-service/app/api/middleware/auth.py](ai-service/app/api/middleware/auth.py) but `settings.api_keys` list is empty.

---

## LLM Configuration

### Switching Backends

**Option 1: OpenAI-style API** (default: SiliconFlow/DeepSeek)
```bash
# .env-local file
USE_OLLAMA=false
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.siliconflow.cn/v1
OPENAI_MODEL=deepseek-ai/DeepSeek-V3.1-Terminus
OPENAI_GRADER_MODEL=deepseek-ai/DeepSeek-V3
SILICONFLOW_API_KEY=sk-...  # Required for SiliconFlow
```

**Option 2: Local Ollama**
```bash
# .env-local file
USE_OLLAMA=true
OLLAMA_BASE_URL=http://192.168.8.231:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_GRADER_MODEL=qwen2.5:7b
OLLAMA_KEEP_ALIVE_INTERVAL=180  # Seconds, 0 to disable
```

### Embedding Configuration

Embeddings are handled separately from LLM:

```bash
# Option 1: OpenAI-style embedding API (default)
EMBEDDING_TYPE=openai_style
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
EMBEDDING_BASE_URL=http://localhost:50009
EMBEDDING_API_KEY=

# Option 2: SiliconFlow embeddings
EMBEDDING_TYPE=siliconflow
SILICONFLOW_API_KEY=sk-...
```

**Embedding Cache**: LRU cache implemented in `app/services/embedding_cache.py` to reduce redundant API calls.

---

## Integration Points

### Java Platform Integration

AI service calls Java backend via `JAVA_API_BASE_URL`:
- Employee config validation: `POST /api/ai/employee/config`
- FAQ/sensitive word sync webhooks ([ai-service/app/api/endpoints/webhook.py](ai-service/app/api/endpoints/webhook.py))

**Test Mode**: For `employee_id="hutao"`, loads local test data from `outer_api_docs/按员工id返回的数据-hutao.json`.

### External APIs

- **Tavily** (`tavily_api_key`) - Web search
- **MinerU** (`mineru_api_url`) - Enhanced PDF parsing
- **OpenAI/SiliconFlow** - LLM + Embeddings

---

## Key Files Reference

| Component | Path | Description |
|-----------|------|-------------|
| LangGraph workflow | [ai-service/app/services/conversation_service.py](ai-service/app/services/conversation_service.py) | 15-node StateGraph for conversation flow |
| RAG hybrid search | [ai-service/app/services/rag_service.py](ai-service/app/services/rag_service.py) | Vector + keyword search with RRF fusion |
| Document service | [ai-service/app/services/document_service.py](ai-service/app/services/document_service.py) | Document CRUD and chunk management |
| MinerU client | [ai-service/app/services/mineru_client.py](ai-service/app/services/mineru_client.py) | PDF parsing with caching/chunking |
| MinerU API | [ai-service/app/api/endpoints/mineru.py](ai-service/app/api/endpoints/mineru.py) | Web interface for PDF processing |
| Chat endpoints | [ai-service/app/api/endpoints/chat.py](ai-service/app/api/endpoints/chat.py) | Streaming chat SSE implementation |
| Session endpoints | [ai-service/app/api/endpoints/session.py](ai-service/app/api/endpoints/session.py) | Session CRUD operations |
| Employee endpoints | [ai-service/app/api/endpoints/employee.py](ai-service/app/api/endpoints/employee.py) | Digital employee management |
| Knowledge base endpoints | [ai-service/app/api/endpoints/knowledge_base.py](ai-service/app/api/endpoints/knowledge_base.py) | KB and document upload |
| Webhook endpoints | [ai-service/app/api/endpoints/webhook.py](ai-service/app/api/endpoints/webhook.py) | Java platform sync webhooks |
| Configuration | [ai-service/app/core/config.py](ai-service/app/core/config.py) | Pydantic Settings with .env detection |
| Logging wrapper | [ai-service/app/core/logging.py](ai-service/app/core/logging.py) | Loguru-based structured logging |
| Database connections | [ai-service/app/core/database.py](ai-service/app/core/database.py) | MongoDB (Motor) connection |
| Chroma connection | [ai-service/app/core/chroma.py](ai-service/app/core/chroma.py) | ChromaDB client wrapper |
| ElasticSearch connection | [ai-service/app/core/elasticsearch.py](ai-service/app/core/elasticsearch.py) | ES client wrapper |
| Data models | [ai-service/app/models/database.py](ai-service/app/models/database.py) | MongoDB document models |
| API schemas | [ai-service/app/models/schemas.py](ai-service/app/models/schemas.py) | Pydantic request/response models |
| App entry | [ai-service/main.py](ai-service/main.py) | FastAPI app with lifespan management |
| Task processor | [ai-service/app/services/task_processor.py](ai-service/app/services/task_processor.py) | Background task queue |
| Embedding cache | [ai-service/app/services/embedding_cache.py](ai-service/app/services/embedding_cache.py) | LRU cache for embeddings |
| Ollama keep-alive | [ai-service/app/services/ollama_keepalive.py](ai-service/app/services/ollama_keepalive.py) | Prevents model unloading |
| RAG evaluation script | [ai-service/scripts/e2e_rag_evaluation.py](ai-service/scripts/e2e_rag_evaluation.py) | End-to-end RAG testing |
| Test configuration | [ai-service/tests/conftest.py](ai-service/tests/conftest.py) | Pytest fixtures and setup |

---

## Common Development Patterns

### Adding New Conversation Nodes

To add a new node to the LangGraph workflow:

1. Define the node method in `ConversationWorkflow` class ([conversation_service.py](ai-service/app/services/conversation_service.py)):
```python
async def my_new_node(self, state: ConversationState) -> ConversationState:
    async with self._time_node("my_new_node", state):
        # Your logic here
        state["some_field"] = "value"
        return state
```

2. Add the node to the graph in `_build_workflow()`:
```python
graph.add_node("my_new_node", self.my_new_node)
```

3. Add edges to connect the node:
```python
graph.add_edge("previous_node", "my_new_node")
graph.add_edge("my_new_node", "next_node")
```

### Adding New API Endpoints

1. Create router function in `app/api/endpoints/` (e.g., `my_feature.py`):
```python
from fastapi import APIRouter, Depends
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

The workflow automatically exports graph structure to Mermaid format:
- Output directory: `graph_debug/` (configurable via `CRAG_GRAPH_DIR`)
- Disable with: `CRAG_DUMP_GRAPH=0`

View the workflow graph at `graph_debug/crag_graph.mmd`.

---

## Troubleshooting

### Common Issues

**Issue**: Ollama responses are slow
- **Cause**: Model being unloaded from memory
- **Fix**: Increase `OLLAMA_KEEP_ALIVE_INTERVAL` or verify keep-alive service is running

**Issue**: Embedding API errors
- **Cause**: Incorrect `EMBEDDING_BASE_URL` or `EMBEDDING_TYPE` mismatch
- **Fix**: Verify embedding service is running and configuration matches

**Issue**: ChromaDB connection errors
- **Cause**: Chroma container not running or wrong port
- **Fix**: `docker-compose ps chroma` - should be port 8101 (Docker) or 8000 (local)

**Issue**: "No module named" errors
- **Cause**: Not using virtual environment
- **Fix**: Always activate venv with `source .venv/bin/activate` or use full python path

**Issue**: Test failures with database connection
- **Cause**: Test fixtures not properly isolated
- **Fix**: Check `tests/conftest.py` for proper `@pytest.fixture` setup

### Performance Optimization

- **Enable streaming**: Use `/api/chat/stream` instead of `/api/chat/message` for better UX
- **Adjust top_k**: Reduce `top_k` in RAG queries for faster retrieval
- **Cache embeddings**: The `embedding_cache.py` LRU cache reduces redundant API calls
- **Monitor node timings**: Check `state["node_timings"]` in logs to identify bottlenecks

---

## Documentation Reference

**Core Features**:
- [联网检索功能使用指南.md](docs/联网检索功能使用指南.md)
- [FAQ多路召回功能实现总结.md](docs/FAQ多路召回功能实现总结.md)
- [异步文档上传使用说明.md](docs/异步文档上传使用说明.md)
- [MinerU客户端使用指南.md](docs/MinerU客户端使用指南.md)
- [MinerU实现总结.md](docs/MinerU实现总结.md)

**Performance**:
- [Ollama模型保活方案.md](docs/Ollama模型保活方案.md)
- [性能优化总结.md](docs/性能优化总结.md)
- [性能优化效果验证.md](docs/性能优化效果验证.md)

**Architecture**:
- [ConversationWorkflow流程图与架构图.md](docs/ConversationWorkflow流程图与架构图.md)
- [API使用文档.md](docs/API使用文档.md)
- [部署指南.md](docs/部署指南.md)
