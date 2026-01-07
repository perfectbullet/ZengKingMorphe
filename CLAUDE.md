# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A LangGraph-based conversational AI service providing RAG (Retrieval-Augmented Generation), multi-turn dialogue, knowledge base management, and web search for digital employee interactions. Built with FastAPI, supporting multiple LLM backends (OpenAI, Ollama), and using a three-tier storage architecture (ChromaDB for vectors, ElasticSearch for keywords, MongoDB for metadata).

**Tech Stack**: FastAPI + LangGraph + OpenAI/Ollama + ChromaDB + ElasticSearch + MongoDB + Tavily Web Search

---

## Essential Development Commands

### Virtual Environment (Critical!)
**ALL Python commands MUST use the virtual environment at `.venv/`** - dependencies are installed there, not globally.

**Windows PowerShell**:
```powershell
# Activate
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/Activate.ps1

# Or run directly
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe script.py
```

**Git Bash/WSL**:
```bash
source D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/activate
```

### Local Development
```powershell
# Start databases only (for local AI service development)
docker-compose up -d mongodb elasticsearch chroma

# Run AI service locally
cd ai-service
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/Activate.ps1
uvicorn main:app --reload --port 8000

# Access API docs
# http://localhost:8000/docs
```

### Full Docker Deployment
```powershell
# Start all services (AI service + databases)
docker-compose up -d

# View logs
docker-compose logs -f ai-service

# Stop all
docker-compose down
```

**Port Mappings** (external:container):
- AI Service: `8100:8000`
- MongoDB: `27117:27017`
- ChromaDB: `8101:8000`
- ElasticSearch: `9320:9200`

### Testing
```powershell
cd ai-service

# Run all tests (pytest)
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe -m pytest tests/

# Run standalone test script
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe tests/test_web_search.py

# Test streaming SSE client
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe scripts/stream_client.py --host http://localhost:8100 --query "test"
```

---

## Architecture Overview

### LangGraph Conversation Workflow

The core conversation engine is a **12-node StateGraph** in [ai-service/app/services/conversation_service.py](ai-service/app/services/conversation_service.py):

**Flow**: `load_config → load_context → validate_input → check_realtime → match_faq → recognize_intent → rag_retrieval → grade_docs → web_search → generate_answer → save_conversation`

**Key Routing Logic**:
- **Realtime queries** (weather, news, stock prices) → bypass RAG, direct to web search
- **FAQ matched** → skip RAG, generate answer directly
- **Low relevance score** (<0.6) → trigger web search as fallback
- **Otherwise** → RAG retrieval → LLM generation

**State Management**: `ConversationState` TypedDict with ~22 fields flows through all nodes, tracking query context, retrieval results, metadata, confidence scores, etc.

**Dual LLM Pattern**: The workflow initializes two separate LLM instances:
- `self.llm` - Answer generation (supports streaming)
- `self.grader_llm` - Document relevance scoring (forced JSON output mode)

### Three-Tier RAG Architecture

Knowledge base documents are stored across **3 databases** ([ai-service/app/services/rag_service.py](ai-service/app/services/rag_service.py)):

1. **ChromaDB** - Vector embeddings for semantic similarity search
2. **ElasticSearch** - BM25 keyword search for exact term matching
3. **MongoDB** - Document metadata, full content, and conversation records

**Hybrid Search Fusion**:
- Uses **RRF (Reciprocal Rank Fusion)** algorithm: `rrf_score = 1/(rank + 60)`
- Vector and keyword results merged by `doc_id`, sorted by combined RRF score
- Returns Top-K most relevant documents

### Database Lifecycle Management

All databases initialized in FastAPI lifespan context ([ai-service/main.py](ai-service/main.py)):
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    await mongodb.connect()      # Async Motor client
    chroma_db.connect()          # HTTP client mode (sync)
    await es_db.connect()        # Async ElasticSearch
    await task_processor.start() # Background task queue
    yield
    # Automatic cleanup on shutdown
```

**Critical**: ChromaDB uses HTTP client mode (`chroma_host:chroma_port`), NOT local persistent mode in production.

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
- `SourceAttribution` - Contains `RAGSource[]` (top 3) + `WebSource[]` (top 5)
- `StreamChunkResponse` - SSE chunks with `type` field (user_query/role/token/done/error)
- All models use Pydantic v2

### Authentication Status
**Currently DISABLED**:
- Middleware exists at [ai-service/app/api/middleware/auth.py](ai-service/app/api/middleware/auth.py)
- All endpoints use `Depends(get_api_key)` but `settings.api_keys` list is empty
- This is legacy code - TODO: either enable auth or remove middleware

### Error Handling Pattern
Global exception handlers ([ai-service/app/api/middleware/error_handler.py](ai-service/app/api/middleware/error_handler.py)):
- `HTTPException` → JSON response with status code
- `RequestValidationError` → 422 with field-level errors
- Generic exceptions → 500 with logged stack trace

**Pattern**: Don't silently catch exceptions - log them and return clear errors via HTTPException.

---

## LLM Configuration

### Switching Backends

**Option 1: OpenAI-style API** (default: SiliconFlow/DeepSeek)
```bash
# .env file
USE_OLLAMA=false
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.siliconflow.cn/v1
OPENAI_MODEL=deepseek-ai/DeepSeek-V3
```

**Option 2: Local Ollama**
```bash
# .env file
USE_OLLAMA=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

### Dual LLM Instances
The system requires **two LLM instances**:
1. **Generation LLM** (`self.llm`) - Answer generation, supports streaming
2. **Grader LLM** (`self.grader_llm`) - Document relevance scoring, JSON output mode

When adding new LLM backends, ensure both instances are properly initialized.

---

## Integration Points

### Java Platform Integration
AI service calls Java backend via `JAVA_API_BASE_URL`:
- Employee config validation: `POST /api/ai/employee/config`
- FAQ/sensitive word sync webhooks ([ai-service/app/api/endpoints/webhook.py](ai-service/app/api/endpoints/webhook.py))

**Test Mode**: For `employee_id="hutao"`, loads local test data from `outer_api_docs/按员工id返回的数据-hutao.json` instead of calling Java API.

### External APIs
- **Tavily** (`tavily_api_key`) - Web search, triggers on realtime queries or low RAG relevance
- **OpenAI/SiliconFlow** - LLM + Embeddings (text-embedding-3-small or BAAI/bge-large-zh-v1.5)

---

## Common Modification Patterns

### Adding a New LangGraph Node
1. Define async method in `ConversationWorkflow` class ([conversation_service.py](ai-service/app/services/conversation_service.py))
2. Add node: `graph.add_node("node_name", self.node_function)`
3. Add conditional edge to integrate into flow
4. Update `ConversationState` TypedDict if new fields needed

### Querying Conversation Records
Use [ai-service/app/api/endpoints/conversation.py](ai-service/app/api/endpoints/conversation.py):
- Multi-dimensional filtering: date range, user_id, employee_id, session_id, keywords
- Pagination (default 20, max 100)
- MongoDB `$regex` for keyword search, `$gte/$lte` for date ranges

### Async Document Upload
Background task processing via `TaskProcessor` ([ai-service/app/services/task_processor.py](ai-service/app/services/task_processor.py)):
- Upload: `POST /api/knowledge_base/documents/upload?async_mode=true`
- Track: `GET /api/knowledge_base/tasks/{task_id}` (returns progress 0-100%)
- Cancel: `DELETE /api/knowledge_base/tasks/{task_id}`

### Streaming Implementation
Two SSE streaming modes ([ai-service/app/api/endpoints/chat.py](ai-service/app/api/endpoints/chat.py#L166-L293)):
1. Native: `POST /api/chat/stream` - Custom format with sources
2. OpenAI-compatible: `POST /v1/chat/completions?stream=true`

**Implementation detail**: Streaming bypasses LangGraph's standard execution - directly invokes `self.llm.astream()` after workflow preparation.

---

## Key Files Reference

| Component | Path |
|-----------|------|
| LangGraph workflow | [ai-service/app/services/conversation_service.py](ai-service/app/services/conversation_service.py) (12 nodes, 800+ lines) |
| RAG hybrid search | [ai-service/app/services/rag_service.py](ai-service/app/services/rag_service.py) (RRF fusion, 300 lines) |
| Chat endpoints | [ai-service/app/api/endpoints/chat.py](ai-service/app/api/endpoints/chat.py) (streaming/sources, 800+ lines) |
| Session management | [ai-service/app/api/endpoints/session.py](ai-service/app/api/endpoints/session.py) (600+ lines) |
| Configuration | [ai-service/app/core/config.py](ai-service/app/core/config.py) (50+ env vars, Pydantic Settings) |
| Schemas | [ai-service/app/models/schemas.py](ai-service/app/models/schemas.py) |
| App entry point | [ai-service/main.py](ai-service/main.py) (lifespan management) |
| Task processor | [ai-service/app/services/task_processor.py](ai-service/app/services/task_processor.py) |
| Docker orchestration | [docker-compose.yml](docker-compose.yml) (4 services) |
| Streaming test client | [ai-service/scripts/stream_client.py](ai-service/scripts/stream_client.py) |

---

## Known Limitations

1. **Sensitive word filtering** - AC automaton engine not implemented
2. **Authentication** - Middleware exists but disabled (empty `api_keys` list)
3. **Test coverage** - Minimal, mostly integration tests
4. **Rate limiting** - Middleware exists but not enforced
5. **FAQ matching** - Purely keyword-based, no semantic similarity (upgrade opportunity)

---

## Documentation Reference

Detailed feature documentation in `docs/`:
- [联网检索功能使用指南.md](docs/联网检索功能使用指南.md)
- [FAQ多路召回功能实现总结.md](docs/FAQ多路召回功能实现总结.md)
- [异步文档上传使用说明.md](docs/异步文档上传使用说明.md)
- [API使用文档.md](docs/API使用文档.md)
- [部署指南.md](docs/部署指南.md)

See also: [README.md](README.md) (project overview), [.github/copilot-instructions.md](.github/copilot-instructions.md) (detailed technical deep-dive)
