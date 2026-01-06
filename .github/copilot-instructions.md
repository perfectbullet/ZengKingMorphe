# Digital Employee AI Service - AI Agent Guide

## Project Overview
LangGraph-based conversational AI service providing RAG (Retrieval-Augmented Generation), multi-turn dialogue, knowledge base management, and web search for digital employee interactions.

**Tech Stack**: FastAPI + LangGraph + OpenAI/Ollama + ChromaDB + ElasticSearch + MongoDB

## Architecture Deep Dive

### LangGraph Workflow (Core Pattern)
Conversation flow is a **StateGraph** in [conversation_service.py](ai-service/app/services/conversation_service.py):
- **12 nodes** orchestrate: config loading → session context → FAQ matching → realtime detection → intent recognition → RAG retrieval → document grading → web search (fallback) → LLM generation → persistence
- **Conditional routing logic**:
  - FAQ matched → skip RAG, generate answer directly
  - Realtime query detected → bypass RAG, go straight to web search
  - Low relevance score (<0.6) → trigger web search fallback
- State flows through `ConversationState` TypedDict (~22 fields) tracking query context, retrieval results, metadata

**Key implementation detail**: The workflow is compiled once during service initialization (`self.workflow = self._build_workflow()`) and reused across all requests.

### Hybrid RAG System
**Three-tier storage** for knowledge base documents ([rag_service.py](ai-service/app/services/rag_service.py)):
1. **ChromaDB** - Vector search (semantic similarity via embeddings)
2. **ElasticSearch** - BM25 keyword search (exact term matching)
3. **MongoDB** - Document metadata and full records

**Search fusion**: Hybrid search uses **RRF (Reciprocal Rank Fusion)** to merge vector + keyword results:
- Default formula: `rrf_score = 1/(rank + 60)`
- Results deduplicated by `doc_id` and sorted by combined score
- Top-K filtering applied after fusion

### Dual LLM Configuration
System supports **two LLM backends** via `use_ollama` flag ([config.py](ai-service/app/core/config.py)):
- `use_ollama=True`: Local Ollama (default: qwen2.5:7b)
- `use_ollama=False`: OpenAI-style APIs (default: DeepSeek-V3 via SiliconFlow)

**Critical distinction**: Two separate LLM instances initialized:
- `self.llm` - Answer generation (supports streaming)
- `self.grader_llm` - Document relevance scoring (forced JSON mode via `format="json"` or `response_format`)

### Multi-Database Lifecycle
All three databases managed in FastAPI lifespan context ([main.py](ai-service/main.py)):
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    await mongodb.connect()      # Async MongoDB connection
    chroma_db.connect()          # HTTP client mode (sync)
    await es_db.connect()        # Async ElasticSearch
    await task_processor.start() # Background task queue
    yield
    # Cleanup happens automatically on shutdown
```

**Important**: ChromaDB uses HTTP client mode (`chroma_host:chroma_port`), NOT local persistent mode in production.

## Development Workflows

### Local Development Setup
```powershell
# Start only databases (leave AI service for local debugging)
docker-compose up -d mongodb elasticsearch chroma

# Run AI service locally (Windows)
cd ai-service
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**API access**: http://localhost:8000/docs (Swagger UI)

**Important - Virtual Environment Usage**:
- **Always use the virtual environment** when running Python scripts in `ai-service/`
- Windows PowerShell: `D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/Activate.ps1` or directly use `D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe script.py`
- Git Bash/WSL: `source D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/activate` or `D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python script.py`
- **Never run Python commands without activating venv first** - dependencies are installed in the virtual environment, not globally
- Example correct command: `cd ai-service; D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe test_web_search.py`

### Full Stack Deployment
```powershell
docker-compose up -d  # Starts 4 services
docker-compose logs -f ai-service  # Watch logs
```

**Port mappings** (external:container):
- AI Service: `8100:8000`
- MongoDB: `27117:27017`
- ChromaDB: `8101:8000`
- ElasticSearch: `9320:9200`

### Testing Patterns
- **Unit tests**: Basic pytest setup in [tests/conftest.py](ai-service/tests/conftest.py) with `anyio_backend` fixture
- **Test suite includes** (13 test files):
  - `test_web_search.py` - Tavily API integration
  - `test_chroma_standalone.py` - ChromaDB connectivity (4 test functions)
  - `test_ollama_embedding.py` - Ollama embedding validation
  - `test_stream_chunks.py` - Streaming chunk persistence
  - `test_sources_attribution.py` - RAG/web source tracking
  - `test_faq_integration.py` - FAQ hybrid search workflow
  - `test_create_rag_document.py` - Document upload with custom chunking
  - `test_embedding_truncation.py` - Token limit handling
  - `test_config_loading.py` - Environment variable validation
  - `test_api.py` - Basic HTTP endpoint testing
  - Additional: `test_loguru_migration.py`, `quick_test_ollama.py`
- **Run tests**: Always use venv - `cd ai-service; D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe -m pytest tests/`
- **Standalone scripts**: Several test files are standalone (not using pytest framework) - run directly with Python

**Note**: Tests focus on integration validation rather than comprehensive unit coverage.

## Project-Specific Conventions

### Authentication Middleware
**Dual auth modes** ([app/api/middleware/auth.py](ai-service/app/api/middleware/auth.py)):
- `verify_api_key()` - Required auth (raises 401 on failure)
- `get_api_key()` - Optional auth (returns None if missing/invalid)

**Current state**: All endpoints use `Depends(get_api_key)` but `settings.api_keys` list is empty, so auth is effectively disabled (legacy pattern, TODO).

### Logging Convention
**Structured logging** via custom wrapper ([app/core/logging.py](ai-service/app/core/logging.py)):
```python
logger.info("Chat message request", user_id=user.id, query=q[:100])  # Truncate PII
logger.error("RAG search failed", error=str(e), exc_info=True)
```
**Rules**:
- Use keyword args for contextual fields (not string interpolation)
- Truncate user queries to prevent PII leakage in logs
- Always include `exc_info=True` for error logs

### Session ID Generation
Auto-generated using MD5 hash pattern when not provided:
```python
f"sess_{hashlib.md5(f'{user_id}_{datetime.utcnow().timestamp()}'.encode()).hexdigest()[:12]}"
```
**Used in**: [chat.py](ai-service/app/api/endpoints/chat.py) endpoint handlers

### Document Chunking
Text splitting uses **paragraph-aware chunking** ([document_service.py](ai-service/app/services/document_service.py)):
- Default: 512 chars with 50 char overlap (configurable via `chunk_size`, `chunk_overlap`)
- Splits on paragraph boundaries (`\n\n`) when possible to preserve semantic coherence
- Each chunk vectorized and stored in **all 3 databases** simultaneously

### Personality-Based System Prompts
LLM generation dynamically builds system prompts from employee config ([conversation_service.py](ai-service/app/services/conversation_service.py:397-440)):
```python
tone_desc = {"professional": "专业严谨", "friendly": "友好亲切", ...}
system_prompt = f"""你是 {name}，{role}。
个性特征：
- 语气风格：{tone_desc[personality.tone]}
- 沟通方式：{style_desc[personality.style]}
上下文信息：{context_text}
用户问题：{user_query}"""
```
**Critical**: Context is built from top 3 retrieved docs (truncated to 500 chars each).

### Error Handling Pattern
Custom exception handlers registered globally ([app/api/middleware/error_handler.py](ai-service/app/api/middleware/error_handler.py)):
- `HTTPException` → JSON response with status code
- `RequestValidationError` → 422 with field-level validation errors
- General exceptions → 500 with logged stack trace

### API Response Schema Pattern
All API responses follow standardized structure ([schemas.py](ai-service/app/models/schemas.py)):
```python
class ChatResponse(BaseModel):
    code: int = 200
    message: str = "success"
    data: ChatResponseData  # Nested data model
```
**Key schemas**:
- `ChatRequest` / `ChatResponse` - Chat endpoints with full source attribution
- `SourceAttribution` - Contains `RAGSource` (top 3) + `WebSource` (top 5) lists
- `StreamChunkResponse` - SSE chunk format with `type` field (user_query/role/token/done/error)
- All models use Pydantic v2 with `model_config = ConfigDict(json_schema_extra=...)` for OpenAPI examples

## Integration Points

### Java Platform Integration
AI service calls Java backend via `JAVA_API_BASE_URL` for:
- Employee config validation: `POST /api/ai/employee/config`
- FAQ/sensitive word sync webhooks ([webhook.py](ai-service/app/api/endpoints/webhook.py))

**Important**: Java platform URLs are hardcoded in config, ensure `.env` has correct `JAVA_API_BASE_URL`.

**Test Mode Pattern** ([session.py:37-51](ai-service/app/api/endpoints/session.py#L37-L51)):
- For `employee_id="hutao"`, external API calls bypass real requests and load local test data from `outer_api_docs/按员工id返回的数据-hutao.json`
- Pattern: Check employee_id → load local JSON → parse via Pydantic `ExternalEmployeeAPIResponse`
- Enables offline development and integration testing without Java backend

### External APIs
- **Tavily** (`tavily_api_key`): Web search fully implemented - triggers on realtime queries or low RAG relevance
- **OpenAI/SiliconFlow**: LLM + Embeddings
  - Embeddings: text-embedding-3-small OR BAAI/bge-large-zh-v1.5 (auto-truncated to max token limit)
  - LLM: DeepSeek-V3 (default) OR qwen2.5:7b (Ollama)

### Database Schemas
- **MongoDB Collections**: `conversations`, `sessions`, `digital_employee_configs`, `knowledge_bases`, `documents`
- **ChromaDB**: Single `"doc"` collection with metadata filters by `kb_id`
- **ElasticSearch**: Indexes prefixed with `digital_employee_*` (configurable via `es_index_prefix`)

## Critical Implementation Notes

### Web Search Integration
**Tavily web search** is now fully implemented ([conversation_service.py:383-460](ai-service/app/services/conversation_service.py#L383-L460)):
- **Trigger conditions**: Realtime queries OR low RAG relevance score (<0.6)
- **Configuration checks**: 
  - Global: `settings.web_search_enabled` and `settings.tavily_api_key`
  - Per-employee: `employee_config.capabilities.web_search_enabled`
- **Search flow**:
  1. Call `TavilySearchResults.ainvoke()` with user query
  2. Format results (title, URL, content truncated to 500 chars, score)
  3. Limit to `settings.web_search_max_results` (default: 5)
  4. Set `state["web_search_used"] = True` if results found
- **Integration with LLM**:
  - Web results added to context as "[网络资料{i}]" (separate from knowledge base "[知识库参考{i}]")
  - System prompt updated to prioritize web results for realtime queries
  - Confidence score: min 0.75 for web-backed answers
- **Persistence**: Web search results saved in `conversations.web_search_results` (top 5 results with rank/title/URL/score)
- **Testing**: Run `python ai-service/test_web_search.py` to validate Tavily API connectivity

**Error handling**: Web search failures don't break workflow - continues with empty results and logs error.

### FAQ Matching Strategy
Simple keyword matching in [conversation_service.py:252-289](ai-service/app/services/conversation_service.py#L252-L289):
1. Check if query contains any FAQ keywords
2. Check if query substring matches FAQ question
3. Return first match with 0.95 confidence

**No semantic similarity** - purely keyword-based (upgrade opportunity).

### Realtime Query Detection
Keyword-based detection using hardcoded categories ([conversation_service.py:291-316](ai-service/app/services/conversation_service.py#L291-L316)):
- `time`: ["今天", "明天", "昨天", "最近", ...]
- `weather`: ["天气", "气温", "降雨", ...]
- `news`: ["新闻", "热点", "最新", ...]
- `market`: ["股价", "汇率", "行情", ...]

When detected, query bypasses RAG and goes directly to web search.

## Common Modification Patterns

### Querying Conversation Records
**Historical data access** via [conversation.py](ai-service/app/api/endpoints/conversation.py):
- Multi-dimensional filtering: date range, user_id, employee_id, session_id, keyword search
- Pagination support (default 20 per page, max 100)
- Full-text search across `user_query` and `ai_response` fields
- Statistics endpoint for conversation analytics (count, avg duration, satisfaction)
- **MongoDB queries**: Use `$regex` for keyword search, `$gte/$lte` for date range filtering

### Adding a New LangGraph Node
1. Define async function in `ConversationWorkflow` class
2. Add node to graph: `graph.add_node("node_name", self.node_function)`
3. Add edges or conditional edges to integrate into flow
4. Update `ConversationState` TypedDict if new fields needed

### Switching LLM Backends
Set in `.env`:
```bash
USE_OLLAMA=false  # Use OpenAI-style API
OPENAI_API_KEY=sk-...
OPENAI_MODEL=deepseek-ai/DeepSeek-V3
```
OR
```bash
USE_OLLAMA=true   # Use local Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

### Adding Knowledge Base Filters
Update employee config `capabilities.kb_ids` to restrict RAG search scope. The `rag_retrieval.search()` call automatically filters by `kb_ids` if provided.

### Streaming Implementation
**Fully integrated SSE streaming** via `sse-starlette` ([chat.py:166-293](ai-service/app/api/endpoints/chat.py#L166-L293)):
- **Two streaming modes**:
  1. Native SSE: `POST /api/chat/stream` - Custom format with sources metadata
  2. OpenAI-compatible: `POST /v1/chat/completions?stream=true` - Standard OpenAI SSE format
- **Token-level streaming**: LLM tokens streamed in real-time via `EventSourceResponse`
- **Chunk persistence**: All stream chunks saved to MongoDB (`stream_chunks` collection) with:
  - Chunk types: `user_query`, `role`, `token`, `done`, `error`
  - Full metadata: `chat_id`, `conversation_id`, `session_id`, `sequence`, `timestamp`
  - Query API: `GET /api/chat/stream/chunks` with multi-dimension filters
- **Test client**: [stream_client.py](ai-service/scripts/stream_client.py) - Full-featured SSE client for testing
- **Implementation detail**: Streaming bypasses LangGraph's standard execution - directly invokes `self.llm.astream()` after workflow preparation

### Source Attribution
**Automatic source tracking** in all chat responses ([chat.py:26-73](ai-service/app/api/endpoints/chat.py#L26-L73)):
- `format_sources()` helper extracts top 3 RAG docs + top 5 web results
- **RAG sources** include: `doc_id`, `kb_id`, `content_snippet` (200 chars), `score`, `chunk_index`
- **Web sources** include: `title`, `url`, `score`, `rank`
- Sources returned in both streaming (`sources` event) and non-streaming modes
- Enables answer traceability and fact-checking workflows

### Async Document Upload
**Background task processing** via `TaskProcessor` ([task_processor.py](ai-service/app/services/task_processor.py)):
- Set `async_mode=true` in document upload to get immediate task ID
- Upload endpoint: `POST /api/knowledge_base/documents/upload?async_mode=true`
- Task tracking: `GET /api/knowledge_base/tasks/{task_id}` returns status/progress/result
- Task cancellation: `DELETE /api/knowledge_base/tasks/{task_id}`
- **Queue system**: `asyncio.Queue` with concurrent worker processing (auto-started in app lifespan)
- **Progress tracking**: Real-time percentage (0-100%) for chunking/vectorization progress

## Known Limitations & TODOs
- **Sensitive word filtering**: AC automaton engine not implemented
- **Auth**: Middleware exists but `api_keys` list empty (auth disabled)
- **Test coverage**: Minimal (basic conftest + standalone DB tests)
- **Rate limiting**: Middleware exists but not enforced

## Documentation Structure
The `docs/` folder contains feature-specific documentation:
- **Core Features**: [联网检索功能使用指南.md](docs/联网检索功能使用指南.md), [FAQ多路召回功能实现总结.md](docs/FAQ多路召回功能实现总结.md), [异步文档上传使用说明.md](docs/异步文档上传使用说明.md)
- **API Documentation**: [API使用文档.md](docs/API使用文档.md), [API-Key-Authentication.md](docs/API-Key-Authentication.md)
- **Technical Deep Dives**: [来源归属功能说明.md](docs/来源归属功能说明.md), [流式Chunk存储与查询功能说明.md](docs/流式Chunk存储与查询功能说明.md), [自定义分段策略RAG文档创建接口说明.md](docs/自定义分段策略RAG文档创建接口说明.md)
- **Migration Guides**: [loguru迁移说明.md](docs/loguru迁移说明.md), [字段重命名迁移说明.md](docs/字段重命名迁移说明.md), [Embedding配置修复说明.md](docs/Embedding配置修复说明.md)
- **Integration**: [Java后端提供的接口-by-刑伟.md](docs/Java后端提供的接口-by-刑伟.md), [数字员工信息集成总结.md](docs/数字员工信息集成总结.md)
- **Deployment**: [部署指南.md](docs/部署指南.md)
- **Project Summary**: [项目总结.md](docs/项目总结.md), [需求补充与完善文档.md](docs/需求补充与完善文档.md)

**Important**: Check `docs/` folder for detailed feature documentation before implementing or modifying features.

## Key Files Reference
- [conversation_service.py](ai-service/app/services/conversation_service.py) - LangGraph workflow (12 nodes, 800+ lines)
- [chat.py](ai-service/app/api/endpoints/chat.py) - Chat endpoints with streaming/sources (800+ lines)
- [conversation.py](ai-service/app/api/endpoints/conversation.py) - Conversation records query API (350 lines)
- [session.py](ai-service/app/api/endpoints/session.py) - Session lifecycle management (600+ lines)
- [rag_service.py](ai-service/app/services/rag_service.py) - Hybrid search (RRF fusion, 300 lines)
- [task_processor.py](ai-service/app/services/task_processor.py) - Async task queue system
- [config.py](ai-service/app/core/config.py) - Pydantic settings (50+ env vars)
- [main.py](ai-service/main.py) - FastAPI app + lifespan management
- [stream_client.py](ai-service/scripts/stream_client.py) - OpenAI-compatible test client
- [docker-compose.yml](docker-compose.yml) - 4-service orchestration

## API Surface Reference
**Core endpoints** (20+ routes across 6 modules):
- `/api/chat/*` - Chat, streaming, OpenAI-compatible endpoints
- `/api/conversation/records` - Query conversation history with filters
- `/api/conversation/statistics` - Conversation analytics
- `/api/session/*` - Session CRUD and conversation listing
- `/api/employee/*` - Employee config and FAQ retrieval
- `/api/knowledge_base/*` - KB management, document upload, task tracking
- `/webhook/*` - Java platform sync notifications (sensitive words, FAQs)

**Important**: All endpoints use `Depends(get_api_key)` but auth is currently disabled (empty `api_keys` list).

## Testing Tools & Scripts
- **Web search**: `python ai-service/tests/test_web_search.py` - Validate Tavily integration
- **ChromaDB**: `python ai-service/tests/test_chroma_standalone.py` - DB connectivity
- **Ollama**: `python ai-service/tests/test_ollama_embedding.py` - Local LLM validation
- **Streaming**: `python ai-service/scripts/stream_client.py --host http://localhost:8100 --query "test"` - Full SSE client
- **Chunks**: `python ai-service/tests/test_stream_chunks.py` - Chunk storage/retrieval validation
- **Always use venv**: `D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe <script>`
