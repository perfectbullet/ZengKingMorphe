# Digital Employee AI Service - AI Agent Guide

## Project Overview
LangGraph-based conversational AI service providing RAG (Retrieval-Augmented Generation), multi-turn dialogue, knowledge base management, and web search capabilities for digital employee interactions.

**Tech Stack**: FastAPI + LangGraph + OpenAI/Ollama + ChromaDB + ElasticSearch + MongoDB

## Architecture Highlights

### LangGraph Workflow (Core Pattern)
Conversation flow is implemented as a **StateGraph** in [app/services/conversation_service.py](ai-service/app/services/conversation_service.py):
- 12 nodes orchestrate: config loading → input validation → realtime detection → intent recognition → RAG retrieval → document grading → web search (fallback) → LLM generation → persistence
- **Conditional routing**: Realtime queries bypass RAG and go directly to web search; low relevance scores (<0.6) trigger web search fallback
- State flows through `ConversationState` TypedDict with ~20 fields tracking query context, retrieval results, and metadata

### Hybrid RAG System
**Three-tier storage** for knowledge base documents ([app/services/rag_service.py](ai-service/app/services/rag_service.py)):
1. **ChromaDB** - Vector search (semantic similarity)
2. **ElasticSearch** - BM25 keyword search
3. **MongoDB** - Document metadata and full records

**Search fusion**: Hybrid search uses RRF (Reciprocal Rank Fusion) to merge vector + keyword results with weighted scoring and deduplication.

### Dual LLM Configuration
System supports **two LLM backends** via `use_ollama` flag ([app/core/config.py](ai-service/app/core/config.py)):
- `use_ollama=True`: Local Ollama (qwen2.5:7b)
- `use_ollama=False`: OpenAI-style APIs (DeepSeek-V3 via SiliconFlow)

Two separate LLM instances: `self.llm` (generation) and `self.grader_llm` (relevance scoring, forced JSON mode).

### Multi-Database Lifecycle
All three databases managed in FastAPI lifespan context ([main.py](ai-service/main.py)):
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    await mongodb.connect()
    chroma_db.connect()  # HTTP client mode
    await es_db.connect()
    yield
    # Cleanup handled automatically
```

## Development Workflows

### Local Development Setup
```powershell
# Start only databases
docker-compose up -d mongodb elasticsearch chroma

# Run AI service locally
cd ai-service
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Full Stack Deployment
```powershell
docker-compose up -d  # Starts 4 services: ai-service, mongodb, chroma, elasticsearch
```

**Port mappings**:
- AI Service: `8100:8000` (external:container)
- MongoDB: `27117:27017`
- ChromaDB: `8101:8000`
- ElasticSearch: `9320:9200`

### Testing Patterns
- **Unit tests**: Basic pytest setup in [tests/conftest.py](ai-service/tests/conftest.py) with `anyio_backend` fixture
- **Standalone ChromaDB testing**: Use [test_chroma_standalone.py](ai-service/test_chroma_standalone.py) to validate ChromaDB connection without full app context
- Run tests: `cd ai-service && python test_chroma_standalone.py`

## Project-Specific Conventions

### Authentication Middleware
**Dual auth modes** ([app/api/middleware/auth.py](ai-service/app/api/middleware/auth.py)):
- `verify_api_key()` - Required auth (raises 401 on failure)
- `get_api_key()` - Optional auth (returns None if missing/invalid)

All endpoints use `Depends(get_api_key)` to access `X-API-Key` header without enforcing auth (legacy pattern, auth is TODO).

### Logging Convention
**Structured logging** via custom wrapper ([app/core/logging.py](ai-service/app/core/logging.py)):
```python
logger.info("Chat message request", user_id=user.id, query=q[:100])  # Truncate PII
logger.error("RAG search failed", error=str(e), exc_info=True)
```
Use keyword args for contextual fields, avoid string interpolation.

### Session ID Generation
Auto-generated if not provided using MD5 hash pattern:
```python
f"sess_{hashlib.md5(f'{user_id}_{datetime.utcnow().timestamp()}'.encode()).hexdigest()[:12]}"
```

### Document Chunking
Text splitting uses **paragraph-aware chunking** with configurable size/overlap ([app/services/document_service.py](ai-service/app/services/document_service.py)):
- Default: 512 chars with 50 char overlap
- Splits on paragraph boundaries when possible
- Each chunk vectorized and stored in all 3 databases

### Error Handling Pattern
Custom exception handlers registered globally ([app/api/middleware/error_handler.py](ai-service/app/api/middleware/error_handler.py)):
- `HTTPException` → JSON response with status code
- `RequestValidationError` → 422 with field-level errors
- General exceptions → 500 with logged stack trace

## Integration Points

### Java Platform Integration
AI service calls Java backend via `JAVA_API_BASE_URL` ([app/core/config.py](ai-service/app/core/config.py)) for:
- Employee config validation (webhook: `POST /api/ai/employee/config`)
- FAQ/sensitive word sync (webhooks in [app/api/endpoints/webhook.py](ai-service/app/api/endpoints/webhook.py))

### External APIs
- **Tavily** (`tavily_api_key`): Web search fallback when RAG fails
- **OpenAI/SiliconFlow**: LLM + Embeddings (text-embedding-3-small or BAAI/bge-large-zh-v1.5)

### Database Schemas
- **MongoDB Collections**: `conversations`, `sessions`, `employee_configs`, `knowledge_bases`, `documents`
- **ChromaDB Collection**: Single `"doc"` collection with metadata filters by `kb_id`
- **ElasticSearch Index**: Prefixed with `digital_employee_*`

## Critical Files
- [conversation_service.py](ai-service/app/services/conversation_service.py) - LangGraph workflow (12 nodes, 441 lines)
- [rag_service.py](ai-service/app/services/rag_service.py) - Hybrid search implementation (301 lines)
- [config.py](ai-service/app/core/config.py) - Pydantic settings with 50+ env vars
- [docker-compose.yml](docker-compose.yml) - 4-service orchestration

## Known Limitations
- Sensitive word filtering engine not implemented (AC automaton TODO)
- Stream endpoint framework exists but not fully integrated with LangGraph
- Test coverage minimal (only conftest + standalone chroma test)
- Auth middleware exists but `api_keys` list not populated (auth effectively disabled)
