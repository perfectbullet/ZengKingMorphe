# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A LangGraph-based conversational AI service providing RAG (Retrieval-Augmented Generation), multi-turn dialogue, knowledge base management, and web search for digital employee interactions. Built with FastAPI, supporting multiple LLM backends (OpenAI, Ollama), and using a three-tier storage architecture (ChromaDB for vectors, ElasticSearch for keywords, MongoDB for metadata).

**Tech Stack**: FastAPI + LangGraph + OpenAI/Ollama + ChromaDB + ElasticSearch + MongoDB + Tavily Web Search + MinerU PDF Parsing

---

## Essential Development Commands

### Virtual Environment (Critical!)
**ALL Python commands MUST use the virtual environment** - dependencies are installed at `.venv/`, not globally.

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

### Local Development (Windows)

**Important**: This project uses `.env-win` for Windows development (auto-detected by `config.py`).

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

### Testing

```powershell
cd ai-service

# Run all tests (pytest)
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe -m pytest tests/

# Run standalone test script
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe tests/test_web_search.py

# MinerU PDF processing test
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe tests/test_mineru_client.py process --file path/to/document.pdf --save-md

# Performance benchmark
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe scripts/performance_test.py
```

---

## Architecture Overview

### LangGraph Conversation Workflow

The core conversation engine is a **12-node StateGraph** ([ai-service/app/services/conversation_service.py](ai-service/app/services/conversation_service.py)):

**Flow**: `load_config → load_context → validate_input → check_realtime → match_faq → recognize_intent → rag_retrieval → grade_docs → web_search → generate_answer → save_conversation`

**Key Routing Logic**:
- **Realtime queries** (weather, news, stock prices) → bypass RAG, direct to web search
- **FAQ matched** → skip RAG, generate answer directly
- **Low relevance score** (<0.6) → trigger web search as fallback
- **Otherwise** → RAG retrieval → LLM generation

**State Management**: `ConversationState` TypedDict with ~22 fields flows through all nodes.

**Dual LLM Pattern**: Two separate LLM instances:
- `self.llm` - Answer generation (supports streaming)
- `self.grader_llm` - Document relevance scoring (forced JSON output mode)

### Three-Tier RAG Architecture

Knowledge base documents stored across **3 databases** ([ai-service/app/services/rag_service.py](ai-service/app/services/rag_service.py)):

1. **ChromaDB** - Vector embeddings for semantic similarity search
2. **ElasticSearch** - BM25 keyword search for exact term matching
3. **MongoDB** - Document metadata, full content, and conversation records

**Hybrid Search Fusion**: Uses **RRF (Reciprocal Rank Fusion)**: `rrf_score = 1/(rank + 60)`

### MinerU PDF Parsing

Enhanced PDF parsing service with caching and parallel processing ([ai-service/app/services/mineru_client.py](ai-service/app/services/mineru_client.py)):

- **Auto-splitting**: Large PDFs split into 8-page chunks for processing
- **MD5 caching**: Results cached in MongoDB `mineru_cache` collection
- **Job tracking**: Processing status tracked in `mineru_jobs` collection
- **Web API**: `/api/mineru/jobs`, `/api/mineru/jobs/{id}/markdown`, `/api/mineru/view`

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

### Authentication Status

**Currently DISABLED**: Middleware exists at [ai-service/app/api/middleware/auth.py](ai-service/app/api/middleware/auth.py) but `settings.api_keys` list is empty.

---

## LLM Configuration

### Switching Backends

**Option 1: OpenAI-style API** (default: SiliconFlow/DeepSeek)
```bash
# .env-win file
USE_OLLAMA=false
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.siliconflow.cn/v1
OPENAI_MODEL=deepseek-ai/DeepSeek-V3.1-Terminus
```

**Option 2: Local Ollama**
```bash
# .env-win file
USE_OLLAMA=true
OLLAMA_BASE_URL=http://192.168.8.231:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_KEEP_ALIVE_INTERVAL=180
```

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

| Component | Path |
|-----------|------|
| LangGraph workflow | [ai-service/app/services/conversation_service.py](ai-service/app/services/conversation_service.py) (12 nodes) |
| RAG hybrid search | [ai-service/app/services/rag_service.py](ai-service/app/services/rag_service.py) (RRF fusion) |
| MinerU client | [ai-service/app/services/mineru_client.py](ai-service/app/services/mineru_client.py) (PDF parsing) |
| MinerU API | [ai-service/app/api/endpoints/mineru.py](ai-service/app/api/endpoints/mineru.py) (Web interface) |
| Chat endpoints | [ai-service/app/api/endpoints/chat.py](ai-service/app/api/endpoints/chat.py) (streaming) |
| Configuration | [ai-service/app/core/config.py](ai-service/app/core/config.py) (Pydantic Settings) |
| App entry | [ai-service/main.py](ai-service/main.py) (lifespan management) |
| Embedding cache | [ai-service/app/services/embedding_cache.py](ai-service/app/services/embedding_cache.py) (LRU cache) |
| Ollama keep-alive | [ai-service/app/services/ollama_keepalive.py](ai-service/app/services/ollama_keepalive.py) |
| Performance test | [ai-service/scripts/performance_test.py](ai-service/scripts/performance_test.py) |

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
