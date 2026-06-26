# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LangGraph-based conversational AI service with RAG, multi-turn dialogue, knowledge base management, and web search for digital employee interactions.
**Stack**: FastAPI + LangGraph + DeepSeek/Qwen/Phi-4 (Ollama/OpenAI/vLLM) + ChromaDB + ElasticSearch + MongoDB + Tavily + RAGAnything

---

## Commands

```bash
# Environment (ALL Python commands must use this)
conda activate morphe

# Run service (local dev)
cd ai-service && uvicorn main:app --reload --port 8000

# Or use startup script (port 8100)
./ai-service/start_ai_service.sh start|status|logs|stop|restart

# Run all tests
cd ai-service && python -m pytest tests/ -v

# Run single test file
cd ai-service && python -m pytest tests/test_chat_stream_v1.py -v

# Run with coverage
cd ai-service && python -m pytest tests/ --cov=app --cov-report=html -v

# Start databases
docker-compose up -d elasticsearch chroma
```

- API docs: http://localhost:8000/docs
- Docker: port 8100, Local: port 8000

---

## Critical Constraints (MUST follow)

- **Python**: ALL commands use conda env `morphe` (`/home/zj/miniconda3/envs/morphe/bin/python`)
- **Logging**: MUST use f-string + Loguru for every `logger.*(...)` call. DO NOT use logger parameter interpolation such as `logger.info("x=%s", x)`, `logger.info("x={}", x)`, or keyword interpolation. Always write `logger.info(f"x={x}")`. Preserve `exc_info=True` for error logs. Preserve truncation for user queries / long content. This rule is mandatory and must not be weakened or removed.
- **Config**: New external service configs use `os.getenv()` in the service file directly, NOT in `config.py` Settings. Keep `config.py` for core services only.
- **Streaming**: Only streaming responses exist. No non-streaming chat endpoint.
- **Env files**: `.env-local` (local dev, gitignored) / `.env` (Docker/prod)
- **After changes**: Run corresponding tests before marking done: `python -m pytest tests/test_xxx.py -v`

---

## Logging Style — Mandatory

All Python logger calls in this repository MUST use f-strings. This is a hard rule for every `logger.*(...)` call (including `logger.bind(...).<method>(...)` chains), enforced by `ai-service/scripts/check_logger_fstring.py`.

Allowed:

````python
logger.info(f"query_changed={query_changed} | query={query[:200]!r}")
logger.error(f"failed | error={e}", exc_info=True)
logger.exception(f"failed | query={query[:200]!r}")
````

Forbidden:

````python
logger.info("query_changed=%s | query=%r", query_changed, query[:200])
logger.info("value={}", value)
logger.info("value={value}", value=value)
````

When converting a parameterized call:

- `logger.info("x=%s", x)` → `logger.info(f"x={x}")`
- `logger.info("x=%r", x)` → `logger.info(f"x={x!r}")` (keep `!r` semantics)
- Preserve `exc_info=True` on `error` / `warning` / `exception` calls.
- Preserve existing truncation (e.g. `query[:200]`, `content[:500]`); do not log full user queries / long content.

Do not weaken, remove, or bypass this rule in future changes.

---

## Architecture

### LangGraph Conversation Workflow

**9-node StateGraph** with conditional routing:
```
load_employee_config → load_session_context → input_validation → classify_query_type
  → [greeting/realtime/normal branches] → generate_answer → save_conversation → END
```

**Routing logic** (in `classify_query_type`): 9 intent categories including math_problem, concept_explain, greeting, realtime_query, noise. Hybrid routing uses local Ollama for simple queries and remote API for complex/RAG.

**Triple LLM backend**:
- **Ollama** (local) — fast/simple queries, greetings
- **DeepSeek via SiliconFlow** (remote OpenAI-compatible) — complex/RAG queries
- **Math LLM via vLLM** — math problems, configured via `MATH_LLM_*` env vars

### Core File Map

| Responsibility | File |
|---|---|
| Graph builder, LLM init | `ai-service/app/services/conversation_service.py` |
| All node implementations | `ai-service/app/services/conversation/conversation_nodes.py` |
| ~40-field state TypedDict | `ai-service/app/services/conversation/conversation_state.py` |
| RAG integration (singleton) | `ai-service/app/services/raganything_wrapper.py` |
| Streaming chat SSE (v1/v2) | `ai-service/app/api/endpoints/chat.py` |
| Pydantic Settings | `ai-service/app/core/config.py` |
| FastAPI app + lifespan | `ai-service/main.py` |

### Service Layer Structure

```
ai-service/app/services/
├── conversation/
│   ├── conversation_nodes.py    # Node implementations
│   ├── conversation_state.py    # State TypedDict
│   └── intent_routing.py        # Intent → route mapping table
├── conversation_service.py      # Graph builder, _build_workflow()
├── raganything_wrapper.py       # RAGAnything singleton
├── query_classifier.py          # Query type classification
└── task_processor.py            # Task processing engine
```

---

## Development Patterns

### Add a Conversation Node

1. Define method in `ConversationNodes` class (`conversation_nodes.py`) using `async with time_node("name", state):`
2. Add node to graph in `_build_workflow()` (`conversation_service.py`): `graph.add_node("name", self.nodes.method)`
3. Wire edges: `graph.add_edge("prev", "name")` or `graph.add_conditional_edges("name", router_func)`

### Add an API Endpoint

1. Create router in `app/api/endpoints/`
2. Register in `main.py`: `app.include_router(router, prefix="/api/xxx", tags=["XXX"])`
3. Response schema: `{ code: int, message: str, data: ... }`

### SSE Streaming Pattern

All chat endpoints use `EventSourceResponse` with async generators. Token buffering applies sentence-level chunking with timeout. No non-streaming alternative exists.

### Configuration Pattern

- Core service configs (DB URLs, API keys for primary services) → `config.py` Settings class
- New external service configs → `os.getenv("KEY", default)` directly in the service file
- LLM routing mode: `LLM_ROUTING_MODE` env var (local_only / remote_only / hybrid)
- Math LLM: separate `MATH_LLM_BASE_URL`, `MATH_LLM_MODEL`, `MATH_LLM_API_KEY` env vars

### Debug LangGraph

- Graph debug output: `graph_debug/` (env: `CRAG_DUMP_GRAPH=1`, `CRAG_GRAPH_DIR`)
- Mermaid format export available

---

## Deep Dive Docs

| Topic | Location |
|-------|----------|
| Workflow details | `docs/ConversationWorkflow详解.md` |
| Architecture diagrams | `docs/ConversationWorkflow流程图与架构图.md` |
| RAG details | `docs/RAG核心流程与代码说明.md` |
| LLM routing & configuration | `docs/llm-configuration.md` |
| Troubleshooting | `docs/troubleshooting.md` |
| Deployment | `docs/部署指南.md` |
| Performance optimization | `docs/性能优化总结.md` |
| Web search integration | `docs/联网检索功能使用指南.md` |
| Ollama keep-alive | `docs/Ollama模型保活方案.md` |
| Embedding integration | `docs/Ollama-Embedding集成说明.md` |
| Employee integration | `docs/数字员工信息集成总结.md` |
| Document upload (async) | `docs/异步文档上传使用说明.md` |
| Custom chunking RAG | `docs/自定义分段策略RAG文档创建接口说明.md` |
