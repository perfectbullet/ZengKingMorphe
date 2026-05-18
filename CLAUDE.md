# CLAUDE.md

## Project Overview

LangGraph-based conversational AI service with RAG, multi-turn dialogue, knowledge base management, and web search for digital employee interactions.
**Stack**: FastAPI + LangGraph + DeepSeek/Qwen/Phi-4 (Ollama/OpenAI/vLLM) + ChromaDB + ElasticSearch + MongoDB + Tavily + RAGAnything

---

## Quick Start

```bash
# 1. Environment
conda activate morphe

# 2. Start databases
docker-compose up -d elasticsearch chroma

# 3. Run service
cd ai-service && uvicorn main:app --reload --port 8000

# 4. Test
python -m pytest tests/ -v

# Or use startup script
./ai-service/start_ai_service.sh start|status|logs|stop
```

- API docs: http://localhost:8000/docs
- Docker: port 8100, Local: port 8000

---

## Critical Constraints (MUST follow)

- **Python**: ALL commands use conda env `morphe` (`/home/zj/miniconda3/envs/morphe/bin/python`)
- **Logging**: f-string + Loguru. Errors MUST have `exc_info=True`. User queries truncated to 100 chars (PII)
- **Config**: New external service configs use `os.getenv()` in the service file directly, NOT in `config.py` Settings
- **Streaming**: Only streaming responses exist. No non-streaming chat endpoint
- **Env files**: `.env-local` (local dev, gitignored) / `.env` (Docker/prod)
- **After changes**: Run corresponding tests before marking done: `python -m pytest tests/test_xxx.py -v`

---

## Architecture

**9-node LangGraph StateGraph** with conditional routing:
`load_employee_config -> load_session_context -> input_validation -> classify_query_type -> [branches] -> generate_answer -> save_conversation`

**Triple LLM**: Ollama (fast/simple) | DeepSeek via SiliconFlow (complex/RAG) | Phi-4 vLLM (math)

**Routing**: classify_query_type -> greeting/realtime/normal -> check_math -> evaluate_complexity -> generate_answer. RAG handled by RAGAnything within generate_answer.

**Core paths**:
- `ai-service/app/services/conversation_service.py` - Graph builder, LLM init
- `ai-service/app/services/conversation/conversation_nodes.py` - All node implementations
- `ai-service/app/services/conversation/conversation_state.py` - ~40-field state TypedDict
- `ai-service/app/services/raganything_wrapper.py` - RAG integration
- `ai-service/app/api/endpoints/chat.py` - Streaming chat SSE (v1/v2)
- `ai-service/app/core/config.py` - Pydantic Settings

-> Workflow details: `docs/ConversationWorkflow详解.md`
-> Architecture diagrams: `docs/ConversationWorkflow流程图与架构图.md`
-> RAG details: `docs/RAG核心流程与代码说明.md`

---

## Development Patterns

### Add Conversation Node
1. Define in `ConversationNodes` class (conversation_nodes.py) with `async with time_node("name", state):`
2. Add to graph in `_build_workflow()` (conversation_service.py): `graph.add_node("name", self.nodes.method)`
3. Wire edges: `graph.add_edge("prev", "name")` / `graph.add_edge("name", "next")`

### Add API Endpoint
1. Create router in `app/api/endpoints/`
2. Register in `main.py`: `app.include_router(router, prefix="/api/xxx", tags=["XXX"])`
3. Response schema: `{ code: int, message: str, data: ... }`

### Debug LangGraph
- Graph debug output: `graph_debug/` (config: `CRAG_DUMP_GRAPH=1`, `CRAG_GRAPH_DIR`)
- Mermaid format export available

---

## Deep Dive Docs

| Topic | Location |
|-------|----------|
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
