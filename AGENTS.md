# AGENTS.md

This file gives coding-agent instructions for this repository. It applies to the whole repository unless a deeper `AGENTS.md` overrides it.

## Project overview

`ZengKingMorphe` is a Digital Employee AI Service. The main application lives in `ai-service/` and is a FastAPI service with LangGraph-based conversation flow, LightRAG knowledge-base support, OpenAI-compatible streaming chat APIs, web search, MinerU document parsing, and integration with MongoDB, ElasticSearch, vLLM/Ollama/OpenAI-compatible LLM services.

## Working directory

- Repository root: infrastructure files, docs, startup scripts, top-level project guidance.
- Main Python service: `ai-service/`.
- Application entry: `ai-service/main.py`.
- Main source package: `ai-service/app/`.
- Tests: `ai-service/tests/`.
- Do not commit generated logs, local env files, uploaded documents, benchmark outputs, temporary JSON/TXT exports, or model artifacts.

## Environment and commands

Prefer Linux/WSL shell commands.

```bash
# Activate the project Python environment when available
conda activate morphe

# Install/update Python dependencies
/home/zj/miniconda3/envs/morphe/bin/python -m pip install -r ai-service/requirements.txt

# Start currently configured docker services from repo root
# Note: docker-compose.yml currently enables ElasticSearch; do not assume MongoDB/Milvus/Neo4j are started by this file.
docker-compose up -d elasticsearch

# Run the API locally on port 8000
cd ai-service
/home/zj/miniconda3/envs/morphe/bin/python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
cd ai-service
/home/zj/miniconda3/envs/morphe/bin/python -m pytest tests/ -v

# Run one focused test file
cd ai-service
/home/zj/miniconda3/envs/morphe/bin/python -m pytest tests/test_api.py -v

# Run coverage
cd ai-service
/home/zj/miniconda3/envs/morphe/bin/python -m pytest tests/ --cov=app --cov-report=html -v
```

The root startup script is:

```bash
./start_ai_service.sh start|status|logs|stop|restart
```

It starts uvicorn on port `8100` and currently expects a root-level `venv/`. For manual development, prefer the `morphe` conda interpreter above unless the user explicitly wants the script path/debugged.

Useful URLs when running locally:

- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health: `http://localhost:8000/health`

## Architecture map

Key files and responsibilities:

| Area | File/Directory |
|---|---|
| FastAPI app, lifespan, router registration | `ai-service/main.py` |
| API endpoints | `ai-service/app/api/endpoints/` |
| Middleware: auth, rate-limit, error handling | `ai-service/app/api/middleware/` |
| Core config, logging, database clients | `ai-service/app/core/` |
| Pydantic schemas and data models | `ai-service/app/models/` |
| Business services | `ai-service/app/services/` |
| LangGraph graph builder / LLM setup | `ai-service/app/services/conversation_service.py` |
| Conversation node implementations | `ai-service/app/services/conversation/conversation_nodes.py` |
| Conversation state TypedDict | `ai-service/app/services/conversation/conversation_state.py` |
| Intent routing table | `ai-service/app/services/conversation/intent_routing.py` |
| LightRAG file knowledge-base integration | `ai-service/app/services/training_lightrag_wrapper.py` |
| Query classification | `ai-service/app/services/query_classifier.py` |
| Tests | `ai-service/tests/` |

## Coding conventions

- Use Python 3.11+ syntax compatible with the current dependency set.
- Keep changes focused. Do not reformat unrelated files.
- Follow the existing async FastAPI style. Prefer async endpoints and async service calls where the surrounding code is async.
- Use existing Pydantic schemas from `app/models/` or add new schemas there for request/response contracts.
- Register new API routers in `ai-service/main.py` with a clear prefix and tag.
- Preserve OpenAI-compatible endpoint behavior under `/api/chat/v1/chat/completions` and `/api/chat/v2/chat/completions` unless explicitly asked to change it.
- Chat completion currently supports streaming responses. Non-streaming requests are not implemented in the v1 handler, so do not add callers/tests that assume non-streaming chat works unless implementing that feature.
- Prefer small helper functions over large endpoint functions when adding business logic.
- For comments and documentation, match nearby style. Existing project comments are often Chinese; keep Chinese comments when they improve maintainability for this repo.

## Logging and error handling

- Use `app.core.logging.get_logger(__name__)` and Loguru-style logging consistent with existing files.
- Include `exc_info=True` when logging caught exceptions that need stack traces.
- Do not log API keys, JWT secrets, full Authorization headers, or full user queries.
- Truncate user-provided text in logs. Existing code commonly logs short previews only.
- Keep FastAPI errors user-safe: internal exception details belong in logs, not response bodies.

## Configuration rules

- Never commit `.env`, `.env-local`, `ai-service/.env*`, secrets, tokens, private API keys, or machine-specific credentials.
- Existing core settings live in `ai-service/app/core/config.py` via `Settings`.
- When adding new external service toggles or URLs, follow the pattern already used by the surrounding service. If the change is local to one service, prefer `os.getenv("KEY", default)` in that service rather than expanding global settings without need.
- Keep production/default values safe. Avoid hardcoding private LAN addresses in new code unless the user explicitly asks and the existing code path already depends on that environment.

## Conversation workflow changes

When adding or changing LangGraph conversation behavior:

1. Add or update the node implementation in `ConversationNodes` under `ai-service/app/services/conversation/conversation_nodes.py`.
2. Update the state shape in `conversation_state.py` when new state fields are required.
3. Register the node and edges in `_build_workflow()` in `conversation_service.py`.
4. Update route decisions in `intent_routing.py` or the relevant classifier/router logic.
5. Add focused tests for the new branch or at least verify the affected endpoint manually with a streaming request.

Keep the graph easy to inspect. If using graph debug output, write it under ignored debug/output directories and do not commit generated `.mmd`, images, or logs.

## RAG, document parsing, and external services

- LightRAG, MongoDB, ElasticSearch, embedding services, rerankers, MinerU, and vLLM/Ollama endpoints may be environment-dependent. Treat failures as integration-environment issues unless the code path is clearly wrong.
- For RAG/document changes, avoid assuming every dependency is locally available. Add graceful fallbacks and clear logs where the current architecture already does so.
- Keep file-upload and document-processing paths safe. Do not introduce path traversal risks; normalize and validate user-controlled filenames/paths.
- Do not commit uploaded documents, extracted images, parsed PDF chunks, large JSON outputs, or benchmark artifacts.

## Testing guidance

Before marking a code change done, run the narrowest useful test command, for example:

```bash
cd ai-service
/home/zj/miniconda3/envs/morphe/bin/python -m pytest tests/test_api.py -v
```

For quick local streaming-chat smoke tests, use the project root and the active local or server Python environment:

```bash
# Local mathematics development environment; the v2 service must already be running.
cd /home/zj/ZengKingMorphe-math/ai-service
/home/zj/miniconda3/envs/morphe/bin/python -m tests.test_chat_stream_v2 -q "求解不等式x的平方减去5x加上6小于0的解" --host http://localhost:8100
/home/zj/miniconda3/envs/morphe/bin/python -m tests.test_chat_stream_v2 -q "北京今天天气怎么样" --host http://localhost:8100

# Equivalent when the conda environment has been activated
conda activate morphe
python -m tests.test_chat_stream_v2 -q "求解不等式x的平方减去5x加上6小于0的解" --host http://localhost:8100
python -m tests.test_chat_stream_v2 -q "北京今天天气怎么样" --host http://localhost:8100

# Test meaningless input handling
python -m tests.test_chat_stream_v2 -q "什么是土豆什么是马铃薯" --host http://localhost:8100
python -m tests.test_chat_stream_v2 -q "负三的值阿巴阿巴阿巴" --host http://localhost:8100
```

For endpoint, routing, or conversation changes, prefer adding or updating tests under `ai-service/tests/`. If tests cannot run because local services or secrets are unavailable, state exactly what was not run and why.

Some tests may depend on application startup side effects and external services. Do not hide failing tests. Report the failure, identify whether it is a code regression or environment/test-staleness issue, and keep the patch minimal.

## Git and review hygiene

- Keep commits small and scoped to the requested task.
- Do not rewrite history or force-push unless explicitly asked.
- Do not change generated files, caches, local settings, or ignored artifacts.
- If a task touches public API behavior, document the behavior change in the relevant README/docs or tests.
- Prefer a direct fix over broad cleanup. If you notice unrelated problems, mention them separately instead of changing them in the same patch.
