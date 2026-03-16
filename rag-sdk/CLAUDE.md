# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LlamaRAG SDK is a Python RAG (Retrieval-Augmented Generation) SDK built on LlamaIndex, optimized for textbook documents (Chinese language and high school math). It supports image description via Qwen2-VL and formula parsing.

**Tech Stack**: Python 3.10+, LlamaIndex + ChromaDB + Ollama + Pydantic Settings + asyncio

**External Services**:
- **Ollama** - Embedding (bge-m3) and multimodal models (qwen2-vl)
- **ChromaDB** - Vector storage (remote: 192.168.8.233:8200, or local)
- **MinerU MCP** - PDF to Markdown parsing (localhost:8001)

---

## Essential Development Commands

### Python Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Or install with all optional dependencies
pip install -e .[all]

# Install for development only
pip install -e .[dev]
```

### Testing

```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/test_parser.py
pytest tests/test_indexer.py
pytest tests/test_retriever.py

# Run with coverage
pytest --cov=src tests/

# Run specific test function
pytest tests/test_parser.py::test_parse_document -v
```

### Code Quality

```bash
# Format code with Black
black src/ tests/ examples/

# Lint with Ruff
ruff check src/ tests/ examples/

# Type check with mypy
mypy src/
```

### Running Examples

```bash
# Basic usage demo
python examples/basic_usage.py

# Document parsing test
python examples/parse_test.py

# Indexing test
python examples/index_test.py

# Retrieval test
python examples/retrieve_test.py

# Full RAG system demo
python examples/rag_system_demo.py
```

---

## Configuration

The SDK uses Pydantic Settings (`src/config.py`) to manage all environment variables. Configuration is loaded from `.env` file in the project root.

**Setup**:
```bash
# Copy example environment file
cp .env.example .env

# Edit with your configuration
nano .env
```

**Key Configuration Categories**:

| Category | Key Env Vars | Default |
|----------|--------------|---------|
| MinerU | `MINERU_MCP_URL`, `USE_LOCAL_MINERU`, `MINERU_OUTPUT_DIR` | http://localhost:8001 |
| Ollama | `OLLAMA_BASE_URL`, `OLLAMA_EMBEDDING_MODEL`, `OLLAMA_CHAT_MODEL` | bge-m3 |
| ChromaDB | `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_COLLECTION_NAME` | documents |
| Image | `QWEN_VL_MODEL`, `ENABLE_IMAGE_DESCRIPTION` | qwen2-vl:latest |
| Indexing | `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` | 512, 50, 5 |
| Retrieval | `USE_HYBRID_RETRIEVAL`, `USE_RERANK`, `RERANK_MODEL` | false, false |

**Accessing Config**:
```python
from src.config import settings

# All config values are available on the settings instance
model = settings.ollama_embedding_model
host = settings.chroma_host
```

---

## Architecture Overview

### Three-Layer Architecture

The SDK follows a three-layer pipeline: **Document Parsing → Indexing → Retrieval**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Document       │    │  Document       │    │  Retrieval      │
│  Parser         │───▶│  Indexer        │───▶│  Engine         │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ MinerUParser    │    │ Chunker         │    │ Retriever       │
│ ImageDescriptor │    │ VectorStore     │    │ Strategies      │
│ ParsedDocument  │    │ DocumentIndexer │    │ RetrievedDoc    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
     Base Classes           Base Classes            Base Classes
```

### 1. Document Parser Layer (`src/document_parser/`)

**Base Classes** (`base.py`):
- `ParsedDocument` - Document with title, content, chunks, images
- `TextChunk` - Text chunk with page, section, metadata
- `ImageInfo` - Image with path, description, position
- `DocumentParser` - Abstract base class for parsers

**Implementations**:
- `MinerUParser` (`mineru_client.py`) - MinerU MCP client for PDF → Markdown
- `ImageDescriptor` (`image_processor.py`) - Qwen2-VL for image descriptions

### 2. Document Indexer Layer (`src/document_indexer/`)

**Base Classes** (`base.py`):
- `ChunkStrategy` - Config for chunking (type, size, overlap)
- `Indexer` - Abstract base class for indexers

**Implementations**:
- `Chunker` (`chunker.py`) - Three strategies: FixedSize, Semantic, Hybrid
- `VectorStore` (`storage.py`) - ChromaDB wrapper with remote/local support
- `DocumentIndexer` (`indexer.py`) - Main indexer combining chunker + embeddings

### 3. Retrieval Layer (`src/retrieval/`)

**Base Classes** (`base.py`):
- `RetrievedDocument` - Result with text, metadata, score, source
- `RetrievalStrategy` - Abstract base class for retrieval strategies

**Implementations**:
- `strategies.py` - VectorRetrieval, HybridRetrieval, RerankRetrieval
- `retriever.py` - Unified `Retriever` class with single/multi-query support

### 4. Integration Layer (`src/rag_system.py`)

**RAGSystem** - High-level API that integrates all layers:
```python
async with RAGSystem(collection_name="textbook") as rag:
    # Parse + index document
    await rag.index_document("path/to/textbook.pdf")

    # Retrieve
    results = await rag.retrieve("什么是导数？", top_k=5)
```

---

## Key Conventions

### Async/Await Pattern

All I/O operations are async. Use `asyncio.run()` for top-level execution:

```python
import asyncio
from src.rag_system import RAGSystem

async def main():
    async with RAGSystem() as rag:
        docs = await rag.retrieve("query")

asyncio.run(main())
```

### Logging

Use loguru for logging (configured via `src/utils.py`):

```python
from src.utils import setup_logger

logger = setup_logger(log_level="INFO", log_file="./logs/app.log")
logger.info("Processing document", count=5)
logger.error("Failed to index", error=str(e), exc_info=True)
```

### Pydantic Models

All data models use Pydantic for validation:

```python
from src.document_parser.base import ParsedDocument, TextChunk

# Create chunk
chunk = TextChunk(
    text="Some content",
    page=1,
    section="Introduction",
    index=0
)
```

### Module Imports

Always import from module `__init__.py` when possible:
```python
# Good
from src.rag_system import RAGSystem
from src.document_parser.mineru_client import MinerUParser

# Also good (via __init__.py)
from src import rag_system
```

---

## Common Patterns

### Adding a New Chunk Strategy

1. Add strategy type to `ChunkStrategy.type` enum
2. Implement in `src/document_indexer/chunker.py`:
```python
class NewChunker(BaseChunker):
    async def chunk(self, text: str, strategy: ChunkStrategy) -> List[str]:
        # Your chunking logic here
        pass
```

### Adding a New Retrieval Strategy

1. Inherit from `RetrievalStrategy` in `src/retrieval/strategies.py`:
```python
class NewRetrieval(RetrievalStrategy):
    async def retrieve(self, query: str, top_k: int, filters=None):
        # Your retrieval logic here
        return documents
```

### Extending RAGSystem

Add new methods to `src/rag_system.py` following the existing pattern:
```python
async def my_new_feature(self, param: str) -> ResultType:
    """Description here"""
    # Access components via properties
    docs = await self.indexer.get_collection_stats(...)
    return result
```

---

## Directory Structure

```
rag-sdk/
├── src/                     # Source code (Python 3.10+)
│   ├── config.py            # Pydantic Settings configuration
│   ├── utils.py             # Utility functions (logging, text processing)
│   ├── rag_system.py        # Main RAGSystem integration class
│   ├── document_parser/     # Document parsing layer
│   │   ├── base.py         # Data models and abstract base
│   │   ├── mineru_client.py # MinerU MCP client
│   │   └── image_processor.py # Qwen2-VL image descriptions
│   ├── document_indexer/    # Document indexing layer
│   │   ├── base.py         # ChunkStrategy and Indexer base classes
│   │   ├── chunker.py      # Chunking strategies
│   │   ├── storage.py      # ChromaDB wrapper
│   │   └── indexer.py      # DocumentIndexer main class
│   └── retrieval/           # Retrieval layer
│       ├── base.py         # RetrievedDocument and RetrievalStrategy
│       ├── strategies.py    # Retrieval strategy implementations
│       └── retriever.py    # Retriever main class
├── tests/                   # pytest tests
├── examples/                # Usage examples
├── data/                    # Data directories
│   ├── sample_pdf/         # Test PDF files
│   └── output/             # MinerU parsing output
├── logs/                    # Log files
├── .env.example             # Environment variables template
├── pyproject.toml          # Project metadata and dependencies
├── requirements.txt        # Pip requirements
└── README.md              # User documentation
```

---

## Troubleshooting

### Common Issues

**Issue**: `Connection refused` when connecting to Ollama
- **Cause**: Ollama service not running
- **Fix**: Ensure `OLLAMA_BASE_URL` is correct and Ollama is running

**Issue**: ChromaDB connection errors
- **Cause**: ChromaDB not running or wrong port
- **Fix**: Check `CHROMA_HOST` and `CHROMA_PORT`, verify ChromaDB is accessible

**Issue**: "No module named" errors
- **Cause**: Dependencies not installed
- **Fix**: Run `pip install -r requirements.txt` or `pip install -e .`

**Issue**: Image description fails
- **Cause**: Qwen2-VL model not available in Ollama
- **Fix**: Pull model: `ollama pull qwen2-vl` or set `ENABLE_IMAGE_DESCRIPTION=false`

### Debugging

Enable debug logging in `.env`:
```bash
LOG_LEVEL=DEBUG
```

Check service connectivity:
```python
import httpx
async with httpx.AsyncClient() as client:
    response = await client.get("http://192.168.8.233:11434/api/tags")
    print(response.json())  # Lists available Ollama models
```

---

## External Dependencies

### Ollama Setup

```bash
# Pull required models
ollama pull bge-m3              # Embedding model
ollama pull qwen2-vl:latest    # Image description model
ollama pull deepseek-coder     # Chat model (optional)

# Verify installation
ollama list
```

### ChromaDB Setup

**Remote** (default):
```bash
docker run -d -p 8200:8000 chromadb/chroma
```

**Local**:
Set `CHROMA_USE_REMOTE=false` in `.env`

### MinerU MCP

```bash
# Run MinerU MCP server
# Follow instructions from MinerU MCP documentation
```

---

## Project Metadata

- **Python Version**: 3.10+
- **License**: MIT
- **Package Name**: llama-rag-sdk
- **Version**: 0.1.0
