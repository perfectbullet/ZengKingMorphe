# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This repository provides production-ready deployments for two BGE models:
- **BGE-Reranker-V2-M3**: A lightweight (~278M parameters) reranking model by BAAI
- **BGE-M3**: A multilingual text embedding model (~567M parameters) by BAAI

Both deployments use vLLM with OpenAI-compatible API, similar to the bge-athenaeum project architecture.

## Quick Start

```bash
# Start services
docker-compose up -d

# Check logs
docker-compose logs -f

# Verify health (Reranker)
curl http://192.168.8.233:8091/health

# Verify health (Embedding)
curl http://192.168.8.233:8092/health

# Run examples tests
cd examples && python test_reranker.py
cd examples && python test_embedding.py
```

## Project Structure

```
.
├── docker-compose.yml          # Service orchestration (vLLM)
├── download_models.sh         # Model download script
├── start.sh                  # Service startup script
├── README.md                 # Deployment documentation
├── CLAUDE.md                 # This file
└── examples/
    ├── __init__.py
    ├── test_reranker.py       # Reranker examples with examples
    ├── test_embedding.py      # Embedding examples with examples
    └── requirements.txt       # Python dependencies
```

## Deployment Configuration

### Services Overview

| Service | Model | Port | Purpose |
|---------|-------|------|---------|
| bge-reranker | BAAI/bge-reranker-v2-m3 | 8091 | Document reranking (Rerank API) |
| bge-m3 | BAAI/bge-m3 | 8092 | Text embedding generation (Embedding API) |

### Reranker Service Details

| Setting | Value |
|---------|-------|
| Image | `docker.m.daocloud.io/vllm/vllm-openai:v0.11.0` |
| Container | `bge-reranker-m3` |
| Port | `8091:8000` |
| GPU Memory | 0.2 |
| GPU | NVIDIA GPU 0 |
| Network | `reranker-network` |
| Model Path | `./models` (mounted to `/model`) |

### Reranker Model Specifications

- **Model**: BAAI/bge-reranker-v2-m3
- **Parameters**: ~278M
- **Context Length**: 8192 tokens
- **Precision**: float16
- **Score Range**: -10 to +10 (higher = more relevant)

### Embedding Service Details

| Setting | Value |
|---------|-------|
| Image | `docker.m.daocloud.io/vllm/vllm-openai:v0.11.0` |
| Container | `bge-m3` |
| Port | `8092:8000` |
| GPU Memory | 0.3 |
| GPU | NVIDIA GPU 0 |
| Network | `reranker-network` |
| Model Path | `./models/BAAI/bge-m3` (mounted to `/model`) |

### Embedding Model Specifications

- **Model**: BAAI/bge-m3
- **Parameters**: ~567M
- **Context Length**: 8192 tokens
- **Precision**: float16
- **Embedding Dimension**: 1024
- **Supported Languages**: 100+

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `NVIDIA_VISIBLE_DEVICES` | 0 | GPU device ID |

## API Endpoints

### Reranker API (Port 8091)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | Model information |
| `/v1/rerank` | POST | Document reranking |

### Rerank API Example

```python
import requests

response = requests.post("http://192.168.8.233:8091/v1/rerank", json={
    "model": "/model",
    "query": "查询文本",
    "documents": ["文档1", "文档2", "文档3"],
    "top_n": 3
})
# Returns: {"results": [{"index": 0, "relevance_score": 8.5}, ...]}
```

### Embedding API (Port 8092)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | Model information |
| `/v1/embeddings` | POST | Generate text embeddings |

### Embedding API Example

```python
import requests

response = requests.post("http://192.168.8.233:8092/v1/embeddings", json={
    "model": "/model",
    "input": ["你好世界", "Hello World"],
    "encoding_format": "float"
})
# Returns: {"data": [{"embedding": [...], "index": 0}, ...]}
```

## examples Usage

### Reranker examples

```python
from examples.test_reranker import BGERerankerClient

client = BGERerankerClient("http://192.168.8.233:8091")

# Basic rerank
results = examples.rerank("查询", ["文档1", "文档2"], top_k=2)
# Returns: [(index, document, score), ...]

# Batch queries
results = examples.rerank_batch(["查询1", "查询2"], documents, top_k=3)
```

### Embedding examples

```python
from examples.test_embedding import BGEEmbeddingClient

client = BGEEmbeddingClient("http://192.168.8.233:8092")

# Generate embedding
embedding = examples.embed("文本内容")
# Returns: List[float] with 1024 dimensions

# Similarity search
results = examples.similarity("查询", ["文档1", "文档2"], top_k=2)
# Returns: [(index, document, score), ...]

# Long text test
client.test_long_text_embedding([1000, 2000, 4000, 8000])
```

### Complete RAG Example

```python
from examples.test_embedding import BGEEmbeddingClient
from examples.test_reranker import BGERerankerClient

embedding_client = BGEEmbeddingClient("http://192.168.8.233:8092")
reranker_client = BGERerankerClient("http://192.168.8.233:8091")

# Knowledge base
knowledge_base = [
    "BGE-M3 is a multilingual embedding model.",
    "BGE-Reranker-V2-M3 is used for reranking.",
]

# Step 1: Coarse retrieval with embeddings
query = "What is BGE?"
candidates = embedding_client.similarity(query, knowledge_base, top_k=4)

# Step 2: Fine reranking
candidate_docs = [doc for idx, doc, score in candidates]
final_results = reranker_client.rerank(query, candidate_docs, top_k=2)
```

## Long Text Testing

BGE-M3 has been tested with Chinese text from 1000 to 8000 characters:

| Text Length | Duration | Embedding Dim |
|-------------|-----------|---------------|
| 1000 chars | ~4.00s | 1024 |
| 2000 chars | ~0.06s | 1024 |
| 4000 chars | ~0.15s | 1024 |
| 8000 chars | ~0.45s | 1024 |

**Notes:**
- All lengths tested successfully
- Embedding dimension is constant at 1024
- First call loads model (~4s), subsequent calls are fast
- Supports max 8192 tokens context length

## Model Storage

### Directory Structure

```
项目目录/
├── models/                 ← Reranker 模型 (~2.2G)
├── models/BAAI/           ← BAAI 模型目录
│   └── bge-m3/           ← Embedding 模型 (~2.2G)
└── docker-compose.yml
```

### Host Path vs Container Path

- **Reranker Host**: `./models/` → **Container**: `/model/`
- **Embedding Host**: `./models/BAAI/bge-m3/` → **Container**: `/model/`

### Model Directory Structure

When using `hf download --local-dir`, directory structure looks like:

```
models/
├── config.json                    ← Used by vLLM (KEEP - 4K)
├── model.safetensors              ← Used by vLLM (KEEP - 2.2G)
├── tokenizer.json                 ← Used by vLLM (KEEP - 17M)
├── sentencepiece.bpe.model        ← Used by vLLM (KEEP - 4.9M)
├── tokenizer_config.json          ← Used by vLLM (KEEP - 4K)
├── special_tokens_map.json        ← Used by vLLM (KEEP - 1K)
│
├── blobs/                         ← HF cache originals (SAFE TO DELETE - 2.2G)
├── snapshots/                     ← HF versioning symlinks (SAFE TO DELETE)
├── refs/                          ← HF references (SAFE TO DELETE)
├── .cache/                        ← HF cache (SAFE TO DELETE)
└── assets/                        ├── Promo images (SAFE TO DELETE)
```

**Important Notes**:
- `hf download --local-dir` creates duplicate files (root dir + blobs)
- Total disk usage: ~4.3G, but only ~2.2G is actually needed
- vLLM (mounted at `/model`) only uses the root directory files
- HF cache directories (`blobs/`, `snapshots/`, etc.) are NOT used by vLLM

**Cleanup Command** (saves ~2.1G):
```bash
rm -rf models/blobs models/snapshots models/refs models/.cache models/assets
rm -rf models/BAAI/bge-m3/blobs models/BAAI/bge-m3/snapshots models/BAAI/bge-m3/refs models/BAAI/bge-m3/.cache models/BAAI/bge-m3/assets
```

### Download Models

```bash
# Use download script (recommended)
./download_models.sh bge-reranker-v2-m3    # Download reranker
./download_models.sh bge-m3                # Download embedding
./download_models.sh all                   # Download all

# From ModelScope (China, recommended)
git clone https://www.modelscope.cn/AI-ModelScope/bge-reranker-v2-m3.git models
git clone https://www.modelscope.cn/AI-ModelScope/bge-m3.git models/BAAI/bge-m3

# Or from HuggingFace with mirror
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models
huggingface-cli download BAAI/bge-m3 --local-dir models/BAAI/bge-m3
```

### Migration

```bash
# Pack entire project
tar czf bge-deployment.tar.gz bge-reranker-v2-m3-deployment/

# Unpack on new machine
tar xzf bge-deployment.tar.gz
cd bge-reranker-v2-m3-deployment && docker-compose up -d
```

## Model Sources

| Model | HuggingFace | ModelScope |
|-------|-------------|------------|
| bge-reranker-v2-m3 | `BAAI/bge-reranker-v2-m3` | `AI-ModelScope/bge-reranker-v2-m3` |
| bge-m3 | `BAAI/bge-m3` | `AI-ModelScope/bge-m3` |

## Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: 11.8+
- **Docker**: 20.10+ with NVIDIA Container Toolkit
- **Python**: 3.8+ (for examples)

## Reference Architecture

This deployment follows the same pattern as `/home/zj/bge-athenaeum`:
- vLLM-based containerized services
- GPU-accelerated inference
- OpenAI-compatible API
- examples library with examples
- Local model directory at `./models/`
