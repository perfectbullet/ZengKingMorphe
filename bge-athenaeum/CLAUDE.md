# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This repository provides production-ready deployments for three BGE models:
- **BGE-Reranker-V2-M3**: A lightweight (~278M parameters) reranking model by BAAI
- **BGE-M3**: A multilingual text embedding model (~567M parameters) by BAAI
- **BGE-Large-ZH**: A Chinese text embedding model (~326M parameters) by BAAI

All deployments use vLLM with OpenAI-compatible API, similar to the bge-athenaeum project architecture.

## Quick Start

```bash
# Start services
docker-compose up -d

# Check logs
docker-compose logs -f

# Verify health (Reranker)
curl http://192.168.8.233:8091/health

# Verify health (BGE-M3)
curl http://192.168.8.233:8092/health

# Verify health (BGE-Large)
curl http://192.168.8.233:8093/health

# Run examples tests
cd examples && python test_reranker.py
cd examples && python test_embedding.py
```

## Project Structure

```
.
├── docker-compose.yml          # Service orchestration (vLLM)
├── download_models.sh         # Model download script
├── README.md                  # Deployment documentation
├── CLAUDE.md                  # This file
└── examples/
    ├── __init__.py
    ├── test_reranker.py       # Reranker client (OpenAI SDK style)
    ├── test_embedding.py      # Embedding client (OpenAI SDK)
    └── requirements.txt       # Python dependencies
```

## Deployment Configuration

### Services Overview

| Service | Model | Port | Purpose |
|---------|-------|------|---------|
| bge-reranker | BAAI/bge-reranker-v2-m3 | 8091 | Document reranking (Rerank API) |
| bge-m3 | BAAI/bge-m3 | 8092 | Multilingual text embedding (Embedding API) |
| bge-large | BAAI/bge-large-zh-v1.5 | 8093 | Chinese text embedding (Embedding API) |

### Reranker Service Details

| Setting | Value |
|---------|-------|
| Image | `docker.m.daocloud.io/vllm/vllm-openai:v0.11.0` |
| Container | `bge-reranker-m3` |
| Port | `8091:8000` |
| GPU Memory | 0.2 |
| GPU | NVIDIA GPU 0 |
| Network | `reranker-network` |
| Model Path | `./models/bge-reranker-m3` (mounted to `/models/bge-reranker-m3`) |

### Reranker Model Specifications

- **Model**: BAAI/bge-reranker-v2-m3
- **Parameters**: ~278M
- **Context Length**: 8192 tokens
- **Precision**: float16
- **Score Range**: -10 to +10 (higher = more relevant)

### BGE-M3 Embedding Service Details

| Setting | Value |
|---------|-------|
| Image | `docker.m.daocloud.io/vllm/vllm-openai:v0.11.0` |
| Container | `bge-m3` |
| Port | `8092:8000` |
| GPU Memory | 0.3 |
| GPU | NVIDIA GPU 0 |
| Network | `reranker-network` |
| Model Path | `./models/BAAI/bge-m3` (mounted to `/model/BAAI/bge-m3`) |

### BGE-M3 Model Specifications

- **Model**: BAAI/bge-m3
- **Parameters**: ~567M
- **Context Length**: 8192 tokens
- **Precision**: float16
- **Embedding Dimension**: 1024
- **Supported Languages**: 100+

### BGE-Large Embedding Service Details

| Setting | Value |
|---------|-------|
| Image | `docker.m.daocloud.io/vllm/vllm-openai:v0.11.0` |
| Container | `bge-large` |
| Port | `8093:8000` |
| GPU Memory | 0.2 |
| GPU | NVIDIA GPU 0 |
| Network | `reranker-network` |
| Model Path | `./models/BAAI/BGE-large` (mounted to `/model/BAAI/BGE-large`) |

### BGE-Large Model Specifications

- **Model**: BAAI/bge-large-zh-v1.5
- **Parameters**: ~326M
- **Context Length**: 512 tokens
- **Precision**: float16
- **Embedding Dimension**: 1024
- **Supported Languages**: Chinese (optimized)

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `NVIDIA_VISIBLE_DEVICES` | 0 | GPU device ID |

### Service Configuration Summary

| Config | Reranker | BGE-M3 | BGE-Large |
|--------|----------|--------|-----------|
| Port | 8091 | 8092 | 8093 |
| GPU Memory | 0.2 | 0.3 | 0.2 |
| Max Context | 8192 | 8192 | 512 |
| Precision | float16 | float16 | float16 |

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
    "model": "bge-reranker-m3",
    "query": "查询文本",
    "documents": ["文档1", "文档2", "文档3"],
    "top_n": 3
})
# Returns: {"results": [{"index": 0, "relevance_score": 8.5}, ...]}
```

### Embedding API (Ports 8092 & 8093)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | Model information |
| `/v1/embeddings` | POST | Generate text embeddings |

### Embedding API Example

```python
import requests

# BGE-M3 (port 8092)
response = requests.post("http://192.168.8.233:8092/v1/embeddings", json={
    "model": "BAAI/bge-m3",
    "input": ["你好世界", "Hello World"],
    "encoding_format": "float"
})

# BGE-Large (port 8093)
response = requests.post("http://192.168.8.233:8093/v1/embeddings", json={
    "model": "BAAI/BGE-large",
    "input": ["你好世界", "Hello World"],
    "encoding_format": "float"
})
# Returns: {"data": [{"embedding": [...], "index": 0}, ...]}
```

## Client Usage

### Reranker Client (OpenAI SDK Style)

```python
from examples.test_reranker import BGERerankerClient

client = BGERerankerClient("http://192.168.8.233:8091")

# Basic rerank
results = client.rerank("查询", ["文档1", "文档2"], top_k=2)
# Returns: [(index, document, score), ...]

# Batch queries
results = client.rerank_batch(["查询1", "查询2"], documents, top_k=3)

# Single document relevance score
score = client.compute_relevance(query, document)
```

### Embedding Client (OpenAI SDK)

```python
from examples.test_embedding import BGEEmbeddingClient

# BGE-M3 client (port 8092)
client_m3 = BGEEmbeddingClient("http://192.168.8.233:8092")

# BGE-Large client (port 8093)
client_large = BGEEmbeddingClient("http://192.168.8.233:8093")

# Generate embedding
embedding = client.embed("文本内容")
# Returns: List[float] with 1024 dimensions

# Similarity search
results = client.similarity("查询", ["文档1", "文档2"], top_k=2)
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

for idx, doc, score in final_results:
    print(f"[{score:.4f}] {doc}")
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

**BGE-Large Note:**
- Max context length: 512 tokens
- Optimized for Chinese text
- Use BGE-M3 for long text (>512 tokens) or multilingual content

## Model Storage

### Directory Structure

```
项目目录/
├── models/                           ← Reranker 模型 (~2.2G)
├── models/BAAI/                      ← BAAI 模型目录
│   ├── bge-m3/                      ← M3 Embedding 模型 (~2.2G)
│   └── BGE-large/                   ← Large Embedding 模型 (~1.2G)
└── docker-compose.yml
```

### Host Path vs Container Path

- **Reranker Host**: `./models/bge-reranker-m3/` → **Container**: `/models/bge-reranker-m3/`
- **BGE-M3 Host**: `./models/BAAI/bge-m3/` → **Container**: `/model/BAAI/bge-m3/`
- **BGE-Large Host**: `./models/BAAI/BGE-large/` → **Container**: `/model/BAAI/BGE-large/`

### Model Directory Structure

When using `download_models.sh`, the essential files are:

```
models/BAAI/bge-m3/
├── config.json                    ← Used by vLLM (KEEP)
├── model.safetensors              ← Used by vLLM (KEEP)
├── tokenizer.json                 ← Used by vLLM (KEEP)
├── tokenizer_config.json          ← Used by vLLM (KEEP)
├── special_tokens_map.json        ← Used by vLLM (KEEP)
└── sentencepiece.bpe.model       ← Used by vLLM (KEEP, if present)
```

**Cleanup Command** (removes HF cache):
```bash
rm -rf models/bge-reranker-m3/blobs models/bge-reranker-m3/snapshots models/bge-reranker-m3/refs
rm -rf models/BAAI/bge-m3/blobs models/BAAI/bge-m3/snapshots models/BAAI/bge-m3/refs
rm -rf models/BAAI/BGE-large/blobs models/BAAI/BGE-large/snapshots models/BAAI/BGE-large/refs
```

### Download Models

```bash
# Use download script (recommended)
./download_models.sh bge-reranker-v2-m3    # Download reranker
./download_models.sh bge-m3                # Download m3 embedding
./download_models.sh bge-large             # Download large embedding
./download_models.sh all                   # Download all

# From ModelScope (China, recommended)
git clone https://www.modelscope.cn/AI-ModelScope/bge-reranker-v2-m3.git models/bge-reranker-m3
git clone https://www.modelscope.cn/AI-ModelScope/bge-m3.git models/BAAI/bge-m3
git clone https://www.modelscope.cn/AI-ModelScope/bge-large-zh-v1.5.git models/BAAI/BGE-large

# Or from HuggingFace with mirror
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-m3
huggingface-cli download BAAI/bge-m3 --local-dir models/BAAI/bge-m3
huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir models/BAAI/BGE-large
```


## Model Sources

| Model | HuggingFace | ModelScope |
|-------|-------------|------------|
| bge-reranker-v2-m3 | `BAAI/bge-reranker-v2-m3` | `AI-ModelScope/bge-reranker-v2-m3` |
| bge-m3 | `BAAI/bge-m3` | `AI-ModelScope/bge-m3` |
| bge-large-zh-v1.5 | `BAAI/bge-large-zh-v1.5` | `AI-ModelScope/bge-large-zh-v1.5` |

## Model Comparison

| Model | Parameters | Dim | Context | Languages | Use Case |
|-------|-----------|-----|---------|-----------|----------|
| bge-reranker-v2-m3 | 278M | - | 8192 | CN/EN | Document reranking |
| bge-m3 | 567M | 1024 | 8192 | 100+ | Multilingual embedding |
| bge-large-zh-v1.5 | 326M | 1024 | 512 | Chinese | Chinese embedding (optimized) |

## Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended for all services)
- **CUDA**: 11.8+
- **Docker**: 20.10+ with NVIDIA Container Toolkit
- **Python**: 3.8+ (for examples)

## Dependencies

```bash
# Install example dependencies
pip install openai requests numpy
```

## Reference Architecture

This deployment follows the same pattern as `/home/zj/bge-athenaeum`:
- vLLM-based containerized services
- GPU-accelerated inference
- OpenAI-compatible API
- Client library with OpenAI SDK style
- Local model directory at `./models/`
