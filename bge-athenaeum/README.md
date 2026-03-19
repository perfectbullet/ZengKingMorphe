# BGE 模型服务部署

本项目提供 **BGE-Reranker-V2-M3**（文档重排序）、**BGE-M3**（多语言文本嵌入）和 **BGE-Large**（中文文本嵌入）三个模型的 vLLM 部署服务，支持 OpenAI 兼容 API。

## 服务概览

| 服务 | 模型 | 端口 | 用途 |
|------|------|------|------|
| bge-reranker | BAAI/bge-reranker-v2-m3 | 8091 | 文档重排序（Rerank API） |
| bge-m3 | BAAI/bge-m3 | 8092 | 多语言文本嵌入生成（Embedding API） |
| bge-large | BAAI/BGE-large | 8093 | 中文文本嵌入生成（最大512字符，Embedding API） |

## 快速启动

### 前置要求

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 Docker 和 NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 下载模型

```bash
# 使用脚本下载模型 (推荐)
./download_models.sh bge-reranker-v2-m3    # 下载 reranker 模型
./download_models.sh bge-m3                # 下载 m3 embedding 模型
./download_models.sh bge-large             # 下载 large embedding 模型
./download_models.sh all                   # 下载所有模型

# 或指定数据源
./download_models.sh -s modelscope bge-m3  # 从 ModelScope 下载
```

### 启动服务

```bash
# 启动所有服务
docker-compose up -d

# 只启动指定服务
docker-compose up -d bge-reranker
docker-compose up -d bge-m3
docker-compose up -d bge-large

# 查看日志
docker-compose logs -f

# 查看服务状态
docker-compose ps
```

### 健康检查

```bash
# Reranker 服务
curl http://192.168.8.233:8091/health
curl http://192.168.8.233:8091/v1/models

# BGE-M3 Embedding 服务
curl http://192.168.8.233:8092/health
curl http://192.168.8.233:8092/v1/models

# BGE-Large Embedding 服务
curl http://192.168.8.233:8093/health
curl http://192.168.8.233:8093/v1/models
```

## Reranker 服务 (端口 8091)

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/models` | GET | 模型信息 |
| `/v1/rerank` | POST | 文档重排序 |

### 使用示例

```bash
curl -X POST http://192.168.8.233:8091/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/model",
    "query": "什么是人工智能？",
    "documents": ["人工智能是计算机科学的一个分支", "今天天气很好"],
    "top_n": 2
  }'
```

### Python 客户端

```bash
cd examples
pip install openai requests numpy
python test_reranker.py
```

```python
from examples.test_reranker import BGERerankerClient

client = BGERerankerClient("http://192.168.8.233:8091")

# 基本重排序
query = "什么是人工智能？"
documents = [
    "人工智能是计算机科学的一个分支。",
    "今天天气很好。",
    "机器学习是人工智能的子领域。",
]
results = examples.rerank(query, documents, top_k=2)

for idx, doc, score in results:
    print(f"[{score:.4f}] {doc}")
```

## Embedding 服务 (端口 8092)

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/models` | GET | 模型信息 |
| `/v1/embeddings` | POST | 生成文本嵌入 |

### 使用示例

```bash
curl -X POST http://192.168.8.233:8092/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/model",
    "input": ["你好世界", "Hello World"],
    "encoding_format": "float"
  }'
```

### Python 客户端

```bash
cd examples
pip install requests numpy
python test_embedding.py
```

```python
from examples.test_embedding import BGEEmbeddingClient

client = BGEEmbeddingClient("http://192.168.8.233:8092")

# 生成嵌入
text = "人工智能是计算机科学的一个分支"
embedding = examples.embed(text)
print(f"嵌入维度: {len(embedding)}")

# 相似度检索
query = "什么是机器学习？"
documents = [
    "机器学习是人工智能的一个重要领域。",
    "今天我去了公园散步。",
]
results = examples.similarity(query, documents, top_k=2)

for idx, doc, score in results:
    print(f"[{score:.4f}] {doc}")
```

## BGE-Large Embedding 服务 (端口 8093)

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/models` | GET | 模型信息 |
| `/v1/embeddings` | POST | 生成文本嵌入 |

### 使用示例

```bash
curl -X POST http://192.168.8.233:8093/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/model",
    "input": ["你好世界", "Hello World"],
    "encoding_format": "float"
  }'
```

### Python 客户端

```bash
cd examples
pip install openai requests numpy
python test_bge_large.py
```

```python
from examples.test_bge_large import BGELargeEmbeddingClient

client = BGELargeEmbeddingClient("http://192.168.8.233:8093")

# 生成嵌入
text = "人工智能是计算机科学的一个分支"
embedding = client.embed(text)
print(f"嵌入维度: {len(embedding)}")

# 相似度检索
query = "什么是机器学习？"
documents = [
    "机器学习是人工智能的一个重要领域。",
    "今天我去了公园散步。",
]
results = client.similarity(query, documents, top_k=2)

for idx, doc, score in results:
    print(f"[{score:.4f}] {doc}")
```

## RAG 完整示例

```python
from examples.test_embedding import BGEEmbeddingClient
from examples.test_reranker import BGERerankerClient

# 知识库
knowledge_base = [
    "BGE-M3 是一个多语言嵌入模型，支持 100+ 种语言。",
    "BGE-Reranker-V2-M3 用于对检索结果进行重新排序。",
    "vLLM 是一个高性能的大语言模型推理引擎。",
    "RAG (Retrieval-Augmented Generation) 结合检索和生成能力。",
]

# 初始化客户端
embedding_client = BGEEmbeddingClient("http://192.168.8.233:8092")
reranker_client = BGERerankerClient("http://192.168.8.233:8091")

# 第一步：使用嵌入模型粗筛候选文档
query = "BGE 模型有什么用途？"
candidates = embedding_client.similarity(query, knowledge_base, top_k=4)

# 第二步：使用 Reranker 精细重排序
candidate_docs = [doc for idx, doc, score in candidates]
final_results = reranker_client.rerank(query, candidate_docs, top_k=2)

print("最终结果:")
for idx, doc, score in final_results:
    print(f"[{score:.4f}] {doc}")
```

## 配置说明

### 服务配置

| 配置项 | Reranker | BGE-M3 | BGE-Large |
|--------|----------|--------|-----------|
| 端口 | 8091 | 8092 | 8093 |
| GPU 显存 | 0.2 | 0.3 | 0.2 |
| 最大上下文 | 8192 tokens | 8192 tokens | 512 tokens (约512字符) |
| 精度 | float16 | float16 | float16 |

### 模型规格

| 模型 | 参数量 | 嵌入维度 | 用途 |
|------|--------|----------|------|
| bge-reranker-v2-m3 | ~278M | - | 文档重排序 |
| bge-m3 | ~567M | 1024 | 多语言文本嵌入 |
| BGE-large | ~326M | 1024 | 中文文本嵌入（最大512字符） |

## 模型存储

### 目录结构

```
项目目录/
├── models/                      ← Reranker 模型 (~2.2G)
├── models/BAAI/                 ← BAAI 模型目录
│   ├── bge-m3/                 ← M3 Embedding 模型 (~2.2G)
│   └── BGE-large/              ← Large Embedding 模型 (~1.2G)
├── docker-compose.yml
└── download_models.sh          ← 模型下载脚本
```

### 清理缓存

使用 `hf download --local-dir` 或 `download_models.sh` 下载后，可以清理缓存目录节省空间：

```bash
rm -rf models/blobs models/snapshots models/refs models/.cache models/assets
rm -rf models/BAAI/bge-m3/blobs models/BAAI/bge-m3/snapshots models/BAAI/bge-m3/refs models/BAAI/bge-m3/.cache models/BAAI/bge-m3/assets
rm -rf models/BAAI/BGE-large/blobs models/BAAI/BGE-large/snapshots models/BAAI/BGE-large/refs models/BAAI/BGE-large/.cache models/BAAI/BGE-large/assets
```

## 服务迁移

```bash
# 在当前机器打包整个项目
tar czf bge-deployment.tar.gz /path/to/bge-athenaeum/

# 传输到新机器后解压
tar xzf bge-deployment.tar.gz
cd bge-athenaeum

# 启动服务
docker-compose up -d
```

## 长文本测试

BGE-M3 经过测试，支持 1000 ~ 8000 个中文字符的嵌入生成：

| 文本长度 | 耗时 | 嵌入维度 |
|---------|------|----------|
| 1000 字符 | ~4.00 秒 | 1024 |
| 2000 字符 | ~0.06 秒 | 1024 |
| 4000 字符 | ~0.15 秒 | 1024 |
| 8000 字符 | ~0.45 秒 | 1024 |

**结论：**
- 所有长度均可成功嵌入
- 嵌入维度恒定为 1024
- 首次调用需要加载模型（约4秒），后续调用响应迅速
- 支持最大 8192 tokens 上下文长度

### 运行长文本测试

**BGE-M3 长文本测试:**

```python
from examples.test_embedding import BGEEmbeddingClient

client = BGEEmbeddingClient("http://192.168.8.233:8092")
client.test_long_text_embedding([1000, 2000, 4000, 8000])
```

**BGE-Large 长文本测试:**

```python
from examples.test_bge_large import BGELargeEmbeddingClient

client = BGELargeEmbeddingClient("http://192.168.8.233:8093")
client.test_long_text_embedding([1000, 2000, 4000, 8000])
```

**注意:**
- **BGE-Large**: 最大支持 **512 字符**（实测中文约 1 token/字符），超过限制会报错。
- **BGE-M3**: 最大支持 8192 tokens，中文约 2 tokens/字符，实际测试最大约 4000 字符。

## 常见问题

### Q: 如何同时运行多个服务？

多个服务共享同一块 GPU，通过 `gpu-memory-utilization` 控制显存使用：
- Reranker: 0.2 (约 1.6GB)
- BGE-M3: 0.3 (约 2.4GB)
- BGE-Large: 0.2 (约 1.6GB)
- 总计: 0.7 (约 5.6GB)

可根据 GPU 显存大小调整此参数。

### Q: 分数范围是多少？

- Reranker: -10 到 +10，越高越相关
- Embedding: -1 到 +1（余弦相似度）

### Q: BGE-Large 的字符限制是多少？

BGE-Large 最大支持 **512 字符**（约 512 tokens），超过此限制会返回错误：
```
Error code: 400 - This model's maximum context length is 512 tokens
```

**解决方案**：
- 使用 BGE-M3 处理长文本（最大 8192 tokens）
- 或对长文本进行分块处理，每块不超过 512 字符

### Q: 模型下载很慢怎么办？

使用 ModelScope 或 HuggingFace 镜像：

```bash
# ModelScope (国内推荐)
./download_models.sh -s modelscope bge-m3

# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
./download_models.sh bge-m3

# 手动下载模型
hf download BAAI/BGE-large --local-dir ./models/BAAI/BGE-large
```

### Q: 如何更新模型？

```bash
# 停止服务
docker-compose down

# 重新下载模型
./download_models.sh bge-m3

# 启动服务
docker-compose up -d
```

## 模型来源

| 模型 | HuggingFace | ModelScope |
|------|-------------|------------|
| bge-reranker-v2-m3 | `BAAI/bge-reranker-v2-m3` | `AI-ModelScope/bge-reranker-v2-m3` |
| bge-m3 | `BAAI/bge-m3` | `AI-ModelScope/bge-m3` |
| BGE-large | `BAAI/BGE-large` | `AI-ModelScope/bge-large-zh-v1.5` |

## 硬件要求

- **GPU**: NVIDIA GPU 8GB+ VRAM (推荐)
- **CUDA**: 11.8+
- **Docker**: 20.10+ with NVIDIA Container Toolkit
