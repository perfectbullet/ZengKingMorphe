# BGE-Reranker-V2-M3 部署文档

## 模型简介

**BGE-Reranker-V2-M3** 是北京智源人工智能研究院（BAAI）开发的轻量级重排序模型，专门用于 RAG 系统中的文档重排序。

| 特性 | 说明 |
|------|------|
| 开发者 | BAAI (北京智源人工智能研究院) |
| 参数量 | ~278M (轻量级) |
| 上下文长度 | 8192 tokens |
| 语言支持 | 中英双语、跨语言 |
| 模型仓库 | [HuggingFace](https://huggingface.co/BAAI/bge-reranker-v2-m3) / [ModelScope](https://modelscope.cn/models/AI-ModelScope/bge-reranker-v2-m3/) |

---

## 一、环境要求

| 组件 | 最低要求 |
|------|----------|
| GPU | NVIDIA GPU (建议 8GB+ 显存) |
| CUDA | 11.8+ |
| Docker | 20.10+ |
| Docker Compose | 2.0+ (可选) |
| Python | 3.8+ (本地部署) |

---

## 二、Docker 部署

### 方式 1: 使用现成的 Docker 镜像

```bash
# 拉取镜像
docker pull wkao/bge-reranker-v2-m3

# 启动服务 (GPU)
docker run -d \
  --name bge-reranker \
  -p 8080:80 \
  --gpus all \
  wkao/bge-reranker-v2-m3

# 启动服务 (CPU)
docker run -d \
  --name bge-reranker \
  -p 8080:80 \
  wkao/bge-reranker-v2-m3
```

### 方式 2: 使用 vLLM 部署

```bash
# 拉取 vLLM 镜像
docker pull vllm/vllm-openai:latest

# 创建模型目录
mkdir -p /data/models

# 启动服务 (自动下载模型)
docker run -d \
  --name bge-reranker-vllm \
  --gpus all \
  -p 8000:8000 \
  -v /data/models:/mnt/models \
  vllm/vllm-openai:latest \
  --model BAAI/bge-reranker-v2-m3 \
  --max-model-len 8192
```

### 方式 3: 使用 Docker Compose

创建 `docker-compose.yml`:

```yaml
version: '3.8'

services:
  bge-reranker:
    image: vllm/vllm-openai:latest
    container_name: bge-reranker-m3
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./models:/mnt/models
    command: >
      --model BAAI/bge-reranker-v2-m3
      --max-model-len 8192
      --host 0.0.0.0
      --port 8000
    restart: unless-stopped
```

启动服务:

```bash
docker-compose up -d
```

### 验证服务

```bash
curl http://localhost:8000/v1/models

curl http://localhost:8080/health
```

---

## 三、Python 调用示例

### 3.1 直接使用 FlagEmbedding 库

首先安装依赖：

```bash
pip install FlagEmbedding
```

**基本用法：**

```python
from FlagEmbedding import FlagReranker

# 初始化 reranker (首次运行会自动下载模型)
reranker = FlagReranker(
    'BAAI/bge-reranker-v2-m3',
    use_fp16=True  # 使用半精度推理，节省显存
)

# 查询和文档
query = "什么是人工智能？"
documents = [
    "人工智能是指由人制造出来的机器所表现出来的智能。",
    "今天天气很好，适合出去散步。",
    "机器学习是人工智能的一个分支。",
    "我昨天吃了一顿美味的晚餐。"
]

# 计算每对 (query, document) 的相关性分数
scores = reranker.compute_score([
    [query, doc] for doc in documents
])

# 打印结果
for doc, score in zip(documents, scores):
    print(f"分数: {score:.4f} | 文档: {doc}")

# 输出:
# 分数: 8.5234 | 文档: 人工智能是指由人制造出来的机器所表现出来的智能。
# 分数: -5.1234 | 文档: 今天天气很好，适合出去散步。
# 分数: 7.8901 | 文档: 机器学习是人工智能的一个分支。
# 分数: -4.5678 | 文档: 我昨天吃了一顿美味的晚餐。

# 按分数排序获取最相关的文档
ranked_results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
print("\n排序结果:")
for i, (doc, score) in enumerate(ranked_results, 1):
    print(f"{i}. [{score:.4f}] {doc}")
```

### 3.2 批量处理

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

# 批量查询
queries = [
    "Python 如何读取文件？",
    "什么是深度学习？"
]

documents = [
    "Python 使用 open() 函数读取文件。",
    "深度学习是机器学习的子集，使用神经网络。",
    "Java 是一种面向对象的编程语言。",
    "TensorFlow 是深度学习框架。"
]

# 为每个查询重排序文档
for query in queries:
    pairs = [[query, doc] for doc in documents]
    scores = reranker.compute_score(pairs)

    print(f"\n查询: {query}")
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    for i, (doc, score) in enumerate(ranked[:3], 1):
        print(f"  {i}. [{score:.4f}] {doc}")
```

### 3.3 RAG 系统集成示例

```python
from FlagEmbedding import FlagReranker

class SimpleRAG:
    def __init__(self, documents, reranker_model='BAAI/bge-reranker-v2-m3'):
        self.documents = documents
        self.reranker = FlagReranker(reranker_model, use_fp16=True)

    def retrieve_and_rerank(self, query, top_k=3):
        """
        检索并重排序文档
        """
        # 构建查询-文档对
        pairs = [[query, doc] for doc in self.documents]

        # 计算相关性分数
        scores = self.reranker.compute_score(pairs)

        # 排序并返回 top_k
        ranked = sorted(zip(self.documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

# 示例文档库
knowledge_base = [
    "BGE-Reranker-V2-M3 是 BAAI 开发的轻量级重排序模型。",
    "该模型支持 8192 tokens 的上下文长度。",
    "模型采用中英双语训练，支持跨语言检索。",
    "Reranker 用于对检索结果进行重新排序，提高最终结果的相关性。",
    "BGE-M3 是多语言嵌入模型，而 BGE-Reranker-V2-M3 是重排序模型。",
    "使用半精度 (FP16) 推理可以节省显存并加速计算。"
]

# 创建 RAG 实例
rag = SimpleRAG(knowledge_base)

# 查询
query = "BGE-Reranker 支持多长的上下文？"

# 检索并重排序
results = rag.retrieve_and_rerank(query, top_k=3)

# 输出结果
print(f"查询: {query}\n")
for i, (doc, score) in enumerate(results, 1):
    print(f"{i}. [分数: {score:.4f}] {doc}")
```

### 3.4 通过 API 调用

如果使用 vLLM 部署的 API 服务：

```python
import requests
import json

class RerankerAPI:
    def __init__(self, base_url="http://192.168.8.233:8091"):
        self.base_url = base_url

    def rerank(self, query, documents, top_k=None):
        """
        通过 API 调用 reranker
        """
        url = f"{self.base_url}/v1/rerank"

        payload = {
            "model": "BAAI/bge-reranker-v2-m3",
            "query": query,
            "documents": documents,
            "top_n": top_k if top_k else len(documents)
        }

        response = requests.post(url, json=payload)
        return response.json()

# 使用示例
api = RerankerAPI()

query = "什么是机器学习？"
documents = [
    "机器学习是人工智能的一个分支。",
    "深度学习基于神经网络结构。",
    "今天气温 25 度。",
    "监督学习需要标注数据。"
]

results = api.rerank(query, documents, top_k=3)

print(f"查询: {query}\n")
for result in results.get('results', []):
    print(f"[{result['relevance_score']:.4f}] {documents[result['index']]}")
```

### 3.5 多轮对话 RAG 示例

```python
from FlagEmbedding import FlagReranker

class ConversationalRAG:
    def __init__(self, documents):
        self.documents = documents
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        self.chat_history = []

    def search(self, query, top_k=3):
        # 结合历史上下文
        context = " ".join([msg for msg in self.chat_history[-3:]])
        enhanced_query = f"{context} {query}".strip() if context else query

        pairs = [[enhanced_query, doc] for doc in self.documents]
        scores = self.reranker.compute_score(pairs)

        ranked = sorted(zip(self.documents, scores), key=lambda x: x[1], reverse=True)
        self.chat_history.append(query)

        return ranked[:top_k]

# 使用
docs = [
    "Python 是一种高级编程语言。",
    "Python 支持面向对象和函数式编程。",
    "Python 的语法简洁易读。",
    "Java 是静态类型语言。",
]

rag = ConversationalRAG(docs)

# 第一轮
print("Q1: Python 有什么特点？")
for doc, score in rag.search("Python 有什么特点？"):
    print(f"  [{score:.4f}] {doc}")

# 第二轮 (利用上下文)
print("\nQ2: 那它的类型系统呢？")
for doc, score in rag.search("类型系统"):
    print(f"  [{score:.4f}] {doc}")
```

---

## 四、性能优化建议

### 4.1 显存优化

```python
# 使用 FP16 半精度
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

# 批量处理减少调用次数
pairs = [[query, doc] for doc in large_document_list]
scores = reranker.compute_score(pairs, batch_size=32)  # 调整 batch_size
```

### 4.2 文档预处理

```python
def truncate_text(text, max_length=512):
    """截断过长文本，保留开头和结尾"""
    if len(text) <= max_length:
        return text
    return text[:max_length//2] + "..." + text[-max_length//2:]

# 对超长文档进行截断
documents = [truncate_text(doc, max_length=512) for doc in documents]
```

---

## 五、常见问题

### Q1: 模型下载很慢怎么办？

```bash
# 使用国内镜像 ModelScope
export HF_ENDPOINT=https://hf-mirror.com
```

或者直接从 ModelScope 下载：

```python
from FlagEmbedding import FlagReranker
reranker = FlagReranker('AI-ModelScope/bge-reranker-v2-m3', use_fp16=True)
```

### Q2: CPU 可以运行吗？

可以，但速度较慢：

```python
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False, device='cpu')
```

### Q3: 如何处理超长文档？

建议对文档进行分段，每段控制在 512-2048 tokens，然后分别 rerank 后取最高分。

### Q4: 分数范围是多少？

分数通常在 -10 到 +10 之间，分数越高表示相关性越强。

---

## 六、参考链接

| 资源 | 链接 |
|------|------|
| HuggingFace | https://huggingface.co/BAAI/bge-reranker-v2-m3 |
| ModelScope | https://modelscope.cn/models/AI-ModelScope/bge-reranker-v2-m3/ |
| GitHub (FlagEmbedding) | https://github.com/FlagOpen/FlagEmbedding |
| Docker Hub | https://hub.docker.com/r/wkao/bge-reranker-v2-m3 |

---

*文档生成时间: 2025-01-27*
