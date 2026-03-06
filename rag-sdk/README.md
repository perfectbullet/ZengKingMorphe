# LlamaRAG SDK

基于 LlamaIndex 的 RAG（检索增强生成）Python SDK，专为教材文档（语文、高中数学）优化，支持图片描述和公式解析。

## 功能特性

- **文档解析**: 集成 MinerU MCP 服务，将 PDF 解析为 Markdown
- **图片描述**: 使用 Qwen2-VL 多模态模型生成图片描述
- **智能分块**: 支持固定大小、语义分块和混合分块策略
- **向量索引**: 基于 LlamaIndex 和 ChromaDB 的高效向量存储
- **灵活检索**: 支持向量检索、混合检索和重排序
- **模块化设计**: 各模块可独立开发和测试

## 项目结构

```
rag-sdk/
├── src/
│   ├── config.py              # 配置管理
│   ├── utils.py               # 工具函数
│   ├── document_parser/       # 文档解析模块
│   │   ├── base.py           # 数据模型和基类
│   │   ├── mineru_client.py  # MinerU MCP 客户端
│   │   └── image_processor.py # 图片描述生成器
│   ├── document_indexer/      # 文档索引模块
│   │   ├── base.py           # 基类
│   │   ├── chunker.py        # 分块策略
│   │   ├── storage.py        # 向量存储
│   │   └── indexer.py        # 索引器
│   └── retrieval/            # 检索模块
│       ├── base.py           # 基类和数据模型
│       ├── strategies.py     # 检索策略
│       └── retriever.py      # 检索器
├── tests/                    # 测试文件
├── examples/                 # 使用示例
├── data/                     # 数据目录
└── logs/                     # 日志目录
```

## 快速开始

### 1. 安装依赖

```bash
# 安装核心依赖
pip install -r requirements.txt

# 或使用 pip 安装（包含所有依赖）
pip install -e .[all]
```

### 2. 配置环境变量

复制 `.env.example` 到 `.env` 并修改配置：

```bash
cp .env.example .env
```

关键配置项：

```bash
# MinerU MCP 服务
MINERU_MCP_URL=http://localhost:8001

# Ollama 服务
OLLAMA_BASE_URL=http://192.168.8.233:11434
OLLAMA_EMBEDDING_MODEL=bge-m3

# ChromaDB
CHROMA_HOST=192.168.8.233
CHROMA_PORT=8200
```

### 3. 基本使用

```python
import asyncio
from src.document_parser.mineru_client import MinerUParser
from src.document_indexer.indexer import DocumentIndexer
from src.retrieval.retriever import Retriever

async def main():
    # 1. 解析文档
    async with MinerUParser() as parser:
        document = await parser.parse("data/sample_pdf/textbook.pdf")
        print(f"文档: {document.title}")
        print(f"文本块: {len(document.chunks)}")

    # 2. 创建索引
    indexer = DocumentIndexer()
    await indexer.create_index([doc.content for doc in [document]])

    # 3. 检索
    retriever = Retriever(vector_store=indexer.vector_store)
    results = await retriever.retrieve("什么是导数？")
    for doc in results:
        print(f"{doc.score:.3f}: {doc.text[:100]}")

asyncio.run(main())
```

## 使用示例

### 文档解析

```python
from src.document_parser.mineru_client import MinerUParser
from src.document_parser.image_processor import ImageDescriptor

async with MinerUParser() as parser:
    async with ImageDescriptor() as descriptor:
        # 解析文档
        document = await parser.parse("textbook.pdf")

        # 生成图片描述
        document.images = await descriptor.describe_images_batch(document.images)

        print(f"标题: {document.title}")
        print(f"文本块: {len(document.chunks)}")
        print(f"图片: {len(document.images)}")
```

### 创建索引

```python
from src.document_indexer.indexer import DocumentIndexer
from src.document_indexer.base import ChunkStrategy

# 自定义分块策略
strategy = ChunkStrategy(
    type="hybrid",
    chunk_size=512,
    chunk_overlap=50
)

indexer = DocumentIndexer(chunk_strategy=strategy)

# 添加文档
await indexer.add_documents(
    documents=["文档1", "文档2"],
    collection_name="textbook"
)

# 获取统计
stats = await indexer.get_collection_stats("textbook")
print(f"文档数量: {stats['count']}")
```

### 检索文档

```python
from src.retrieval.retriever import Retriever
from src.document_indexer.storage import VectorStore

vector_store = VectorStore(collection_name="textbook")

# 使用重排序检索
retriever = Retriever(
    vector_store=vector_store,
    use_rerank=True
)

# 执行检索
results = await retriever.retrieve(
    query="牛顿第二定律的内容",
    top_k=5
)

for doc in results:
    print(f"[{doc.score:.3f}] {doc.text}")
```

## 分块策略

### 固定大小分块

```python
strategy = ChunkStrategy(
    type="fixed",
    chunk_size=512,
    chunk_overlap=50
)
```

### 语义分块

```python
strategy = ChunkStrategy(type="semantic")
```

### 混合分块

```python
strategy = ChunkStrategy(
    type="hybrid",
    chunk_size=512,
    chunk_overlap=50,
    max_chunk_size=1024
)
```

## 检索策略

### 向量检索

```python
retriever = Retriever(vector_store=store, use_hybrid=False)
```

### 混合检索（向量 + 关键词）

```python
retriever = Retriever(vector_store=store, use_hybrid=True)
```

### 重排序检索

```python
retriever = Retriever(vector_store=store, use_rerank=True)
```

## 运行示例

```bash
# 基本使用示例
python examples/basic_usage.py

# 文档解析测试
python examples/parse_test.py

# 索引测试
python examples/index_test.py

# 检索测试
python examples/retrieve_test.py
```

## 运行测试

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/test_parser.py
pytest tests/test_indexer.py
pytest tests/test_retriever.py

# 带覆盖率报告
pytest --cov=src tests/
```

## 与 LangGraph 集成

```python
from langgraph_sdk import Client
from src.retrieval.retriever import Retriever
from src.document_indexer.storage import VectorStore

# 初始化 RAG 系统
vector_store = VectorStore(collection_name="textbook")
retriever = Retriever(vector_store=vector_store)

# 在 LangGraph 中使用
async def retrieval_node(state):
    query = state["query"]
    documents = await retriever.retrieve(query, top_k=5)
    return {"documents": documents}

# 创建 LangGraph workflow
client = Client()
workflow = await client.create_workflow()
await client.create_node(workflow_id=workflow, function=retrieval_node)
```

## API 文档

### MinerUParser

文档解析器，调用 MinerU MCP 服务。

- `async parse(file_path: str) -> ParsedDocument`: 解析单个文档
- `async parse_batch(file_paths: List[str]) -> List[ParsedDocument]`: 批量解析

### DocumentIndexer

文档索引器，处理分块和向量化。

- `async create_index(documents: List[str]) -> str`: 创建索引
- `async add_documents(documents: List[str]) -> List[str]`: 添加文档
- `async get_collection_stats(collection_name: str) -> Dict`: 获取统计信息

### Retriever

检索器，提供统一的检索接口。

- `async retrieve(query: str, top_k: int) -> List[RetrievedDocument]`: 检索文档
- `async retrieve_multiple(queries: List[str]) -> List[RetrievedDocument]`: 多查询检索

## 依赖服务

### MinerU MCP

```bash
# 使用 Docker 启动
docker run -d -p 8001:8001 mineru/mcp-server
```

### Ollama

```bash
# 安装 Ollama
curl https://ollama.ai/install.sh | sh

# 下载模型
ollama pull bge-m3
ollama pull qwen2-vl
ollama pull deepseek-coder
```

### ChromaDB

```bash
# 使用 Docker 启动远程服务
docker run -d -p 8200:8000 chromadb/chroma
```

## 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `MINERU_MCP_URL` | MinerU 服务地址 | http://localhost:8001 |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | http://192.168.8.233:11434 |
| `OLLAMA_EMBEDDING_MODEL` | Embedding 模型 | bge-m3 |
| `CHROMA_HOST` | ChromaDB 主机 | 192.168.8.233 |
| `CHROMA_PORT` | ChromaDB 端口 | 8200 |
| `CHUNK_SIZE` | 分块大小 | 512 |
| `CHUNK_OVERLAP` | 分块重叠 | 50 |
| `TOP_K` | 检索返回数 | 5 |
| `SIMILARITY_THRESHOLD` | 相似度阈值 | 0.7 |

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
