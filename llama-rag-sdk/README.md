# LlamaRAG SDK

基于 LlamaIndex 的 RAG（检索增强生成）Python SDK，专为教材文档（中文、高中数学）优化，支持图片描述和公式解析。

## 功能特性

- **文档解析**: 集成 MinerU API 服务，将 PDF 解析为 Markdown
  - 支持结构化内容解析（content_list.json）
  - 支持分页解析、公式解析、表格解析
  - 进度回调支持
  - 直接返回 JSON 或下载 ZIP
- **图片描述**: 使用 Qwen2-VL 多模态模型生成图片描述
- **智能分块**: 支持固定大小、语义分块和混合分块策略
- **向量索引**: 基于 LlamaIndex 和 ChromaDB 的高效向量存储
- **vLLM 集成**: 使用 vLLM OpenAI 兼容 API 进行 Embedding
- **灵活检索**: 统一的混合检索 + Rerank 策略
- **模块化设计**: 各模块可独立开发和测试

## 项目结构

```
rag-sdk/
├── src/
│   ├── config.py              # 配置管理
│   ├── utils.py               # 工具函数
│   ├── document_parser/       # 文档解析模块
│   │   ├── base.py           # 数据模型和基类
│   │   ├── mineru_client.py  # MinerU API 客户端
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
# MinerU API 服务
MINERU_API_URL=http://192.168.8.231:8000

# vLLM 服务（Embedding）
VLLM_EMBEDDING_BASE_URL=http://192.168.8.233:8092
VLLM_EMBEDDING_MODEL=BAAI/bge-m3

# ChromaDB
CHROMA_HOST=192.168.8.233
CHROMA_PORT=8200
```

### 3. 基本使用

```python
import asyncio
from src.document_parser import MinerUParser, ParseOptions, ReturnOptions
from src.document_indexer.indexer import DocumentIndexer
from src.retrieval.retriever import Retriever

async def main():
    # 1. 解析文档（使用自定义选项）
    async with MinerUParser() as parser:
        parse_options = ParseOptions(
            backend="pipeline",  # 传统后端，显存占用低
            lang="ch",  # 中文
            formula_enable=True,  # 启用公式解析
        )
        return_options = ReturnOptions(
            return_md=True,
            return_content_list=True,  # 结构化内容
            return_images=True,
        )

        document = await parser.parse(
            "data/sample_pdf/textbook.pdf",
            parse_options=parse_options,
            return_options=return_options,
        )

        print(f"文档: {document.title}")
        print(f"文本块: {len(document.chunks)}")
        print(f"图片: {len(document.images)}")

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

#### 基本解析

```python
from src.document_parser import MinerUParser

async with MinerUParser() as parser:
    document = await parser.parse("textbook.pdf")

    print(f"标题: {document.title}")
    print(f"文本块: {len(document.chunks)}")
    print(f"图片: {len(document.images)}")
```

#### 自定义解析选项

```python
from src.document_parser import MinerUParser, ParseOptions, ReturnOptions

async with MinerUParser() as parser:
    # 自定义解析选项
    parse_options = ParseOptions(
        backend="pipeline",  # pipeline/vlm-auto-engine/hybrid-auto-engine
        parse_method="auto",  # auto/txt/ocr
        lang="ch",
        start_page=0,
        end_page=10,  # 只解析前 10 页
        formula_enable=True,
        table_enable=True,
    )

    # 自定义返回选项
    return_options = ReturnOptions(
        return_md=True,
        return_content_list=True,
        return_images=True,
    )

    document = await parser.parse(
        "textbook.pdf",
        parse_options=parse_options,
        return_options=return_options,
    )
```

#### 带进度回调

```python
from src.document_parser import MinerUParser, STAGE_NAMES

def on_progress(stage: str, percent: float):
    name = STAGE_NAMES.get(stage, stage)
    print(f"[{percent:5.1f}%] {name}")

async with MinerUParser() as parser:
    document = await parser.parse(
        "textbook.pdf",
        progress_callback=on_progress,
    )
```

#### 直接返回 JSON（不下载 ZIP）

```python
async with MinerUParser() as parser:
    result = await parser.parse_to_memory("textbook.pdf")

    # 直接获取 JSON 结果
    content_list = result.get("content_list", [])
    markdown = result.get("md_content", "")
```

#### 批量解析

```python
async with MinerUParser() as parser:
    file_paths = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    documents = await parser.parse_batch(file_paths)

    for doc in documents:
        print(f"{doc.title}: {len(doc.chunks)} 块")
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

# 统一检索器（混合检索 + Rerank）
retriever = Retriever(vector_store=vector_store)

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

当前 Retriever 统一使用混合检索 + Rerank 策略：

```python
from src.retrieval.retriever import Retriever

retriever = Retriever(vector_store=store)
# 内部使用：向量检索 + 关键词检索（混合）+ BGE Rerank
```

## 运行示例

```bash
# 基本使用示例
python examples/basic_usage.py

# 文档解析测试（包含 6 个示例）
python examples/parse_test.py

# 索引测试
python examples/index_test.py

# 检索测试
python examples/retrieve_test.py

# vLLM Embedding 测试
python examples/test_vllm_with_llamaindex.py

# RAG 系统完整演示
python examples/rag_system_demo.py
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

## API 文档

### MinerUParser

MinerU API 客户端，用于解析 PDF 文档。

#### 初始化

```python
MinerUParser(
    api_url: Optional[str] = None,  # MinerU API 地址
    output_dir: Optional[str] = None,  # 输出目录
    timeout: Optional[int] = None,  # 超时时间（秒）
)
```

#### 主要方法

- `async parse(file_path: str, parse_options: ParseOptions, return_options: ReturnOptions, progress_callback: Callable) -> ParsedDocument`
  - 解析单个文档
  - 支持进度回调

- `async parse_to_memory(file_path: str, parse_options: ParseOptions, return_options: ReturnOptions) -> Dict`
  - 直接返回 JSON，不下载 ZIP

- `async parse_batch(file_paths: List[str], ...) -> List[ParsedDocument]`
  - 批量解析文档（并发）

- `async health_check() -> bool`
  - 检查 MinerU API 服务是否可访问

#### ParseOptions

```python
@dataclass
class ParseOptions:
    backend: str = "pipeline"  # pipeline/vlm-auto-engine/hybrid-auto-engine
    parse_method: str = "auto"  # auto/txt/ocr
    lang: str = "ch"
    formula_enable: bool = True
    table_enable: bool = True
    start_page: int = 0
    end_page: Optional[int] = None
```

#### ReturnOptions

```python
@dataclass
class ReturnOptions:
    return_md: bool = True
    return_middle_json: bool = False
    return_model_output: bool = False
    return_content_list: bool = True
    return_images: bool = True
```

### DocumentIndexer

文档索引器，处理分块和向量化。

- `async create_index(documents: List[str]) -> str`: 创建索引
- `async add_documents(documents: List[str]) -> List[str]`: 添加文档
- `async get_collection_stats(collection_name: str) -> Dict`: 获取统计信息

### Retriever

检索器，提供统一的检索接口。内部使用混合检索 + Rerank 策略。

- `async retrieve(query: str, top_k: int) -> List[RetrievedDocument]`: 检索文档
- `async retrieve_multiple(queries: List[str]) -> List[RetrievedDocument]`: 多查询检索

## 依赖服务

### MinerU API

```bash
# 使用 Docker 启动 MinerU API 服务
docker run -d -p 8000:8000 mineru/mineru-api

# 或使用本地安装
# 参考: https://github.com/opendatalab/MinerU
```

### vLLM

**BGE-M3 Embedding 模型**：

vLLM 服务应运行在 `http://192.168.8.233:8092`，提供 OpenAI 兼容的 API 接口。

配置：
```bash
VLLM_EMBEDDING_BASE_URL=http://192.168.8.233:8092
VLLM_EMBEDDING_API_BASE=http://192.168.8.233:8092/v1
VLLM_EMBEDDING_MODEL=BAAI/bge-m3
VLLM_API_KEY=not-needed
```

### ChromaDB

```bash
# 使用 Docker 启动远程服务
docker run -d -p 8200:8000 chromadb/chroma
```

### Ollama (图片描述)

```bash
# 下载 Qwen2-VL 模型
ollama pull qwen2-vl:latest
```

### BGE Reranker

```bash
# 使用 Docker 启动 BGE Reranker 服务
docker run -d -p 6006:6006 wkao/bge-reranker-v2-m3:latest
```

## 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| MinerU 相关配置 | | |
| `MINERU_API_URL` | MinerU API 地址 | http://192.168.8.231:8000 |
| `MINERU_OUTPUT_DIR` | 输出目录 | ./data/output |
| `MINERU_BACKEND` | 解析后端 | pipeline |
| `MINERU_PARSE_METHOD` | 解析方法 | auto |
| `MINERU_LANG` | 语言代码 | ch |
| `MINERU_TIMEOUT` | 请求超时（秒） | 1800 |
| `MINERU_FORMULA_ENABLE` | 启用公式解析 | true |
| `MINERU_TABLE_ENABLE` | 启用表格解析 | true |
| vLLM Embedding 配置 | | |
| `VLLM_EMBEDDING_BASE_URL` | vLLM Embedding 服务地址 | http://192.168.8.233:8092 |
| `VLLM_EMBEDDING_MODEL` | Embedding 模型 | BAAI/bge-m3 |
| ChromaDB 配置 | | |
| `CHROMA_HOST` | ChromaDB 主机 | 192.168.8.233 |
| `CHROMA_PORT` | ChromaDB 端口 | 8200 |
| 检索配置 | | |
| `CHUNK_SIZE` | 分块大小 | 512 |
| `CHUNK_OVERLAP` | 分块重叠 | 50 |
| `TOP_K` | 检索返回数 | 5 |
| `USE_RERANK` | 是否使用重排序 | true |
| `RERANK_BASE_URL` | BGE Reranker 服务地址 | http://192.168.8.233:8091 |

## vLLM 迁移说明

本项目已完成从 Ollama 到 vLLM 的 Embedding 功能迁移：

- 使用 `llama-index-embeddings-openai` 替代 `llama-index-embeddings-ollama`
- 通过 vLLM OpenAI 兼容 API 调用 BGE-M3 模型
- Ollama 仅保留用于 Qwen2-VL 图片描述功能

详细文档请参考：`docs/test_vllm_with_llamaindex.md`

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
