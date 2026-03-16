# LlamaRAG SDK 实现总结

## 项目概述

LlamaRAG SDK 是一个基于 LlamaIndex 的 RAG（检索增强生成）Python SDK，专为教材文档（语文、高中数学）优化，支持图片描述和公式解析。

**项目位置**: `/home/zj/ZengKingMorphe/rag-sdk/`

## 已实现功能

### 1. 核心模块

#### 1.1 配置管理 (`src/config.py`)
- 基于 Pydantic Settings 的配置管理
- 支持从 `.env` 文件加载配置
- 提供配置验证和默认值

#### 1.2 工具函数 (`src/utils.py`)
- 日志配置（loguru）
- 文本分块工具
- 文件名清理
- 文本截断
- 元数据合并

### 2. 文档解析模块 (`src/document_parser/`)

#### 2.1 数据模型 (`base.py`)
- `ParsedDocument`: 解析后的文档对象
- `TextChunk`: 文本块模型
- `ImageInfo`: 图片信息模型

#### 2.2 MinerU 客户端 (`mineru_client.py`)
- `MinerUParser`: MinerU MCP 服务客户端
- 支持异步解析和批量解析
- 自动提取 Markdown 内容和图片

#### 2.3 图片描述生成器 (`image_processor.py`)
- `ImageDescriptor`: 使用 Qwen2-VL 生成图片描述
- 支持批量图片描述生成
- 可配置并发数

### 3. 文档索引模块 (`src/document_indexer/`)

#### 3.1 分块策略 (`chunker.py`)
- `FixedSizeChunker`: 固定大小分块
- `SemanticChunker`: 基于语义结构的分块（标题/段落）
- `HybridChunker`: 混合分块策略

#### 3.2 向量存储 (`storage.py`)
- `VectorStore`: ChromaDB 存储封装
- 支持远程和本地存储
- 提供增删改查和统计功能

#### 3.3 索引器 (`indexer.py`)
- `DocumentIndexer`: 整合分块和向量化的主索引器
- 支持 Ollama Embedding 模型
- 提供索引创建、文档添加和删除功能

### 4. 检索模块 (`src/retrieval/`)

#### 4.1 数据模型 (`base.py`)
- `RetrievedDocument`: 检索结果模型
- `RetrievalStrategy`: 检索策略基类

#### 4.2 检索策略 (`strategies.py`)
- `VectorRetrieval`: 纯向量检索
- `HybridRetrieval`: 混合检索（向量 + 关键词）
- `RerankRetrieval`: 重排序检索

#### 4.3 检索器 (`retriever.py`)
- `Retriever`: 统一的检索接口
- 支持单查询和多查询检索
- 支持结果去重

### 5. RAG 系统集成 (`src/rag_system.py`)

- `RAGSystem`: 完整的 RAG 系统集成类
- 提供文档解析、索引和检索的一站式接口
- 支持异步上下文管理器

## 测试覆盖

### 测试文件
- `tests/test_parser.py`: 文档解析模块测试
- `tests/test_indexer.py`: 索引模块测试
- `tests/test_retriever.py`: 检索模块测试

### 测试特性
- 单元测试覆盖各模块核心功能
- 支持异步测试（pytest-asyncio）
- 提供测试夹具（fixtures）

## 示例代码

### 基础示例
- `examples/basic_usage.py`: 基本使用流程
- `examples/parse_test.py`: 文档解析测试
- `examples/index_test.py`: 索引创建测试
- `examples/retrieve_test.py`: 检索功能测试
- `examples/rag_system_demo.py`: RAG 系统完整演示

## 项目结构

```
rag-sdk/
├── src/                      # 源代码
│   ├── config.py            # 配置管理
│   ├── utils.py             # 工具函数
│   ├── rag_system.py        # RAG 系统集成
│   ├── document_parser/     # 文档解析模块
│   ├── document_indexer/    # 文档索引模块
│   └── retrieval/           # 检索模块
├── tests/                   # 测试文件
├── examples/                # 示例代码
├── data/                    # 数据目录
│   ├── sample_pdf/         # 测试 PDF
│   └── output/             # 解析输出
├── logs/                    # 日志目录
├── pyproject.toml          # 项目配置
├── requirements.txt        # 依赖列表
├── README.md              # 使用文档
└── .env.example           # 环境变量模板
```

## 依赖服务

### 必需服务
1. **Ollama** - Embedding 和多模态模型
   - 地址: http://192.168.8.233:11434
   - 模型: bge-m3, qwen2-vl

2. **ChromaDB** - 向量存储
   - 地址: http://192.168.8.233:8200
   - 支持远程和本地存储

### 可选服务
1. **MinerU MCP** - 文档解析
   - 地址: http://localhost:8001
   - 如不使用，可跳过文档解析功能

## 使用方式

### 快速开始

```python
import asyncio
from src.rag_system import RAGSystem

async def main():
    # 创建 RAG 系统
    async with RAGSystem() as rag:
        # 添加文档
        await rag.add_text_documents(["文档1", "文档2"])

        # 检索
        results = await rag.retrieve("查询内容")
        for doc in results:
            print(f"[{doc.score:.3f}] {doc.text}")

asyncio.run(main())
```

### 与 LangGraph 集成

```python
from src.rag_system import RAGSystem

# 初始化 RAG 系统
rag = RAGSystem()

# 在 LangGraph 中使用
async def retrieval_node(state):
    query = state["query"]
    documents = await rag.retrieve(query, top_k=5)
    return {"documents": documents}
```

## 待扩展功能

1. **更多文档格式支持**: PPT、DOCX 等
2. **分布式索引**: 支持多节点索引
3. **检索结果缓存**: 提高重复查询性能
4. **多模态检索**: 文本 + 图片混合检索
5. **A/B 测试框架**: 支持策略对比

## 技术栈

- **文档解析**: MinerU MCP
- **索引框架**: LlamaIndex
- **Embedding**: Ollama (bge-m3)
- **向量存储**: ChromaDB
- **多模态**: Qwen2-VL
- **异步框架**: asyncio + aiohttp
- **配置管理**: Pydantic Settings
- **日志**: loguru
- **测试**: pytest + pytest-asyncio

## 文件统计

- **Python 源文件**: 25 个
- **测试文件**: 3 个
- **示例文件**: 5 个
- **总代码行数**: 约 2000+ 行

## 下一步

1. 安装依赖: `pip install -r requirements.txt`
2. 配置环境变量: 复制 `.env.example` 到 `.env`
3. 启动服务: Ollama、ChromaDB、MinerU MCP
4. 运行示例: `python examples/basic_usage.py`
5. 运行测试: `pytest tests/`

## 注意事项

1. 确保所有依赖服务正常运行
2. 首次使用时需要下载 Ollama 模型
3. ChromaDB 远程服务需要网络访问
4. 大量文档索引时注意显存和内存使用

## 许可证

MIT License
