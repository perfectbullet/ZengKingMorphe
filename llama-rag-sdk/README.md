# LlamaRAG SDK

基于 LlamaIndex 的 RAG（检索增强生成）Python SDK，专为教材文档（中文、高中数学）优化，支持图片描述和公式解析。

## 功能特性

- **文档解析**: 基于 mineru-client SDK 集成 MinerU API 服务，将 PDF 解析为 Markdown
  - 支持结构化内容解析（content_list.json）
  - 支持分页解析、公式解析、表格解析
  - 进度回调支持
  - 直接返回 JSON 或下载 ZIP
  - 支持批量并发解析
- **结构感知分块**: MinerU 专用的智能分块策略
  - 按章节边界自然分割，保持语义完整性
  - 智能标题识别（支持 type=title 和 text_level）
  - 丰富元数据（标题路径、页码范围、块类型、图片引用）
  - 可通过环境变量配置（CHUNK_SIZE, CHUNK_OVERLAP）
  - 适配 BGE-M3 的 8192 token 容量
- **图片描述**: 使用 Qwen2-VL 多模态模型生成图片描述
- **智能分块**: 支持固定大小、语义分块和混合分块策略
- **向量索引**: 基于 LlamaIndex 和 ChromaDB 的高效向量存储
- **vLLM 集成**: 使用 vLLM OpenAI 兼容 API 进行 Embedding
- **灵活检索**: 统一的混合检索 + Rerank 策略
- **查询扩展**: LLM 驱动的智能查询扩展（专为教育场景优化）
  - 自动分解复合查询（如"导数和积分的关系" → "导数的定义" + "积分的定义"）
  - 数学术语同义词扩展（如"微商" → "导数"、"变化率"）
  - Few-shot 示例学习，准确识别需要/不需要分解的查询
  - 可通过 `QUERY_EXPANSION_ENABLED` 环境变量启用
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
│       ├── retriever.py      # 检索器
│       ├── query_expansion.py # 查询扩展
│       └── llm_client.py     # LLM 客户端
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

# ChromaDB
CHROMA_HOST=192.168.8.233
CHROMA_PORT=8200

# 查询扩展（可选，需要 LLM 服务）
QUERY_EXPANSION_ENABLED=true
QUERY_EXPANSION_MAX_EXPANSIONS=3

# LLM 配置（用于查询扩展）
LLM_PROVIDER=ollama
LLM_BASE_URL=http://192.168.8.233:11434
LLM_MODEL=qwen2.5:14b
```

### 3. 基本使用

