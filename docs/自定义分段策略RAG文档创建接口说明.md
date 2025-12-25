# 自定义分段策略的RAG文档创建接口说明

## 概述

新增接口 `POST /api/knowledge_base/documents/create_with_segment` 为Java平台提供自定义文档分段策略支持，实现灵活的文本预处理、分段切分和合并功能。

## 接口信息

- **路径**: `/api/knowledge_base/documents/create_with_segment`
- **方法**: `POST`
- **Content-Type**: `application/json`
- **认证**: API Key (可选，当前auth已禁用)

## 请求参数

### 主要参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| team_id | int | 是 | 团队ID |
| dataset_id | int | 是 | 知识库ID（Java端） |
| resource_id | int | 是 | 系统资源ID |
| document_name | string | 是 | 文档名称 |
| start_time | string | 否 | 生效开始时间 |
| end_time | string | 否 | 生效结束时间 |
| segment_flag | int | 是 | 分段策略：0=自动分段，1=自定义分段 |
| segment_vo | object | 否 | 自定义分段配置（segment_flag=1时必填） |
| rag_data_set_id | string | 是 | RAG系统知识库ID（格式：kb_xxx） |
| resource_url | string | 是 | 文档下载URL（支持HTTP/HTTPS） |

### segment_vo 配置（自定义分段参数）

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| team_id | int | - | 团队ID |
| dataset_document_id | int | - | RAG文档ID |
| rag_document_id | string | null | RAG系统文档ID |
| is_space_flag | int | 0 | 文本预处理-删除连续空格/换行/制表符：0=不启用，1=启用 |
| is_menu_flag | int | 0 | 文本预处理-删除目录/页眉/页脚：0=不启用，1=启用 |
| segment_type | int | 0 | 分段方式：0=换行切分，1=分段标识符切分 |
| is_segment_union_flag | int | 0 | 换行切分-是否分段合并：0=否，1=是 |
| segment_union_max_length | int | 512 | 分段最大长度（字符数，范围：128-2048） |
| segment_identifier_type | int | 0 | 标识符类型：0=系统内置，1=自定义 |
| identifier_default | string | "" | 系统内置标识符bitmap（7位字符串，如"1111111"） |
| identifier_customize | string | "" | 自定义标识符列表（逗号分隔，如"###,===,---"） |

## 系统内置标识符说明

`identifier_default` 是一个7位的bitmap字符串，每一位代表一个系统内置标识符是否启用（'1'=启用，'0'=禁用）：

| 位置 | 标识符 | 说明 |
|------|--------|------|
| 0 | `......` | 省略号（6个点） |
| 1 | `。` | 中文句号 |
| 2 | `.` | 英文句号 |
| 3 | `！` | 中文感叹号 |
| 4 | `!` | 英文感叹号 |
| 5 | `？` | 中文问号 |
| 6 | `?` | 英文问号 |

**示例**：
- `"1111111"` - 启用所有标识符
- `"0110101"` - 仅启用中文句号、英文句号、中文问号、英文问号
- `"1000000"` - 仅启用省略号

## 分段策略详解

### 1. 自动分段（segment_flag=0）

使用系统默认配置进行文档切分：
- **默认chunk_size**: 512字符
- **默认chunk_overlap**: 50字符
- **分隔符**: 根据文件类型自动选择（支持Markdown、HTML、纯文本）

### 2. 换行切分（segment_type=0）

基于换行符切分文档，支持分段合并：
- 以 `\n\n`（段落）和 `\n`（行）为主要分隔符
- 如果 `is_segment_union_flag=1`，将小段合并至 `segment_union_max_length`
- 适用于结构化文本（如Markdown、文章段落）

### 3. 标识符切分（segment_type=1）

使用指定标识符进行切分，采用LangChain的RecursiveCharacterTextSplitter串联机制：

#### 系统内置标识符（segment_identifier_type=0）
```python
# 示例：identifier_default="1111111"
# 分隔符优先级：
['......', '。', '.', '！', '!', '？', '?', '\n\n', '\n', ' ', '']
```

#### 自定义标识符（segment_identifier_type=1）
```python
# 示例：identifier_customize="###,===,---"
# 分隔符优先级：
['###', '===', '---', '\n\n', '\n', ' ', '']
```

**工作机制**：RecursiveCharacterTextSplitter按优先级依次尝试分隔符，直到满足chunk_size约束。

## 文本预处理功能

### 删除连续空格/换行/制表符（is_space_flag=1）

- 多个连续空格 → 单个空格
- 3个以上换行 → 2个换行（保留段落分隔）
- 制表符 → 单个空格

**正则表达式**：
```python
text = re.sub(r' {2,}', ' ', text)         # 空格
text = re.sub(r'\n{3,}', '\n\n', text)    # 换行
text = re.sub(r'\t+', ' ', text)          # 制表符
```

### 删除目录/页眉/页脚（is_menu_flag=1）

使用启发式规则移除常见的非内容元素：
- 目录标题（"目录"、"Table of Contents"）
- 章节标题（"第X章"、"Chapter X"）
- 页眉页脚（"页眉"、"页脚"、"Page X"）

**注意**：此功能基于正则匹配，可能需要根据实际文档格式调整。

## 请求示例

### 示例1：自动分段

```json
{
  "team_id": 40,
  "dataset_id": 3,
  "resource_id": 48907,
  "document_name": "产品手册.pdf",
  "segment_flag": 0,
  "rag_data_set_id": "kb_c8507c336f48",
  "resource_url": "https://example.com/manual.pdf"
}
```

### 示例2：自定义分段（系统标识符）

```json
{
  "team_id": 40,
  "dataset_id": 3,
  "resource_id": 48907,
  "document_name": "首饰雕蜡工艺-全本.txt",
  "segment_flag": 1,
  "segment_vo": {
    "team_id": 40,
    "dataset_document_id": 16,
    "is_space_flag": 1,
    "is_menu_flag": 1,
    "segment_type": 1,
    "is_segment_union_flag": 1,
    "segment_union_max_length": 700,
    "segment_identifier_type": 0,
    "identifier_default": "1111111",
    "identifier_customize": ""
  },
  "rag_data_set_id": "kb_c8507c336f48",
  "resource_url": "https://education-test-private.oss-cn-beijing.aliyuncs.com/text/file.txt"
}
```

### 示例3：自定义标识符切分

```json
{
  "team_id": 50,
  "dataset_id": 5,
  "resource_id": 50001,
  "document_name": "API文档.md",
  "segment_flag": 1,
  "segment_vo": {
    "team_id": 50,
    "dataset_document_id": 25,
    "is_space_flag": 1,
    "is_menu_flag": 0,
    "segment_type": 1,
    "is_segment_union_flag": 1,
    "segment_union_max_length": 800,
    "segment_identifier_type": 1,
    "identifier_default": "",
    "identifier_customize": "##,===,---"
  },
  "rag_data_set_id": "kb_api_docs",
  "resource_url": "https://example.com/api-docs.md"
}
```

## 响应格式

### 成功响应（HTTP 200）

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "task_id": "document_task_202512240930_abc12345",
    "rag_document_id": "doc_1a2b3c4d5e6f",
    "status": "processing",
    "team_id": 40,
    "dataset_id": 3,
    "resource_id": 48907,
    "document_name": "首饰雕蜡工艺-全本.txt",
    "rag_data_set_id": "kb_c8507c336f48"
  }
}
```

### 错误响应

#### 知识库不存在（HTTP 404）

```json
{
  "code": 404,
  "message": "Knowledge base kb_xxx not found"
}
```

#### 文件下载失败（HTTP 400）

```json
{
  "code": 400,
  "message": "Failed to download file: HTTP 403"
}
```

#### 内部错误（HTTP 500）

```json
{
  "code": 500,
  "message": "Failed to create RAG document"
}
```

## 任务状态查询

创建文档后，使用返回的 `task_id` 查询处理状态：

**请求**：
```
GET /api/knowledge_base/documents/tasks/{task_id}
```

**响应示例**：
```json
{
  "status": "success",
  "task": {
    "task_id": "document_task_202512240930_abc12345",
    "kb_id": "kb_c8507c336f48",
    "filename": "首饰雕蜡工艺-全本.txt",
    "status": "running",
    "progress": 45.5,
    "total_chunks": 120,
    "processed_chunks": 55,
    "doc_id": "doc_1a2b3c4d5e6f",
    "started_at": "2025-12-24T09:30:15Z"
  }
}
```

**任务状态值**：
- `pending` - 等待处理
- `running` - 处理中
- `completed` - 已完成
- `failed` - 失败
- `cancelled` - 已取消

## 实现架构

### 核心组件

1. **Schema层** (`schemas.py`)
   - `SegmentVo`: 分段配置模型
   - `CreateRagDocumentRequest`: 请求模型
   - `CreateRagDocumentResponse`: 响应模型

2. **API层** (`knowledge_base.py`)
   - 接口端点实现
   - URL下载
   - 参数验证
   - 任务提交

3. **服务层** (`document_service.py`)
   - `_preprocess_text()`: 文本预处理（正则清洗）
   - `_chunk_text()`: 动态分段切分（支持自定义配置）
   - RecursiveCharacterTextSplitter串联机制

4. **任务层** (`task_processor.py`)
   - 异步任务队列
   - chunk_config参数传递
   - 进度追踪

### 数据流

```
Java平台 → API端点 
  ↓
下载文件 → 提交任务 
  ↓
TaskProcessor → 异步队列
  ↓
DocumentService → 预处理 → 分段 → 向量化
  ↓
三重存储（ChromaDB + ElasticSearch + MongoDB）
```

## 技术细节

### 分段算法

使用LangChain的 `RecursiveCharacterTextSplitter`：

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=separators,  # 动态确定
    length_function=len,
    is_separator_regex=False
)
```

**优势**：
- 按优先级递归尝试分隔符
- 自动处理边界情况
- 保持语义完整性

### 向量化存储

切分后的每个chunk会同步存储到三个数据库：

1. **ChromaDB**: 向量检索（语义搜索）
2. **ElasticSearch**: BM25关键词检索
3. **MongoDB**: 元数据和全文存储

## 兼容性说明

### 向后兼容

- 原有 `/documents/upload` 端点保持不变
- 默认使用全局 `chunk_size`/`chunk_overlap` 配置
- 新端点为独立接口，不影响现有功能

### Java平台集成

接口参数完全匹配Java端请求格式（驼峰命名自动转为蛇形命名）：
- `teamId` → `team_id`
- `segmentVo` → `segment_vo`
- `ragDataSetId` → `rag_data_set_id`

## 测试说明

### 运行测试脚本

```bash
# 确保服务运行在 http://localhost:8100
cd ai-service

# 激活虚拟环境（Windows）
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/Activate.ps1

# 运行测试
python tests/test_create_rag_document.py
```

### 测试前准备

1. 确保知识库 `kb_316a7dbc75d0` 存在（或修改测试脚本中的kb_id）
2. 替换 `resource_url` 为真实可访问的文件URL
3. 检查AI服务日志确认处理流程

### Swagger UI测试

访问 `http://localhost:8100/docs` 使用交互式API文档测试接口。

## 日志追踪

关键日志点：

```python
# 接口层
logger.info("Create RAG document with segment config", 
            team_id=..., kb_id=..., segment_flag=...)

# 预处理层
logger.info("Applied space/newline/tab preprocessing")
logger.info("Applied TOC/header/footer removal")

# 分段层
logger.info("Using system default identifiers", separators=...)
logger.info("Using custom identifiers", separators=...)
logger.info("Chunked document with RecursiveCharacterTextSplitter",
            chunks_count=..., avg_chunk_size=...)
```

## 常见问题

### Q: 为什么选择字符数而不是token数作为chunk_size？

A: 简化计算，避免token化开销。实际embeddings时会在 `embeddings.py` 中自动截断超限文本。

### Q: 自定义标识符如何与系统默认标识符结合？

A: 不支持混合使用。`segment_identifier_type=0` 时使用系统标识符，`=1` 时使用自定义标识符。

### Q: is_menu_flag删除目录功能准确吗？

A: 基于简单正则匹配，适用于常见格式。复杂文档可能需要增强规则或使用AI识别。

### Q: 文件下载超时如何处理？

A: 当前超时设置为120秒（`aiohttp.ClientTimeout(total=120)`），可根据需要调整。

## 未来优化方向

1. **增强目录识别**：引入LLM或机器学习模型识别TOC
2. **分段预览**：支持分段配置预览（不实际处理）
3. **批量导入**：支持多文件批量创建
4. **配置模板**：保存常用分段配置为模板
5. **分段质量评分**：基于语义完整性评估分段效果

## 参考资料

- [LangChain RecursiveCharacterTextSplitter文档](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter)
- [项目架构指南](../.github/copilot-instructions.md)
- [异步文档上传说明](./异步文档上传使用说明.md)
