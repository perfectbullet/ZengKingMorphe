# MinerU 客户端使用指南

## 概述

MinerU 客户端是一个增强的 PDF 解析工具，提供以下功能：

- **智能分页处理**：自动将大 PDF 文档拆分为每 8 页的小文档进行处理
- **MD5 缓存机制**：基于文件内容、页码范围和修改时间的 MD5 哈希实现缓存
- **MongoDB 状态跟踪**：使用 `mineru_` 前缀的集合记录处理状态
- **并行处理**：支持多个小文档并行请求（最多 3 个并发）
- **自动合并**：完成后自动合并所有小文档的结果

## 配置

在 `.env` 或 `.env-win` 文件中添加以下配置：

```bash
# MinerU PDF 解析配置
MINERU_API_URL=http://192.168.8.231:8000/file_parse
MINERU_ENABLED=true
MINERU_PAGES_PER_CHUNK=8
MINERU_TIMEOUT=600
MINERU_CACHE_TTL_DAYS=30
```

### 配置说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `MINERU_API_URL` | MinerU API 端点 URL | `http://192.168.8.231:8000/file_parse` |
| `MINERU_ENABLED` | 是否启用 MinerU（优先使用基本解析） | `false` |
| `MINERU_PAGES_PER_CHUNK` | 每个小文档的页数 | `8` |
| `MINERU_TIMEOUT` | API 请求超时时间（秒） | `600` |
| `MINERU_CACHE_TTL_DAYS` | 缓存保留天数 | `30` |

## MongoDB 集合

客户端使用两个 MongoDB 集合（都带 `mineru_` 前缀）：

### 1. mineru_cache
存储已处理的小文档缓存结果。

**文档结构**：
```json
{
  "_id": "md5_hash_value",
  "result": { /* API 返回的完整结果 */ },
  "start_page": 0,
  "end_page": 8,
  "created_at": "2025-01-08T10:00:00Z",
  "accessed_at": "2025-01-08T10:00:00Z"
}
```

**缓存键计算**：
```python
hash_input = f"{file_path}_{start_page}_{end_page}_{file_mtime}_{file_size}"
cache_key = md5(hash_input.encode()).hexdigest()
```

### 2. mineru_jobs
跟踪整个 PDF 文档的处理任务。

**文档结构**：
```json
{
  "_id": "job_id",
  "file_path": "/path/to/file.pdf",
  "file_name": "file.pdf",
  "status": "processing|completed|failed",
  "total_chunks": 10,
  "processed_chunks": 5,
  "failed_chunks": 0,
  "created_at": "2025-01-08T10:00:00Z",
  "updated_at": "2025-01-08T10:05:00Z",
  "results": [ /* 各个小文档的结果 */ ],
  "final_result": { /* 合并后的最终结果 */ },
  "error_message": "错误信息（如果失败）"
}
```

## 使用方法

### 方式一：通过 DocumentProcessor 集成使用

在文档上传时启用 MinerU 解析：

```python
from app.services.document_service import document_processor

# 方法 1：通过 metadata 参数指定
doc_id = await document_processor.process_document(
    file_path="/path/to/document.pdf",
    filename="document.pdf",
    kb_id="kb_123",
    metadata={"use_mineru": True}  # 启用 MinerU
)

# 方法 2：全局启用（修改 .env 中的 MINERU_ENABLED=true）
# 然后正常上传文档即可
doc_id = await document_processor.process_document(
    file_path="/path/to/document.pdf",
    filename="document.pdf",
    kb_id="kb_123"
)
```

### 方式二：直接使用 MineruClient

```python
from app.services.mineru_client import get_mineru_client

# 获取客户端实例
client = get_mineru_client()

# 处理 PDF
async with client:
    result = await client.process_pdf(
        file_path="/path/to/document.pdf",
        use_cache=True  # 使用缓存
    )

# 结果结构
{
    "status": "success",
    "chunks": 10,  # 处理的小文档数量
    "content": [ /* 合并后的内容 */ ],
    "metadata": {
        "total_pages": 80,
        "merged_at": "2025-01-08T10:05:00Z"
    }
}
```

### 查询任务状态

```python
from app.services.mineru_client import get_mineru_client

client = get_mineru_client()

# 查询任务状态
job_status = await client.get_job_status(job_id="job_id_here")

if job_status:
    print(f"状态: {job_status['status']}")
    print(f"进度: {job_status['processed_chunks']}/{job_status['total_chunks']}")
```

### 清理缓存

```python
from app.services.mineru_client import get_mineru_client

client = get_mineru_client()

# 清理 30 天以前的缓存
deleted_count = await client.clear_cache(older_than_days=30)
print(f"已清理 {deleted_count} 条缓存记录")
```

### 方式三：通过测试脚本

```bash
# 处理 PDF 文件
python tests/test_mineru_client.py process --file path/to/document.pdf

# 处理并保存 Markdown 结果
python tests/test_mineru_client.py process --file path/to/document.pdf --save-md
# Markdown 文件保存到: ai-service/docs/mineru_output/<文件名>.md

# 查询任务状态
python tests/test_mineru_client.py status --job-id job_id_here

# 清理缓存
python tests/test_mineru_client.py clear-cache --days 30
```

### 方式四：通过 Web API

```bash
# 获取最近的任务列表
GET /api/mineru/jobs?limit=10

# 获取任务详情
GET /api/mineru/jobs/{job_id}

# 获取任务的 Markdown 内容
GET /api/mineru/jobs/{job_id}/markdown

# 访问 Web 管理页面
GET /api/mineru/view
```

## 工作流程

```
1. 接收 PDF 文件
   ↓
2. 计算总页数
   ↓
3. 拆分为每 8 页的小文档（例如 80 页 → 10 个小文档）
   ↓
4. 对每个小文档：
   a. 计算 MD5（文件路径 + 页码范围 + 文件修改时间 + 文件大小）
   b. 检查 MongoDB 缓存（mineru_cache 集合）
   c. 如果缓存存在 → 使用缓存结果
   d. 如果缓存不存在 → 调用 MinerU API → 保存到缓存
   ↓
5. 合并所有小文档的结果
   ↓
6. 保存完整结果到 mineru_jobs
   ↓
7. 返回最终结果
```

## API 请求参数

MinerU 客户端调用 API 时使用以下参数：

```python
{
    'return_middle_json': 'true',
    'return_model_output': 'true',
    'return_md': 'true',
    'return_images': 'false',
    'start_page_id': str(start_page),  # 例如: 0
    'end_page_id': str(end_page),      # 例如: 8
    'parse_method': 'auto',
    'lang_list': 'ch',
    'output_dir': './output',
    'server_url': 'string',
    'return_content_list': 'true',
    'backend': 'pipeline',
    'table_enable': 'true',
    'response_format_zip': 'false',
    'formula_enable': 'true'
}
```

## 缓存策略

### 如何判断小文档已处理？

使用 MD5 哈希作为缓存键，包含以下信息：
- 文件路径
- 起始页码
- 结束页码
- 文件修改时间（mtime）
- 文件大小

**优点**：
- 如果文件被修改（mtime 或 size 变化），会自动重新处理
- 相同文件的相同页码范围会使用缓存
- 不依赖文件内容本身，避免大文件的 MD5 计算

### 缓存失效

缓存会在以下情况失效：
1. 文件被修改（mtime 或 size 变化）
2. 手动清理缓存（调用 `clear_cache()` 方法）
3. 超过 TTL 时间（默认 30 天）

## 性能优化

1. **并行处理**：最多 3 个并发 API 请求，避免过载
2. **智能缓存**：已处理的小文档直接返回缓存结果
3. **分页处理**：避免单次请求处理大文件超时
4. **异步 I/O**：所有文件操作和 API 请求都是异步的

## 错误处理

### API 调用失败

如果某个小文档的 API 调用失败：
- 该小文档会被标记为失败
- 任务会记录 `failed_chunks` 计数
- 任务状态变为 `failed`
- 错误信息保存在 `error_message` 字段

### 自动降级

如果 MinerU API 不可用或失败，`DocumentProcessor` 会自动降级到基本的 PyPDF 解析：

```python
try:
    # 尝试使用 MinerU
    result = await client.process_pdf(file_path)
except Exception as e:
    logger.warning("MinerU 失败，降级到基本解析")
    result = await self._extract_pdf_basic(file_path)
```

## 监控和日志

客户端使用结构化日志记录：

```python
logger.info(
    "Splitting PDF into chunks",
    file=file_path,
    total_pages=total_pages,
    pages_per_chunk=self.pages_per_chunk
)

logger.info(
    "Using cached result for chunk",
    cache_key=cache_key,
    start_page=start_page,
    end_page=end_page
)
```

## 注意事项

1. **临时文件**：处理过程中会创建 `temp_pdf_chunks/` 目录存放小文档，处理完成后自动删除
2. **MongoDB 索引**：确保在 `mineru_cache._id` 和 `mineru_jobs._id` 上有索引
3. **网络超时**：默认 10 分钟超时，大文件可能需要调整
4. **内存使用**：处理大文件时会将所有结果加载到内存中合并

## 完整示例

```python
import asyncio
from app.services.mineru_client import get_mineru_client
from app.services.document_service import document_processor

async def example_with_mineru():
    """示例：使用 MinerU 处理 PDF"""

    # 方式 1：通过 DocumentProcessor
    doc_id = await document_processor.process_document(
        file_path="D:/documents/example.pdf",
        filename="example.pdf",
        kb_id="kb_001",
        metadata={"use_mineru": True}
    )
    print(f"文档已处理: {doc_id}")

    # 方式 2：直接使用 MineruClient
    client = get_mineru_client()
    async with client:
        result = await client.process_pdf(
            file_path="D:/documents/example.pdf",
            use_cache=True
        )

        print(f"状态: {result['status']}")
        print(f"处理的小文档数: {result['chunks']}")
        print(f"总页数: {result['metadata']['total_pages']}")

        # 获取任务 ID
        job_id = result.get("job_id")

    # 查询任务状态
    if job_id:
        job_status = await client.get_job_status(job_id)
        print(f"任务状态: {job_status}")

# 运行示例
asyncio.run(example_with_mineru())
```

## 故障排查

### 问题：API 调用超时

**解决方案**：
1. 增加 `MINERU_TIMEOUT` 配置值
2. 减少 `MINERU_PAGES_PER_CHUNK` 值（例如改为 5 页）

### 问题：缓存未命中

**检查**：
- 文件是否被修改（检查 mtime 和 size）
- 缓存集合中是否存在记录
- MD5 计算是否正确

### 问题：MongoDB 连接失败

**检查**：
- `MONGODB_URI` 配置是否正确
- MongoDB 服务是否运行
- 网络连接是否正常
