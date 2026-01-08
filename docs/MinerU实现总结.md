# MinerU 客户端实现总结

## 实现内容

### 1. MineruClient 类
**文件**: [ai-service/app/services/mineru_client.py](ai-service/app/services/mineru_client.py)

**核心功能**:
- ✅ PDF 自动分页处理（每 8 页一个小文档）
- ✅ MD5 哈希缓存机制（基于文件路径+页码+时间戳+大小）
- ✅ MongoDB 状态跟踪（`mineru_cache` 和 `mineru_jobs` 集合）
- ✅ 并行处理支持（最多 3 个并发请求）
- ✅ 自动合并小文档结果
- ✅ 任务状态查询 API
- ✅ 缓存清理功能

### 2. DocumentProcessor 集成
**文件**: [ai-service/app/services/document_service.py](ai-service/app/services/document_service.py:177-268)

**集成方式**:
- 新增 `use_mineru` 参数支持
- `_extract_pdf_with_mineru()` 方法实现增强解析
- 自动降级机制（MinerU 失败时回退到基本 PyPDF 解析）

### 3. 配置管理
**文件**: [ai-service/app/core/config.py](ai-service/app/core/config.py:178-202)

**新增配置项**:
```bash
MINERU_API_URL=http://192.168.8.231:8000/file_parse
MINERU_ENABLED=false
MINERU_PAGES_PER_CHUNK=8
MINERU_TIMEOUT=600
MINERU_CACHE_TTL_DAYS=30
```

### 4. 数据库设计

#### mineru_cache 集合
```json
{
  "_id": "md5_hash_value",
  "result": { /* API 返回结果 */ },
  "start_page": 0,
  "end_page": 8,
  "created_at": "2025-01-08T10:00:00Z",
  "accessed_at": "2025-01-08T10:00:00Z"
}
```

#### mineru_jobs 集合
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
  "results": [],
  "final_result": {},
  "error_message": null
}
```

## 缓存设计说明

### MD5 计算方式（已优化 v2）
```python
# 使用绝对路径确保一致性
abs_path = os.path.abspath(file_path)

# 只使用文件大小，不使用时间戳（避免精度问题）
file_stat = os.stat(abs_path)
hash_input = f"{abs_path}_{start_page}_{end_page}_{file_stat.st_size}"
cache_key = hashlib.md5(hash_input.encode()).hexdigest()
```

### 优化历史

**问题（v1）**: 使用 `mtime`（修改时间）导致重复缓存
- `mtime` 是浮点数，在不同时间点精度可能不同
- 相同文件和页码范围产生不同的 MD5
- 数据库中出现 46 条记录，实际应该只有 23 条（每个页码范围一条）

**修复（v2）**: 移除 `mtime`，只使用文件大小
- 确保相同的文件和页码范围始终产生相同的 MD5
- 添加了重复清理脚本：`scripts/clean_duplicate_cache.py`

### 优势
1. **稳定性**: 相同文件+页码范围始终产生相同的 MD5
2. **性能**: 不需要读取整个文件内容计算 MD5
3. **简洁**: 只使用必要的标识信息（路径+页码+大小）
4. **合理**: 文件内容变化时，大小通常也会变化

### 判断是否已处理
查询 `mineru_cache` 集合中是否存在该 `cache_key`：
- 存在 → 使用缓存结果
- 不存在 → 调用 API 处理

## 使用方法

### 方式一：通过 DocumentProcessor
```python
# 启用 MinerU
doc_id = await document_processor.process_document(
    file_path="/path/to/file.pdf",
    filename="file.pdf",
    kb_id="kb_123",
    metadata={"use_mineru": True}
)
```

### 方式二：直接使用 MineruClient
```python
from app.services.mineru_client import get_mineru_client

client = get_mineru_client()
async with client:
    result = await client.process_pdf("/path/to/file.pdf", use_cache=True)
```

### 测试脚本
```bash
# 处理 PDF
python scripts/test_mineru_client.py process --file path/to/file.pdf

# 查询任务状态
python scripts/test_mineru_client.py status --job-id job_id_here

# 清理缓存
python scripts/test_mineru_client.py clear-cache --days 30
```

## 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 接收 PDF 文件                                             │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 计算总页数，拆分为每 8 页的小文档                          │
│    例如: 80 页 → 10 个小文档 (0-7, 8-15, ..., 72-79)        │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 为每个小文档创建任务记录到 mineru_jobs                    │
└────────────────────┬────────────────────────────────────────┘
                     ↓
         ┌───────────┴───────────┐
         ↓                       ↓
┌─────────────────┐     ┌─────────────────┐
│ 4a. 计算 MD5    │     │ 4b. 查询缓存    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ↓
         ┌───────────┴───────────┐
         ↓                       ↓
┌─────────────────┐     ┌─────────────────┐
│ 5a. 缓存命中    │     │ 5b. 缓存未命中  │
│ 使用缓存结果    │     │ 调用 MinerU API │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │                       ↓
         │              ┌─────────────────┐
         │              │ 保存到缓存      │
         │              └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. 并行处理所有小文档（最多 3 个并发）                        │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. 合并所有小文档结果                                        │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. 更新任务状态为 completed，保存最终结果                    │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 9. 返回合并后的结果                                         │
└─────────────────────────────────────────────────────────────┘
```

## API 调用参数

每个小文档调用 MinerU API 时使用的参数：

```python
{
    'return_middle_json': 'true',
    'return_model_output': 'true',
    'return_md': 'true',
    'return_images': 'false',
    'start_page_id': '0',      # 起始页码
    'end_page_id': '8',        # 结束页码
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

## 性能优化

1. **并行处理**: 使用 `asyncio.Semaphore(3)` 限制最多 3 个并发请求
2. **智能缓存**: 已处理的小文档直接返回，无需重新调用 API
3. **分页处理**: 避免单次请求处理大文件导致超时
4. **异步 I/O**: 所有文件操作和网络请求都是异步的

## 错误处理

### 小文档处理失败
- 记录 `failed_chunks` 计数
- 任务状态标记为 `failed`
- 保存错误信息到 `error_message`

### 自动降级
DocumentProcessor 中的 `_extract_pdf_with_mineru()` 方法会捕获异常：
```python
try:
    result = await client.process_pdf(file_path)
except Exception as e:
    logger.warning("MinerU 失败，降级到基本解析")
    return await self._extract_pdf_basic(file_path)
```

## 文件清单

| 文件 | 说明 |
|------|------|
| [ai-service/app/services/mineru_client.py](ai-service/app/services/mineru_client.py) | MinerU 客户端实现 |
| [ai-service/app/services/document_service.py](ai-service/app/services/document_service.py) | DocumentProcessor 集成（修改） |
| [ai-service/app/core/config.py](ai-service/app/core/config.py) | 配置项（新增） |
| [docs/MinerU客户端使用指南.md](docs/MinerU客户端使用指南.md) | 详细使用文档 |
| [scripts/test_mineru_client.py](ai-service/scripts/test_mineru_client.py) | 测试脚本 |

## 依赖项

需要安装以下 Python 包（已在项目中）：
```bash
pip install httpx          # 异步 HTTP 客户端
pip install pypdf          # PDF 处理
pip install motor          # MongoDB 异步驱动
```

## 后续优化建议

1. **进度回调**: 添加处理进度回调，实时通知前端
2. **断点续传**: 支持中断后继续处理
3. **批量清理**: 定期自动清理过期缓存
4. **监控指标**: 添加 Prometheus 监控指标
5. **限流保护**: 添加 API 调用限流
6. **结果压缩**: 对缓存结果进行压缩节省存储
