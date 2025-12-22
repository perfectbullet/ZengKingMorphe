# 流式Chunk存储与查询功能说明

## 概述

本功能为OpenAI兼容的流式对话接口（`/v1/chat/completions`）添加了完整的chunk存储和查询能力。系统会自动将每次流式输出的chunk保存到MongoDB，支持多维度查询和重放。

## 功能特性

### 1. 自动Chunk存储

流式对话过程中，系统会自动保存以下类型的chunk：
- **user_query**: 用户提问（包含完整的messages历史）- **role**: 初始角色声明（assistant）
- **token**: 每个生成的token内容
- **done**: 完成标记（包含usage和metadata）
- **error**: 错误信息

每个chunk包含以下信息：
```python
{
    "chunk_id": "chatcmpl-xxx_chunk_1",        # 唯一标识
    "conversation_id": "conv_abc123",          # 会话ID
    "session_id": "sess_20251222_abc",         # 会话ID
    "user_id": "user_123456",                  # 用户ID
    "employee_id": "hutao",                    # 员工ID
    "chat_id": "chatcmpl-abc123",              # OpenAI格式的chat ID
    "chunk_type": "token",                     # chunk类型
    "chunk_data": {...},                       # 完整的chunk数据（OpenAI格式）
    "sequence": 42,                            # 序列号
    "timestamp": "2025-12-22T10:30:00.123Z",   # 时间戳
    "created_at": "2025-12-22T10:30:00.123Z"   # 创建时间
}
```

### 2. 多维度查询

新增查询接口：`GET /api/chat/stream/chunks`

支持的查询参数：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `user_id` | string | 否 | 用户ID |
| `employee_id` | string | 否 | 数字员工ID |
| `session_id` | string | 否 | 会话ID |
| `chat_id` | string | 否 | OpenAI格式的chat completion ID |
| `chunk_type` | string | 否 | chunk类型过滤（user_query/role/token/done/error） |
| `start_date` | string | 否 | 开始日期（YYYY-MM-DD） |
| `end_date` | string | 否 | 结束日期（YYYY-MM-DD） |
| `page` | integer | 否 | 页码（默认1） |
| `page_size` | integer | 否 | 每页数量（默认50，最大200） |

## API使用示例

### 1. 流式对话请求

```bash
curl -X POST "http://localhost:8100/api/chat/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "qwen3:32b",
    "messages": [
      {"role": "user", "content": "介绍一下首饰制作的雕蜡工艺"}
    ],
    "stream": true,
    "employee_id": "hutao",
    "user_id": "user_123456",
    "session_id": "sess_20251222_abc123"
  }'
```

**响应示例**（SSE流）：
```json
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1703232000,"model":"qwen3:32b","user_message":"介绍一下首饰制作的雕蜡工艺","messages":[{"role":"user","content":"介绍一下首饰制作的雕蜡工艺"}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1703232000,"model":"qwen3:32b","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1703232000,"model":"qwen3:32b","choices":[{"index":0,"delta":{"content":"雕"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1703232000,"model":"qwen3:32b","choices":[{"index":0,"delta":{"content":"蜡"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1703232000,"model":"qwen3:32b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":150,"total_tokens":170},"metadata":{"conversation_id":"conv_xyz","confidence":0.92,"kb_used":["kb_jewelry"],"web_search_used":false,"sources":{...}}}

data: [DONE]
```

### 2. 查询Chunks

#### 2.1 按session_id查询
```bash
curl -X GET "http://localhost:8100/api/chat/stream/chunks?session_id=sess_20251222_abc123&page=1&page_size=50" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

#### 2.2 按user_id和employee_id查询
```bash
curl -X GET "http://localhost:8100/api/chat/stream/chunks?user_id=user_123456&employee_id=hutao&page=1&page_size=50" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

#### 2.3 按chat_id查询（精确查询某次对话的所有chunks）
```bash
curl -X GET "http://localhost:8100/api/chat/stream/chunks?chat_id=chatcmpl-abc123&page=1&page_size=100" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

#### 2.4 只查询token类型的chunks
```bash
curl -X GET "http://localhost:8100/api/chat/stream/chunks?session_id=sess_20251222_abc123&chunk_type=token&page=1&page_size=100" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

#### 2.5 查询用户提问历史
```bash
curl -X GET "http://localhost:8100/api/chat/stream/chunks?user_id=user_123456&chunk_type=user_query&page=1&page_size=50" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

#### 2.6 按日期范围查询
```bash
curl -X GET "http://localhost:8100/api/chat/stream/chunks?user_id=user_123456&start_date=2025-12-01&end_date=2025-12-22&page=1&page_size=50" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**响应示例**：
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "chunks": [
      {
        "chunk_id": "chatcmpl-abc123_chunk_1",
        "conversation_id": null,
        "session_id": "sess_20251222_abc123",
        "user_id": "user_123456",
        "employee_id": "hutao",
        "chat_id": "chatcmpl-abc123",
        "chunk_type": "user_query",
        "chunk_data": {
          "id": "chatcmpl-abc123",
          "object": "chat.completion.chunk",
          "created": 1703232000,
          "model": "qwen3:32b",
          "user_message": "介绍一下首饰制作的雕蜡工艺",
          "messages": [
            {"role": "user", "content": "介绍一下首饰制作的雕蜡工艺"}
          ]
        },
        "sequence": 1,
        "timestamp": "2025-12-22T10:30:00.100Z",
        "created_at": "2025-12-22T10:30:00.100Z"
      },
      {
        "chunk_id": "chatcmpl-abc123_chunk_2",
        "conversation_id": "conv_xyz",
        "session_id": "sess_20251222_abc123",
        "user_id": "user_123456",
        "employee_id": "hutao",
        "chat_id": "chatcmpl-abc123",
        "chunk_type": "role",
        "chunk_data": {
          "id": "chatcmpl-abc123",
          "object": "chat.completion.chunk",
          "created": 1703232000,
          "model": "qwen3:32b",
          "choices": [
            {
              "index": 0,
              "delta": {"role": "assistant", "content": ""},
              "finish_reason": null
            }
          ]
        },
        "sequence": 2,
        "timestamp": "2025-12-22T10:30:00.123Z",
        "created_at": "2025-12-22T10:30:00.123Z"
      },
      {
        "chunk_id": "chatcmpl-abc123_chunk_3",
        "chunk_type": "token",
        "sequence": 3,
        "chunk_data": {
          "id": "chatcmpl-abc123",
          "choices": [{"delta": {"content": "雕"}}]
        },
        ...
      }
    ],
    "pagination": {
      "page": 1,
      "page_size": 50,
      "total_count": 156,
      "total_pages": 4,
      "has_next": true,
      "has_prev": false
    }
  }
}
```

## 使用场景

### 1. 对话重放
通过`chat_id`查询所有chunks（包括user_query和token），按sequence排序，可以精确重放整个对话过程，包括用户提问和AI回复。

### 2. 用户提问分析
通过`chunk_type=user_query`过滤，可以提取所有用户提问历史，用于：
- 用户意图分析
- 热门问题统计
- 问题分类和标注
- 训练数据收集

### 3. 用户行为分析
通过`user_id`和日期范围查询，分析用户的对话模式、频率、内容偏好等。

### 4. 员工性能监控
通过`employee_id`查询，统计数字员工的响应速度、token生成速率、错误率等指标。

### 5. 调试与排障
当流式对话出现问题时，可以通过`session_id`或`chat_id`查询完整的chunk序列，定位问题节点。

### 6. 计费与统计
统计每个用户/员工的token消耗量（done类型的chunk包含usage信息）。

## 测试脚本

使用提供的测试脚本：

```bash
cd ai-service
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe test_stream_chunks.py
```

测试脚本会：
1. 发起一次流式对话请求
2. 等待2秒确保数据保存
3. 执行4种不同的查询场景
4. 展示查询结果和统计信息

## 数据库索引

MongoDB中`stream_chunks`集合已创建以下索引以优化查询性能：

- `chunk_id` (unique)
- `session_id` + `sequence`
- `user_id` + `created_at` (descending)
- `employee_id` + `created_at` (descending)
- `chat_id` + `sequence`
- `created_at` (descending)

## 注意事项

1. **存储容量**: 流式chunk数据量较大，建议定期清理历史数据或设置TTL索引
2. **性能影响**: 每个token都会触发一次数据库写入，高并发场景建议使用批量写入优化
3. **查询限制**: 单次查询最多返回200条记录，大量数据需要分页查询
4. **时区**: 所有时间戳使用UTC时区（ISO 8601格式，带Z后缀）

## 技术实现

### 数据模型
- 文件：`ai-service/app/models/database.py`
- 类：`StreamChunkModel`

### 索引配置
- 文件：`ai-service/app/core/database.py`
- 位置：`_create_indexes()` 方法

### API端点
- 文件：`ai-service/app/api/endpoints/chat.py`
- 流式输出：`generate_openai_stream_response()`
- 查询接口：`query_stream_chunks()`

### Schema定义
- 文件：`ai-service/app/models/schemas.py`
- 类：`StreamChunkQuery`, `StreamChunkResponse`
