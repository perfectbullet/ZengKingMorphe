# API 使用文档

## 基础信息

**Base URL**: `http://localhost:8000`  
**API 文档**: `http://localhost:8000/docs`  
**认证方式**: JWT Token / API Key

## 认证

### 方式一：JWT Token
```bash
# Header
Authorization: Bearer <your-jwt-token>
```

### 方式二：API Key
```bash
# Header
X-API-Key: <your-api-key>
```

## 核心 API

### 1. 对话接口

#### 1.1 同步对话
```http
POST /api/chat/message
Content-Type: application/json

{
  "user_id": "user_001",
  "employee_id": "DE001",
  "session_id": "sess_001",  // 可选，不提供则自动生成
  "query": "如何重置密码？",
  "context": {
    "platform": "web"
  }
}
```

**响应**:
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "conversation_id": "conv_abc123",
    "session_id": "sess_001",
    "answer": "您可以通过以下步骤重置密码...",
    "intent": "password_reset",
    "confidence": 0.95,
    "kb_used": ["kb_001"],
    "web_search_used": false,
    "timestamp": "2025-12-16T10:00:00Z"
  }
}
```

#### 1.2 流式对话（SSE）
```http
POST /api/chat/stream
Content-Type: application/json

{
  "user_id": "user_001",
  "employee_id": "DE001",
  "query": "介绍一下你们的产品"
}
```

**SSE 响应流**:
```
data: {"type": "start", "session_id": "sess_001"}

data: {"type": "token", "content": "我们"}

data: {"type": "token", "content": "的产品"}

data: {"type": "done", "conversation_id": "conv_001"}
```

#### 1.3 获取会话信息
```http
GET /api/chat/session/{session_id}
```

**响应**:
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "session_id": "sess_001",
    "user_id": "user_001",
    "employee_id": "DE001",
    "created_at": "2025-12-16T10:00:00Z",
    "last_activity": "2025-12-16T10:30:00Z",
    "message_count": 5,
    "context": []
  }
}
```

#### 1.4 结束会话
```http
DELETE /api/chat/session/{session_id}
```

### 2. 数字员工管理

#### 2.1 创建数字员工
```http
POST /api/ai/digital-employee/create
Content-Type: application/json

{
  "employee_id": "DE001",
  "name": "小智",
  "domain": "客户服务",
  "role": "客服助手",
  "description": "专业的客户服务数字员工",
  "personality": {
    "tone": "professional",
    "style": "friendly",
    "language": "zh-CN",
    "formality": "moderate"
  },
  "capabilities": {
    "kb_ids": ["kb_001", "kb_002"],
    "web_search_enabled": true,
    "max_context_turns": 10
  },
  "greeting": "您好！我是客服助手小智，有什么可以帮助您的？",
  "hot_questions": [
    "如何重置密码？",
    "如何联系客服？"
  ]
}
```

#### 2.2 更新数字员工
```http
PUT /api/ai/digital-employee/{employee_id}
Content-Type: application/json

{
  "name": "小智（升级版）",
  "description": "更新后的描述"
}
```

#### 2.3 查询数字员工
```http
GET /api/ai/digital-employee/{employee_id}
```

#### 2.4 删除数字员工
```http
DELETE /api/ai/digital-employee/{employee_id}
```

#### 2.5 列出数字员工
```http
GET /api/ai/digital-employee/list?domain=客户服务&page=1&page_size=20
```

### 3. 知识库管理

#### 3.1 创建知识库
```http
POST /api/knowledge-base/create
Content-Type: multipart/form-data

name=产品知识库
description=产品相关的知识库
category=产品
```

#### 3.2 列出知识库
```http
GET /api/knowledge-base/list?category=产品&page=1&page_size=20
```

#### 3.3 上传文档
```http
POST /api/knowledge-base/documents/upload
Content-Type: multipart/form-data

kb_id=kb_001
category=产品文档
files=@document1.pdf
files=@document2.docx
```

**支持的文件格式**:
- PDF (.pdf)
- Word (.docx)
- 文本 (.txt)
- Markdown (.md)
- HTML (.html)

#### 3.4 列出文档
```http
GET /api/knowledge-base/documents/list?kb_id=kb_001&status=completed&page=1&page_size=20
```

### 4. Webhook 接口

#### 4.1 FAQ 同步通知
```http
POST /api/ai/faq/sync-notify
Content-Type: application/json
X-API-Key: <your-api-key>

{
  "event_type": "create",
  "word_ids": ["faq_001", "faq_002"],
  "timestamp": "2025-12-16T10:00:00Z"
}
```

#### 4.2 敏感词同步通知
```http
POST /api/ai/sensitive-words/sync-notify
Content-Type: application/json
X-API-Key: <your-api-key>

{
  "event_type": "update",
  "word_ids": ["sen_001"],
  "timestamp": "2025-12-16T10:00:00Z"
}
```

#### 4.3 专业词同步通知
```http
POST /api/ai/professional-words/sync-notify
Content-Type: application/json
X-API-Key: <your-api-key>

{
  "event_type": "delete",
  "word_ids": ["prof_001"],
  "timestamp": "2025-12-16T10:00:00Z"
}
```

## 错误处理

### 错误响应格式
```json
{
  "code": 400,
  "message": "Invalid request",
  "error": {
    "type": "ValidationError",
    "details": "Missing required field: user_id"
  }
}
```

### 常见错误码
- `400` - 请求参数错误
- `401` - 未授权（Token无效或过期）
- `403` - 禁止访问（权限不足）
- `404` - 资源不存在
- `429` - 请求过于频繁（触发限流）
- `500` - 服务器内部错误
- `503` - 服务暂时不可用

## 限流规则

- 每用户每分钟最大请求数：60
- 每会话每分钟最大请求数：10
- 最大并发会话数：100

## 使用示例

### Python
```python
import requests

# 发起对话
response = requests.post(
    "http://localhost:8000/api/chat/message",
    json={
        "user_id": "user_001",
        "employee_id": "DE001",
        "query": "如何使用这个产品？"
    }
)

data = response.json()
print(data["data"]["answer"])
```

### JavaScript
```javascript
// 发起对话
fetch('http://localhost:8000/api/chat/message', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    user_id: 'user_001',
    employee_id: 'DE001',
    query: '如何使用这个产品？'
  })
})
.then(response => response.json())
.then(data => console.log(data.data.answer));
```

### cURL
```bash
# 发起对话
curl -X POST http://localhost:8000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "employee_id": "DE001",
    "query": "如何使用这个产品？"
  }'
```

## 最佳实践

### 1. 会话管理
- 使用相同的 `session_id` 保持对话上下文
- 会话 30 分钟无活动自动过期
- 及时调用结束会话接口释放资源

### 2. 错误处理
- 始终检查响应状态码
- 实现重试机制（指数退避）
- 记录错误日志

### 3. 性能优化
- 使用流式接口提升响应速度
- 批量上传文档
- 复用会话减少初始化开销

### 4. 安全建议
- 使用 HTTPS
- 定期轮换 API Key
- 实施请求签名
- 记录审计日志
