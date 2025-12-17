# API Key 认证说明

## 概述

本项目已从 JWT Token 认证迁移到 API Key 认证方式。所有 API 端点现在使用 `X-API-Key` HTTP 头进行身份验证。

## 配置 API Keys

### 环境变量配置

在 `.env` 文件中配置有效的 API keys（多个 key 使用逗号分隔）：

```bash
API_KEYS=your-api-key-1,your-api-key-2,your-api-key-3
```

### 示例配置

```bash
# 生产环境建议使用强随机字符串
API_KEYS=sk_live_abc123def456,sk_live_xyz789uvw012
```

## 使用 API Key

### HTTP 请求示例

所有需要认证的 API 请求都需要在 HTTP 头中包含 `X-API-Key`：

```bash
curl -X POST "http://localhost:8000/api/chat/message" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-1" \
  -d '{
    "user_id": "user123",
    "employee_id": "DE001",
    "query": "你好"
  }'
```

### Python 示例

```python
import requests

headers = {
    "X-API-Key": "your-api-key-1",
    "Content-Type": "application/json"
}

data = {
    "user_id": "user123",
    "employee_id": "DE001",
    "query": "你好"
}

response = requests.post(
    "http://localhost:8000/api/chat/message",
    headers=headers,
    json=data
)

print(response.json())
```

### JavaScript 示例

```javascript
const response = await fetch('http://localhost:8000/api/chat/message', {
  method: 'POST',
  headers: {
    'X-API-Key': 'your-api-key-1',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    user_id: 'user123',
    employee_id: 'DE001',
    query: '你好'
  })
});

const data = await response.json();
console.log(data);
```

## 需要认证的端点

以下端点需要 API Key 认证：

### 对话接口
- `POST /api/chat/message` - 同步对话
- `POST /api/chat/stream` - 流式对话

### 会话管理
- `GET /api/chat/session/{session_id}` - 获取会话信息
- `DELETE /api/chat/session/{session_id}` - 结束会话

### 数字员工管理
- `POST /api/ai/digital-employee/create` - 创建数字员工
- `PUT /api/ai/digital-employee/{employee_id}` - 更新数字员工
- `GET /api/ai/digital-employee/{employee_id}` - 获取数字员工信息
- `DELETE /api/ai/digital-employee/{employee_id}` - 删除数字员工
- `GET /api/ai/digital-employee/list` - 获取数字员工列表

### 知识库管理
- `POST /api/knowledge-base/create` - 创建知识库
- `POST /api/knowledge-base/{kb_id}/upload` - 上传文档
- `GET /api/knowledge-base/{kb_id}` - 获取知识库信息
- `POST /api/knowledge-base/{kb_id}/search` - 知识库检索

### 对话记录
- `GET /api/conversation/records` - 获取对话记录
- `GET /api/conversation/statistics` - 获取统计数据

### Webhook（Java 平台集成）
- `POST /api/ai/sensitive-words/sync-notify` - 敏感词同步
- `POST /api/ai/professional-words/sync-notify` - 专业词同步
- `POST /api/ai/faq/sync-notify` - FAQ 同步

## 错误处理

### 401 Unauthorized

**缺少 API Key**

```json
{
  "detail": "API key missing"
}
```

**无效的 API Key**

```json
{
  "detail": "Invalid API key"
}
```

### 示例错误处理

```python
try:
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 401:
        print("认证失败：API Key 无效或缺失")
    else:
        print(f"请求失败：{e}")
```

## 安全建议

1. **密钥管理**
   - 使用强随机字符串作为 API Key
   - 定期轮换 API Keys
   - 不要在代码中硬编码 API Keys

2. **传输安全**
   - 生产环境使用 HTTPS
   - 不要在 URL 中传递 API Key

3. **访问控制**
   - 为不同的客户端/应用分配不同的 API Key
   - 记录和监控 API Key 的使用情况
   - 发现异常时及时撤销相应的 API Key

4. **环境隔离**
   - 开发、测试、生产环境使用不同的 API Keys
   - 不要共享生产环境的 API Keys

## 迁移说明

### 从 JWT 迁移

如果您之前使用 JWT Token 认证，需要做以下调整：

**之前（JWT）：**
```bash
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." ...
```

**现在（API Key）：**
```bash
curl -H "X-API-Key: your-api-key-1" ...
```

### 代码迁移

**Python 示例：**

```python
# 之前
headers = {
    "Authorization": f"Bearer {jwt_token}",
    "Content-Type": "application/json"
}

# 现在
headers = {
    "X-API-Key": api_key,
    "Content-Type": "application/json"
}
```

## 常见问题

### Q: 如何生成安全的 API Key？

A: 可以使用以下方法生成：

```python
import secrets
api_key = f"sk_{secrets.token_urlsafe(32)}"
print(api_key)
```

```bash
# 或使用命令行
openssl rand -base64 32
```

### Q: 可以有多个 API Key 吗？

A: 可以。在 `.env` 文件中使用逗号分隔多个 key：

```bash
API_KEYS=key1,key2,key3
```

### Q: API Key 有过期时间吗？

A: 当前版本的 API Key 不会自动过期。如需实现过期机制，请考虑在数据库中存储 API Key 及其过期时间。

### Q: 如何撤销某个 API Key？

A: 从 `.env` 文件的 `API_KEYS` 列表中删除该 key，然后重启服务。

## 参考

- [API 使用文档](./API使用文档.md)
- [部署指南](./部署指南.md)
