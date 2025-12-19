# Embedding配置修复说明

## 问题描述

环境变量 `EMBEDDING_TYPE=siliconflow` 未能正确加载到 `settings.embedding_type` 中，导致系统使用默认值 `openai_style`。

**错误日志：**
```json
{
  "embedding_type": "openai_style",  // 错误：应该是 "siliconflow"
  "embedding_model": "BAAI/bge-large-zh-v1.5",
  "embedding_base_url": "http://192.168.8.230:50009",
  "event": "Creating embedding function"
}
```

## 根本原因

### 1. 缺少默认值
在 [config.py](../ai-service/app/core/config.py) 中，以下字段使用了 `...` 作为默认值，表示必填字段：

```python
embedding_type: str = Field(..., description="...")  # ❌ 必填字段
embedding_base_url: str = Field(..., description="...")  # ❌ 必填字段
embedding_api_url: str = Field(..., description="...")  # ❌ 必填字段
```

当Pydantic无法从环境变量读取时（如`.env`文件未正确加载），会导致配置加载失败或使用未定义的默认值。

### 2. 硬编码参数
在 [chroma.py](../ai-service/app/core/chroma.py) 中，SiliconFlow初始化时硬编码了参数：

```python
# ❌ 硬编码，未使用配置值
embedder = SiliconFlowEmbeddings(
    model="BAAI/bge-large-zh-v1.5",  # 应使用 settings.embedding_model
    api_key=settings.siliconflow_api_key,
    url="https://api.siliconflow.cn/v1/embeddings"  # 应使用 settings.embedding_api_url
)
```

即使环境变量配置正确，仍会使用硬编码的错误URL。

## 修复方案

### 修复1: 为配置字段添加默认值

**文件：** `ai-service/app/core/config.py`

```python
# ✅ 修复后
embedding_type: str = Field(
    default="openai_style",  # 添加默认值
    description="Embedding type: openai_style or siliconflow"
)
embedding_model: str = Field(default="BAAI/bge-large-zh-v1.5")
embedding_base_url: str = Field(
    default="http://localhost:50009",  # 添加默认值
    description="Embedding service base URL"
)
embedding_api_url: str = Field(
    default="http://localhost:50009",  # 添加默认值
    description="Embedding API URL"
)
```

**原因：**
- 提供默认值可以避免配置加载失败
- 保证服务在.env文件缺失时仍能启动（使用默认配置）
- 环境变量优先级高于默认值，正确配置会覆盖默认值

### 修复2: 使用配置值替代硬编码

**文件：** `ai-service/app/core/chroma.py`

```python
# ✅ 修复后
if settings.embedding_type == "siliconflow":
    embedder = SiliconFlowEmbeddings(
        model=settings.embedding_model,  # 使用配置
        api_key=settings.siliconflow_api_key or settings.embedding_api_key,  # 支持两个key
        base_url=settings.embedding_api_url  # 使用配置
    )
    logger.info(
        "Using SiliconFlow embeddings",
        model=settings.embedding_model,
        base_url=settings.embedding_api_url,
        has_api_key=bool(settings.siliconflow_api_key or settings.embedding_api_key)
    )
```

**改进点：**
1. 使用 `settings.embedding_model` 替代硬编码的 `"BAAI/bge-large-zh-v1.5"`
2. 使用 `settings.embedding_api_url` 替代硬编码的URL
3. API key支持两种环境变量（`siliconflow_api_key` 或 `embedding_api_key`）
4. 日志记录完整配置信息，便于调试

## 验证修复

### 1. 运行配置测试

```powershell
# 激活虚拟环境
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/Activate.ps1

# 运行测试
cd ai-service
python test_config_loading.py
```

**预期输出：**
```
📦 Embedding 配置:
  EMBEDDING_TYPE: siliconflow  ✓
  EMBEDDING_MODEL: BAAI/bge-large-zh-v1.5
  EMBEDDING_BASE_URL: http://192.168.8.230:50009
  EMBEDDING_API_URL: http://192.168.8.230:50009
  
✅ 验证结果:
  ✓ EMBEDDING_TYPE: siliconflow
  ✓ SiliconFlow API Key: 已配置
  ✓ 所有配置验证通过
```

### 2. 重启服务查看日志

```powershell
# 重启Docker服务
docker-compose restart ai-service

# 查看日志
docker-compose logs -f ai-service
```

**预期日志：**
```json
{
  "embedding_type": "siliconflow",  // ✓ 正确
  "embedding_model": "BAAI/bge-large-zh-v1.5",
  "embedding_base_url": "http://192.168.8.230:50009",
  "embedding_api_url": "http://192.168.8.230:50009",
  "event": "Creating embedding function"
}
```

### 3. 测试embedding功能

上传文档或查询测试，验证embedding正常工作：

```powershell
# 测试聊天接口
curl -X POST http://localhost:8100/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "employee_id": "hutao",
    "query": "测试embedding配置"
  }'
```

## 相关文件

- ✅ 修复配置定义: [ai-service/app/core/config.py](../ai-service/app/core/config.py)
- ✅ 修复硬编码: [ai-service/app/core/chroma.py](../ai-service/app/core/chroma.py)
- ✅ 测试脚本: [ai-service/test_config_loading.py](../ai-service/test_config_loading.py)
- 📄 环境配置: [ai-service/.env](../ai-service/.env)

## 环境变量配置示例

### SiliconFlow模式

```env
EMBEDDING_TYPE=siliconflow
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
EMBEDDING_API_URL=https://api.siliconflow.cn/v1/embeddings
SILICONFLOW_API_KEY=sk-your-api-key
```

### OpenAI风格模式（本地服务）

```env
EMBEDDING_TYPE=openai_style
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
EMBEDDING_BASE_URL=http://192.168.8.230:50009
EMBEDDING_API_KEY=your-api-key  # 可选
```

## 注意事项

1. **优先级**：环境变量 > 默认值
2. **API Key**：SiliconFlow模式支持 `SILICONFLOW_API_KEY` 或 `EMBEDDING_API_KEY`
3. **URL格式**：
   - SiliconFlow: 使用 `EMBEDDING_API_URL`（完整endpoint URL）
   - OpenAI-style: 使用 `EMBEDDING_BASE_URL`（基础URL，会自动拼接 `/v1/embeddings`）
4. **重启生效**：修改.env后需重启服务才能生效

## 未来改进建议

1. **配置验证**：启动时验证embedding配置有效性
2. **自动降级**：SiliconFlow失败时自动切换到OpenAI-style
3. **健康检查**：定期测试embedding服务可用性
4. **配置热更新**：支持无需重启的配置更新

---

**修复日期**: 2025-12-19  
**修复版本**: v1.0.1
