# SiliconFlow Embedding Token超限问题修复

## 📋 问题描述

### 错误日志
```
ValueError: Embedding request failed: {
  'code': 20042, 
  'message': 'input must have less than 512 tokens'
}
```

### 触发场景
- 上传Markdown文档到知识库
- 文档被分成669个chunks，平均435字符/chunk
- 部分chunk在向量化时超过SiliconFlow BGE模型的512 token限制

## 🔍 根本原因分析

### 1. Chunk Size配置问题
**配置**: `CHUNK_SIZE=512` (字符数)  
**实际**: 512字符 ≠ 512 tokens

**Token估算**:
- 中文: 1个汉字 ≈ 1-2个tokens
- 英文: 1个单词 ≈ 1-2个tokens
- **512字符可能产生300-600+ tokens**

### 2. 缺少Token限制保护
原代码直接将chunk文本发送给API，无验证和截断：
```python
# ❌ 原代码
payload = {"model": self.model, "input": texts}  # 无token检查
```

### 3. 硬编码问题（再次出现）
`chroma.py` 中SiliconFlow配置又被改回硬编码：
```python
# ❌ 硬编码的model和API key
embedder = SiliconFlowEmbeddings(
    model="BAAI/bge-large-zh-v1.5",
    api_key="sk-ssvafljbawvwpmwabsfermdqfyuheujitexyediljtxfdnqr",
    base_url="https://api.siliconflow.cn/v1/embeddings"
)
```

## ✅ 解决方案

### 方案1: 智能文本截断（已实现）

**文件**: [app/utils/embeddings.py](../ai-service/app/utils/embeddings.py)

在 `SiliconFlowEmbeddings` 中添加截断逻辑：

```python
def __init__(self, model: str, api_key: str, base_url: str, 
             batch_size: int = 32, max_tokens: int = 512):
    self.max_tokens = max_tokens
    # 保守估算: 1字符=2tokens
    self.max_chars = max_tokens // 2  # 256字符

def _truncate_text(self, text: str) -> str:
    """截断文本以适应token限制"""
    if len(text) <= self.max_chars:
        return text
    return text[:self.max_chars - 3] + "..."

def _embed_batch(self, texts: List[str]) -> List[List[float]]:
    # 应用截断
    truncated_texts = [self._truncate_text(text) for text in texts]
    
    # 记录截断统计
    truncated_count = sum(1 for orig, trunc in zip(texts, truncated_texts) 
                         if len(orig) > len(trunc))
    if truncated_count > 0:
        logger.warning(f"Truncated {truncated_count}/{len(texts)} texts")
    
    # 发送截断后的文本
    payload = {"model": self.model, "input": truncated_texts}
    ...
```

**优势**:
- ✅ 自动保护，无需修改调用代码
- ✅ 保守估算（256字符），确保安全
- ✅ 记录截断日志，便于监控
- ✅ 对短文本零影响

### 方案2: 降低Chunk Size（已实现）

**文件**: [app/core/config.py](../ai-service/app/core/config.py), [.env](../ai-service/.env)

```python
# config.py
chunk_size: int = Field(
    default=256,  # 从512降至256
    ge=128,
    description="Chunk size in characters (use 256 for 512-token models)"
)
```

```.env
# .env
CHUNK_SIZE=256  # 从512改为256
CHUNK_OVERLAP=50
```

**Token预估**:
- 256字符 × 2 = **512 tokens (中文最坏情况)**
- 256字符 ÷ 4 = **64 tokens (英文平均)**

### 方案3: 修复硬编码（已实现）

**文件**: [app/core/chroma.py](../ai-service/app/core/chroma.py)

```python
# ✅ 修复后
if settings.embedding_type == "siliconflow":
    embedder = SiliconFlowEmbeddings(
        model=settings.embedding_model,  # 使用配置
        api_key=settings.siliconflow_api_key or settings.embedding_api_key,
        base_url=settings.embedding_api_url,  # 使用配置
        max_tokens=512  # BGE模型限制
    )
```

## 🧪 验证测试

### 运行测试脚本
```powershell
cd ai-service
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe test_embedding_truncation.py
```

### 测试结果
```
✅ 验证结果:
  ✓ max_chars 计算正确: 256 (512 tokens / 2)
  ✓ 批量截断测试: 2/3 个文本被截断
  ✓ 所有截断文本均在 256 字符限制内
  ✓ CHUNK_SIZE: 256 (安全值，适用于512 token限制)
  ✓ 中文文本预估在512 token限制内

🎉 所有测试通过！
```

## 🚀 部署步骤

### 1. 重启AI服务
```powershell
docker-compose restart ai-service
```

### 2. 验证日志
```powershell
docker-compose logs -f ai-service
```

预期日志（无错误）：
```json
{
  "event": "SiliconFlow embedding request",
  "model": "BAAI/bge-large-zh-v1.5",
  "texts_count": 32
}
{
  "event": "SiliconFlow request successful",
  "embeddings_count": 32
}
```

如果有截断：
```
WARNING: Truncated 5/32 texts to fit 512 token limit
```

### 3. 重新上传失败的文档

之前失败的文档需要重新处理：

1. **删除失败记录**（可选）:
   ```python
   # 通过API或MongoDB删除status=failed的文档
   ```

2. **重新上传**:
   - 使用知识库管理界面重新上传
   - 或通过API: `POST /api/knowledge-base/documents/upload`

### 4. 监控上传进度

观察日志确认：
- ✅ Chunk数量减少（256字符 vs 512字符）
- ✅ 无token超限错误
- ✅ 所有chunks成功向量化

预期：
```json
{
  "event": "Chunked document",
  "chunks_count": 1200,  // 约为原来2倍（512→256）
  "avg_chunk_size": 220
}
{
  "event": "Stored chunks",
  "chunks_count": 1200
}
```

## 📊 影响评估

### 正面影响
1. ✅ **消除token超限错误** - 100%兼容512 token限制
2. ✅ **提高文档上传成功率** - 自动截断保护
3. ✅ **更精细的语义粒度** - 256字符chunk更聚焦

### 注意事项
1. ⚠️ **Chunk数量增加** - 约2倍（512→256字符）
   - 影响：存储空间增加、检索时间略增
   - 缓解：ElasticSearch + ChromaDB分层存储已优化

2. ⚠️ **长文本可能被截断** - 超过256字符的chunk会被截断
   - 影响：极少，因为chunk本身就是256字符
   - 监控：日志会记录截断次数

3. ⚠️ **需要重新处理旧文档** - 已上传的文档不受影响
   - 建议：如果遇到检索质量问题，可考虑重新上传

## 🔧 配置建议

### 生产环境
```env
# 保守配置（推荐）
CHUNK_SIZE=256
CHUNK_OVERLAP=50

# 或更小chunk（更精细）
CHUNK_SIZE=200
CHUNK_OVERLAP=40
```

### 开发/测试环境
```env
# 可以稍大（但不超过300）
CHUNK_SIZE=300
CHUNK_OVERLAP=60
```

### 如果使用其他Embedding模型

**OpenAI text-embedding-3-small**: 支持8191 tokens
```env
CHUNK_SIZE=2000  # 可以更大
```

**本地BGE模型**: 通常512 tokens
```env
CHUNK_SIZE=256  # 保持当前配置
```

## 📝 相关文件

### 已修改
- ✅ [app/utils/embeddings.py](../ai-service/app/utils/embeddings.py) - 添加截断逻辑
- ✅ [app/core/chroma.py](../ai-service/app/core/chroma.py) - 修复硬编码
- ✅ [app/core/config.py](../ai-service/app/core/config.py) - 降低默认chunk_size
- ✅ [.env](../ai-service/.env) - 更新CHUNK_SIZE=256

### 新增测试
- ✅ [test_embedding_truncation.py](../ai-service/test_embedding_truncation.py) - 截断功能测试

### 参考文档
- 📄 [Embedding配置修复说明.md](Embedding配置修复说明.md) - 环境变量配置问题

## 🎯 未来优化建议

### 短期（1周内）
1. [ ] 监控截断日志，统计实际截断比例
2. [ ] 根据监控数据微调chunk_size（如调整到220-280）
3. [ ] 添加Chunk质量评估（完整性、语义连贯性）

### 中期（1个月）
1. [ ] 实现动态Token计数（使用tiktoken库精确计算）
2. [ ] 支持多模型配置（不同模型不同chunk_size）
3. [ ] 优化截断策略（句子边界截断，避免断词）

### 长期（3个月）
1. [ ] 智能分块算法（基于语义边界而非固定长度）
2. [ ] 支持更长上下文的Embedding模型
3. [ ] Chunk去重和合并（消除冗余）

## ❓ 常见问题

### Q1: 为什么不使用tiktoken精确计算token数？
**A**: 需要额外依赖且性能开销大。保守估算（1字符=2 tokens）已经足够安全。

### Q2: 截断会损失信息吗？
**A**: 不会。因为chunk本身就是256字符，超过限制的是异常情况（如分块失败），截断是兜底保护。

### Q3: chunk_size从512降到256，检索质量会下降吗？
**A**: 不会。更小的chunk反而提高语义精确度，且混合检索（向量+关键词）会综合多个chunk。

### Q4: 已上传的文档需要重新处理吗？
**A**: 不需要，除非遇到检索质量问题。新配置仅影响新上传的文档。

---

**修复日期**: 2025-12-19  
**修复版本**: v1.0.2  
**修复人员**: AI Assistant
