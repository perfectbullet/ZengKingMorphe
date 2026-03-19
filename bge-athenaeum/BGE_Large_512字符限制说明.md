# BGE-Large 512字符限制说明

## 重要限制

**BGE-Large 模型最大支持 512 字符**（约 512 tokens）

## 实测数据

| 字符数 | 状态 | 耗时 | 备注 |
|--------|------|------|------|
| 100 | ✅ 成功 | 0.16秒 | 首次加载 |
| 200 | ✅ 成功 | 0.03秒 | 正常响应 |
| 300 | ✅ 成功 | 0.03秒 | 正常响应 |
| 400 | ✅ 成功 | 0.03秒 | 正常响应 |
| 500 | ✅ 成功 | 0.03秒 | 接近上限 |
| 512 | ✅ 成功 | 0.03秒 | **理论最大值** |
| 600+ | ❌ 失败 | - | 超过限制 |

## 错误信息

当文本超过 512 字符时，会返回以下错误：

```
Error code: 400 - {'error': {'message': "This model's maximum context length is 512 tokens. However, your messages resulted in 615 tokens. Please reduce the length of your messages.", ...}}
```

## Token/字符比

- **中文文本**: 约 1.02 tokens/字符
- **推荐**: 实际使用按 **1 字符 = 1 token** 计算
- **安全限制**: **512 字符**

## 解决方案

### 方案 1: 使用 BGE-M3（推荐）

```python
from examples.test_embedding import BGEEmbeddingClient

# BGE-M3 支持 8192 tokens (约 4000 中文字符)
client = BGEEmbeddingClient("http://192.168.8.233:8092")
embedding = client.embed(long_text)  # 支持长文本
```

### 方案 2: 文本分块

```python
def split_text_for_bge_large(text: str, max_length: int = 512) -> List[str]:
    """将长文本分割为不超过 512 字符的块"""
    chunks = []
    for i in range(0, len(text), max_length):
        chunks.append(text[i:i + max_length])
    return chunks

# 使用示例
from examples.test_bge_large import BGELargeEmbeddingClient

client = BGELargeEmbeddingClient("http://192.168.8.233:8093")
chunks = split_text_for_bge_large(long_text)
embeddings = client.embed(chunks)  # 批量处理
```

### 方案 3: 预检查长度

```python
def check_text_length(text: str, max_length: int = 512) -> bool:
    """检查文本是否超过限制"""
    if len(text) > max_length:
        raise ValueError(f"文本长度 {len(text)} 超过 BGE-Large 限制 {max_length}")
    return True

# 使用示例
try:
    check_text_length(text)
    embedding = client.embed(text)
except ValueError as e:
    print(f"错误: {e}")
    # 使用其他方案处理
```

## 使用建议

### ✅ 适合使用 BGE-Large 的场景

- **短文本嵌入** (< 512 字符)
  - 文章标题
  - 用户问题
  - 简短描述
  - 关键词

- **中文专用场景**
  - 纯中文文档
  - 中文问答系统
  - 中文语义搜索

- **高性能需求**
  - 实时检索
  - 低延迟要求
  - 批量处理加速 (6x)

### ❌ 不适合使用 BGE-Large 的场景

- **长文档处理** (> 512 字符)
  - 完整文章
  - 长篇报告
  - 完整章节
  → **改用 BGE-M3**

- **多语言场景**
  - 中英文混合
  - 跨语言检索
  - 多语言文档
  → **改用 BGE-M3**

- **复杂文档结构**
  - 包含表格、公式
  - 多段落长文本
  → **改用 BGE-M3 或分块处理**

## 配置对比

| 模型 | 最大字符 | 最大 tokens | 用途 |
|------|---------|------------|------|
| **BGE-Large** | **512** | **512** | 中文短文本 |
| BGE-M3 | ~4000 | 8192 | 多语言长文本 |
| Reranker-M3 | ~4000 | 8192 | 文档重排序 |

## 代码示例

### 基本使用

```python
from examples.test_bge_large import BGELargeEmbeddingClient

client = BGELargeEmbeddingClient("http://192.168.8.233:8093")

# 短文本嵌入 (推荐)
text = "人工智能是计算机科学的重要分支"
embedding = client.embed(text)  # ✅ 长度 < 512
```

### 错误示例

```python
# 长文本会失败 ❌
long_text = "这是一段很长的文本..." * 100  # > 512 字符
embedding = client.embed(long_text)  # ❌ Error 400
```

### 正确处理长文本

```python
# 方案 1: 使用 BGE-M3
from examples.test_embedding import BGEEmbeddingClient
m3_client = BGEEmbeddingClient("http://192.168.8.233:8092")
embedding = m3_client.embed(long_text)  # ✅ 支持 8192 tokens

# 方案 2: 分块处理
chunks = [long_text[i:i+512] for i in range(0, len(long_text), 512)]
embeddings = client.embed(chunks)  # ✅ 批量处理
```

## 总结

**BGE-Large 的核心价值在于**:
- ✅ 中文短文本语义理解准确
- ✅ 响应速度快 (30ms)
- ✅ 批量性能优秀 (6x 加速)
- ✅ 内存占用低 (0.2 GPU)

**核心限制**:
- ⚠️ **最大 512 字符**
- ⚠️ 中文专用
- ⚠️ 不支持长文本

**选择建议**:
- **短文本 + 中文** → BGE-Large
- **长文本/多语言** → BGE-M3
