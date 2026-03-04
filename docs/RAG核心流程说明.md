# RAG核心流程说明

本文档详细说明项目中两个核心流程：
1. 文档的索引和Embedding流程
2. 文档从提问到召回文档的流程

---

## 一、文档的索引和Embedding流程

### 1.1 整体流程图

```
文档上传
    ↓
创建文档记录（MongoDB）
    ↓
文本提取（根据文件类型）
    ↓
文本预处理（删除空格、目录等）
    ↓
分块处理（语义分块/传统分块/MinerU分块）
    ↓
生成分层摘要（chunk/section/document）
    ↓
生成Embedding（Ollama/SiliconFlow）
    ↓
多数据库存储
    ├─ ChromaDB（向量索引）
    ├─ ElasticSearch（关键词索引）
    └─ MongoDB（元数据）
```

### 1.2 API入口

| 接口 | 路径 | 用途 |
|------|------|------|
| 文件上传 | `/api/documents/upload` | 支持多文件上传 |
| Java平台接口 | `/api/documents/create_with_segment` | 从URL下载，支持自定义分段 |

### 1.3 文档处理核心类

**DocumentProcessor** - `ai-service/app/services/document_service.py`

主要方法：
- `process_document()` - 文档处理主流程
- `_extract_pdf()` - PDF文本提取
- `_extract_docx()` - DOCX文本提取
- `_extract_txt()` - TXT文本提取
- `_extract_markdown()` - Markdown文本提取
- `_extract_html()` - HTML文本提取

### 1.4 文本提取策略

#### PDF提取策略

```python
# 支持的提取方式
1. 基础提取：PyPDF提取文本层
2. 扫描PDF检测：文本<100字符自动使用OCR
3. MinerU增强解析：支持表格、公式、多列布局
4. 外部MinerU JSON文件支持
```

#### 支持的文件格式

```python
.supported_formats = {
    ".pdf": self._extract_pdf,
    ".docx": self._extract_docx,
    ".txt": self._extract_txt,
    ".md": self._extract_markdown,
    ".html": self._extract_html,
    ".mp4": self._extract_mp4,  # 待实现
}
```

### 1.5 分块策略

#### 1.5.1 语义分块（默认）

**算法流程：**

```
1. 句子分割（按中文标点符号和换行）
   ↓
2. 滑动窗口（默认3个句子）
   ↓
3. 相似度计算（相邻窗口余弦相似度）
   ↓
4. 边界检测（相似度低于阈值时分割）
   ↓
5. 大小约束（确保chunk在min_size和max_size之间）
```

**参数配置：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `semantic_chunk_similarity_threshold` | 0.75 | 相似度阈值 |
| `semantic_chunk_min_size` | 100 | 最小chunk大小（字符） |
| `semantic_chunk_max_size` | 500 | 最大chunk大小（字符） |
| `semantic_chunk_window_size` | 3 | 窗口大小（句子数） |

**自适应调整：**
- Markdown/PDF格式使用更大的chunk
- 根据文档内容自动调整分块参数

#### 1.5.2 MinerU结构化分块

**适用场景：** 启用 `mineru_structure_aware_chunking` 时

**特点：**
- 利用MinerU的标题层级结构
- 按章节/标题进行智能分块
- 支持图片引用和描述
- 保留文档结构信息

**分块策略：**
- `by_title` - 按标题层级分块
- `by_page` - 按页面分块
- `hybrid` - 混合策略（推荐）

#### 1.5.3 传统分块（降级方案）

**适用场景：** 语义分块失败时

**策略：**
- 动态计算chunk大小（根据文档长度）
- 支持自定义分隔符
- 支持换行、标识符等分割方式

### 1.6 分层摘要生成

#### 三层摘要结构

| 层级 | 内容 | 长度限制 | 用途 |
|------|------|----------|------|
| Chunk级摘要 | 每个chunk的核心内容 | ≤50字 | 快速浏览 |
| Section级摘要 | 相关chunk组的主要观点 | ≤100字 | 章节概览 |
| Document级摘要 | 全文整体内容和要点 | ≤200字 | 文档总览 |

**生成配置：**

```python
enable_hierarchical_summary: true  # 启用分层摘要
summary_chunk_level: true          # 生成chunk摘要
summary_section_level: true        # 生成section摘要
summary_document_level: true       # 生成document摘要
```

**摘要语言：** 所有文档摘要均使用中文生成

### 1.7 Embedding生成流程

#### 1.7.1 支持的Embedding服务

| 服务类型 | 配置参数 | 用途 |
|----------|----------|------|
| Ollama Embeddings | `OLLAMA_BASE_URL`, `EMBEDDING_OLLAMA_MODEL` | 本地向量生成（默认） |
| SiliconFlow Embeddings | `SILICONFLOW_API_KEY`, `EMBEDDING_TYPE=siliconflow` | 云端向量生成 |
| OpenAI-style Embeddings | `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL` | 通用API向量生成 |

#### 1.7.2 OllamaEmbeddings实现

**关键特性：**

```python
# 智能截断策略
max_tokens: 1024          # 最大token数
max_chars: 384           # 实际字符限制 (1024 / 2.5)

# 多级降级策略
512字符 → 384字符 → 256字符

# 缓存机制
LRU缓存减少重复API调用

# 批处理
8个文本一批，质量优化

# 错误处理
自动重试和降级策略
```

**API调用：**

```bash
POST {base_url}/api/embed
{
    "model": "bge-large-zh-v1.5:2k",
    "input": "文本内容",
    "keep_alive": 0
}
```

#### 1.7.3 Embedding数据流

```
传入文本
    ↓
检查缓存
    ↓
超长文本截断（多级降级）
    ↓
批量处理（8个一批）
    ↓
生成向量
    ↓
存入缓存
    ↓
返回embedding
```

**传入Embedding的文本格式：**

| 分块类型 | 文本格式 |
|----------|----------|
| 普通分块 | `chunk.content` |
| MinerU分块 | `"{title_str}\n\n{chunk.content}"`（标题作为前缀） |

### 1.8 多数据库存储

#### ChromaDB - 向量数据库

| 字段 | 说明 |
|------|------|
| collection | `doc` |
| documents | chunk文本内容 |
| metadatas | doc_id, kb_id, chunk_index, summary等 |
| ids | chunk_id |
| 用途 | 语义相似度搜索 |

#### ElasticSearch - 关键词搜索

| 字段 | 说明 |
|------|------|
| index | `digital_employee_doc`（可配置前缀） |
| fields | content, summary（权重3倍） |
| analyzer | BM25分词 |
| 用途 | 精确关键词匹配 |

#### MongoDB - 元数据存储

| Collection | 用途 |
|------------|------|
| `documents` | 文档元数据（文件名、大小、状态等） |
| `document_chunks` | chunk详细信息（含embedding ID） |

### 1.9 MinerU PDF解析增强

**特性：**
- **自动分块：** 大PDF按8页分块处理
- **MD5缓存：** 避免重复处理
- **任务跟踪：** 处理状态实时更新
- **块合并：** 自动合并处理后的分块

**配置：**

```python
mineru_pages_per_chunk: 8        # 每块页数
mineru_api_url: "..."            # MinerU服务地址
mineru_structure_aware_chunking: true  # 结构化分块
```

### 1.10 关键配置参数

#### 分块配置

```python
# 基础配置
chunk_size: 256              # 默认字符数
chunk_overlap: 50           # 重叠字符数

# 语义分块
semantic_chunk_similarity_threshold: 0.75
semantic_chunk_min_size: 100
semantic_chunk_max_size: 500
semantic_chunk_window_size: 3

# MinerU配置
mineru_structure_aware_chunking: true
mineru_chunking_strategy: "hybrid"  # by_title, by_page, hybrid
```

#### Embedding配置

```python
# Ollama配置
embedding_ollama_model: "bge-large-zh-v1.5:2k"
ollama_base_url: "http://localhost:11434"

# 字符限制
max_tokens: 1024
max_chars: 384  # 实际限制
```

---

## 二、文档从提问到召回文档的流程

### 2.1 整体工作流架构

系统基于**LangGraph的17节点StateGraph工作流**，核心RAG流程路径如下：

```
用户提问
    ↓
classify_query_type（查询分类）
    ↓
evaluate_complexity（复杂度评估）
    ↓
rewrite_query（查询重写）
    ↓
recognize_intent（意图识别）
    ↓
knowledge_retrieval（知识检索 ★）
    ↓
grade_documents（文档评分 ★）
    ↓
[compress_context]（上下文压缩）
    ↓
[web_search]（网络搜索，低相关时）
    ↓
generate_answer（答案生成）
```

### 2.2 LangGraph工作流关键节点

#### knowledge_retrieval 节点

**位置：** `ai-service/app/services/conversation/conversation_nodes.py:599-706`

**功能：** 执行混合搜索（向量+关键词）

**输入：**
- `user_query` - 原始用户查询
- `rewritten_query` - 重写后的查询
- `employee_config` - 员工配置（含kb_ids）

**输出：**
- `retrieved_docs` - 召回文档列表
- `kb_used` - 使用的知识库ID

**处理逻辑：**

```python
# 获取知识库ID
kb_ids = state["employee_config"].get("kb_ids", [])
search_query = state.get("rewritten_query", state["user_query"])

# 特殊处理：数学教材知识库
if MATH_KB_ID in kb_ids:
    math_results = await math_textbook_retrieval.search(
        query=search_query,
        kb_ids=[MATH_KB_ID],
        top_k=5,
        use_hybrid=True,
        enable_rerank=True
    )
else:
    # 标准RAG检索
    all_results = await rag_retrieval.search(
        query=search_query,
        kb_ids=kb_ids if kb_ids else None,
        top_k=5,
        use_hybrid=True,
        enable_rerank=True
    )
```

#### grade_documents 节点

**位置：** `ai-service/app/services/conversation/conversation_nodes.py:708-807`

**功能：** 计算相关性评分，触发直接匹配检测

**处理逻辑：**

```python
# 相关性评分计算
if not docs:
    state["relevance_score"] = 0.0
else:
    top_doc = docs[0]
    rerank_score = top_doc.get("rerank_score")
    state["relevance_score"] = max(0.0, min(1.0, float(
        rerank_score or top_doc.get("score", 0.0)
    )))

# 直接匹配检测（仅数学教材）
is_math_kb = MATH_KB_ID in kb_used
content_type = top_doc.get("content_type")

if is_math_kb and content_type in ['qa', 'teaching_script'] and relevance_score > 0.9:
    state["final_answer"] = top_doc.get("context_text", "")
    return state  # 跳过LLM生成
```

#### compress_context 节点

**位置：** `ai-service/app/services/conversation/conversation_nodes.py:788-869`

**功能：** 上下文压缩，减少token使用

**触发条件：**
- `context_compression_enabled = true`
- 总上下文长度 > 1500字符
- 压缩到500字以内

### 2.3 混合检索实现

#### RAGRetrieval 类

**位置：** `ai-service/app/services/rag_service.py`

#### 混合搜索流程

```
查询输入
    ↓
[向量搜索] ChromaDB
    ↓
[关键词搜索] ElasticSearch
    ↓
[RRF融合] 倒数排名融合
    ↓
[重排序] BGE Reranker（可选）
    ↓
返回top_k结果
```

#### 向量搜索

```python
async def _vector_search(self, query, kb_ids, top_k):
    # 构建过滤器
    where_filter = None
    if kb_ids:
        where_filter = {"kb_id": {"$in": kb_ids}}

    # 查询ChromaDB
    results = await chroma_db.query_documents(
        collection_name="doc",
        query_texts=[query],
        n_results=top_k,
        where=where_filter
    )

    # 转换距离为相似度 (0-1)
    similarity = max(0.0, 1.0 - distance)
```

#### 关键词搜索

```python
async def _keyword_search(self, query, kb_ids, top_k):
    # 构建ES查询
    es_query = {
        "query": {
            "bool": {
                "must": [{
                    "multi_match": {
                        "query": query,
                        "fields": ["summary^3", "content"],  # 摘要权重3倍
                        "type": "best_fields"
                    }
                }]
            }
        }
    }

    # 归一化分数到0-1范围
    normalized_score = min(1.0, score / 10.0)
```

### 2.4 RRF（倒数排名融合）

#### RRF公式

```
score(d) = Σ 1 / (k + rank_i(d))

其中：
- k = 60（默认常数）
- rank_i(d) = 文档d在第i个检索结果中的排名
```

#### RRF融合实现

```python
def _rrf_fusion(self, vector_results, keyword_results, k=60):
    doc_scores = {}

    # 添加向量搜索分数
    for rank, result in enumerate(vector_results, start=1):
        chunk_id = f"{result['doc_id']}_{result['chunk_index']}"
        if chunk_id not in doc_scores:
            doc_scores[chunk_id] = self._create_doc_fusion_entry(result)
        doc_scores[chunk_id]["rrf_score"] += 1.0 / (k + rank)

    # 添加关键词搜索分数
    for rank, result in enumerate(keyword_results, start=1):
        chunk_id = f"{result['doc_id']}_{result['chunk_index']}"
        if chunk_id not in doc_scores:
            doc_scores[chunk_id] = self._create_doc_fusion_entry(result)
        doc_scores[chunk_id]["rrf_score"] += 1.0 / (k + rank)

    # 按RRF分数排序
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_docs
```

#### 文档融合条目结构

```python
{
    "content": "...",
    "doc_id": "...",
    "chunk_id": "...",
    "kb_id": "...",
    "rrf_score": 0.0,         # RRF融合后的最终分数
    "vector_score": 0.0,      # 向量搜索分数
    "keyword_score": 0.0,     # 关键词搜索分数
    "vector_rank": None,      # 向量搜索排名
    "keyword_rank": None,     # 关键词搜索排名
    # 其他元数据...
}
```

### 2.5 重排序（Reranker）

#### Reranker类型

| 类型 | 实现类 | 特点 |
|------|--------|------|
| BGE API Reranker | `BGEAPIReranker` | 无文本长度限制，默认 |
| BGE Local Reranker | `BGELocalReranker` | 本地运行，需FlagEmbedding |
| Hybrid Reranker | `HybridReranker` | 结合语义、关键词和位置偏置 |

#### 重排序调用流程

```python
async def _rerank(self, query, documents, top_k):
    # 获取重排序器
    reranker = get_reranker(reranker_type=settings.reranker_type)

    # 提取文档内容
    docs_to_rerank = [doc.get("content", "") for doc in documents]

    # 执行重排序
    rerank_results = await reranker.rerank(query, docs_to_rerank, top_k=len(documents))

    # 重新排序并添加rerank_score
    reranked_docs = []
    for idx, score in rerank_results:
        doc = documents[idx].copy()
        doc["rerank_score"] = float(score)
        doc["score"] = float(score)  # 统一使用rerank_score
        reranked_docs.append(doc)

    return reranked_docs[:top_k]
```

#### BGE API Reranker配置

```bash
BGE_RERANKER_API_URL=http://192.168.8.233:8091
BGE_RERANKER_API_KEY=sk-aaabbbcccdddeeefffggghhhiiijjjkkk
BGE_RERANKER_MODEL=bge-reranker-v2-m3
RERANKER_TYPE=bge_api
```

### 2.6 相关性评分和阈值过滤

#### 相关性评分计算

```python
# grade_documents节点
if not docs:
    state["relevance_score"] = 0.0
else:
    top_doc = docs[0]
    rerank_score = top_doc.get("rerank_score")
    state["relevance_score"] = max(0.0, min(1.0, float(
        rerank_score or top_doc.get("score", 0.0)
    )))
```

#### 阈值过滤和路由

```python
# compress_context节点的条件路由
if state.get("relevance_score", 0) < settings.relevance_threshold:
    # 低相关性，触发网络搜索
    next_node = "web_search"
else:
    # 高相关性，直接生成答案
    next_node = "generate_answer"
```

**默认阈值：** `settings.relevance_threshold = 0.6`（60%）

### 2.7 特殊处理：数学教材直接匹配

**触发条件：**
1. 查询匹配数学教材知识库
2. `content_type` 为 `qa` 或 `teaching_script`
3. 相关性评分 > 0.9

**处理方式：**
- 跳过LLM生成
- 直接使用预生成的 `context_text`
- 支持 `teaching_script_tts` 语音播报文本

### 2.8 特殊处理：FAQ检索

**FAQ混合搜索流程：**

```python
async def faq_hybrid_search(self, query, employee_id, faq_sim_threshold=0.0, faq_top_k=3):
    # 向量搜索 + 关键词搜索
    vector_results = await self._faq_vector_search(query, employee_id, faq_top_k * 2)
    keyword_results = await self._faq_keyword_search(query, employee_id, faq_top_k * 2)

    # RRF融合
    fused_results = self._faq_rrf_fusion(vector_results, keyword_results, k=60)

    # 多重条件阈值过滤
    filtered_results = [
        result for result in fused_results
        if (result.get("rrf_score", 0.0) >= faq_sim_threshold or
            result.get("vector_score", 0.0) >= faq_sim_threshold or
            (result.get("keyword_score", 0.0) >= faq_sim_threshold and
             result.get("vector_score", 0.0) >= faq_sim_threshold * 0.7))
    ]

    return filtered_results[:faq_top_k]
```

### 2.9 关键配置参数

#### RAG配置

```python
# 检索参数
top_k: 5                    # 返回文档数
rerank_enabled: true        # 启用重排序
relevance_threshold: 0.6    # 相关度阈值
use_hybrid_search: true     # 启用混合搜索

# RRF配置
rrf_k: 60                  # RRF常数

# 重排序配置
reranker_type: "bge_api"   # bge_api, bge, hybrid
```

### 2.10 数据流向总结

```
用户查询
    ↓
查询重写（可选）
    ↓
混合搜索
    ├─ 向量搜索 (ChromaDB)
    └─ 关键词搜索 (ElasticSearch)
    ↓
RRF融合
    ↓
重排序（BGE Reranker）
    ↓
相关性评分
    ↓
阈值判断
    ├─ 低相关 → 网络搜索fallback
    └─ 高相关 → 上下文压缩 → 答案生成
```

### 2.11 关键设计特点

| 特点 | 说明 |
|------|------|
| 三重召回 | 向量语义 + 关键词匹配 + 重排序优化 |
| RRF融合 | 平衡语义和关键词搜索结果 |
| 动态路由 | 根据相关性决定是否使用网络搜索 |
| 特殊处理 | 数学教材直接匹配、FAQ快速通道 |
| 性能优化 | 上下文压缩、本地缓存、批处理 |

---

## 三、参考资料

### 相关代码文件

| 组件 | 路径 |
|------|------|
| 文档处理服务 | `ai-service/app/services/document_service.py` |
| RAG检索服务 | `ai-service/app/services/rag_service.py` |
| 重排序服务 | `ai-service/app/services/reranker_service.py` |
| 语义分块 | `ai-service/app/services/semantic_chunking.py` |
| Embedding工具 | `ai-service/app/utils/embeddings.py` |
| 会话工作流 | `ai-service/app/services/conversation_service.py` |
| ChromaDB连接 | `ai-service/app/core/chroma.py` |
| ElasticSearch连接 | `ai-service/app/core/elasticsearch.py` |

### 配置文件

| 文件 | 用途 |
|------|------|
| `.env` | Docker环境配置 |
| `.env-local` | 本地开发配置 |
| `app/core/config.py` | 配置参数定义 |

### 相关文档

| 文档 | 路径 |
|------|------|
| MinerU实现总结 | `docs/MinerU实现总结.md` |
| MinerU使用指南 | `docs/MinerU客户端使用指南.md` |
| 性能优化总结 | `docs/性能优化总结.md` |
| ConversationWorkflow流程图 | `docs/ConversationWorkflow流程图与架构图.md` |
