# RAG核心流程与代码说明

本文档详细说明项目中两个核心流程：文档索引/Embedding流程和文档召回流程，并附带关键代码和类的详细说明。

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

### 1.3 核心类与代码

#### 1.3.1 DocumentProcessor

**文件位置：** `ai-service/app/services/document_service.py:43-1292`

**类结构：**
```python
class DocumentProcessor:
    """文档处理服务，支持多种文件格式和分块策略。"""

    def __init__(self):
        self.supported_formats = {
            ".pdf": self._extract_pdf,
            ".docx": self._extract_docx,
            ".txt": self._extract_txt,
            ".md": self._extract_markdown,
            ".html": self._extract_html,
            ".mp4": self._extract_mp4,  # 待实现
        }
```

**关键方法：**

| 方法 | 行号 | 功能 |
|------|------|------|
| `process_document()` | 56-258 | 文档处理主流程 |
| `_extract_text()` | 260-282 | 文本提取入口 |
| `_extract_pdf()` | 284-298 | PDF提取（支持MinerU） |
| `_detect_scanned_pdf()` | 316-364 | 扫描PDF检测 |
| `_extract_pdf_with_mineru()` | 379-439 | MinerU增强提取 |
| `_preprocess_text()` | 591-626 | 文本预处理 |
| `_chunk_text()` | 671-824 | 文档分块入口 |
| `_semantic_chunk_text()` | 950-1027 | 语义分块 |
| `_process_chunks()` | 1092-1216 | Chunk处理和存储 |

**代码示例 - process_document主流程：**
```python
async def process_document(
    self,
    file_path: str,
    filename: str,
    kb_id: str,
    enhance: int,
    category: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
    chunk_config: Optional[Dict[str, Any]] = None,
    doc_id: Optional[str] = None,
    resource_id: Optional[int] = None,
) -> str:
    """处理文档：提取文本、分块、向量化、存储。"""

    try:
        # 1. 创建文档记录（MongoDB）
        doc_model = DocumentModel(
            doc_id=doc_id,
            filename=filename,
            kb_id=kb_id,
            enhance=enhance,
            category=category,
            status="processing",
            segment_config=chunk_config,
        )
        await db.documents.insert_one(doc_model.model_dump())

        # 2. 文本提取（支持MinerU）
        text_content = await self._extract_text(
            file_path, file_ext, use_mineru=use_mineru
        )

        # 3. 文本预处理
        if chunk_config:
            text_content = self._preprocess_text(text_content, chunk_config)

        # 4. 文档分块（返回 chunks 和 hierarchical_summary_data）
        chunk_result = await self._chunk_text(
            text_content, doc_id, kb_id, file_ext=file_ext,
            chunk_config=chunk_config, mineru_structured_data=mineru_structured_data
        )

        if isinstance(chunk_result, tuple):
            chunks, hierarchical_summary_data = chunk_result
        else:
            chunks = chunk_result
            hierarchical_summary_data = None

        # 5. 处理chunks（生成embedding，存储到数据库）
        await self._process_chunks(chunks, doc_id, kb_id, task_id=task_id)

        # 6. 更新文档状态
        await db.documents.update_one(
            {"doc_id": doc_id},
            {"$set": {"status": "completed", "chunks_count": len(chunks)}}
        )

        return doc_id

    except Exception as e:
        # 处理失败时更新状态
        await db.documents.update_one(
            {"doc_id": doc_id},
            {"$set": {"status": "failed", "error_message": str(e)}}
        )
        raise
```

#### 1.3.2 SemanticChunker

**文件位置：** `ai-service/app/services/semantic_chunking.py:61-356`

**类结构：**
```python
class SemanticChunker:
    """
    使用embedding进行语义分块的类。
    算法：
    1. 按句子分割文本
    2. 计算滑动窗口的embeddings
    3. 计算相邻窗口的相似度
    4. 在相似度低于阈值时识别边界
    5. 将句子合并为chunks，考虑min/max大小约束
    """

    # Markdown/PDF需要更大的chunk
    MARKDOWN_FORMATS = {'.md', '.pdf'}

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
        window_size: Optional[int] = None,
        file_ext: Optional[str] = None
    ):
        self.similarity_threshold = similarity_threshold or settings.semantic_chunk_similarity_threshold

        # 自动根据文件类型调整chunk大小
        if file_ext and file_ext.lower() in self.MARKDOWN_FORMATS:
            self.min_chunk_size = min_chunk_size or 500   # Markdown/PDF使用更大chunk
            self.max_chunk_size = max_chunk_size or 2000
        else:
            self.min_chunk_size = min_chunk_size or 100   # 普通文本默认
            self.max_chunk_size = max_chunk_size or 500

        self.window_size = window_size or settings.semantic_chunk_window_size
```

**关键方法：**

```python
def _split_into_sentences(self, text: str) -> List[str]:
    """按中文/英文标点符号分割句子"""
    parts = self.sentence_pattern.split(text)
    # ... 处理逻辑

async def _compute_similarities(self, windows: List[str]) -> List[float]:
    """计算相邻窗口的余弦相似度"""
    embedder = get_embedding()
    similarities = []
    for i in range(len(windows) - 1):
        emb1 = embedder.embed_query(windows[i])
        emb2 = embedder.embed_query(windows[i + 1])
        similarity = self._cosine_similarity(emb1, emb2)
        similarities.append(similarity)
    return similarities

async def _find_chunk_boundaries(
    self,
    sentences: List[str],
    similarities: List[float]
) -> List[ChunkBoundary]:
    """识别最优分块边界"""
    boundaries = []
    current_size = 0

    for i, similarity in enumerate(similarities):
        current_size += len(sentences[i])

        # 检查是否是好的边界点
        is_low_similarity = similarity < self.similarity_threshold
        exceeds_max = current_size >= self.max_chunk_size
        is_end = i == len(similarities) - 1

        if is_low_similarity or exceeds_max or is_end:
            if current_size >= self.min_chunk_size or is_end:
                boundaries.append(ChunkBoundary(
                    index=i,
                    position=sum(len(s) for s in sentences[:i + 1]),
                    text=" ".join(sentences[:i + 1]),
                    similarity_score=similarity,
                    cumulative_size=current_size
                ))
                current_size = 0

    return boundaries
```

#### 1.3.3 HierarchicalSummarizer

**文件位置：** `ai-service/app/services/semantic_chunking.py:359-658`

**类结构：**
```python
class HierarchicalSummarizer:
    """
    创建多层摘要：chunk、section和document级别。
    分层结构：
    - Chunk level: 每个语义分块的摘要
    - Section level: 相关分块组的摘要（按主题分组）
    - Document level: 整个文档的摘要
    """

    def __init__(self):
        self.chunk_enabled = settings.summary_chunk_level
        self.section_enabled = settings.summary_section_level
        self.document_enabled = settings.summary_document_level
```

**摘要生成结构：**
```python
async def summarize(
    self,
    chunks: List[SemanticChunk],
    doc_id: str
) -> HierarchicalSummary:
    """生成文档chunks的分层摘要。"""

    summary = HierarchicalSummary(
        chunk_summaries={},
        section_summaries=[]
    )

    # Chunk级摘要
    if self.chunk_enabled:
        await self._summarize_chunks(chunks, summary, doc_id)

    # Section级摘要
    if self.section_enabled and len(chunks) >= self.section_threshold:
        await self._summarize_sections(chunks, summary, doc_id)

    # Document级摘要
    if self.document_enabled and len(chunks) >= self.document_threshold:
        await self._summarize_document(chunks, summary, doc_id)

    return summary
```

**摘要提示词模板：**
```python
def _create_chunk_summary_prompt(self, content: str) -> str:
    """创建chunk级摘要提示词"""
    return f"""请用简洁的中文总结以下文本的核心内容，不超过50字：

{content}

总结："""

def _create_section_summary_prompt(self, content: str) -> str:
    """创建section级摘要提示词"""
    return f"""请用简洁的中文总结以下章节内容的主要观点，不超过100字：

{content}

总结："""

def _create_document_summary_prompt(self, content: str) -> str:
    """创建document级摘要提示词"""
    return f"""请用简洁的中文总结以下文档的整体内容和主要要点，不超过200字：

{content}

总结："""
```

#### 1.3.4 OllamaEmbeddings

**文件位置：** `ai-service/app/utils/embeddings.py:70-420`

**类结构：**
```python
class OllamaEmbeddings(Embeddings):
    """
    Ollama embedding实现，使用/api/embeddings端点。
    特性：
    - 智能截断（384 → 320 → 256字符多级降级）
    - LRU缓存
    - 批处理（8个一批）
    - 自动重试
    """

    TRUNCATE_LIMITS = [384, 320, 256]  # 智能降级截断限制

    def __init__(
        self,
        model: str,
        base_url: str,
        batch_size: int = 32,
        max_tokens: int = 1024,
        max_chars: int = 384,
        enable_fallback: bool = True,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.truncator = TextTruncator(max_tokens=max_tokens)
        self.request_delay = 0.5  # 500ms延迟
        self.max_retries = 3
        self.retry_delay = 1.0
```

**智能截断实现：**
```python
def _truncate_text(self, text: str, level: int = 0) -> tuple[str, int]:
    """智能截断文本，支持降级策略"""
    for level, limit in enumerate(self.TRUNCATE_LIMITS):
        if len(text) <= limit:
            return text, -1, len(text)

    # 需要截断
    for level, limit in enumerate(self.TRUNCATE_LIMITS):
        truncated = text[:limit - 3] + "..."
        if len(truncated) <= limit:
            return truncated, level, len(text)
```

**批量Embedding：**
```python
def embed_documents(self, texts: List[str]) -> List[List[float]]:
    """使用批量请求embed多个文档。"""
    BATCH_SIZE = 8  # 质量优化，>=16会导致质量下降

    embeddings = []
    cache_hits = 0
    request_count = 0

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        # 检查缓存
        uncached_count = sum(1 for t in batch if embedding_cache.get(t, self.model) is None)

        if uncached_count == 0:
            # 全部命中缓存
            for text in batch:
                cached = embedding_cache.get(text, self.model)
                embeddings.append(np.array(cached, dtype=float))
        else:
            # 批量请求
            if request_count > 0:
                time.sleep(self.request_delay)

            batch_embeddings = self._embed_batch(batch)
            embeddings.extend(batch_embeddings)
            request_count += 1

    return embeddings
```

### 1.4 数据模型

#### SemanticChunk（语义分块结果）
**文件位置：** `ai-service/app/services/semantic_chunking.py:35-42`
```python
@dataclass
class SemanticChunk:
    """语义连贯的文本块"""
    content: str                      # 块内容
    start_pos: int                    # 起始位置
    end_pos: int                      # 结束位置
    chunk_index: int                  # 块索引
    summary: Optional[str] = None      # 摘要
    embedding: Optional[List[float]] = None  # 向量
```

#### HierarchicalSummary（分层摘要）
**文件位置：** `ai-service/app/services/semantic_chunking.py:54-58`
```python
@dataclass
class HierarchicalSummary:
    """分层摘要"""
    chunk_summaries: Dict[int, str]         # chunk_index -> summary
    section_summaries: List[SectionSummary]  # section级摘要列表
    document_summary: Optional[str] = None   # 文档级摘要
```

#### DocumentChunkModel（文档块模型）
**文件位置：** `ai-service/app/models/database.py:112-133`
```python
class DocumentChunkModel(BaseModel):
    # 基础字段
    chunk_id: str
    doc_id: str
    kb_id: str
    content: str
    chunk_index: int
    summary: Optional[Dict[str, str]] = None
    vector_id: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: datetime
    updated_at: Optional[datetime] = None

    # MinerU结构化字段
    page_idx: Optional[int] = None
    page_indices: List[int] = []
    block_types: List[str] = []
    image_references: List[str] = []
    image_captions: List[str] = []
    title_path: List[str] = []
    structure_level: Optional[int] = None
```

### 1.5 分块策略详解

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
- Markdown/PDF格式使用更大的chunk（min=500, max=2000）
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

### 1.7 多数据库存储

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

### 1.8 MinerU PDF解析增强

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

### 2.2 核心类与代码

#### 2.2.1 RAGRetrieval

**文件位置：** `ai-service/app/services/rag_service.py:13-696`

**类结构：**
```python
class RAGRetrieval:
    """RAG检索服务，支持混合搜索（向量+关键词）和重排序。"""

    def __init__(self):
        self._reranker = None

    async def search(
        self,
        query: str,
        kb_ids: Optional[List[str]] = None,
        top_k: int = 5,
        use_hybrid: bool = True,
        enable_rerank: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """搜索相关文档，使用RAG。"""
        # 自动选择是否启用重排序
        if enable_rerank is None:
            enable_rerank = settings.rerank_enabled

        if use_hybrid:
            return await self._hybrid_search(query, kb_ids, top_k, enable_rerank)
        else:
            return await self._vector_search(query, kb_ids, top_k, enable_rerank)
```

**混合搜索流程：**
```python
async def _hybrid_search(
    self,
    query: str,
    kb_ids: Optional[List[str]],
    top_k: int,
    enable_rerank: bool = False
) -> List[Dict[str, Any]]:
    """混合搜索：向量 + BM25 + RRF + 重排序"""

    # 获取更多结果用于重排序
    fetch_count = top_k * 2 if enable_rerank else top_k

    # 1. 向量搜索（ChromaDB）
    vector_results = await self._vector_search(query, kb_ids, fetch_count, enable_rerank=False)

    # 2. 关键词搜索（ElasticSearch）
    keyword_results = await self._keyword_search(query, kb_ids, fetch_count)

    # 3. RRF融合
    fused_results = self._rrf_fusion(vector_results, keyword_results, k=60)

    # 4. 重排序
    if enable_rerank and fused_results:
        fused_results = await self._rerank(query, fused_results, top_k)

    return fused_results[:top_k]
```

**向量搜索实现：** (`rag_service.py:125-204`)
```python
async def _vector_search(
    self,
    query: str,
    kb_ids: Optional[List[str]],
    top_k: int,
    enable_rerank: bool = False
) -> List[Dict[str, Any]]:
    """基于ChromaDB的向量语义搜索"""

    # 构建过滤器
    where_filter = None
    if kb_ids:
        where_filter = {"kb_id": {"$in": kb_ids}}

    # 查询Chroma
    results = await chroma_db.query_documents(
        collection_name="doc",
        query_texts=[query],
        n_results=top_k,
        where=where_filter
    )

    # 格式化结果并转换距离为相似度
    documents = []
    if results and results.get("documents"):
        for i, doc_text in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results.get("distances") else 1.0
            similarity = max(0.0, 1.0 - distance)  # 距离转相似度
            metadata = results["metadatas"][0][i] if results.get("metadatas") else {}

            documents.append({
                "content": doc_text,
                "score": similarity,
                "doc_id": metadata.get("doc_id"),
                "kb_id": metadata.get("kb_id"),
                "chunk_index": metadata.get("chunk_index"),
                "content_type": metadata.get("content_type", "unknown"),
                "context_text": metadata.get("context_text", ""),
                "source": "vector",
                # MinerU结构化元数据
                "page_idx": metadata.get("page_idx"),
                "has_images": metadata.get("has_images", False),
                "block_types": metadata.get("block_types", "").split("|") if metadata.get("block_types") else [],
                "structure_level": metadata.get("structure_level", 0),
            })

    return documents
```

**关键词搜索实现：** (`rag_service.py:206-290`)
```python
async def _keyword_search(
    self,
    query: str,
    kb_ids: Optional[List[str]],
    top_k: int
) -> List[Dict[str, Any]]:
    """基于ElasticSearch的BM25关键词搜索"""

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

    # 添加kb_id过滤器
    if kb_ids:
        es_query["query"]["bool"]["filter"] = [{"terms": {"kb_id": kb_ids}}]

    # 搜索
    results = await es_db.search(index="doc", query=es_query, size=top_k)

    # 格式化结果
    documents = []
    if results and results.get("hits"):
        for hit in results["hits"]["hits"]:
            source = hit["_source"]
            score = hit["_score"]
            normalized_score = min(1.0, score / 10.0)  # 归一化到0-1

            documents.append({
                "content": source.get("content", ""),
                "score": normalized_score,
                "doc_id": source.get("doc_id"),
                "kb_id": source.get("kb_id"),
                "chunk_index": source.get("chunk_index"),
                "content_type": source.get("content_type", "unknown"),
                "context_text": source.get("context_text", ""),
                "source": "keyword",
                # MinerU结构化字段
                "page_idx": source.get("page_idx"),
                "page_indices": source.get("page_indices", []),
                "block_types": source.get("block_types", []),
                "image_count": source.get("image_count", 0),
                "image_references": source.get("image_references", []),
                "image_captions": source.get("image_captions", []),
                "title_path": source.get("title_path", []),
                "structure_level": source.get("structure_level", 0),
            })

    return documents
```

**RRF融合实现：** (`rag_service.py:372-424`)
```python
def _rrf_fusion(
    self,
    vector_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    倒数排名融合算法。
    公式: score(d) = Σ 1 / (k + rank_i(d))
    """
    doc_scores: Dict[str, Dict[str, Any]] = {}

    # 添加向量搜索分数
    for rank, result in enumerate(vector_results, start=1):
        chunk_id = f"{result['doc_id']}_{result['chunk_index']}"
        if chunk_id not in doc_scores:
            doc_scores[chunk_id] = self._create_doc_fusion_entry(result, result["score"], rank, "vector")
        doc_scores[chunk_id]["rrf_score"] += 1.0 / (k + rank)

    # 添加关键词搜索分数
    for rank, result in enumerate(keyword_results, start=1):
        chunk_id = f"{result['doc_id']}_{result['chunk_index']}"
        if chunk_id not in doc_scores:
            doc_scores[chunk_id] = self._create_doc_fusion_entry(result, result["score"], rank, "keyword")
        else:
            doc_scores[chunk_id]["keyword_score"] = result["score"]
            doc_scores[chunk_id]["keyword_rank"] = rank
        doc_scores[chunk_id]["rrf_score"] += 1.0 / (k + rank)

    # 按RRF分数排序
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_docs
```

#### 2.2.2 BGEAPIReranker

**文件位置：** `ai-service/app/services/reranker_service.py:187-306`

**类结构：**
```python
class BGEAPIReranker(BaseReranker):
    """
    使用BGE Reranker API的重排序器。
    API格式:
        POST {base_url}/v1/rerank
        Headers: Authorization: Bearer {api_key}
        Body: {"model": "...", "query": "...", "documents": [...], "top_n": k}

    优势：
    - 真正的重排序（非embedding相似度）
    - 无文本长度限制
    - 单个API调用处理所有文档
    """

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        timeout: int = 120,
    ):
        self.base_url = (base_url or settings.bge_reranker_api_url).rstrip("/")
        self.api_key = api_key or settings.bge_reranker_api_key
        self.model = model or "/model"
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """使用BGE Reranker API重排序文档。"""

        url = f"{self.base_url}/v1/rerank"
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self._headers) as response:
                if response.status != 200:
                    raise RuntimeError(f"BGE API error: status={response.status}")
                data = await response.json()

        # 解析结果
        results = data.get("results", [])
        indexed_scores = [(r.get("index", i), r.get("relevance_score", 0.0))
                         for i, r in enumerate(results)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_k]
```

#### 2.2.3 ConversationNodes

**文件位置：** `ai-service/app/services/conversation/conversation_nodes.py:46-1271`

**knowledge_retrieval节点：** (`conversation_nodes.py:599-706`)
```python
async def knowledge_retrieval(self, state: ConversationState) -> ConversationState:
    """
    知识库检索 - 混合搜索相关文档。

    特殊处理：数学教材知识库使用math_textbook_retrieval
    """
    async with time_node("knowledge_retrieval", state):
        kb_ids = state["employee_config"].get("kb_ids", [])
        search_query = state.get("rewritten_query", state["user_query"])

        all_results = []

        # 数学教材知识库特殊处理
        if MATH_KB_ID in kb_ids:
            math_results = await math_textbook_retrieval.search(
                query=search_query,
                kb_ids=[MATH_KB_ID],
                top_k=5,
                use_hybrid=True,
                enable_rerank=True
            )
            all_results.extend(math_results)

            # 其他知识库使用标准检索
            other_kb_ids = [kb_id for kb_id in kb_ids if kb_id != MATH_KB_ID]
            if other_kb_ids:
                standard_results = await rag_retrieval.search(
                    query=search_query,
                    kb_ids=other_kb_ids,
                    top_k=5,
                    use_hybrid=True,
                    enable_rerank=True
                )
                all_results.extend(standard_results)
        else:
            # 标准RAG检索
            all_results = await rag_retrieval.search(
                query=search_query,
                kb_ids=kb_ids if kb_ids else None,
                top_k=5,
                use_hybrid=True,
                enable_rerank=True
            )

        state["retrieved_docs"] = all_results[:5]
        state["kb_used"] = list(set([
            doc.get("kb_id") for doc in all_results if doc.get("kb_id")
        ]))

    return state
```

**grade_documents节点：** (`conversation_nodes.py:708-786`)
```python
async def grade_documents(self, state: ConversationState) -> ConversationState:
    """
    文档评分 - 计算相关性评分，触发直接匹配检测。

    relevance_score来自top文档的rerank_score
    """
    async with time_node("grade_documents", state):
        docs = state.get("retrieved_docs", [])

        if not docs:
            state["relevance_score"] = 0.0
        else:
            top_doc = docs[0]
            rerank_score = top_doc.get("rerank_score")
            state["relevance_score"] = max(0.0, min(1.0, float(rerank_score)))

            # 直接匹配检测（仅数学教材）
            content_type = top_doc.get("content_type")
            kb_used = state.get("kb_used", [])
            is_math_kb = MATH_KB_ID in kb_used

            if state["relevance_score"] > 0.8 and content_type in ("qa", "teaching_script") and is_math_kb:
                direct_content = top_doc.get("context_text", "")
                if direct_content:
                    state["final_answer"] = direct_content  # 跳过LLM生成
                    state["direct_match"] = {
                        "content_type": content_type,
                        "doc_id": top_doc.get("doc_id"),
                        "chunk_id": top_doc.get("chunk_id"),
                    }

    return state
```

**compress_context节点：** (`conversation_nodes.py:788-869`)
```python
async def compress_context(self, state: ConversationState) -> ConversationState:
    """
    上下文压缩 - 减少token使用。

    触发条件：
    - context_compression_enabled = True
    - 总上下文长度 > 1500字符
    - 压缩到500字以内
    """
    async with time_node("compress_context", state):
        state["compressed_context"] = None

        compression_enabled = getattr(settings, 'context_compression_enabled', False)
        if not compression_enabled:
            return state

        docs = state.get("retrieved_docs", [])
        web_results = state.get("web_search_results", [])
        total_length = (
            sum(len(doc.get('content', '')) for doc in docs) +
            sum(len(r.get('content', '')) for r in web_results)
        )

        if total_length < 1500:
            return state

        # 使用LLM压缩上下文
        compress_prompt = f"""请将以下文档内容压缩成最精炼的关键信息...

用户问题: {state["user_query"]}

压缩要求:
1. 只保留与用户问题相关的信息
2. 去除重复和冗余内容
3. 使用简洁的语言
4. 压缩后的内容不超过500字
5. 保留关键数据和事实

压缩后的内容:"""

        response = await llm.ainvoke(compress_prompt)
        state["compressed_context"] = response.content.strip()

    return state
```

### 2.3 RRF（倒数排名融合）

#### RRF公式
```
score(d) = Σ 1 / (k + rank_i(d))

其中：
- k = 60（默认常数）
- rank_i(d) = 文档d在第i个检索结果中的排名
```

#### 文档融合条目结构
```python
{
    "content": "...",           # 文档内容
    "doc_id": "...",           # 文档ID
    "chunk_id": "...",         # chunk ID
    "kb_id": "...",            # 知识库ID
    "chunk_index": 0,          # chunk索引
    "rrf_score": 0.0,         # RRF融合后的最终分数
    "vector_score": 0.0,      # 向量搜索分数
    "keyword_score": 0.0,     # 关键词搜索分数
    "vector_rank": None,       # 向量搜索排名
    "keyword_rank": None,      # 关键词搜索排名
    # 其他元数据...
}
```

### 2.4 相关性评分和阈值过滤

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

### 2.5 特殊处理：数学教材直接匹配

**触发条件：**
1. 查询匹配数学教材知识库
2. `content_type` 为 `qa` 或 `teaching_script`
3. 相关性评分 > 0.8

**处理方式：**
- 跳过LLM生成
- 直接使用预生成的 `context_text`
- 支持 `teaching_script_tts` 语音播报文本

### 2.6 特殊处理：FAQ检索

**FAQ混合搜索流程：**
```python
async def faq_hybrid_search(
    self,
    query: str,
    employee_id: str,
    faq_sim_threshold: float = 0.0,
    faq_top_k: int = 3
) -> List[Dict[str, Any]]:
    # 1. 向量搜索（ChromaDB faq collection）
    vector_results = await self._faq_vector_search(query, employee_id, faq_top_k * 2)

    # 2. 关键词搜索（ElasticSearch faq index）
    keyword_results = await self._faq_keyword_search(query, employee_id, faq_top_k * 2)

    # 3. RRF融合
    fused_results = self._faq_rrf_fusion(vector_results, keyword_results, k=60)

    # 4. 多重条件阈值过滤
    filtered_results = [
        result for result in fused_results
        if (
            # 优先使用RRF分数（综合考虑向量和关键词）
            (result.get("rrf_score", 0.0) >= faq_sim_threshold) or
            # 向量分数单独判断（语义相似度高）
            (result.get("vector_score", 0.0) >= faq_sim_threshold) or
            # 关键词分数高但向量分数也要有一定匹配度
            (
                result.get("keyword_score", 0.0) >= faq_sim_threshold and
                result.get("vector_score", 0.0) >= faq_sim_threshold * 0.7
            )
        )
    ]

    return filtered_results[:faq_top_k]
```

### 2.7 数据流向总结

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

### 2.8 关键设计特点

| 特点 | 说明 |
|------|------|
| 三重召回 | 向量语义 + 关键词匹配 + 重排序优化 |
| RRF融合 | 平衡语义和关键词搜索结果 |
| 动态路由 | 根据相关性决定是否使用网络搜索 |
| 特殊处理 | 数学教材直接匹配、FAQ快速通道 |
| 性能优化 | 上下文压缩、本地缓存、批处理 |

---

## 三、配置参数汇总

### 3.1 分块配置

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

### 3.2 Embedding配置

```python
# Ollama配置
ollama_base_url: "http://localhost:11434"

# 字符限制
max_tokens: 1024
max_chars: 384  # 实际限制
```

### 3.3 RAG配置

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
bge_reranker_api_url: "http://192.168.8.233:8091"
```

### 3.4 分层摘要配置

```python
enable_hierarchical_summary: true  # 启用分层摘要
summary_chunk_level: true          # 生成chunk摘要
summary_section_level: true        # 生成section摘要
summary_document_level: true       # 生成document摘要
summary_max_tokens: 500            # 最大token数
```

---

## 四、关键文件索引

| 功能 | 文件路径 |
|------|----------|
| 文档处理服务 | `ai-service/app/services/document_service.py` |
| 语义分块 | `ai-service/app/services/semantic_chunking.py` |
| RAG检索服务 | `ai-service/app/services/rag_service.py` |
| 重排序服务 | `ai-service/app/services/reranker_service.py` |
| Embedding工具 | `ai-service/app/utils/embeddings.py` |
| 会话工作流节点 | `ai-service/app/services/conversation/conversation_nodes.py` |
| 会话工作流 | `ai-service/app/services/conversation_service.py` |
| ChromaDB连接 | `ai-service/app/core/chroma.py` |
| ElasticSearch连接 | `ai-service/app/core/elasticsearch.py` |
| 数据库模型 | `ai-service/app/models/database.py` |
| 配置定义 | `ai-service/app/core/config.py` |

---

## 五、相关文档参考

| 文档 | 路径 |
|------|------|
| MinerU实现总结 | `docs/MinerU实现总结.md` |
| MinerU使用指南 | `docs/MinerU客户端使用指南.md` |
| 性能优化总结 | `docs/性能优化总结.md` |
| ConversationWorkflow流程图 | `docs/ConversationWorkflow流程图与架构图.md` |
| API使用文档 | `docs/API使用文档.md` |
| 部署指南 | `docs/部署指南.md` |
