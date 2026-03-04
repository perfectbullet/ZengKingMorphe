# RAG核心流程 - 关键代码和类说明

本文档详细说明两个核心流程涉及的关键类和代码位置。

---

## 一、文档的索引和Embedding流程

### 1.1 整体流程架构

```
DocumentProcessor.process_document()
    ↓
_extract_text()            # 文本提取
    ↓
_preprocess_text()         # 文本预处理
    ↓
_chunk_text()              # 文档分块
    ↓
HierarchicalSummarizer.summarize()  # 分层摘要
    ↓
_process_chunks()          # Embedding生成和多数据库存储
```

### 1.2 核心类

#### 1.2.1 DocumentProcessor
**文件位置：** `ai-service/app/services/document_service.py:43-1292`

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
    # 1. 插入文档记录（MongoDB）
    doc_model = DocumentModel(
        doc_id=doc_id,
        filename=filename,
        kb_id=kb_id,
        status="processing",
        ...
    )
    await db.documents.insert_one(doc_model.model_dump())

    # 2. 文本提取
    text_content = await self._extract_text(file_path, file_ext, use_mineru)

    # 3. 文本预处理
    if chunk_config:
        text_content = self._preprocess_text(text_content, chunk_config)

    # 4. 文档分块（返回 chunks 和 hierarchical_summary_data）
    chunk_result = await self._chunk_text(
        text_content, doc_id, kb_id, file_ext=file_ext,
        chunk_config=chunk_config, mineru_structured_data=mineru_structured_data
    )

    # 5. 处理chunks（生成embedding，存储到数据库）
    await self._process_chunks(chunks, doc_id, kb_id, task_id=task_id)
```

#### 1.2.2 SemanticChunker
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

    MARKDOWN_FORMATS = {'.md', '.pdf'}  # 需要更大chunk的格式

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
        window_size: Optional[int] = None,
        file_ext: Optional[str] = None
    ):
        # 自动根据文件类型调整chunk大小
        if file_ext and file_ext.lower() in self.MARKDOWN_FORMATS:
            self.min_chunk_size = 500   # Markdown/PDF使用更大chunk
            self.max_chunk_size = 2000
        else:
            self.min_chunk_size = 100   # 普通文本默认
            self.max_chunk_size = 500

    def _split_into_sentences(self, text: str) -> List[str]:
        """按中文/英文标点符号分割句子"""
        parts = self.sentence_pattern.split(text)
        # ... 处理逻辑

    async def _compute_similarities(self, windows: List[str]) -> List[float]:
        """计算相邻窗口的余弦相似度"""
        embedder = get_embedding()
        for i in range(len(windows) - 1):
            emb1 = embedder.embed_query(windows[i])
            emb2 = embedder.embed_query(windows[i + 1])
            similarity = self._cosine_similarity(emb1, emb2)
            similarities.append(similarity)

    async def chunk_text(self, text: str) -> List[SemanticChunk]:
        """分块主方法"""
        sentences = self._split_into_sentences(text)
        windows = self._create_text_windows(sentences)
        similarities = await self._compute_similarities(windows)
        boundaries = await self._find_chunk_boundaries(sentences, similarities)
        # ... 生成chunks
```

#### 1.2.3 HierarchicalSummarizer
**文件位置：** `ai-service/app/services/semantic_chunking.py:359-658`

**摘要生成结构：**
```python
class HierarchicalSummarizer:
    """
    三层摘要生成：
    - Chunk level: 每个chunk的核心内容总结（≤50字）
    - Section level: 相关chunks的主要观点总结（≤100字）
    - Document level: 文档整体内容和要点总结（≤200字）
    """

    async def summarize(self, chunks: List[SemanticChunk], doc_id: str):
        # 1. Chunk级摘要
        if self.chunk_enabled:
            await self._summarize_chunks(chunks, summary, doc_id)

        # 2. Section级摘要
        if self.section_enabled and len(chunks) >= self.section_threshold:
            await self._summarize_sections(chunks, summary, doc_id)

        # 3. Document级摘要
        if self.document_enabled and len(chunks) >= self.document_threshold:
            await self._summarize_document(chunks, summary, doc_id)
```

#### 1.2.4 OllamaEmbeddings
**文件位置：** `ai-service/app/utils/embeddings.py:70-420`

**关键特性：**
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

    def _truncate_text(self, text: str, level: int = 0) -> tuple[str, int]:
        """智能截断文本，支持降级策略"""
        truncated_text, trunc_level, original_length = self.truncator.truncate(text)
        return truncated_text, trunc_level

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量embedding，使用8个一批的批处理"""
        BATCH_SIZE = 8
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            # 检查缓存
            uncached_count = sum(1 for t in batch if not cached)
            if uncached_count == 0:
                # 全部命中缓存
                cache_hits += len(batch)
            else:
                # 批量请求
                batch_embeddings = self._embed_batch(batch)
                embeddings.extend(batch_embeddings)

    async def _process_chunks(self, chunks: List[DocumentChunkModel], ...):
        """Chunk处理和存储"""
        # 1. 构建chunk_texts（MinerU分块带标题前缀）
        chunk_texts = []
        for chunk in valid_chunks:
            if hasattr(chunk, 'title_path') and chunk.title_path:
                title_str = " > ".join(chunk.title_path)
                chunk_texts.append(f"{title_str}\n\n{chunk.content}")
            else:
                chunk_texts.append(chunk.content)

        # 2. 存储到ChromaDB（自动生成embedding）
        await chroma_db.add_documents(
            collection_name="doc",
            documents=chunk_texts,
            metadatas=chunk_metadatas,
            ids=chunk_ids,
        )

        # 3. 存储到ElasticSearch
        for chunk in valid_chunks:
            es_document = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "summary": chunk_summary,
                "chunk_index": chunk.chunk_index,
                # MinerU结构化字段
                "page_idx": chunk.page_idx,
                "title_path": chunk.title_path,
                ...
            }
            await es_db.index_document(index="doc", doc_id=chunk.chunk_id, document=es_document)

        # 4. 存储到MongoDB
        await db.document_chunks.insert_many(chunk_docs)
```

### 1.3 数据模型

#### SemanticChunk（语义分块结果）
**文件位置：** `ai-service/app/services/semantic_chunking.py:35-42`
```python
@dataclass
class SemanticChunk:
    """语义连贯的文本块"""
    content: str              # 块内容
    start_pos: int            # 起始位置
    end_pos: int              # 结束位置
    chunk_index: int          # 块索引
    summary: Optional[str] = None  # 摘要
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

### 1.4 支持的文件格式

**DocumentProcessor.supported_formats** (`document_service.py:47-54`)
```python
self.supported_formats = {
    ".pdf": self._extract_pdf,
    ".docx": self._extract_docx,
    ".txt": self._extract_txt,
    ".md": self._extract_markdown,
    ".html": self._extract_html,
    ".mp4": self._extract_mp4,  # 待实现
}
```

---

## 二、文档从提问到召回文档的流程

### 2.1 整体流程架构

```
用户提问
    ↓
LangGraph ConversationWorkflow
    ↓
knowledge_retrieval 节点 (conversation_nodes.py:599-706)
    ├─ 混合搜索（RAGRetrieval）
    │   ├─ _vector_search (ChromaDB)
    │   ├─ _keyword_search (ElasticSearch)
    │   └─ _rrf_fusion
    └─ 重排序 (BGEAPIReranker)
    ↓
grade_documents 节点 (conversation_nodes.py:708-786)
    ├─ 计算相关性评分
    └─ 直接匹配检测（数学教材）
    ↓
compress_context 节点（可选）
    ↓
generate_answer 节点
```

### 2.2 核心类

#### 2.2.1 RAGRetrieval
**文件位置：** `ai-service/app/services/rag_service.py:13-696`

**类结构：**
```python
class RAGRetrieval:
    """
    RAG检索服务，支持混合搜索（向量+关键词）和重排序。
    """

    async def search(
        self,
        query: str,
        kb_ids: Optional[List[str]] = None,
        top_k: int = 5,
        use_hybrid: bool = True,
        enable_rerank: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """搜索入口，自动选择混合/向量搜索"""
        if use_hybrid:
            return await self._hybrid_search(query, kb_ids, top_k, enable_rerank)
        else:
            return await self._vector_search(query, kb_ids, top_k, enable_rerank)

    async def _hybrid_search(
        self,
        query: str,
        kb_ids: Optional[List[str]],
        top_k: int,
        enable_rerank: bool = False
    ):
        """混合搜索：向量 + BM25 + RRF + 重排序"""
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
                "source": "vector",
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
                "source": "keyword",
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
        """
        使用BGE Reranker API重排序文档。
        返回: List[(original_index, score)], 按分数降序
        """
        url = f"{self.base_url}/v1/rerank"
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self._headers) as response:
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
    """
    async with time_node("compress_context", state):
        compression_enabled = getattr(settings, 'context_compression_enabled', False)
        if not compression_enabled:
            return state

        docs = state.get("retrieved_docs", [])
        total_length = sum(len(doc.get('content', '')) for doc in docs)

        if total_length < 1500:
            return state

        # 使用LLM压缩上下文到500字以内
        compress_prompt = f"""请将以下文档内容压缩成最精炼的关键信息...
        压缩后的内容不超过500字"""
        response = await llm.ainvoke(compress_prompt)
        state["compressed_context"] = response.content.strip()

    return state
```

### 2.3 数据结构

#### RRF融合条目
```python
{
    "content": "...",           # 文档内容
    "doc_id": "...",           # 文档ID
    "chunk_id": "...",         # chunk ID
    "kb_id": "...",            # 知识库ID
    "chunk_index": 0,          # chunk索引
    "rrf_score": 0.0,         # RRF融合分数
    "vector_score": 0.0,      # 向量搜索分数
    "keyword_score": 0.0,     # 关键词搜索分数
    "vector_rank": None,       # 向量搜索排名
    "keyword_rank": None,      # 关键词搜索排名
}
```

#### 重排序结果
```python
# BGEAPIReranker.rerank() 返回格式
List[Tuple[int, float]]
# 例如: [(2, 0.95), (0, 0.87), (1, 0.72)]
# 每个元组是 (原始索引, 相关性分数)，按分数降序排序
```

### 2.4 FAQ检索（特殊流程）

**faq_hybrid_search方法：** (`rag_service.py:426-481`)
```python
async def faq_hybrid_search(
    self,
    query: str,
    employee_id: str,
    faq_sim_threshold: float = 0.0,
    faq_top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    FAQ混合搜索（向量+关键词双路召回+RRF融合）。
    """
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
            result.get("rrf_score", 0.0) >= faq_sim_threshold or
            result.get("vector_score", 0.0) >= faq_sim_threshold or
            (
                result.get("keyword_score", 0.0) >= faq_sim_threshold and
                result.get("vector_score", 0.0) >= faq_sim_threshold * 0.7
            )
        )
    ]

    return filtered_results[:faq_top_k]
```

---

## 三、配置参数汇总

### 3.1 分块配置
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enable_semantic_chunking` | True | 启用语义分块 |
| `semantic_chunk_similarity_threshold` | 0.75 | 相似度阈值 |
| `semantic_chunk_min_size` | 100 | 最小chunk大小 |
| `semantic_chunk_max_size` | 500 | 最大chunk大小 |
| `semantic_chunk_window_size` | 3 | 窗口大小 |
| `chunk_size` | 256 | 传统分块大小 |
| `chunk_overlap` | 50 | 分块重叠 |

### 3.2 Embedding配置
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `embedding_ollama_model` | bge-large-zh-v1.5:2k | Ollama模型 |
| `ollama_base_url` | http://localhost:11434 | Ollama地址 |
| `max_tokens` | 1024 | 最大token数 |
| `max_chars` | 384 | 实际字符限制 |

### 3.3 RAG配置
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `top_k` | 5 | 返回文档数 |
| `rerank_enabled` | True | 启用重排序 |
| `reranker_type` | bge_api | 重排序器类型 |
| `bge_reranker_api_url` | - | BGE API地址 |
| `relevance_threshold` | 0.6 | 相关度阈值 |
| `context_compression_enabled` | False | 启用上下文压缩 |

### 3.4 分层摘要配置
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enable_hierarchical_summary` | True | 启用分层摘要 |
| `summary_chunk_level` | True | chunk级摘要 |
| `summary_section_level` | True | section级摘要 |
| `summary_document_level` | True | document级摘要 |
| `summary_max_tokens` | 500 | 最大token数 |

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
| ChromaDB连接 | `ai-service/app/core/chroma.py` |
| ElasticSearch连接 | `ai-service/app/core/elasticsearch.py` |
| 数据库模型 | `ai-service/app/models/database.py` |
| 配置定义 | `ai-service/app/core/config.py` |
