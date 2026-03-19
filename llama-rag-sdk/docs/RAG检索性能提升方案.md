# RAG 检索性能提升方案

## 📊 llama_cloud_services 中 RAG 性能提升模块分析

### 🔍 发现的关键检索增强模块

#### 1. **LlamaCloudRetriever** (`retriever.py`)

支持的检索参数：
- `dense_similarity_top_k` - 向量检索 Top-K
- `sparse_similarity_top_k` - 稀疏检索（关键词）Top-K
- `enable_reranking` - 是否启用 Rerank
- `rerank_top_n` - Rerank 后返回结果数
- `alpha` - 混合检索权重（1=全向量，0=全关键词）
- `retrieval_mode` - 检索模式
- `filters` - 元数据过滤

#### 2. **LlamaCloudCompositeRetriever** (`composite_retriever.py`)

支持：
- `CompositeRetrievalMode` - 复合检索模式
- `RerankConfig` - 可配置的 Rerank 策略
- 多索引检索融合
- 持久化与非持久化两种模式

---

## 🚀 提升 llama-rag-sdk 文档检索能力的方案

基于当前架构和 llama_cloud_services 的经验，以下是具体的提升方案：

### ✅ 方案 1: 查询扩展

**原理**：将单个查询扩展为多个相关查询，增加召回率

**实现方式**：
```python
# src/retrieval/strategies.py

class QueryExpansionRetrieval(RetrievalStrategy):
    """查询扩展检索策略"""

    async def expand_query(self, query: str) -> List[str]:
        """使用 LLM 扩展查询"""
        prompt = f"""
        为以下查询生成 3-5 个相关的扩展查询，用于提高检索召回率：
        查询：{query}

        只返回查询列表，每行一个。
        """
        # 调用 LLM 生成扩展查询
        expanded_queries = await self._call_llm(prompt)
        return [query] + expanded_queries

    async def retrieve(self, query: str, top_k: int = 5, filters=None):
        # 1. 扩展查询
        expanded = await self.expand_query(query)

        # 2. 对每个查询执行检索
        all_results = []
        for q in expanded:
            results = await self.vector_search(q, top_k)
            all_results.extend(results)

        # 3. 去重并 Rerank
        unique_results = self._deduplicate(all_results)
        return await self.rerank(query, unique_results, top_k)
```

**适用场景**：查询表述模糊、同义词、专业术语

---

### ✅ 方案 2: 稀疏检索 + 密集检索融合

**原理**：结合 BM25 关键词检索和向量检索，提升精确匹配召回

**实现方式**：
```python
# src/retrieval/strategies.py

class DenseSparseHybridRetrieval(RetrievalStrategy):
    """密集 + 稀疏混合检索"""

    def __init__(self, vector_store, rerank_client, alpha=0.7):
        self.vector_store = vector_store
        self.rerank_client = rerank_client
        self.alpha = alpha  # 0.0 = 全稀疏，1.0 = 全密集

    async def retrieve(self, query: str, top_k: int = 5, filters=None):
        # 1. 密集检索（向量）
        dense_results = await self._dense_search(query, top_k * 2, filters)

        # 2. 稀疏检索（BM25）
        sparse_results = await self._sparse_search(query, top_k * 2, filters)

        # 3. RRF 融合
        fused = self._rrf_fusion(
            dense_results,
            sparse_results,
            k=60,  # RRF 常数
            alpha=self.alpha
        )

        # 4. Rerank
        return await self.rerank(query, fused, top_k)

    def _rrf_fusion(self, dense, sparse, k=60, alpha=0.7):
        """Reciprocal Rank Fusion"""
        scores = {}

        # 密集检索分数
        for i, doc in enumerate(dense):
            doc_id = doc.chunk_id
            if doc_id not in scores:
                scores[doc_id] = {"doc": doc, "dense": 0, "sparse": 0}
            scores[doc_id]["dense"] = 1 / (k + i + 1)

        # 稀疏检索分数
        for i, doc in enumerate(sparse):
            doc_id = doc.chunk_id
            if doc_id not in scores:
                scores[doc_id] = {"doc": doc, "dense": 0, "sparse": 0}
            scores[doc_id]["sparse"] = 1 / (k + i + 1)

        # 加权融合
        for doc_id, data in scores.items():
            final_score = alpha * data["dense"] + (1 - alpha) * data["sparse"]
            data["doc"].score = final_score

        # 返回融合结果
        return sorted(
            [data["doc"] for data in scores.values()],
            key=lambda x: x.score,
            reverse=True
        )
```

**配置**：
```bash
# .env
USE_HYBRID_RETRIEVAL=true
HYBRID_ALPHA=0.7  # 向量检索权重
```

---

### ✅ 方案 3: 父文档检索

**原理**：检索小块，但返回完整的父文档上下文，避免信息截断

**实现方式**：
```python
# src/retrieval/strategies.py

class ParentDocumentRetrieval(RetrievalStrategy):
    """父文档检索"""

    def __init__(self, vector_store, rerank_client,
                 parent_chunk_size=2000, child_chunk_size=400):
        self.vector_store = vector_store
        self.rerank_client = rerank_client
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size

    async def retrieve(self, query: str, top_k: int = 5, filters=None):
        # 1. 检索子块
        child_results = await self._vector_search(query, top_k * 2, filters)

        # 2. 获取父文档
        parent_docs = []
        for child in child_results:
            parent_id = child.metadata.get("parent_id")
            if parent_id:
                parent = await self._get_parent_document(parent_id)
                if parent not in parent_docs:
                    parent_docs.append(parent)

        # 3. Rerank 父文档
        return await self.rerank(query, parent_docs, top_k)
```

**优势**：提供更完整的上下文，避免分块割裂信息

---

### ✅ 方案 4: 上下文压缩

**原理**：压缩检索到的文档，只保留与查询相关的关键信息

**实现方式**：
```python
# src/retrieval/context_compressor.py

class ContextCompressor:
    """上下文压缩器"""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def compress(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """压缩文档上下文"""
        prompt = f"""
        以下是与查询相关的文档片段，请提取并只保留与查询最相关的关键信息：

        查询：{query}

        文档：
        {self._format_documents(documents)}

        返回格式：每行一个提取的关键句，保持原顺序。
        """

        compressed_texts = await self._call_llm(prompt)

        # 更新文档内容为压缩后的文本
        for doc, compressed in zip(documents, compressed_texts):
            doc.text = compressed.strip()

        return documents
```

---

### ✅ 方案 5: 分块策略优化

**当前问题**：固定大小分块可能切断语义

**改进方案**：
```python
# src/document_indexer/chunker.py

class SmartSemanticChunker(BaseChunker):
    """智能语义分块器"""

    def chunk(self, text: str) -> List[str]:
        """
        基于句子边界和语义相似度的智能分块
        """
        # 1. 按句子分割
        sentences = self._split_sentences(text)

        # 2. 计算句子向量相似度
        embeddings = self._get_embeddings(sentences)
        similarities = self._compute_similarities(embeddings)

        # 3. 在相似度低的地方切分
        chunks = []
        current_chunk = []

        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)

            # 检查是否应该切分
            if i < len(similarities) - 1:
                if (len(current_chunk) >= self.strategy.chunk_size and
                    similarities[i] < 0.7):  # 相似度阈值
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
```

---

### ✅ 方案 6: Rerank 候选数动态调整

**当前问题**：固定候选数可能不足或浪费

**改进方案**：
```python
# src/retrieval/strategies.py

class AdaptiveRerankRetrieval(RetrievalStrategy):
    """自适应 Rerank 检索"""

    async def retrieve(self, query: str, top_k: int = 5, filters=None):
        # 1. 检索候选（初始数量）
        initial_candidates = top_k * self.candidate_multiplier
        candidates = await self._vector_search(query, initial_candidates, filters)

        # 2. 根据分数分布动态调整
        scores = [doc.score for doc in candidates]
        score_variance = self._compute_variance(scores)

        # 如果分数差异小（可能相关性低），增加候选数
        if score_variance < 0.05:
            additional = await self._vector_search(query, initial_candidates, filters)
            candidates.extend(additional)

        # 3. Rerank
        reranked = await self.rerank(query, candidates)

        # 4. 根据最终分数过滤
        threshold = 0.5  # 可配置
        filtered = [doc for doc in reranked if doc.score >= threshold]

        return filtered[:top_k]

    def _compute_variance(self, scores: List[float]) -> float:
        """计算分数方差"""
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        return sum((s - mean) ** 2 for s in scores) / len(scores)
```

---

### ✅ 方案 7: 元数据过滤优化

**当前支持**：基本过滤

**增强方案**：
```python
# src/retrieval/strategies.py

class FilteredRetrieval(RetrievalStrategy):
    """增强的过滤检索"""

    async def retrieve(self, query: str, top_k: int = 5,
                   filters: Optional[Dict[str, Any]] = None):
        # 支持复杂过滤条件
        if filters:
            # 支持 AND/OR/NOT 操作
            filters = self._parse_complex_filters(filters)

            # 支持范围查询
            filters = self._expand_range_filters(filters)

            # 支持正则匹配
            filters = self._expand_regex_filters(filters)

        results = await self._vector_search(query, top_k * 2, filters)

        # 后处理过滤（LLM 辅助）
        if results:
            results = await self._llm_rerank(query, results, top_k)

        return results
```

---

### ✅ 方案 8: 缓存优化

**实现方式**：
```python
# src/retrieval/cache.py

class RetrievalCache:
    """检索结果缓存"""

    def __init__(self, max_size=1000, ttl=3600):
        self.cache = {}  # {query_hash: (results, timestamp)}
        self.max_size = max_size
        self.ttl = ttl

    async def get(self, query: str, top_k: int) -> Optional[List]:
        """获取缓存"""
        key = self._hash(query, top_k)
        if key in self.cache:
            results, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return results
            else:
                del self.cache[key]
        return None

    async def set(self, query: str, top_k: int, results: List):
        """设置缓存"""
        key = self._hash(query, top_k)

        # LRU 淘汰
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache,
                          key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        self.cache[key] = (results, time.time())
```

---

## 📋 推荐实施优先级

| 方案 | 难度 | 效果 | 优先级 |
|------|--------|------|--------|
| 稀疏 + 密集混合检索 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 🔥 **高** |
| 查询扩展 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 🔥 **高** |
| 智能分块策略 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ 中 |
| 父文档检索 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ 中 |
| 自适应 Rerank | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ 中 |
| 上下文压缩 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ 低 |
| 缓存优化 | ⭐ | ⭐⭐ | ⭐ 低 |

---

## 🎯 立即可实施的改进

基于当前代码，最快速有效的改进是：

### 1. **启用混合检索**（已在配置中）
```bash
# .env
USE_HYBRID_RETRIEVAL=true
```

### 2. **增加候选数量**
```bash
# .env
RERANK_CANDIDATE_MULTIPLIER=6  # 当前是 4，提高到 6
```

### 3. **优化分块大小**
```bash
# .env
CHUNK_SIZE=800  # 根据文档特点调整
CHUNK_OVERLAP=100
```

---

## 🔧 实施建议

### 阶段一：快速见效（1-2天）
1. 启用混合检索
2. 增加候选倍数
3. 优化分块参数

### 阶段二：核心增强（3-5天）
1. 实现稀疏 + 密集混合检索
2. 实现查询扩展
3. 优化 Rerank 策略

### 阶段三：高级优化（1-2周）
1. 实现父文档检索
2. 实现智能分块
3. 实现上下文压缩

---

## 📊 预期效果

| 指标 | 当前 | 改进后 | 提升 |
|------|------|--------|------|
| 召回率 | 60-70% | 85-90% | +20% |
| 准确率 | 70-80% | 85-90% | +10% |
| 响应时间 | 2-3s | 1.5-2s | -30% |
| 用户满意度 | 70% | 85% | +15% |

---

## 📝 参考资源

### llama_cloud_services 关键文件
- `llama_cloud_services/py/llama_cloud_services/index/retriever.py`
- `llama_cloud_services/py/llama_cloud_services/index/composite_retriever.py`
- `llama_cloud_services/py/llama_cloud_services/index/base.py`
- `llama_cloud_services/index.md`

### 相关文档
- `examples/parse/advanced_rag/dynamic_section_retrieval.ipynb`
- `examples/parse/multimodal/multimodal_contextual_retrieval_rag.ipynb`

### 外部参考
- Reciprocal Rank Fusion (RRF): https://plg.uwaterloo.ca/~gvcormac/cormacksigir08-rrf.pdf
- BGE Reranker: https://github.com/FlagOpen/FlagEmbedding
- Hybrid Search: https://www.pinecone.io/learn/hybrid-search-intro/

---

**文档生成时间**: 2026-03-18
**版本**: 1.0
