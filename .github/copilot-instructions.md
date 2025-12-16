# 数字员工项目 - AI 代理开发指南

> 基于项目最新代码实现状态、需求文档（`docs/需求补充与完善文档.md`）与技术架构梳理。本指南面向所有 AI 编程助手，确保实现与最新需求同步。
> 
> **最后更新**: 2025-12-16  
> **项目状态**: 开发中 (Phase 1 - FastAPI框架 + LangGraph工作流骨架)

---

## 一、项目总览

### 1.1 系统形态
- **前端**: Vue.js SPA，负责用户交互与会话管理
- **AI 引擎**: Python FastAPI + LangGraph，核心对话与检索逻辑
- **数据存储**: MongoDB（会话/文档/FAQ）、Chroma（向量库）、ElasticSearch（全文检索）
- **外部服务**: OpenAI/Claude/DeepSeek/Qwen（LLM）、Tavily（联网检索）、Java 平台（词库管理）

### 1.2 核心职责划分

#### AI 服务（FastAPI + LangGraph）
- ✅ REST/SSE 对话接口（`POST /api/chat/message`, `POST /api/chat/stream`）
- ✅ 会话与上下文管理（多轮对话、超时控制）
- ✅ 知识库检索与混合搜索（Chroma + ElasticSearch）
- ✅ 任务编排与工作流（LangGraph DAG）
- ✅ 实时联网检索（Tavily集成、CRAG模式）
- ✅ 敏感词/敏感问题检测（调用Java接口、AC自动机）
- ✅ 日志记录与审计（MongoDB存储）

#### Java 平台
- ✅ 词库管理 API（敏感词/专业词的增删改查）
- ✅ Webhook 通知（变更推送）
- ⚠️  **不参与对话链路**，仅提供词库数据

### 1.3 核心指标
- **回答准确性**: 知识库召回率、相关性评分 ≥ 0.6
- **实时性**: 实时查询判断准确率、联网检索耗时 < 5s
- **合规性**: 敏感词拦截率 100%、审计日志完整性
- **性能**: 平均响应时间 < 2s、支持并发会话数 ≥ 100

---

## 二、技术架构（简化流程图）

```
┌─── Vue 前端 (SPA)
│       ├─ 会话列表 / 消息展示
│       ├─ 流式输出渲染
│       └─ 知识库管理界面
│
├─── FastAPI 网关层
│       ├─ 请求校验 (XSS/SQL/长度/必填)
│       ├─ JWT + API Key 鉴权
│       ├─ 令牌桶限流
│       └─ 统一错误体处理
│
└─── LangGraph 工作流
        ├─ 节点1: 意图识别 (规则 + Few-shot)
        ├─ 节点2: 实时性判断 (关键词库匹配)
        ├─ 节点3: 知识库检索 (RAG - 向量 + 全文混合)
        ├─ 节点4: 联网检索 (Tavily - CRAG降级)
        ├─ 节点5: 答案生成 (LLM流式)
        └─ 节点6: 敏感词/问题过滤 + 日志记录

存储层:
├─ MongoDB: conversations / knowledge_docs / sessions / audit_logs
├─ Chroma: vector embeddings (text-embedding-3-small)
├─ ElasticSearch: full-text index with IK analyzer
└─ Redis (可选): 缓存 + 限流计数
```

---

## 三、API 接口规范

### 3.1 对话接口

#### 3.1.1 同步对话 (POST /api/chat/message)

**请求**:
```json
{
  "user_id": "user_001",
  "employee_id": "DE001",
  "session_id": "sess_001",  // 可选，自动生成
  "query": "如何重置密码？",
  "context": {
    "platform": "web",
    "ip": "192.168.1.1"
  }
}
```

**响应** (成功 - 200):
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "conversation_id": "conv_001",
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

#### 3.1.2 流式对话 (POST /api/chat/stream - SSE)

**请求**: 同上

**响应** (Server-Sent Events):
```
data: {"type": "start", "session_id": "sess_001", "conversation_id": "conv_001"}
data: {"type": "token", "content": "您好"}
data: {"type": "token", "content": "，"}
data: {"type": "token", "content": "我是"}
data: {"type": "token", "content": "数字员工"}
data: {"type": "done", "conversation_id": "conv_001", "intent": "greeting", "confidence": 0.98}
```

**注意**:
- SSE 是唯一支持的流式协议
- 事件顺序固定: `start` → 多个 `token` → `done` / `error`
- 客户端可随时中断连接
- 单次流最多推送 `chunk_size * 4` token（内存安全）

#### 3.1.3 会话管理

```http
# 获取会话
GET /api/chat/session/{session_id}

# 结束会话
DELETE /api/chat/session/{session_id}

# 获取会话历史
GET /api/chat/session/{session_id}/messages?limit=10&offset=0
```

### 3.2 知识库管理接口

```http
# 上传文档
POST /api/knowledge-base/upload
Content-Type: multipart/form-data
{
  "kb_id": "kb_001",
  "file": <binary>,
  "doc_type": "pdf|docx|txt|markdown"
}

# 获取知识库列表
GET /api/knowledge-base/list

# 删除文档
DELETE /api/knowledge-base/{doc_id}
```

### 3.3 错误处理规范

所有错误统一格式:
```json
{
  "code": 400,
  "message": "Invalid request",
  "error": {
    "type": "ValidationError",
    "details": "Missing required field: user_id",
    "request_id": "req_xyz123"  // 追踪ID
  }
}
```

**标准错误码**:
| 状态码 | 类型 | 说明 | 处理建议 |
|-------|------|------|--------|
| 400 | ValidationError | 参数验证失败 | 检查请求体格式 |
| 401 | Unauthorized | Token 无效/过期 | 重新登录 |
| 403 | Forbidden | 权限不足/IP黑名单 | 检查权限配置 |
| 404 | NotFound | 资源不存在 | 确认资源ID |
| 429 | RateLimitExceeded | 触发限流 | 等待或升级配额 |
| 500 | ServerError | 服务内部错误 | 查看日志，重试 |
| 503 | ServiceUnavailable | LLM/数据库不可用 | 转人工或重试 |

---

## 四、LangGraph 工作流设计

### 4.1 工作流概览

**核心节点**:

```python
# 工作流状态定义
class ConversationState(TypedDict):
    messages: List[dict]                # 多轮对话历史
    user_query: str                     # 当前用户查询
    user_id: str                        # 用户ID
    session_id: str                     # 会话ID
    employee_id: str                    # 数字员工ID
    employee_config: dict               # 员工配置（风格、知识库等）
    
    # 意图识别阶段
    is_realtime_query: bool             # 是否实时查询
    realtime_category: str              # 实时查询分类（天气/股价/新闻等）
    realtime_detect_reason: str         # 判断理由
    intent: str                         # 意图标签
    entities: dict                      # 命名实体
    
    # 检索阶段
    retrieved_docs: List[dict]          # 知识库检索结果
    relevance_score: float              # 最高相关性评分
    web_search_results: List[dict]      # 联网检索结果
    
    # 生成与过滤阶段
    final_answer: str                   # 最终回答
    confidence: float                   # 置信度
    has_sensitive: bool                 # 是否包含敏感内容
    
    # 元数据
    context: dict                       # 请求上下文
    kb_used: List[str]                  # 使用的知识库ID
    web_search_used: bool               # 是否使用了联网检索
    conversation_id: str                # 对话ID
    response_time_ms: int               # 响应耗时
    error: Optional[str]                # 错误信息
```

### 4.2 工作流节点细节

#### 节点1: intent_recognition（意图识别）

**目标**: 快速判断用户意图，分类为：greeting / password_reset / faq / complaint / other

**实现策略**:
- 基于关键词的规则匹配（高性能）
- Few-shot 示例（可选，用于复杂情况）
- 返回 intent + 置信度

**代码示例**:
```python
async def intent_recognition(state: ConversationState) -> ConversationState:
    query = state["user_query"]
    
    intent_patterns = {
        "greeting": ["你好", "hi", "hello", "早上好"],
        "password_reset": ["重置", "忘记", "密码", "reset", "forget password"],
        "faq": ["怎样", "如何", "是什么", "how", "what"],
    }
    
    for intent, keywords in intent_patterns.items():
        if any(kw in query.lower() for kw in keywords):
            state["intent"] = intent
            break
    
    return state
```

#### 节点2: check_realtime_query（实时性判断）

**目标**: 检测是否为实时信息查询，若是则直接跳到联网，避免知识库过时信息干扰

**触发条件**:
- 包含实时关键词（时间/天气/股价/新闻等）
- 简化意图识别为"realtime"

**实时关键词库** (config驱动):
```yaml
realtime_keywords:
  time: ["今天", "明天", "昨天", "最近", "现在", "本周", "当前"]
  weather: ["天气", "气温", "降雨", "温度", "阴晴"]
  news: ["新闻", "热点", "最新", "资讯", "动态", "头条"]
  market: ["股价", "汇率", "行情", "股市", "期货"]
  sports: ["比分", "赛况", "赛程", "球赛", "体育"]
  traffic: ["路况", "航班", "火车", "高铁", "交通"]
```

**代码示例**:
```python
async def check_realtime_query(state: ConversationState) -> ConversationState:
    query = state["user_query"].lower()
    
    for category, keywords in settings.realtime_keywords.items():
        if any(kw in query for kw in keywords):
            state["is_realtime_query"] = True
            state["realtime_category"] = category
            state["realtime_detect_reason"] = f"Matched keyword: {category}"
            break
    
    return state
```

#### 节点3: rag_search（知识库检索）

**目标**: 混合检索（向量 + 全文）+ RRF排序

**流程**:
1. 并行查询 Chroma（向量）& ElasticSearch（全文）
2. RRF 融合排序
3. 过滤 < 0.6 相关性的结果

**代码示例**:
```python
async def rag_search(state: ConversationState) -> ConversationState:
    if state["is_realtime_query"]:
        # 实时查询则跳过知识库
        return state
    
    rag_service = RAGRetrieval()
    results = await rag_service.search(
        query=state["user_query"],
        kb_ids=state["employee_config"].get("kb_ids", []),
        top_k=5,
        use_hybrid=True
    )
    
    state["retrieved_docs"] = results
    state["relevance_score"] = results[0]["score"] if results else 0.0
    
    return state
```

#### 节点4: web_search（联网检索 - CRAG降级）

**目标**: 两种触发路径：
1. **直连**: 实时查询 → 直接联网 (skip RAG)
2. **降级**: RAG 低相关性 (< 0.6) → 触发联网

**实现**:
```python
async def web_search(state: ConversationState) -> ConversationState:
    should_search = (
        state["is_realtime_query"] or 
        state["relevance_score"] < settings.relevance_threshold
    )
    
    if should_search:
        web_searcher = WebSearcher()
        results = await web_searcher.search(
            query=state["user_query"],
            max_results=settings.web_search_max_results,
            timeout=settings.web_search_timeout
        )
        state["web_search_results"] = results
        state["web_search_used"] = len(results) > 0
    
    return state
```

#### 节点5: answer_generation（答案生成）

**目标**: LLM 生成 + 流式推送

**策略** (可配置):
- **仅联网**: 高质量实时结果
- **仅知识库**: 高置信度的FAQ/标准答案
- **混合**: 联网+知识库，带来源标注

**代码示例**:
```python
async def answer_generation(state: ConversationState) -> ConversationState:
    prompt = build_prompt(
        query=state["user_query"],
        docs=state["retrieved_docs"],
        web_results=state["web_search_results"],
        context=state["context"]
    )
    
    llm = OpenAIChat()
    async for chunk in llm.astream(prompt):
        state["final_answer"] += chunk
        yield state  # 流式推送给客户端
    
    return state
```

#### 节点6: sensitive_filter（敏感词过滤）

**目标**: 输入 & 输出双向检测 + 拦截

**层级**:
- **Critical**: 拒绝服务 + 警告
- **Warning**: 替换敏感词 + 记录
- **Notice**: 仅记录

**代码示例**:
```python
async def sensitive_filter(state: ConversationState) -> ConversationState:
    # 拉取敏感词库（缓存）
    sensitive_words = await fetch_sensitive_words()
    
    # 检测输入
    input_violations = detect_sensitive(state["user_query"], sensitive_words)
    if any(v["level"] == "critical" for v in input_violations):
        state["error"] = "Your query contains prohibited content"
        return state
    
    # 检测输出
    output_violations = detect_sensitive(state["final_answer"], sensitive_words)
    if output_violations:
        for violation in output_violations:
            if violation["level"] in ["critical", "warning"]:
                state["final_answer"] = replace_sensitive(
                    state["final_answer"],
                    violation["word"],
                    violation.get("replacement", "***")
                )
        state["has_sensitive"] = True
    
    return state
```

### 4.3 工作流条件路由

```python
# 条件路由示例
def should_do_rag_search(state: ConversationState) -> bool:
    return not state["is_realtime_query"]

def should_do_web_search(state: ConversationState) -> bool:
    return (
        state["is_realtime_query"] or 
        state["relevance_score"] < settings.relevance_threshold
    )

# 工作流定义
graph = StateGraph(ConversationState)
graph.add_node("intent_recognition", intent_recognition)
graph.add_node("check_realtime_query", check_realtime_query)
graph.add_node("rag_search", rag_search)
graph.add_node("web_search", web_search)
graph.add_node("answer_generation", answer_generation)
graph.add_node("sensitive_filter", sensitive_filter)

graph.add_edge("START", "intent_recognition")
graph.add_edge("intent_recognition", "check_realtime_query")
graph.add_conditional_edges(
    "check_realtime_query",
    should_do_rag_search,
    {"true": "rag_search", "false": "web_search"}
)
graph.add_edge("rag_search", "web_search")
graph.add_edge("web_search", "answer_generation")
graph.add_edge("answer_generation", "sensitive_filter")
graph.add_edge("sensitive_filter", "END")
```

---

## 五、知识库与检索（RAG）

### 5.1 数据流

```
用户上传文档 (PDF/DOCX/TXT)
    ↓
文件解析 (PyPDF / python-docx / markdown)
    ↓
分块 (chunk_size=512, overlap=50)
    ↓
嵌入向量化 (text-embedding-3-small via OpenAI)
    ↓
存储:
├─ MongoDB: doc_id / kb_id / chunk_index / metadata
├─ Chroma: vector + metadata
└─ ElasticSearch: full-text index (IK分词)
    ↓
用户查询时：混合检索 → RRF排序 → 返回Top-K
```

### 5.2 混合检索 (Chroma + ElasticSearch)

**向量搜索** (Chroma):
```python
# 语义相似度，对意义理解有优势
results_vector = await chroma.query(
    collection="doc",
    query_embeddings=embed(user_query),
    n_results=10,
    where={"kb_id": {"$in": kb_ids}}
)
```

**全文搜索** (ElasticSearch):
```python
# 关键词匹配，召回精准
results_keyword = await es.search(
    index="doc_index",
    query={
        "match": {
            "content": user_query
        }
    },
    size=10
)
```

**RRF 融合**:
```python
def reciprocal_rank_fusion(vector_results, keyword_results, k=60):
    """
    RRF公式: score = sum(1/(k + rank))
    """
    scores = {}
    for rank, doc in enumerate(vector_results):
        scores[doc["id"]] = scores.get(doc["id"], 0) + 1/(k + rank)
    for rank, doc in enumerate(keyword_results):
        scores[doc["id"]] = scores.get(doc["id"], 0) + 1/(k + rank)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 5.3 检索结果过滤

**相关性阈值**:
- 若最高相关性 < 0.6 → 触发联网检索
- 若无检索结果 → 转人工或联网

**配置** (`app/core/config.py`):
```python
relevance_threshold: float = Field(default=0.6)  # 相关性阈值
confidence_threshold: float = Field(default=0.7)  # 生成置信度阈值
```

---

## 六、联网检索（Tavily + CRAG）

### 6.1 集成方案

**Tavily API** (实时信息源):
```python
from tavily import TavilyClient

client = TavilyClient(api_key=settings.tavily_api_key)

results = client.search(
    query="latest Tesla stock price",
    include_answer=True,
    max_results=5
)
```

### 6.2 缓存与限频

**缓存策略**:
- 同一查询 5 分钟内不重复请求
- 结果存储在 Redis，TTL=300s

**限频**:
- 每个 user_id 每分钟最多 5 次联网检索
- 全局每秒最多 10 次联网检索

---

## 七、敏感词与合规

### 7.1 词库数据流

```
Java 平台词库管理系统
    │
    ├─ GET /api/java/sensitive-words/list
    │  返回: [{ id, word, level, match_type, action, replacement }, ...]
    │
    ├─ GET /api/java/professional-words/list
    │  返回: [{ id, word, category, synonyms, kb_id }, ...]
    │
    └─ 推送变更通知 (Webhook)
       POST /api/ai/sensitive-words/sync-notify
       POST /api/ai/professional-words/sync-notify

AI 侧缓存更新:
    ↓
构建 AC 自动机 (敏感词)
构建 Trie 树 (专业词)
    ↓
运行时检测:
├─ 输入检测 (user_query)
└─ 输出检测 (final_answer)
```

### 7.2 检测实现 (AC 自动机)

**库**: `pyahocorasick`

```python
from ahocorasick import Automaton

def build_sensitive_automaton(words: List[str]) -> Automaton:
    A = Automaton()
    for word in words:
        A.add_word(word, word)
    A.make_automaton()
    return A

def detect_sensitive(text: str, automaton: Automaton) -> List[str]:
    matches = []
    for end_index, matched_word in automaton.iter(text.lower()):
        matches.append(matched_word)
    return matches
```

### 7.3 拦截策略

| 级别 | 说明 | 处理 | 示例 |
|------|------|------|------|
| Critical | 涉及法律/政治/极端 | ❌ 拒绝 + 告警 | "违禁词汇" |
| Warning | 不当但可替换 | ⚠️ 替换为\*\*\* | "脏话" |
| Notice | 仅统计追踪 | 📝 仅记录 | "过期术语" |

---

## 八、安全与鉴权

### 8.1 认证方案

**双模式**:

1. **JWT Token** (Web 前端)
   ```python
   # 获取 Token
   POST /api/auth/login
   { "username": "xxx", "password": "xxx" }
   → { "access_token": "eyJ0eX..." }
   
   # 请求头
   Authorization: Bearer <token>
   ```

2. **API Key** (第三方集成)
   ```
   X-API-Key: api_key_123456
   ```

### 8.2 限流

**令牌桶算法**:
- 每用户每分钟: 60 请求（可配置）
- 每会话并发: 1 活跃请求
- 全局并发会话: 100 个

**配置** (`app/core/config.py`):
```python
rate_limit_per_minute: int = Field(default=60)
rate_limit_per_session: int = Field(default=10)
max_concurrent_sessions: int = Field(default=100)
```

### 8.3 输入验证

**必填字段**:
- `user_id` (字符串, ≤ 100字)
- `employee_id` (字符串, ≤ 100字)
- `query` (字符串, 1-1000字)

**防护**:
- XSS: HTML 转义
- SQL 注入: 参数化查询
- 路径遍历: 白名单校验

---

## 九、数据库 Schema

### 9.1 MongoDB 集合

**conversations**:
```json
{
  "_id": ObjectId,
  "conversation_id": "conv_001",
  "session_id": "sess_001",
  "user_id": "user_001",
  "employee_id": "DE001",
  "messages": [
    { "role": "user", "content": "...", "timestamp": ISODate },
    { "role": "assistant", "content": "...", "timestamp": ISODate }
  ],
  "metadata": {
    "intent": "password_reset",
    "kb_used": ["kb_001"],
    "web_search_used": false,
    "relevance_score": 0.85,
    "confidence": 0.95,
    "response_time_ms": 1200
  },
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**knowledge_docs**:
```json
{
  "_id": ObjectId,
  "doc_id": "doc_001",
  "kb_id": "kb_001",
  "title": "用户手册.pdf",
  "content_type": "pdf",
  "chunks": [
    {
      "chunk_index": 0,
      "text": "...",
      "vector_id": "vec_001",  // Chroma ID
      "es_id": "es_001"         // ElasticSearch ID
    }
  ],
  "metadata": {
    "source_url": "...",
    "upload_user": "admin",
    "file_size": 5242880,
    "page_count": 10
  },
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**sessions**:
```json
{
  "_id": ObjectId,
  "session_id": "sess_001",
  "user_id": "user_001",
  "employee_id": "DE001",
  "created_at": ISODate,
  "last_activity": ISODate,
  "message_count": 5,
  "context": {}  // 自定义上下文
}
```

**audit_logs**:
```json
{
  "_id": ObjectId,
  "event_type": "sensitive_word_detected" | "rate_limit_exceeded" | "api_call",
  "user_id": "user_001",
  "query": "...",
  "sensitive_words_detected": ["word1", "word2"],
  "action_taken": "blocked" | "replaced" | "logged",
  "timestamp": ISODate
}
```

### 9.2 ElasticSearch 索引

**doc_index**:
```json
{
  "mappings": {
    "properties": {
      "doc_id": { "type": "keyword" },
      "kb_id": { "type": "keyword" },
      "title": { "type": "text", "analyzer": "ik_max_word", "boost": 2 },
      "content": { "type": "text", "analyzer": "ik_max_word" },
      "chunk_index": { "type": "integer" },
      "created_at": { "type": "date" }
    }
  }
}
```

### 9.3 Chroma 集合

**doc collection**:
```python
{
  "ids": ["id1", "id2", ...],
  "embeddings": [[0.1, 0.2, ...], ...],
  "documents": ["chunk text 1", "chunk text 2", ...],
  "metadatas": [
    { "doc_id": "doc_001", "kb_id": "kb_001", "chunk_index": 0 },
    ...
  ]
}
```

---

## 十、开发指南

### 10.1 代码结构

```
ai-service/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── chat.py              # 对话接口
│   │   │   ├── session.py           # 会话管理
│   │   │   ├── knowledge_base.py    # 知识库管理
│   │   │   ├── conversation.py      # 多轮对话历史
│   │   │   ├── employee.py          # 数字员工配置
│   │   │   └── webhook.py           # Webhook回调
│   │   └── middleware/
│   │       ├── auth.py              # JWT + API Key 鉴权
│   │       ├── rate_limit.py        # 限流中间件
│   │       └── error_handler.py     # 统一错误处理
│   ├── core/
│   │   ├── config.py                # 配置管理
│   │   ├── database.py              # MongoDB 连接
│   │   ├── chroma.py                # Chroma 向量库
│   │   ├── elasticsearch.py         # ElasticSearch 连接
│   │   └── logging.py               # 日志配置
│   ├── models/
│   │   ├── database.py              # ORM 模型（可选）
│   │   └── schemas.py               # Pydantic 请求/响应
│   └── services/
│       ├── conversation_service.py  # LangGraph 工作流
│       ├── rag_service.py           # 混合检索服务
│       ├── document_service.py      # 文档处理服务
│       └── web_search_service.py    # 联网检索服务
├── main.py                          # FastAPI 应用入口
└── requirements.txt
```

### 10.2 新功能开发流程

1. **需求对齐**
   - 查看 `docs/需求补充与完善文档.md` 最新需求
   - 确认 API 接口、错误码、数据格式

2. **数据模型**
   - 在 `app/models/schemas.py` 定义 Pydantic 模型
   - 在 `app/core/database.py` 定义 MongoDB collection schema

3. **服务实现**
   - 在 `app/services/` 实现业务逻辑
   - 集成 LangGraph 节点（如需）
   - 添加日志记录 & 错误处理

4. **API 端点**
   - 在 `app/api/endpoints/` 添加路由
   - 应用鉴权、限流中间件
   - 返回标准错误体

5. **测试**
   - 单元测试: `tests/test_*.py`
   - 集成测试: FastAPI TestClient
   - 参考 `tests/conftest.py` 搭建测试环境

### 10.3 代码规范

**Async 优先**:
```python
# ✅ 推荐
async def search():
    result = await db.query()
    return result

# ❌ 避免
def search():
    result = db.query()  # 阻塞操作
    return result
```

**错误处理**:
```python
# ✅ 推荐
try:
    result = await rag_service.search(query)
except Exception as e:
    logger.error("Search failed", query=query, error=str(e), exc_info=True)
    raise HTTPException(status_code=500, detail="Search failed")

# ❌ 避免
result = rag_service.search(query)  # 无错误处理
```

**日志**:
```python
from app.core.logging import get_logger

logger = get_logger(__name__)

# ✅ 推荐
logger.info("User query processed", user_id=user_id, intent=intent)

# ❌ 避免
print(f"Processing query for {user_id}")  # 使用print
```

### 10.4 配置管理

所有配置存储在 `app/core/config.py`，通过环境变量覆盖:

```bash
# .env 示例
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
MONGODB_URI=mongodb://user:pass@host:27017/db
CHROMA_HOST=localhost
CHROMA_PORT=8001
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200

# 运行时
python main.py  # 自动读取 .env
```

---

## 十一、部署与运维

### 11.1 Docker Compose

**启动完整栈**:
```bash
docker-compose up -d
```

**服务清单**:
- `ai-service`: FastAPI (port 8000)
- `mongodb`: 数据库 (port 27017)
- `elasticsearch`: 全文搜索 (port 9200)
- `chroma`: 向量库 (port 8001)

### 11.2 健康检查

```bash
# 应用健康
curl http://localhost:8000/health

# 数据库连接
curl http://localhost:8000/health/db

# 返回格式
{
  "status": "ok",
  "timestamp": "2025-12-16T10:00:00Z",
  "databases": {
    "mongodb": "connected",
    "elasticsearch": "connected",
    "chroma": "connected"
  }
}
```

### 11.3 日志与监控

**日志路径**: `logs/` 目录
**日志级别**: 通过 `LOG_LEVEL` 环境变量控制
**Prometheus 指标**: `http://localhost:8000/metrics`

---

## 十二、常见问题与扩展

### 12.1 实时查询 vs RAG 查询

| 场景 | 触发条件 | 流程 | 使用时机 |
|------|---------|------|--------|
| 直接联网 | 含实时关键词 | skip RAG → 联网 → 生成 | 天气/股价/新闻 |
| RAG + 降级 | 高相关性 ≥ 0.6 | RAG → 生成 | 常见FAQ/内部规范 |
| RAG + 联网 | 低相关性 < 0.6 | RAG + 联网 → 融合生成 | 边界问题 |

### 12.2 SSE 流式中断处理

**客户端**:
```javascript
// 中断流
eventSource.close();
```

**服务端** (自动):
```python
# FastAPI + sse_starlette 自动检测客户端断开
async def stream_response():
    try:
        async for chunk in llm.astream(prompt):
            yield chunk
    except asyncio.CancelledError:
        logger.info("Stream cancelled by client")
```

### 12.3 多知识库路由

**配置 employee_config**:
```json
{
  "employee_id": "DE001",
  "employee_config": {
    "kb_ids": ["kb_001", "kb_002"],      // 绑定知识库
    "style": "formal",                   // 对话风格
    "max_response_length": 1000,
    "preferred_sources": ["FAQ", "docs"]
  }
}
```

**路由逻辑**:
```python
# 优先级: 指定kb_ids > 默认kb > 联网
async def route_knowledge_base(state):
    kb_ids = state["employee_config"].get("kb_ids", [])
    if not kb_ids:
        kb_ids = settings.default_kb_ids
    state["kb_ids"] = kb_ids
    return state
```

### 12.4 性能优化

**缓存策略**:
- 敏感词库: Redis (TTL=3600s)
- 专业词库: Memory (定期同步)
- 检索结果: Redis (TTL=600s)

**并发优化**:
- 混合检索并行: `asyncio.gather(vector_search, keyword_search)`
- 批量处理: 支持单次查询多个用户

---

## 十三、参考资源

- 📄 需求文档: [docs/需求补充与完善文档.md](../docs/需求补充与完善文档.md)
- 🏗️ 架构设计: [docs-history/技术架构详细设计.md](../docs-history/技术架构详细设计.md)
- 📚 LangGraph示例: [docs-history/langgraph_crag.ipynb](../docs-history/langgraph_crag.ipynb)
- 🚀 部署指南: [docs/部署指南.md](../docs/部署指南.md)
- 📖 API文档: [docs/API使用文档.md](../docs/API使用文档.md)

---

## 十四、关键文件快速导航

| 用途 | 文件路径 |
|------|--------|
| 应用入口 | [ai-service/main.py](../ai-service/main.py) |
| 配置管理 | [ai-service/app/core/config.py](../ai-service/app/core/config.py) |
| 对话接口 | [ai-service/app/api/endpoints/chat.py](../ai-service/app/api/endpoints/chat.py) |
| 工作流 | [ai-service/app/services/conversation_service.py](../ai-service/app/services/conversation_service.py) |
| 混合检索 | [ai-service/app/services/rag_service.py](../ai-service/app/services/rag_service.py) |
| 数据模型 | [ai-service/app/models/schemas.py](../ai-service/app/models/schemas.py) |
| 错误处理 | [ai-service/app/api/middleware/error_handler.py](../ai-service/app/api/middleware/error_handler.py) |
| 容器编排 | [docker-compose.yml](../docker-compose.yml) |

---

## 附录: AI 代理操作清单

在实现任何功能前，请确保：

- [ ] 查看 `docs/需求补充与完善文档.md` 的相关字段/接口定义
- [ ] 确认 API 请求/响应格式、错误码、数据类型
- [ ] 添加 `app.core.logging.get_logger()` 日志记录
- [ ] 应用 `Depends(get_current_user_optional)` 鉴权
- [ ] 应用 `Depends(rate_limit_middleware)` 限流
- [ ] 返回标准错误体格式 `{"code": ..., "message": ..., "error": {...}}`
- [ ] SSE 遵循 `start → token → done/error` 顺序
- [ ] MongoDB操作使用异步驱动 `motor`
- [ ] 编写单元测试 (参考 `tests/conftest.py`)
- [ ] 更新本文档的"参考资源"、"关键文件"部分

---

**版本历史**:
- v1.0 (2025-12-16): 初版，基于项目现状梳理
- 更新计划: 每月同步代码和需求文档变更
- **MongoDB**：`conversations`、`knowledge_docs`（含 chunks→vector_id）、`faqs`、`intents`、`custom_dictionaries` 等集合必须按文档 schema 写入。
- **Chroma**：`faq_collection`、`doc_collection` 用 text-embedding-3-small；向量 ID 回写到 Mongo。
- **ElasticSearch**：`faq_index`、`doc_index`，启用 IK 分词 + 字段权重 (`question^2`)。
- **词库/敏感词**：
  - 拉取：`GET /api/java/sensitive-words/list`、`GET /api/java/professional-words/list`（过滤参数见文档）。
  - Webhook：`POST /api/ai/sensitive-words/sync-notify` / `professional-words/sync-notify`。需实现缓存、增量更新、审计日志。
  - 专业词库用于查询扩展、实体识别、召回加权。

## 实时检索（CRAG 强化）
- `check_realtime_query`：规则 + 关键词库（时间/天气/股价/...）；命中后直接联网，绕过知识库。
- 联网工具：HTTP 客户端 + JSON 模板，需缓存/限频，并记录失败重试策略。
- 答案融合策略：
  1. 仅联网结果、
  2. 仅知识库、
  3. 混合（需标注来源）。

## 安全与合规
- 输入过滤：必填字段校验、query ≤ 1000 字符、特殊字符转义、路径遍历防护。
- 输出过滤：敏感词/敏感问题、引用合法性检查。
- 审计：保存敏感拦截记录、阈值配置、人工复核入口。

## 代码与实现约定
- **LangGraph**：工作流节点至少包含：`intent_recognition → route_strategy → (rag_search|faq_search|realtime_search) → rerank → answer_generation → quality_check → sensitive_filter`，并暴露 `should_route_realtime`、`need_human` 等条件。
- **FastAPI**：使用 `Depends` 注入鉴权、限流；SSE 用 `EventSourceResponse`；所有接口 async。
- **检索融合**：实现 RRF，加上专业词扩展 & 置信度阈值；低于阈值的答案走转人工话术。
- **任务编排**：长耗时任务走 Celery/Redis（可选）或 asyncio 后台任务；Webhook 通知需幂等。

## 开发阶段（建议）
1. **Phase 1**：FastAPI 框架 + 会话/上下文 + LangGraph 骨架 + SSE。 
2. **Phase 2**：知识库管理（上传/切片/入库）、FAQ/Mongo/Chroma/ES 管道 + 混合检索。 
3. **Phase 3**：敏感/专业词集成、联网检索、质量评估、日志与监控。 
4. **Phase 4**：性能优化、容器化、前端联调、监控告警、部署脚本。

## 参考文件
- `docs/需求补充与完善文档.md`（最新需求、接口、流程）
- `requirements.md`（初版需求）
- `docs/数字员工项目-AI研发工作规划指南.md`（架构规划）
- `docs/关于FAQ问答库.md`、`docs/关于Agent对话风格分类.md`

## AI 代理操作提示
- 任何实现前先查对 `docs/需求补充与完善文档.md` 的字段/接口详情。
- 生成代码需覆盖：鉴权、限流、错误体、日志、SSE、异步/await、敏感词钩子、知识库融合等必备逻辑。
- 回答问题时优先引用具体文件/段落，提供 JSON/HTTP 示例或 LangGraph 代码片段。
- 容器化与部署脚本必须包含 Vue + ai-service + Mongo + Chroma + ES。
- 若需求存在歧义，记录假设并在输出中显式说明，以便后续复核。
