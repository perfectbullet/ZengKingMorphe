# 数字员工项目 AI 代理指南

## 项目概述

这是一个**数字员工（Digital Employee）**产品的早期规划阶段，专注于AI研发方向。项目当前处于**需求分析和架构设计阶段**，主要包含需求文档、技术规划指南和原型截图。

**团队组成**：4人小团队（1前端 + 1Java后端 + 1AI研发 + 1其他）  
**AI研发核心关注**：Agent回答的准确性（知识库召回、意图理解、答案生成质量）

## 核心业务领域

### 1. 对话管理系统
- **对话设定**：知识库配置、开场白/热门问题设定、角色人设
- **对话规则**：异常处理规则、安全规则、对话流程控制
- **高级功能**：插件系统、专业词库集成、二元评价机制（好/不好）

### 2. 意图识别引擎
- **意图配置**：意图名称、意图描述的结构化管理
- **调优工具**：辅助编写意图描述的智能工具
- **高级设置**：支持直接配置意图识别提示词（Prompt Engineering）

### 3. 知识库系统（双模式）
- **RAG文档库**：文档上传、知识片段管理、向量检索
- **FAQ问答库**：结构化问答对、语义检索、答案一致性保证
- **关键特性**：支持多知识库、专业词库增强检索准确性

### 4. 词库管理
- **专有名称词库**：行业术语、产品名称、人名地名等实体词库
- **敏感词库**：内容安全过滤、合规性保障
- **管理功能**：版本管理、权限管理、导入导出

### 5. 对话记录分析
- 对话记录查看、搜索、筛选、导出
- 对话统计分析、用户行为洞察

## 技术架构方向

基于文档 `docs/数字员工项目-AI研发工作规划指南.md` 的规划：

### 系统架构图
```
┌─────────────┐
│  Vue前端界面 │
└──────┬──────┘
       │ REST API / WebSocket
┌──────▼───────────────────────────┐
│         API Gateway              │
│      (Java后端服务)               │
└──────┬───────────────────────────┘
       │
   ┌───┴────┬─────────┬──────────┐
   │        │         │          │
┌──▼──┐ ┌──▼──┐  ┌───▼───┐ ┌───▼────┐
│对话 │ │任务 │  │知识库│ │AI引擎 │
│管理 │ │编排 │  │检索  │ │服务   │
└─────┘ └─────┘  └──────┘ └────────┘
                             ↑
                        (AI研发负责)
```

### AI核心模块（AI研发工作重点）
```
LangGraph工作流编排
        ↓
对话引擎 → 知识库RAG → 意图识别
   ↓           ↓            ↓
上下文管理  Chroma向量库   实体提取
多轮对话    FAQ检索        准确性优化
流式响应    ElasticSearch  提示词工程
```

### 确定的技术栈（AI研发侧）
- **编排框架**：LangGraph（状态机工作流）
- **向量数据库**：Chroma（文档embedding存储）
- **全文检索**：ElasticSearch（关键词精确匹配）
- **文档数据库**：MongoDB（对话记录、知识片段）
- **LLM集成**：OpenAI API / Claude API（支持流式响应）
- **容器化**：Docker + Docker Compose

### 数据检索策略（提升准确性的关键）
```
用户问题
    ↓
意图识别 → 选择检索策略
    ↓
┌────────┴────────┐
│                 │
向量检索          全文检索
(Chroma)         (ElasticSearch)
语义相似度        关键词精确匹配
    │                 │
    └────────┬────────┘
             ↓
         结果融合排序
             ↓
         上下文增强 → LLM生成
             ↓
         答案质量评估
```

## 代码生成约定

### LangGraph工作流编排模式
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

# 状态定义
class ConversationState(TypedDict):
    messages: Annotated[list, add]
    user_query: str
    intent: str
    retrieved_docs: list
    final_answer: str

# 构建对话图
def build_conversation_graph():
    graph = StateGraph(ConversationState)
    
    # 添加节点
    graph.add_node("intent_recognition", recognize_intent)
    graph.add_node("knowledge_retrieval", retrieve_knowledge)
    graph.add_node("answer_generation", generate_answer)
    
    # 添加边（工作流）
    graph.add_edge("intent_recognition", "knowledge_retrieval")
    graph.add_edge("knowledge_retrieval", "answer_generation")
    graph.add_edge("answer_generation", END)
    
    graph.set_entry_point("intent_recognition")
    return graph.compile()
```

### Chroma向量库集成模式
```python
import chromadb
from chromadb.config import Settings

# 初始化Chroma客户端
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# 创建FAQ集合
faq_collection = client.create_collection(
    name="faq_knowledge",
    metadata={"description": "FAQ问答库"}
)

# 添加文档
faq_collection.add(
    documents=["如何重置密码？点击忘记密码链接..."],
    metadatas=[{"category": "账户管理", "source": "faq_001"}],
    ids=["faq_001"]
)

# 语义检索
results = faq_collection.query(
    query_texts=["忘记密码怎么办"],
    n_results=3
)
```

### ElasticSearch关键词检索模式
```python
from elasticsearch import Elasticsearch

es = Elasticsearch(['http://localhost:9200'])

# 全文检索（提高准确性）
def keyword_search(query: str, index: str = "knowledge_base"):
    response = es.search(
        index=index,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^2", "keywords", "answer"],
                    "type": "best_fields"
                }
            }
        }
    )
    return response['hits']['hits']
```

### 对话风格（见 `docs/关于Agent对话风格分类.md`）
根据场景选择合适的风格：
- **Professional（专业型）**：企业级项目、正式汇报、技术文档
- **Tutorial（教学型）**：新手引导、技术分享、详细解释
- **Concise（简洁型）**：快速原型、代码审查、精炼直接
- **Analytical（分析型）**：技术选型、方案对比、权衡利弊

### FAQ知识库实现模式（见 `docs/关于FAQ问答库.md`）
```python
# 标准FAQ数据结构（存储在MongoDB）
{
    "id": "faq_001",
    "question": "问题文本",
    "answer": "标准答案",
    "category": "分类",
    "keywords": ["关键词1", "关键词2"],
    "related_questions": ["faq_002"],  # 关联问题
    "vector_id": "chroma_vec_001"  # Chroma向量ID
}

# 混合检索策略（向量 + 关键词）
async def hybrid_search(query: str, top_k: int = 5):
    # 1. Chroma语义检索
    semantic_results = chroma_collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    # 2. ElasticSearch关键词检索
    keyword_results = es.search(
        index="faq",
        body={"query": {"match": {"question": query}}}
    )
    
    # 3. 结果融合与排序（RRF算法）
    merged = merge_and_rank(semantic_results, keyword_results)
    return merged[:top_k]
```

### Docker容器化配置
```yaml
# docker-compose.yml
version: '3.8'
services:
  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_data:/chroma/chroma
  
  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
  
  mongodb:
    image: mongo:7.0
    ports:
      - "27017:27017"
    volumes:
      - ./mongo_data:/data/db
  
  ai-service:
    build: ./ai-service
    depends_on:
      - chroma
      - elasticsearch
      - mongodb
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

## 开发工作流

### 阶段1：需求分析（当前阶段）
- 上传需求文档和原型图到对话
- 生成技术方案文档和系统架构
- 提取AI功能点和开发优先级

### 阶段2：核心模块开发
1. **对话引擎**：多轮对话、上下文管理、LLM集成
2. **知识库RAG**：文档向量化、语义检索、上下文增强
3. **意图识别**：分类模型、实体提取、提示词工程
4. **词库管理**：专有名称、敏感词过滤

### 阶段3：系统集成
- API开发、前后端联调
- 性能优化、缓存策略
- 安全加固、敏感词过滤

### 阶段4：测试上线
- 单元测试、集成测试
- 部署方案、监控告警

## 关键文件位置

- **需求文档**：`requirements.md`
- **原型截图**：`docs/prototype_screenshots/*.png`
- **技术规划**：`docs/数字员工项目-AI研发工作规划指南.md`
- **FAQ设计**：`docs/关于FAQ问答库.md`
- **对话风格**：`docs/关于Agent对话风格分类.md`

## AI代理工作建议

### 生成代码时
1. **优先参考**规划指南中的代码模板和架构设计
2. **使用LangGraph**构建状态机工作流，避免传统的if-else逻辑
3. **使用异步模式**处理LLM调用（支持流式响应）
4. **实现会话管理**时使用MongoDB持久化对话记录
5. **知识库检索**采用Chroma向量检索 + ElasticSearch关键词检索的混合策略
6. **关注准确性**：添加检索结果相似度阈值、答案质量评估、专业词库增强

### 回答问题时
1. 根据用户背景选择合适的**对话风格**（见文档分类）
2. 提供**具体代码示例**而非泛泛而谈
3. 对比多个方案时使用**表格或对比图**
4. 技术选型时说明**优缺点和适用场景**

### 分析需求时
1. 识别**AI能力需求**（LLM、向量检索、意图识别等）
2. 区分**核心功能**与**辅助功能**，优先级排序
3. 提出**技术风险**和**替代方案**
4. 考虑**性能优化**和**成本控制**（API调用费用）

## 注意事项

- ⚠️ 项目当前**无代码实现**，仅处于规划阶段
- ⚠️ 原型截图显示了UI设计方向，但未定义API接口
- ⚠️ 技术栈选型为**建议方案**，需根据实际情况调整
- ⚠️ 重点关注**AI研发工作**，后端架构配合工程师完成
