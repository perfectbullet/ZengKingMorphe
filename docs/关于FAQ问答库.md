## 📚 什么是 FAQ 问答库

**FAQ（Frequently Asked Questions）问答库** 是一个结构化的知识库，存储了常见问题及其标准答案。Agent 可以通过检索这个库来快速、准确地回答用户问题。

### 🎯 核心价值
- ✅ **一致性**：确保相同问题得到统一答案
- ✅ **准确性**：避免 AI 幻觉（hallucination）
- ✅ **时效性**：及时更新业务规则和政策
- ✅ **可控性**：人工审核标准答案，质量可控
- ✅ **可追溯**：知道答案来源，便于改进

---

## 🏗️ FAQ 问答库的构建方法

### 方案一：基于向量数据库的语义检索（推荐）⭐

#### 架构图
```
用户问题 → Embedding → 向量检索 → 匹配FAQ → 返回答案
                ↓
            向量数据库
         (Pinecone/Qdrant/
          Chroma/Milvus)
```

#### 实现步骤

**1. 数据准备**
```json
// faq_data.json
[
  {
    "id": "faq_001",
    "question": "如何重置密码？",
    "answer":  "请点击登录页面的'忘记密码'链接，输入注册邮箱，我们会发送重置链接到您的邮箱。",
    "category": "账户管理",
    "keywords": ["密码", "重置", "忘记"],
    "related_questions": ["faq_002", "faq_003"]
  },
  {
    "id": "faq_002",
    "question":  "支持哪些支付方式？",
    "answer": "我们支持：1) 支付宝 2) 微信支付 3) 信用卡（Visa/MasterCard）4) 银行转账",
    "category":  "支付相关",
    "keywords": ["支付", "付款", "充值"]
  }
]
```

**2. 生成向量嵌入**
```python
from openai import OpenAI
import json

client = OpenAI()

def create_embeddings(faq_data):
    """将 FAQ 转换为向量"""
    embeddings = []
    
    for item in faq_data:
        # 组合问题和关键词以提高检索准确性
        text = f"{item['question']} {' '.join(item['keywords'])}"
        
        response = client. embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        
        embeddings.append({
            "id": item["id"],
            "vector": response.data[0].embedding,
            "metadata": item
        })
    
    return embeddings

# 加载 FAQ 数据
with open('faq_data.json', 'r', encoding='utf-8') as f:
    faq_data = json.load(f)

# 生成向量
embeddings = create_embeddings(faq_data)
```

**3. 存储到向量数据库**
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 初始化 Qdrant
client = QdrantClient(url="http://localhost:6333")

# 创建集合
client.create_collection(
    collection_name="faq_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# 插入数据
points = [
    PointStruct(
        id=idx,
        vector=emb["vector"],
        payload=emb["metadata"]
    )
    for idx, emb in enumerate(embeddings)
]

client.upsert(collection_name="faq_collection", points=points)
```

**4. 检索与回答**
```python
def search_faq(user_question, top_k=3, threshold=0.75):
    """检索最相关的 FAQ"""
    
    # 将用户问题转为向量
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_question
    )
    query_vector = response.data[0].embedding
    
    # 向量检索
    search_result = client.search(
        collection_name="faq_collection",
        query_vector=query_vector,
        limit=top_k
    )
    
    # 过滤低相似度结果
    results = []
    for hit in search_result: 
        if hit.score >= threshold:
            results.append({
                "question": hit.payload["question"],
                "answer": hit.payload["answer"],
                "category": hit.payload["category"],
                "confidence": hit.score
            })
    
    return results

# 使用示例
user_query = "我忘记了登录密码怎么办？"
answers = search_faq(user_query)

for ans in answers:
    print(f"匹配度:  {ans['confidence']:.2f}")
    print(f"问题: {ans['question']}")
    print(f"答案: {ans['answer']}\n")
```

---

### 方案二：基于关键词的精确匹配

#### 适用场景
- FAQ 数量少（<100条）
- 问题表述标准化
- 需要极快响应速度

#### 实现代码
```python
import re
from collections import defaultdict

class KeywordFAQ:
    def __init__(self, faq_data):
        self.faq_data = faq_data
        self.keyword_index = self._build_index()
    
    def _build_index(self):
        """构建关键词倒排索引"""
        index = defaultdict(list)
        
        for item in self.faq_data:
            for keyword in item['keywords']:
                index[keyword. lower()].append(item['id'])
        
        return index
    
    def search(self, query):
        """基于关键词匹配"""
        query_lower = query.lower()
        matched_ids = set()
        
        # 查找包含关键词的 FAQ
        for keyword, ids in self.keyword_index. items():
            if keyword in query_lower:
                matched_ids.update(ids)
        
        # 返回匹配的 FAQ
        results = [
            faq for faq in self.faq_data 
            if faq['id'] in matched_ids
        ]
        
        return results

# 使用示例
faq_system = KeywordFAQ(faq_data)
results = faq_system.search("如何付款")
```

---

### 方案三：混合检索（最佳实践）🏆

结合向量检索 + 关键词过滤 + 规则引擎

```python
class HybridFAQ:
    def __init__(self, vector_db, faq_data):
        self.vector_db = vector_db
        self. faq_data = faq_data
        self.rules = self._load_rules()
    
    def _load_rules(self):
        """加载业务规则"""
        return {
            "urgent_keywords": ["紧急", "无法登录", "支付失败"],
            "category_boost": {
                "账户安全": 1.2,
                "支付相关": 1.1
            }
        }
    
    def search(self, query, context=None):
        # Step 1: 规则优先匹配
        for keyword in self.rules["urgent_keywords"]:
            if keyword in query: 
                return self._get_urgent_faq(keyword)
        
        # Step 2: 向量语义检索
        candidates = self.vector_db.search(query, top_k=10)
        
        # Step 3: 根据上下文重新排序
        if context: 
            candidates = self._rerank_by_context(candidates, context)
        
        # Step 4: 应用分类权重
        for item in candidates:
            category = item['category']
            if category in self.rules["category_boost"]:
                item['score'] *= self.rules["category_boost"][category]
        
        # Step 5: 返回 Top 3
        candidates. sort(key=lambda x: x['score'], reverse=True)
        return candidates[:3]
```

---

## 🔧 集成到 Agent 的方法

### 方法 1: Function Calling（推荐）

```python
# 定义 FAQ 检索工具
tools = [
    {
        "type": "function",
        "function": {
            "name":  "search_faq",
            "description": "在 FAQ 知识库中搜索问题答案",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户的问题"
                    },
                    "category": {
                        "type":  "string",
                        "enum": ["账户管理", "支付相关", "功能使用"],
                        "description": "问题分类（可选）"
                    }
                },
                "required":  ["query"]
            }
        }
    }
]

# Agent 调用示例
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "我忘记密码了"}
    ],
    tools=tools
)

# 如果 Agent 决定调用 FAQ 工具
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    
    # 执行 FAQ 检索
    faq_results = search_faq(args['query'])
    
    # 将结果返回给 Agent
    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "我忘记密码了"},
            response.choices[0].message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content":  json.dumps(faq_results, ensure_ascii=False)
            }
        ]
    )
```

### 方法 2: RAG (Retrieval Augmented Generation)

```python
def rag_answer(user_question):
    # 1. 检索相关 FAQ
    relevant_faqs = search_faq(user_question, top_k=3)
    
    # 2. 构建增强上下文
    context = "\n\n".join([
        f"Q: {faq['question']}\nA: {faq['answer']}"
        for faq in relevant_faqs
    ])
    
    # 3. 生成回答
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"""你是客服助手。请基于以下 FAQ 知识库回答问题。
                如果知识库中没有相关信息，请诚实告知。
                
                FAQ 知识库：
                {context}
                """
            },
            {
                "role": "user",
                "content": user_question
            }
        ]
    )
    
    return response.choices[0].message.content
```

### 方法 3: Prompt 预填充

```python
system_prompt = f"""
你是智能客服助手。以下是常见问题解答：

{load_faq_as_text()}

请优先使用上述 FAQ 回答用户问题。如果 FAQ 中没有相关信息，
再使用你的知识回答，并标注"此信息来自通用知识，非官方 FAQ"。
"""
```

---

## 📊 FAQ 数据结构设计

### 基础结构
```json
{
  "id": "unique_id",
  "question": "标准问题",
  "answer": "标准答案",
  "category":  "分类",
  "keywords": ["关键词数组"],
  "created_at": "2025-01-01",
  "updated_at": "2025-12-11",
  "version": "1.2"
}
```

### 进阶结构
```json
{
  "id": "faq_advanced_001",
  "question": "如何申请退款？",
  "answer":  "{{answer_template}}",
  "answer_template": "根据您的订单状态（{order_status}），退款流程为：{refund_steps}",
  "dynamic_fields": ["order_status", "refund_steps"],
  "category": "售后服务",
  "subcategory": "退款相关",
  "keywords": ["退款", "退货", "取消订单"],
  "intent": "refund_request",
  "confidence_threshold": 0.8,
  "related_faqs": ["faq_002", "faq_015"],
  "follow_up_questions": [
    "需要我帮您查询订单状态吗？",
    "您是因为什么原因申请退款？"
  ],
  "escalation_rule": {
    "condition": "refund_amount > 1000",
    "action": "transfer_to_human"
  },
  "multimedia":  {
    "video_tutorial": "https://example.com/videos/refund. mp4",
    "images": ["step1.png", "step2.png"]
  },
  "language": "zh-CN",
  "alternatives": {
    "en": "How to request a refund?",
    "ja": "返金を申請するには？"
  },
  "metadata": {
    "view_count": 1523,
    "helpful_votes": 142,
    "last_reviewed": "2025-12-01",
    "reviewer":  "admin@example.com"
  }
}
```

---

## 🛠️ 推荐工具栈

| 组件 | 开源方案 | 商业方案 |
|------|----------|----------|
| **向量数据库** | Qdrant, Milvus, Chroma | Pinecone, Weaviate Cloud |
| **Embedding 模型** | sentence-transformers | OpenAI, Cohere |
| **检索框架** | LangChain, LlamaIndex | - |
| **FAQ 管理平台** | 自建后台 | Zendesk, Intercom |
| **评估工具** | RAGAS, TruLens | - |

---

## 📈 FAQ 质量评估

```python
def evaluate_faq_system(test_cases):
    """评估 FAQ 系统性能"""
    metrics = {
        "accuracy": 0,      # 准确率
        "coverage": 0,      # 覆盖率
        "avg_confidence": 0 # 平均置信度
    }
    
    correct = 0
    total = len(test_cases)
    confidences = []
    
    for test in test_cases:
        results = search_faq(test['question'])
        
        if results:
            # 检查是否匹配正确答案
            if results[0]['id'] == test['expected_faq_id']:
                correct += 1
            confidences.append(results[0]['confidence'])
        
    metrics["accuracy"] = correct / total
    metrics["coverage"] = len([r for r in test_cases if search_faq(r['question'])]) / total
    metrics["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0
    
    return metrics
```

---

## 🎯 最佳实践建议

1. **持续更新**：每月审核 FAQ，添加新问题
2. **A/B 测试**：测试不同答案的用户满意度
3. **用户反馈**：允许用户标记答案是否有帮助
4. **多语言支持**：使用多语言 Embedding 模型
5. **版本控制**：像代码一样管理 FAQ 变更
6. **监控指标**：跟踪命中率、准确率、用户满意度

需要我帮您构建一个具体场景的 FAQ 问答库吗？比如针对您之前提到的 txt2img 异步接口的 FAQ？