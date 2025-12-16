# 数字员工项目 - AI 代理开发指南

> 基于 LangGraph 的数字员工 AI 服务开发指南，面向 AI 编程助手。
> 
> **最后更新**: 2025-12-16  
> **项目状态**: 开发中 (Phase 1)

---

## 项目概述

数字员工 AI 服务，提供：
- **对话引擎**: 基于 LangGraph 的多轮对话管理
- **知识检索**: RAG（向量 + 全文混合检索）
- **联网检索**: 实时信息查询（Tavily）
- **合规过滤**: 敏感词检测与拦截
- **多模态**: 支持流式（SSE）和同步对话

## 技术栈

- **后端**: FastAPI + LangGraph
- **AI**: OpenAI/Claude/DeepSeek/Qwen
- **存储**: MongoDB + Chroma（向量）+ ElasticSearch（全文）
- **联网**: Tavily API
- **容器化**: Docker + Docker Compose

## 项目结构

```
ai-service/
├── app/
│   ├── api/endpoints/      # API 端点
│   ├── core/              # 配置、数据库、日志
│   ├── models/            # 数据模型
│   ├── services/          # 业务逻辑
│   └── utils/             # 工具函数
├── tests/                 # 测试
├── main.py               # 应用入口
└── requirements.txt      # 依赖
```

## 核心工作流（LangGraph）

节点链路:
```
意图识别 → 实时性判断 → 知识库检索/联网检索 → 答案生成 → 敏感词过滤
```

**关键节点**:
1. **intent_recognition**: 快速意图分类（greeting/faq/complaint等）
2. **check_realtime_query**: 检测是否需要实时信息（天气/股价/新闻等）
3. **rag_search**: 混合检索（向量相似度 + 关键词匹配），RRF融合排序
4. **web_search**: 联网降级（实时查询或低相关性时触发）
5. **answer_generation**: LLM流式生成
6. **sensitive_filter**: 敏感词双向检测（输入+输出）

## 开发规范

### API 设计
- 统一响应格式: `{ code, message, data, error }`
- 流式接口使用 SSE (Server-Sent Events)
- 错误码标准化（400/401/403/429/500/503）

### 代码约定
- 所有 API 使用 `async/await`
- LangGraph 状态使用 `TypedDict` 定义
- 中间件顺序: CORS → Auth → RateLimit → ErrorHandler
- 配置通过 `pydantic.BaseSettings` 管理

### 检索规则
- 相关性阈值: 0.6（低于触发联网）
- 置信度阈值: 0.7
- 混合检索: Chroma（向量） + ElasticSearch（全文） + RRF融合
- 实时查询: 直接联网，跳过知识库

### 安全要求
- 输入验证: query ≤ 1000 字符，必填字段检查
- 敏感词: 3级拦截（Critical拒绝/Warning替换/Notice记录）
- 鉴权: JWT + API Key 双模式
- 限流: 令牌桶算法，用户级 + 全局级

## 核心配置

关键环境变量 (`.env`):
```bash
# LLM
OPENAI_API_KEY=sk-...
DEFAULT_MODEL=gpt-4o-mini

# 数据库
MONGODB_URL=mongodb://localhost:27017
CHROMA_URL=http://localhost:8001
ELASTICSEARCH_URL=http://localhost:9200

# 联网检索
TAVILY_API_KEY=tvly-...

# 安全
JWT_SECRET_KEY=...
JWT_ALGORITHM=HS256
JWT_EXPIRY=3600

# 阈值
RELEVANCE_THRESHOLD=0.6
CONFIDENCE_THRESHOLD=0.7
```

## 开发工作流

### 本地开发
```bash
# 启动数据库
docker-compose up -d mongodb elasticsearch chroma

# 安装依赖
cd ai-service && pip install -r requirements.txt

# 运行服务
uvicorn main:app --reload --port 8000
```

### 测试
```bash
pytest
pytest --cov=app --cov-report=html
```

### 部署
```bash
docker-compose up -d
```

## 重要文档参考

- **详细需求**: `docs/需求补充与完善文档.md`（3000+ 行完整需求）
- **API 文档**: `docs/API使用文档.md`
- **部署指南**: `docs/部署指南.md`
- **项目总结**: `docs/项目总结.md`
- **README**: `README.md`

## 开发阶段

- [x] **Phase 1**: FastAPI 框架 + LangGraph 骨架 + SSE
- [ ] **Phase 2**: 知识库管理（上传/切片/向量化/检索）
- [ ] **Phase 3**: 敏感词集成 + 联网检索 + 审计日志
- [ ] **Phase 4**: 性能优化 + 监控 + 前端联调

## AI 助手提示

1. **优先参考**: 所有实现前先查 `docs/需求补充与完善文档.md` 的详细接口和字段定义
2. **代码生成**: 必须包含鉴权、限流、错误处理、SSE、敏感词检测等必备逻辑
3. **问题回答**: 引用具体文件/段落，提供 JSON/HTTP/LangGraph 代码示例
4. **容器化**: 部署脚本需包含 Vue + ai-service + MongoDB + Chroma + ElasticSearch
5. **歧义处理**: 记录假设并显式说明，便于后续复核

---

**注意**: 本文档为简化版指南。详细的 API 规范、数据模型、工作流实现、数据库 schema 等请参阅 `docs/` 目录下的完整文档。
