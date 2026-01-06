# Digital Employee AI Service - Claude 指导文档

## 项目概述

这是一个基于 LangGraph 的**数字员工 AI 服务**，提供智能对话、知识库检索（RAG）、联网搜索、多轮对话管理等功能。

**核心价值**：为企业提供可定制的 AI 数字员工，支持不同个性、知识库和专业领域，实现智能客服和知识问答。

**技术栈**：FastAPI + LangGraph + OpenAI/Ollama + ChromaDB + ElasticSearch + MongoDB

**主要功能**：
- 智能对话（同步/流式）
- RAG 知识库检索（向量 + 关键词混合搜索）
- 联网搜索（Tavily API）
- FAQ 问答匹配
- 数字员工管理（个性化配置）
- 文档上传与向量化

---

## 项目结构

```
ZengKingMorphe/
├── ai-service/                    # 主应用目录
│   ├── app/
│   │   ├── api/                  # API 层
│   │   │   ├── endpoints/        # 各功能模块端点
│   │   │   │   ├── chat.py       # 对话接口（核心）
│   │   │   │   ├── session.py    # 会话管理
│   │   │   │   ├── employee.py   # 数字员工管理
│   │   │   │   ├── knowledge_base.py  # 知识库操作
│   │   │   │   ├── conversation.py    # 对话记录查询
│   │   │   │   └── webhook.py    # 外部同步通知
│   │   │   └── middleware/       # 中间件（鉴权、限流、错误处理）
│   │   ├── core/                 # 核心基础设施
│   │   │   ├── config.py         # 配置管理（Pydantic Settings）
│   │   │   ├── database.py       # MongoDB 连接
│   │   │   ├── chroma.py         # ChromaDB 连接
│   │   │   ├── elasticsearch.py  # ElasticSearch 连接
│   │   │   └── logging.py        # 日志配置（Loguru）
│   │   ├── models/               # 数据模型
│   │   │   ├── database.py       # MongoDB 模型
│   │   │   └── schemas.py        # API 请求/响应 Schema
│   │   ├── services/             # 业务逻辑层（核心）
│   │   │   ├── conversation_service.py  # LangGraph 对话工作流
│   │   │   ├── rag_service.py           # RAG 检索服务
│   │   │   ├── document_service.py      # 文档处理
│   │   │   └── task_processor.py        # 异步任务队列
│   │   └── utils/                # 工具函数
│   ├── tests/                    # 测试文件
│   ├── scripts/                  # 脚本工具
│   ├── main.py                   # 应用入口
│   ├── requirements.txt          # Python 依赖
│   └── .env.example             # 环境变量模板
├── docs/                         # 项目文档（中文）
├── docker-compose.yml            # 多服务编排
└── README.md                     # 项目说明
```

**核心文件说明**：
- [main.py](ai-service/main.py) - FastAPI 应用入口，管理数据库生命周期
- [conversation_service.py](ai-service/app/services/conversation_service.py) - LangGraph 工作流（12 个节点，800+ 行）
- [rag_service.py](ai-service/app/services/rag_service.py) - 混合搜索（RRF 融合算法）
- [chat.py](ai-service/app/api/endpoints/chat.py) - 对话 API 端点（流式/非流式）
- [config.py](ai-service/app/core/config.py) - 环境变量配置（50+ 配置项）

---

## 关键约定 ⚠️

### 1. 虚拟环境使用（非常重要！）

**所有 Python 命令必须使用虚拟环境**，依赖安装在虚拟环境而非全局。

**Windows PowerShell**：
```powershell
# 激活虚拟环境
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/Activate.ps1

# 或直接使用虚拟环境的 Python
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe script.py
```

**Git Bash/WSL**：
```bash
source D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/activate
# 或
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python script.py
```

**错误示例**（不要这样）：
```bash
python script.py  # ❌ 使用全局 Python，依赖缺失
```

**正确示例**：
```bash
cd ai-service
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe test_web_search.py
```

### 2. 日志规范

使用 Loguru 结构化日志，关键字段使用命名参数：

```python
from app.core.logging import logger

logger.info("Chat message request", user_id=user.id, query=query[:100])  # 截断 PII
logger.error("RAG search failed", error=str(e), exc_info=True)  # 异常日志包含堆栈
```

**规则**：
- ✅ 使用关键字参数，不要字符串拼接
- ✅ 用户查询截断到 100 字符（防止 PII 泄露）
- ✅ 错误日志必须包含 `exc_info=True`
- ❌ 不要：`logger.info(f"User {user.id} asked: {query}")`

### 3. API 响应格式

所有 API 响应遵循统一结构：

```python
class ChatResponse(BaseModel):
    code: int = 200  # 状态码
    message: str = "success"  # 消息
    data: ChatResponseData  # 实际数据
```

**关键 Schema**（[schemas.py](ai-service/app/models/schemas.py)）：
- `ChatRequest` / `ChatResponse` - 对话请求和响应
- `SourceAttribution` - 来源归属（RAG 来源 + 网络来源）
- `StreamChunkResponse` - SSE 流式数据块
- 所有模型使用 Pydantic v2

### 4. 错误处理

全局异常处理器在 [error_handler.py](ai-service/app/api/middleware/error_handler.py)：

- `HTTPException` → JSON 响应 + 状态码
- `RequestValidationError` → 422 错误 + 字段级验证信息
- 通用异常 → 500 错误 + 日志记录堆栈

**模式**：不要捕获所有异常后静默处理，应该抛出 HTTPException 或记录日志后返回明确错误。

### 5. 认证状态（重要！）

**当前认证已禁用**：
- 中间件存在但 `settings.api_keys` 列表为空
- 所有端点使用 `Depends(get_api_key)` 但实际不验证
- 这是遗留代码，TODO: 启用认证或移除相关代码

---

## 核心工作流程

### LangGraph 对话流程

对话流程是 12 个节点的状态图（[conversation_service.py](ai-service/app/services/conversation_service.py)）：

```
用户查询 → 加载配置 → 加载会话上下文 → FAQ 匹配 → 实时检测
   ↓
意图识别 → RAG 检索 → 文档评分 → 联网搜索（fallback）→ LLM 生成 → 持久化
```

**关键路由逻辑**：
- **FAQ 匹配成功** → 跳过 RAG，直接生成答案
- **实时查询**（天气/新闻/股价）→ 跳过 RAG，直接联网搜索
- **RAG 相关性低**（<0.6）→ 触发联网搜索补充
- **其他** → RAG 检索 + LLM 生成

**状态管理**：通过 `ConversationState` TypedDict 传递 22 个字段，包括查询上下文、检索结果、元数据等。

### RAG 混合检索

**三层存储**（[rag_service.py](ai-service/app/services/rag_service.py)）：
1. **ChromaDB** - 向量搜索（语义相似度）
2. **ElasticSearch** - BM25 关键词搜索（精确匹配）
3. **MongoDB** - 文档元数据和完整记录

**搜索融合（RRF 算法）**：
```python
rrf_score = 1 / (rank + 60)  # 默认公式
```
- 向量结果和关键词结果按 RRF 分数融合
- 按 `doc_id` 去重
- 取 Top-K 结果

### 数据库生命周期

所有数据库在 FastAPI lifespan 中管理（[main.py](ai-service/main.py)）：

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    await mongodb.connect()      # 异步连接
    chroma_db.connect()          # HTTP 客户端模式（同步）
    await es_db.connect()        # 异步连接
    await task_processor.start() # 后台任务队列
    yield
    # 自动清理
```

**注意**：ChromaDB 使用 HTTP 客户端模式（`chroma_host:chroma_port`），生产环境不用本地持久模式。

---

## 开发模式

### 本地开发

**步骤 1：启动数据库**
```powershell
docker-compose up -d mongodb elasticsearch chroma
```

**步骤 2：激活虚拟环境并运行**
```powershell
cd ai-service
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**API 文档**：http://localhost:8000/docs

### 测试

**运行测试**（必须使用虚拟环境）：
```powershell
cd ai-service
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe -m pytest tests/
```

**独立测试脚本**（不使用 pytest）：
```powershell
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe tests/test_web_search.py
```

**主要测试文件**：
- `test_web_search.py` - Tavily API 测试
- `test_chroma_standalone.py` - ChromaDB 连接测试
- `test_ollama_embedding.py` - Ollama 嵌入测试
- `test_stream_chunks.py` - 流式数据块存储测试
- `stream_client.py` ([scripts/](ai-service/scripts/)) - SSE 客户端

### Docker 部署

**完整部署**：
```powershell
docker-compose up -d  # 启动 4 个服务
docker-compose logs -f ai-service  # 查看日志
```

**端口映射**（外部:容器）：
- AI Service: `8100:8000`
- MongoDB: `27117:27017`
- ChromaDB: `8101:8000`
- ElasticSearch: `9320:9200`

---

## 重要提醒

### 已知限制

1. **敏感词过滤**：AC 自动机引擎未实现
2. **认证**：中间件存在但实际禁用（`api_keys` 为空）
3. **测试覆盖率**：较低，主要是集成测试
4. **限流**：中间件存在但未强制执行

### TODO 项

- 启用 API Key 认证或移除相关代码
- 实现敏感词过滤引擎
- 提高测试覆盖率
- FAQ 匹配改用语义相似度（当前仅关键词匹配）

### 文档参考

详细文档见 `docs/` 目录：
- [联网检索功能使用指南.md](docs/联网检索功能使用指南.md)
- [FAQ多路召回功能实现总结.md](docs/FAQ多路召回功能实现总结.md)
- [异步文档上传使用说明.md](docs/异步文档上传使用说明.md)
- [API使用文档.md](docs/API使用文档.md)
- [部署指南.md](docs/部署指南.md)

### 外部集成

**Java 平台集成**（通过 `JAVA_API_BASE_URL`）：
- 员工配置验证：`POST /api/ai/employee/config`
- FAQ/敏感词同步 Webhook（[webhook.py](ai-service/app/api/endpoints/webhook.py)）

**测试模式**：对于 `employee_id="hutao"`，会加载本地测试数据（[outer_api_docs/](outer_api_docs/)），跳过 Java API 调用。

---

## 快速参考

**常用命令**：
```powershell
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f ai-service

# 本地运行（仅启动数据库）
docker-compose up -d mongodb elasticsearch chroma
cd ai-service
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/Activate.ps1
uvicorn main:app --reload

# 运行测试
cd ai-service
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe -m pytest tests/

# 停止服务
docker-compose down
```

**环境变量配置**（[.env.example](ai-service/.env.example)）：
```bash
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
MONGODB_URL=mongodb://localhost:27017
JWT_SECRET_KEY=your-jwt-secret-key
USE_OLLAMA=false  # true=本地 Ollama, false=OpenAI API
```

**LLM 切换**：
- OpenAI API：`USE_OLLAMA=false`, `OPENAI_API_KEY=sk-...`
- Ollama 本地：`USE_OLLAMA=true`, `OLLAMA_BASE_URL=http://localhost:11434`

---

## 关键文件路径速查

| 功能 | 文件路径 |
|------|---------|
| LangGraph 工作流 | [ai-service/app/services/conversation_service.py](ai-service/app/services/conversation_service.py) |
| RAG 检索 | [ai-service/app/services/rag_service.py](ai-service/app/services/rag_service.py) |
| 对话 API | [ai-service/app/api/endpoints/chat.py](ai-service/app/api/endpoints/chat.py) |
| 配置管理 | [ai-service/app/core/config.py](ai-service/app/core/config.py) |
| 数据模型 | [ai-service/app/models/schemas.py](ai-service/app/models/schemas.py) |
| 应用入口 | [ai-service/main.py](ai-service/main.py) |
| 异步任务 | [ai-service/app/services/task_processor.py](ai-service/app/services/task_processor.py) |
| Docker 编排 | [docker-compose.yml](docker-compose.yml) |
| 项目说明 | [README.md](README.md) |
| Copilot 指导 | [.github/copilot-instructions.md](.github/copilot-instructions.md) |
