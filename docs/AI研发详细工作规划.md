# AI研发详细工作规划

## 文档信息
- **版本**：v1.0
- **创建日期**：2025-12-13
- **适用角色**：AI方向研发工程师
- **技术栈**：LangGraph + MongoDB + ElasticSearch + Chroma

---

## 一、工作规划总览

### 1.1 项目背景

**数字员工产品**是一个企业级AI对话助手平台，核心目标是提升Agent回答的**准确性**。

**团队配置**：4人小团队（1前端 + 1AI研发 + 2其他）

**AI研发职责范围**：
- ✅ 完整的后端服务开发（FastAPI REST API + WebSocket）
- ✅ 业务服务层（对话管理、知识库管理、任务编排）
- ✅ AI核心引擎（LangGraph工作流 + 对话引擎 + 知识检索）
- ✅ 数据库集成（MongoDB + Chroma + ElasticSearch）

### 1.2 技术架构

```
┌─────────────┐
│  Vue前端界面 │
└──────┬──────┘
       │ REST API / WebSocket
┌──────▼───────────────────────────────┐
│      Python AI引擎服务            │
│   (FastAPI + LangGraph)          │
│      (AI研发负责)                │
├──────────────────────────────────────┤
│  业务服务层：                      │
│  - 对话管理服务（会话/上下文）      │
│  - 知识库管理服务（文档/向量化）    │
│  - 任务编排服务（异步调度）        │
├──────────────────────────────────────┤
│  LangGraph工作流编排：            │
│  - 对话引擎（多轮对话/流式响应）   │
│  - 知识库RAG（向量检索/上下文增强）│
│  - 意图识别（分类/实体提取）       │
└──────┬───────────────────────────────┘
       │
   ┌───┴────┬─────────┬──────────┐
   │        │         │          │
┌──▼──┐ ┌──▼──┐  ┌───▼───┐ ┌───▼────┐
│MongoDB│ │Chroma│  │ElasticSearch│ │外部LLM│
└─────┘ └─────┘  └──────┘ └────────┘
```

### 1.3 开发时间规划

| 阶段 | 时间 | 目标 |
|------|------|------|
| Phase 1 | 2周 | 基础服务搭建 |
| Phase 2 | 2周 | 知识库与检索 |
| Phase 3 | 1.5周 | 意图识别与质量保障 |
| Phase 4 | 1.5周 | 系统集成与优化 |
| **总计** | **7周** | **完整AI引擎** |

---

## 二、Phase 1：基础服务搭建（2周）

### 目标
搭建FastAPI服务框架，实现基本对话能力

### 2.1 第1-2天：项目初始化

#### 任务清单
- [ ] 创建项目目录结构
- [ ] 配置开发环境
- [ ] 初始化Git仓库
- [ ] 配置依赖管理

#### 项目目录结构
```
digital-employee/
├── ai-service/                 # AI引擎服务
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI应用入口
│   │   ├── config.py          # 配置管理
│   │   ├── api/               # API路由
│   │   │   ├── __init__.py
│   │   │   ├── chat.py        # 对话接口
│   │   │   ├── knowledge.py   # 知识库接口
│   │   │   └── admin.py       # 管理接口
│   │   ├── services/          # 业务服务层
│   │   │   ├── __init__.py
│   │   │   ├── conversation.py    # 对话管理
│   │   │   ├── knowledge_base.py  # 知识库管理
│   │   │   └── task_manager.py    # 任务编排
│   │   ├── core/              # 核心模块
│   │   │   ├── __init__.py
│   │   │   ├── langgraph_workflow.py  # LangGraph工作流
│   │   │   ├── dialogue_engine.py     # 对话引擎
│   │   │   ├── retrieval.py           # 知识检索
│   │   │   └── intent.py              # 意图识别
│   │   ├── models/            # 数据模型
│   │   │   ├── __init__.py
│   │   │   ├── conversation.py
│   │   │   ├── knowledge.py
│   │   │   └── intent.py
│   │   ├── utils/             # 工具函数
│   │   │   ├── __init__.py
│   │   │   ├── logger.py
│   │   │   ├── db.py          # 数据库工具
│   │   │   └── llm_client.py  # LLM客户端
│   │   └── middleware/        # 中间件
│   │       ├── __init__.py
│   │       ├── auth.py        # 鉴权
│   │       ├── rate_limit.py  # 限流
│   │       └── error_handler.py
│   ├── tests/                 # 测试
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
├── frontend/                  # 前端（由前端工程师负责）
├── docker-compose.yml
└── README.md
```

#### 依赖管理（requirements.txt）
```txt
# Web框架
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
python-multipart==0.0.6

# LangGraph
langgraph==0.0.25
langchain==0.1.4
langchain-openai==0.0.5

# 数据库
pymongo==4.6.1
motor==3.3.2  # 异步MongoDB驱动

# 向量数据库
chromadb==0.4.22

# 全文检索
elasticsearch==8.11.0

# LLM客户端
openai==1.10.0

# 异步任务
celery==5.3.6
redis==5.0.1

# 工具库
python-dotenv==1.0.0
pydantic-settings==2.1.0
loguru==0.7.2
tenacity==8.2.3  # 重试
httpx==0.26.0

# 开发工具
pytest==7.4.4
pytest-asyncio==0.23.3
black==23.12.1
ruff==0.1.13
```

#### 配置管理（app/config.py）
```python
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """应用配置"""
    
    # 应用配置
    APP_NAME: str = "Digital Employee AI Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API配置
    API_V1_PREFIX: str = "/api/v1"
    API_PORT: int = 8000
    
    # 数据库配置
    MONGODB_URI: str = Field(..., env="MONGODB_URI")
    MONGODB_DB_NAME: str = "digital_employee"
    
    # Chroma配置
    CHROMA_HOST: str = Field(default="localhost", env="CHROMA_HOST")
    CHROMA_PORT: int = Field(default=8001, env="CHROMA_PORT")
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    
    # ElasticSearch配置
    ES_HOST: str = Field(default="localhost", env="ES_HOST")
    ES_PORT: int = Field(default=9200, env="ES_PORT")
    
    # LLM配置
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    LLM_MODEL: str = "gpt-4"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2000
    
    # Redis配置（Celery）
    REDIS_URI: str = "redis://localhost:6379/0"
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # 安全配置
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # 限流配置
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

### 2.2 第3-5天：FastAPI服务框架

#### 任务清单
- [ ] FastAPI应用初始化
- [ ] REST API路由设计
- [ ] WebSocket接口实现
- [ ] 鉴权中间件
- [ ] 错误处理中间件

#### FastAPI应用入口（app/main.py）
```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn

from app.config import settings
from app.api import chat, knowledge, admin
from app.middleware.error_handler import error_handler_middleware
from app.utils.logger import setup_logger
from app.utils.db import init_databases, close_databases

# 设置日志
logger = setup_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("Starting Digital Employee AI Service...")
    await init_databases()
    logger.info("Databases initialized")
    
    yield
    
    # 关闭时执行
    logger.info("Shutting down...")
    await close_databases()
    logger.info("Databases closed")

# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 错误处理中间件
app.middleware("http")(error_handler_middleware)

# 注册路由
app.include_router(chat.router, prefix=f"{settings.API_V1_PREFIX}/chat", tags=["Chat"])
app.include_router(knowledge.router, prefix=f"{settings.API_V1_PREFIX}/knowledge", tags=["Knowledge"])
app.include_router(admin.router, prefix=f"{settings.API_V1_PREFIX}/admin", tags=["Admin"])

@app.get("/")
async def root():
    """根路径"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
```

#### 对话API路由（app/api/chat.py）
```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import json

from app.services.conversation import ConversationService
from app.middleware.auth import get_current_user
from app.utils.logger import logger

router = APIRouter()
conversation_service = ConversationService()

class ChatRequest(BaseModel):
    """对话请求"""
    message: str
    session_id: Optional[str] = None
    stream: bool = True

class ChatResponse(BaseModel):
    """对话响应"""
    session_id: str
    message: str
    confidence: float
    sources: list

@router.post("/message", response_model=ChatResponse)
async def chat_message(
    request: ChatRequest,
    user = Depends(get_current_user)
):
    """同步对话接口"""
    try:
        result = await conversation_service.process_message(
            user_id=user["user_id"],
            message=request.message,
            session_id=request.session_id,
            stream=False
        )
        return result
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    user = Depends(get_current_user)
):
    """流式对话接口"""
    try:
        async def generate():
            async for chunk in conversation_service.process_message(
                user_id=user["user_id"],
                message=request.message,
                session_id=request.session_id,
                stream=True
            ):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    """WebSocket实时对话"""
    await websocket.accept()
    session_id = None
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # 创建会话（首次连接）
            if not session_id:
                session_id = await conversation_service.create_session(
                    user_id=message_data.get("user_id"),
                    system_prompt="你是一个数字员工助手"
                )
                await websocket.send_json({
                    "type": "session_created",
                    "session_id": session_id
                })
            
            # 处理消息
            async for chunk in conversation_service.process_message(
                user_id=message_data.get("user_id"),
                message=message_data.get("message"),
                session_id=session_id,
                stream=True
            ):
                await websocket.send_json(chunk)
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        if session_id:
            await conversation_service.end_session(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1000)

@router.get("/session/{session_id}")
async def get_session(
    session_id: str,
    user = Depends(get_current_user)
):
    """获取会话信息"""
    session = await conversation_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    user = Depends(get_current_user)
):
    """结束会话"""
    await conversation_service.end_session(session_id)
    return {"message": "Session ended"}
```

### 2.3 第6-8天：LangGraph工作流搭建

#### 任务清单
- [ ] 定义对话状态
- [ ] 实现核心节点
- [ ] 构建工作流图
- [ ] 测试工作流

#### LangGraph工作流（app/core/langgraph_workflow.py）
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
import asyncio

from app.utils.logger import logger

# 定义对话状态
class ConversationState(TypedDict):
    """对话状态定义"""
    # 基础信息
    messages: Annotated[List[Dict], add]  # 对话历史
    user_query: str                        # 当前用户问题
    user_id: str                           # 用户ID
    session_id: str                        # 会话ID
    
    # 处理中间结果
    has_sensitive: bool                    # 是否包含敏感词
    intent: str                            # 识别的意图
    entities: Dict[str, Any]               # 提取的实体
    retrieved_docs: List[Dict]             # 检索到的文档
    context: Dict[str, Any]                # 上下文信息
    
    # 输出结果
    final_answer: str                      # 最终答案
    confidence: float                      # 置信度
    sources: List[str]                     # 来源
    error: str                             # 错误信息

# 节点实现
async def validate_input(state: ConversationState) -> ConversationState:
    """输入验证"""
    logger.info(f"Validating input: {state['user_query']}")
    
    # 基础验证
    if not state["user_query"] or len(state["user_query"]) < 1:
        state["error"] = "输入不能为空"
        return state
    
    if len(state["user_query"]) > 1000:
        state["error"] = "输入过长，请控制在1000字符以内"
        return state
    
    return state

async def check_sensitive_words(state: ConversationState) -> ConversationState:
    """敏感词检测"""
    logger.info("Checking sensitive words...")
    
    from app.services.sensitive_filter import check_sensitive
    
    has_sensitive, matched_words = await check_sensitive(state["user_query"])
    state["has_sensitive"] = has_sensitive
    
    if has_sensitive:
        logger.warning(f"Sensitive words detected: {matched_words}")
        state["final_answer"] = "抱歉，您的问题包含敏感内容，无法回答。"
        state["confidence"] = 0.0
    
    return state

async def recognize_intent(state: ConversationState) -> ConversationState:
    """意图识别"""
    logger.info("Recognizing intent...")
    
    from app.core.intent import IntentRecognizer
    
    recognizer = IntentRecognizer()
    intent, confidence = await recognizer.classify(
        query=state["user_query"],
        context=state.get("context", {})
    )
    
    state["intent"] = intent
    state["confidence"] = confidence
    
    logger.info(f"Intent: {intent}, Confidence: {confidence}")
    return state

async def extract_entities(state: ConversationState) -> ConversationState:
    """实体提取"""
    logger.info("Extracting entities...")
    
    from app.core.intent import EntityExtractor
    
    extractor = EntityExtractor()
    entities = await extractor.extract(state["user_query"])
    
    state["entities"] = entities
    logger.info(f"Entities: {entities}")
    return state

async def route_to_knowledge(state: ConversationState) -> ConversationState:
    """知识库路由"""
    logger.info(f"Routing to knowledge base for intent: {state['intent']}")
    
    # 根据意图选择知识库
    from app.services.knowledge_base import get_knowledge_bases_by_intent
    
    kb_ids = await get_knowledge_bases_by_intent(state["intent"])
    state["context"]["kb_ids"] = kb_ids
    
    return state

async def hybrid_search(state: ConversationState) -> ConversationState:
    """混合检索"""
    logger.info("Performing hybrid search...")
    
    from app.core.retrieval import HybridRetrieval
    
    retrieval = HybridRetrieval()
    docs = await retrieval.search(
        query=state["user_query"],
        kb_ids=state["context"].get("kb_ids", []),
        top_k=5
    )
    
    state["retrieved_docs"] = docs
    logger.info(f"Retrieved {len(docs)} documents")
    return state

async def enhance_with_context(state: ConversationState) -> ConversationState:
    """上下文增强"""
    logger.info("Enhancing with context...")
    
    # 获取历史对话
    recent_messages = state["messages"][-5:]  # 最近5轮
    
    # 构建上下文
    context_text = "\n\n".join([
        f"Q: {msg['content']}" if msg['role'] == 'user' else f"A: {msg['content']}"
        for msg in recent_messages
    ])
    
    state["context"]["conversation_history"] = context_text
    return state

async def generate_answer(state: ConversationState) -> ConversationState:
    """生成答案"""
    logger.info("Generating answer...")
    
    from app.core.dialogue_engine import DialogueEngine
    
    engine = DialogueEngine()
    answer = await engine.generate(
        query=state["user_query"],
        retrieved_docs=state["retrieved_docs"],
        context=state["context"]
    )
    
    state["final_answer"] = answer["content"]
    state["confidence"] = answer["confidence"]
    state["sources"] = answer.get("sources", [])
    
    return state

async def check_answer_quality(state: ConversationState) -> ConversationState:
    """答案质量检查"""
    logger.info(f"Checking answer quality (confidence: {state['confidence']})")
    
    # 低置信度处理
    if state["confidence"] < 0.7:
        logger.warning("Low confidence answer")
        state["final_answer"] = (
            "抱歉，我不太确定这个问题的答案。"
            "您可以换个说法试试，或者联系人工客服。"
        )
    
    return state

async def filter_output(state: ConversationState) -> ConversationState:
    """输出过滤"""
    logger.info("Filtering output...")
    
    from app.services.sensitive_filter import check_sensitive
    
    # 敏感词过滤
    has_sensitive, matched_words = await check_sensitive(state["final_answer"])
    if has_sensitive:
        logger.warning(f"Sensitive words in output: {matched_words}")
        # 替换敏感词
        for word in matched_words:
            state["final_answer"] = state["final_answer"].replace(word, "***")
    
    return state

async def save_conversation(state: ConversationState) -> ConversationState:
    """保存对话记录"""
    logger.info("Saving conversation...")
    
    from app.services.conversation import save_to_db
    
    await save_to_db(
        session_id=state["session_id"],
        user_query=state["user_query"],
        answer=state["final_answer"],
        intent=state["intent"],
        confidence=state["confidence"],
        sources=state["sources"]
    )
    
    return state

# 构建工作流图
def build_conversation_graph() -> StateGraph:
    """构建对话工作流图"""
    graph = StateGraph(ConversationState)
    
    # 添加节点
    graph.add_node("validate_input", validate_input)
    graph.add_node("check_sensitive", check_sensitive_words)
    graph.add_node("recognize_intent", recognize_intent)
    graph.add_node("extract_entities", extract_entities)
    graph.add_node("route_knowledge", route_to_knowledge)
    graph.add_node("hybrid_search", hybrid_search)
    graph.add_node("enhance_context", enhance_with_context)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("check_quality", check_answer_quality)
    graph.add_node("filter_output", filter_output)
    graph.add_node("save_conversation", save_conversation)
    
    # 添加边（工作流）
    graph.add_edge("validate_input", "check_sensitive")
    
    # 条件边：敏感词检测
    graph.add_conditional_edges(
        "check_sensitive",
        lambda state: "reject" if state.get("has_sensitive") else "continue",
        {
            "reject": "save_conversation",
            "continue": "recognize_intent"
        }
    )
    
    graph.add_edge("recognize_intent", "extract_entities")
    graph.add_edge("extract_entities", "route_knowledge")
    graph.add_edge("route_knowledge", "hybrid_search")
    graph.add_edge("hybrid_search", "enhance_context")
    graph.add_edge("enhance_context", "generate_answer")
    graph.add_edge("generate_answer", "check_quality")
    graph.add_edge("check_quality", "filter_output")
    graph.add_edge("filter_output", "save_conversation")
    graph.add_edge("save_conversation", END)
    
    # 设置入口
    graph.set_entry_point("validate_input")
    
    return graph.compile()

# 创建全局工作流实例
conversation_graph = build_conversation_graph()
```

### 2.4 第9-10天：对话管理服务

#### 任务清单
- [ ] 会话管理
- [ ] 上下文存储
- [ ] 流式响应
- [ ] 集成LangGraph

#### 对话管理服务（app/services/conversation.py）
```python
from typing import Optional, Dict, List, AsyncGenerator
import uuid
from datetime import datetime

from app.core.langgraph_workflow import conversation_graph, ConversationState
from app.utils.db import get_mongodb
from app.utils.logger import logger

class ConversationService:
    """对话管理服务"""
    
    def __init__(self):
        self.db = None
    
    async def _init_db(self):
        """初始化数据库连接"""
        if not self.db:
            self.db = await get_mongodb()
    
    async def create_session(
        self,
        user_id: str,
        system_prompt: str = "你是一个数字员工助手"
    ) -> str:
        """创建对话会话"""
        await self._init_db()
        
        session_id = f"session_{uuid.uuid4().hex}"
        
        session_doc = {
            "session_id": session_id,
            "user_id": user_id,
            "start_time": datetime.utcnow(),
            "status": "active",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                    "timestamp": datetime.utcnow()
                }
            ],
            "metadata": {}
        }
        
        await self.db.conversations.insert_one(session_doc)
        logger.info(f"Session created: {session_id}")
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict]:
        """获取会话信息"""
        await self._init_db()
        return await self.db.conversations.find_one({"session_id": session_id})
    
    async def end_session(self, session_id: str):
        """结束会话"""
        await self._init_db()
        
        await self.db.conversations.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "status": "ended",
                    "end_time": datetime.utcnow()
                }
            }
        )
        logger.info(f"Session ended: {session_id}")
    
    async def process_message(
        self,
        user_id: str,
        message: str,
        session_id: Optional[str] = None,
        stream: bool = True
    ) -> AsyncGenerator[Dict, None] | Dict:
        """处理用户消息"""
        await self._init_db()
        
        # 创建或获取会话
        if not session_id:
            session_id = await self.create_session(user_id)
        
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        # 构建初始状态
        initial_state: ConversationState = {
            "messages": session["messages"],
            "user_query": message,
            "user_id": user_id,
            "session_id": session_id,
            "has_sensitive": False,
            "intent": "",
            "entities": {},
            "retrieved_docs": [],
            "context": {},
            "final_answer": "",
            "confidence": 0.0,
            "sources": [],
            "error": ""
        }
        
        # 执行工作流
        if stream:
            # 流式响应
            async for state in conversation_graph.astream(initial_state):
                # 检查是否有最终答案
                if "final_answer" in state and state["final_answer"]:
                    yield {
                        "type": "answer",
                        "content": state["final_answer"],
                        "confidence": state.get("confidence", 0.0),
                        "sources": state.get("sources", []),
                        "session_id": session_id
                    }
        else:
            # 同步响应
            final_state = await conversation_graph.ainvoke(initial_state)
            return {
                "session_id": session_id,
                "message": final_state["final_answer"],
                "confidence": final_state["confidence"],
                "sources": final_state["sources"]
            }
```

### 2.5 第11-14天：数据库初始化与测试

#### 任务清单
- [ ] MongoDB集合创建
- [ ] Chroma集合创建
- [ ] ElasticSearch索引创建
- [ ] 数据库工具类
- [ ] 单元测试
- [ ] 集成测试

#### 数据库工具（app/utils/db.py）
```python
from motor.motor_asyncio import AsyncIOMotorClient
import chromadb
from chromadb.config import Settings as ChromaSettings
from elasticsearch import AsyncElasticsearch

from app.config import settings
from app.utils.logger import logger

# 全局数据库连接
_mongo_client: AsyncIOMotorClient = None
_chroma_client = None
_es_client: AsyncElasticsearch = None

async def init_databases():
    """初始化所有数据库连接"""
    global _mongo_client, _chroma_client, _es_client
    
    # MongoDB
    _mongo_client = AsyncIOMotorClient(settings.MONGODB_URI)
    logger.info("MongoDB connected")
    
    # Chroma
    _chroma_client = chromadb.Client(ChromaSettings(
        chroma_api_impl="rest",
        chroma_server_host=settings.CHROMA_HOST,
        chroma_server_http_port=settings.CHROMA_PORT
    ))
    logger.info("Chroma connected")
    
    # ElasticSearch
    _es_client = AsyncElasticsearch(
        hosts=[f"http://{settings.ES_HOST}:{settings.ES_PORT}"]
    )
    logger.info("ElasticSearch connected")
    
    # 初始化集合和索引
    await init_collections()

async def close_databases():
    """关闭所有数据库连接"""
    global _mongo_client, _es_client
    
    if _mongo_client:
        _mongo_client.close()
    if _es_client:
        await _es_client.close()

async def get_mongodb():
    """获取MongoDB数据库"""
    return _mongo_client[settings.MONGODB_DB_NAME]

def get_chroma():
    """获取Chroma客户端"""
    return _chroma_client

async def get_elasticsearch():
    """获取ElasticSearch客户端"""
    return _es_client

async def init_collections():
    """初始化MongoDB集合和索引"""
    db = await get_mongodb()
    
    # 创建对话集合索引
    await db.conversations.create_index("session_id", unique=True)
    await db.conversations.create_index("user_id")
    await db.conversations.create_index("start_time")
    
    # 创建知识库集合索引
    await db.knowledge_docs.create_index("doc_id", unique=True)
    await db.knowledge_docs.create_index("kb_id")
    
    # 创建FAQ集合索引
    await db.faqs.create_index("faq_id", unique=True)
    await db.faqs.create_index("category")
    
    logger.info("MongoDB collections initialized")
    
    # 初始化Chroma集合
    chroma = get_chroma()
    try:
        chroma.create_collection(name="faq_collection")
        chroma.create_collection(name="doc_collection")
        logger.info("Chroma collections initialized")
    except Exception as e:
        logger.info(f"Chroma collections already exist: {e}")
    
    # 初始化ElasticSearch索引
    es = await get_elasticsearch()
    faq_index = {
        "mappings": {
            "properties": {
                "faq_id": {"type": "keyword"},
                "question": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "answer": {"type": "text"},
                "keywords": {"type": "keyword"},
                "category": {"type": "keyword"}
            }
        }
    }
    
    try:
        await es.indices.create(index="faq_index", body=faq_index)
        logger.info("ElasticSearch indexes initialized")
    except Exception as e:
        logger.info(f"ElasticSearch indexes already exist: {e}")
```

---

## 三、Phase 2：知识库与检索（2周）

### 目标
实现知识库管理和混合检索策略

### 3.1 第15-17天：知识库管理服务

（继续补充...）

---

## 四、快速启动指南

### 4.1 环境配置

1. **创建.env文件**：
```bash
MONGODB_URI=mongodb://localhost:27017
CHROMA_HOST=localhost
CHROMA_PORT=8001
ES_HOST=localhost
ES_PORT=9200
OPENAI_API_KEY=your_openai_api_key
SECRET_KEY=your_secret_key_here
```

2. **安装依赖**：
```bash
cd ai-service
pip install -r requirements.txt
```

3. **启动数据库**（使用Docker Compose）：
```bash
docker-compose up -d mongodb chroma elasticsearch
```

4. **运行服务**：
```bash
python -m app.main
```

### 4.2 测试API

```bash
# 健康检查
curl http://localhost:8000/health

# 创建对话
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好",
    "stream": false
  }'
```

---

## 五、后续阶段概要

### Phase 2：知识库与检索（2周）
- FAQ问答库实现
- RAG文档库实现
- 混合检索策略（Chroma + ElasticSearch + RRF）
- 专业词库集成

### Phase 3：意图识别与质量保障（1.5周）
- 意图识别模块（基于LLM）
- 实体提取（NER）
- 敏感词过滤系统
- 答案质量评估

### Phase 4：系统集成与优化（1.5周）
- 异步任务队列（Celery）
- Docker容器化
- 性能优化（缓存、限流）
- 监控告警
- 前后端联调

---

## 六、关键技术点总结

### 1. LangGraph工作流编排
- 状态机模式
- 条件分支
- 异步执行
- 错误处理

### 2. 混合检索策略
- 向量检索（Chroma）：语义相似度
- 关键词检索（ElasticSearch）：精确匹配
- RRF融合算法：结果排序
- 重排序：质量优化

### 3. 流式响应
- Server-Sent Events (SSE)
- WebSocket
- 异步生成器
- 首Token时延优化

### 4. 数据库设计
- MongoDB：灵活的文档存储
- Chroma：高效的向量检索
- ElasticSearch：强大的全文检索

---

## 七、常见问题与解决方案

### 1. LLM调用失败
**解决方案**：
- 实现重试机制（tenacity库）
- 降级到备用模型
- 缓存常见问题答案

### 2. 检索准确率不足
**解决方案**：
- 调整相似度阈值
- 优化分块策略
- 增加专业词库
- 使用重排序模型

### 3. 性能瓶颈
**解决方案**：
- 实现缓存层（Redis）
- 批处理优化
- 异步处理
- 水平扩展

---

## 八、下一步计划

完成Phase 1后，继续按照规划推进：
1. ✅ Phase 1：基础服务搭建（2周）
2. ⏭️ Phase 2：知识库与检索（2周）
3. ⏭️ Phase 3：意图识别与质量保障（1.5周）
4. ⏭️ Phase 4：系统集成与优化（1.5周）

**预计总时长**：7周完成完整AI引擎

---

**本文档持续更新中...**
