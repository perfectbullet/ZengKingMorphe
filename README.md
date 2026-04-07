# Digital Employee AI Service

## 项目简介

基于 LangGraph 的数字员工 AI 服务，提供知识库检索（RAG）、联网检索、多轮对话管理、敏感词过滤等功能。

## 技术栈

- **后端框架**: FastAPI
- **AI 引擎**: LangGraph + OpenAI/Claude/DeepSeek/Qwen
- **向量数据库**: Chroma
- **全文检索**: ElasticSearch
- **数据库**: MongoDB
- **联网检索**: Tavily
- **容器化**: Docker + Docker Compose

## 快速开始

### 1. 环境要求

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+ (本地开发)

### 2. 配置环境变量

复制示例配置文件并修改：

```bash
cd ai-service
cp .env.example .env
```

编辑 `.env` 文件，填入必要的 API 密钥：

```bash
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
JWT_SECRET_KEY=your-jwt-secret-key
```

### 3. 启动服务

使用 Docker Compose 启动所有服务：

```bash
docker-compose up -d
```

查看服务状态：

```bash
docker-compose ps
```

查看日志：

```bash
docker-compose logs -f ai-service
```

### 4. 访问服务

- **API 文档 (Swagger)**: http://localhost:8000/docs
- **API 文档 (ReDoc)**: http://localhost:8000/redoc
- **健康检查**: http://localhost:8000/health
- **MongoDB**: mongodb://localhost:27017
- **ElasticSearch**: http://localhost:9200
- **Chroma**: http://localhost:8001

## 本地开发

### 1. 安装依赖

```bash
# 虚拟环境已创建，依赖已安装
# 如果没有创建虚拟环境
python3 -m venv venv

# 如需重新安装，使用项目 Python 解释器：
venv/bin/python -m pip install -r ai-service/requirements.txt

# 或激活虚拟环境后安装
source venv/bin/activate
pip install -r ai-service/requirements.txt
# 安装 RAG-Anything 
cd RAG-Anything
# 只安装类似客户端的mineru
pip install mineru==3.0.8
# 安装
pip install -e .
```

### 2. 启动数据库服务

```bash
# 只启动数据库服务
docker-compose up -d mongodb elasticsearch chroma
```

### 3. 运行开发服务器

**方式一：使用启动脚本（推荐）**

```bash
# 启动服务（后台运行）
./start_ai_service.sh start

# 查看服务状态
./start_ai_service.sh status

# 查看实时日志
./start_ai_service.sh logs

# 停止服务
./start_ai_service.sh stop

# 重启服务
./start_ai_service.sh restart
```

**方式二：手动启动**

```bash
cd ai-service

# 使用项目 Python 解释器
/home/zj/miniconda3/envs/morphe/bin/python -m uvicorn main:app --reload --host 0.0.0.0 --port 8100

# 或激活虚拟环境后运行
source ../.venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 项目结构

```
ai-service/
├── app/
│   ├── api/
│   │   ├── endpoints/        # API 端点
│   │   └── middleware/       # 中间件（鉴权、限流、错误处理）
│   ├── core/                 # 核心配置（数据库、日志、配置）
│   ├── models/               # 数据模型（数据库模型、API schemas）
│   ├── services/             # 业务逻辑服务
│   └── utils/                # 工具函数
├── tests/                    # 测试文件
├── chroma_db/               # Chroma 数据持久化
├── main.py                  # 应用入口
├── requirements.txt         # Python 依赖
├── .env.example            # 环境变量示例
└── Dockerfile              # Docker 构建文件
```

## API 接口

### 对话接口

- `POST /api/chat/message` - 同步对话
- `POST /api/chat/stream` - 流式对话 (SSE)
- `GET /api/chat/session/{id}` - 获取会话信息
- `DELETE /api/chat/session/{id}` - 结束会话

### 数字员工管理

- `POST /api/employee/create` - 创建数字员工
- `GET /api/employee/{employee_id}` - 获取数字员工信息
- `PUT /api/employee/{employee_id}` - 更新数字员工配置
- `DELETE /api/employee/{employee_id}` - 删除数字员工
- `GET /api/employee/list` - 列出数字员工

### Webhook 接口

- `POST /api/thesaurus_sensitive/update` - 敏感词同步更新
- `POST /api/thesaurus_major/update` - 专业词同步更新
- `POST /api/dataset_faq/update` - FAQ 更新

## 开发计划

当前进度按照 8 个阶段进行：

- [x] **Phase 1**: 基础框架搭建
  - [x] FastAPI 服务框架
  - [x] 数据库连接（MongoDB, Chroma, ElasticSearch）
  - [x] 鉴权和限流中间件
  - [x] 错误处理
  - [x] API 端点框架
- [ ] **Phase 2**: 知识库系统
- [ ] **Phase 3**: 对话引擎模块
- [ ] **Phase 4**: 对话记录管理
- [ ] **Phase 5**: 词库管理
- [ ] **Phase 6**: 意图识别模块
- [ ] **Phase 7**: 数字员工管理
- [ ] **Phase 8**: Docker 化与部署

详细计划见 `docs/需求补充与完善文档.md`。

## 测试

运行测试：

```bash
cd ai-service
pytest
```

运行测试并生成覆盖率报告：

```bash
pytest --cov=app --cov-report=html
```

## 故障排查

### 服务无法启动

1. 检查 Docker 服务是否运行
2. 检查端口是否被占用
3. 查看服务日志：`docker-compose logs ai-service`

### 数据库连接失败

1. 确认数据库容器正在运行：`docker-compose ps`
2. 检查环境变量配置
3. 查看数据库日志：`docker-compose logs mongodb`

### API 返回 500 错误

1. 查看应用日志：`docker-compose logs -f ai-service`
2. 检查 OpenAI API Key 是否有效
3. 确认所有依赖服务正常运行

## License

MIT
