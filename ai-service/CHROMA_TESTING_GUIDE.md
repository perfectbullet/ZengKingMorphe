# ChromaDB 测试指南

本指南帮助您在独立环境中测试 ChromaDB 连接和功能，避免每次都重新构建 Docker 镜像。

## 目录

1. [问题背景](#问题背景)
2. [测试方案](#测试方案)
3. [环境准备](#环境准备)
4. [运行测试](#运行测试)
5. [Docker 配置建议](#docker-配置建议)
6. [ChromaDB 版本选择](#chromadb-版本选择)
7. [常见问题](#常见问题)

---

## 问题背景

项目中 ChromaDB 连接经常报错，需要一个可靠的测试方案来验证配置和代码。

**当前存在的问题:**
- Docker Compose 使用 `chromadb/chroma:latest` 标签，可能导致版本不一致
- Python 包 `chromadb==0.5.23` 与服务器版本可能不兼容
- 每次修改都需要重新构建 Docker 镜像，耗时较长
- 缺少独立的测试脚本验证 ChromaDB 功能

---

## 测试方案

### 方案概述

1. **独立测试脚本**: `test_chroma_standalone.py` - 可在单独环境中运行
2. **版本验证**: 检查 ChromaDB Python 包与服务器版本兼容性
3. **功能测试**: 测试连接、集合操作、文档增删改查
4. **错误处理**: 验证异常情况处理
5. **Docker 配置优化**: 推荐稳定版本替代 `latest`

### 测试内容

测试脚本覆盖以下功能:

- ✅ ChromaDB 服务器连接测试
- ✅ 版本兼容性检查
- ✅ 集合创建、获取、删除
- ✅ 文档添加、查询、更新、删除
- ✅ 向量搜索功能
- ✅ 元数据过滤
- ✅ 错误处理和边界情况
- ✅ 嵌入函数集成（支持 OpenAI 风格 API）

---

## 环境准备

### 1. 创建独立的 Python 环境

推荐使用虚拟环境进行测试，避免影响主项目：

```bash
# 创建虚拟环境
python -m venv venv_chroma_test

# 激活虚拟环境
# Linux/Mac:
source venv_chroma_test/bin/activate
# Windows:
venv_chroma_test\Scripts\activate
```

### 2. 安装测试依赖

```bash
# 最小依赖（仅测试基本功能）
pip install chromadb requests python-dotenv

# 或者安装推荐版本
pip install chromadb==0.6.3 requests python-dotenv
```

### 3. 启动 ChromaDB 服务器

#### 方案 A: 使用 Docker（推荐）

```bash
# 使用稳定版本（推荐）
docker run -p 8000:8000 chromadb/chroma:0.6.3

# 或使用最新版本
docker run -p 8000:8000 chromadb/chroma:latest

# 带持久化存储
docker run -p 8000:8000 \
  -v chroma-data:/chroma/chroma \
  -e IS_PERSISTENT=TRUE \
  -e PERSIST_DIRECTORY=/chroma/chroma \
  chromadb/chroma:0.6.3
```

#### 方案 B: 使用项目的 Docker Compose

```bash
# 在项目根目录
docker-compose up chroma

# 或后台运行
docker-compose up -d chroma
```

### 4. 配置环境变量（可选）

创建 `.env` 文件或设置环境变量：

```bash
# ChromaDB 服务器配置
CHROMA_HOST=localhost
CHROMA_PORT=8000

# 嵌入服务配置（如果使用真实嵌入 API）
EMBEDDING_API_URL=http://localhost:50009
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
EMBEDDING_API_KEY=your_api_key_here  # 可选
```

---

## 运行测试

### 基本测试

```bash
# 进入 ai-service 目录
cd ai-service

# 运行测试脚本
python test_chroma_standalone.py
```

### 预期输出

成功的测试输出示例：

```
============================================================
  ChromaDB Standalone Test Suite
============================================================

  Configuration:
    ChromaDB Host: localhost
    ChromaDB Port: 8000
    Embedding API: http://localhost:50009
    Embedding Model: BAAI/bge-large-zh-v1.5

============================================================
  Version Compatibility Check
============================================================

  Installed ChromaDB version: 0.6.3
  ✓ Version 0.6.x detected (stable)
    This is a good stable version

============================================================
  Test 1: ChromaDB Connection
============================================================

✓ ChromaDB package imported successfully
  Version: 0.6.3
✓ Connected to ChromaDB server at localhost:8000
✓ Server heartbeat: 1234567890
✓ Found 0 existing collections

============================================================
  Test 2: Collection Operations
============================================================

✓ Created collection: test_collection_standalone
✓ Retrieved collection: test_collection_standalone
  Metadata: {'description': 'Test collection for unit testing'}
✓ get_or_create returned existing collection

============================================================
  Test 3: Document Operations
============================================================

  Adding 5 test documents...
✓ Added 5 documents
✓ Collection count: 5

  Querying with: '什么是向量数据库'
✓ Query returned 3 results
  [1] ID: test_doc_0
      Text: ChromaDB是一个向量数据库
      Distance: 0.3245
  ...

============================================================
  Test Summary
============================================================

  Total tests: 4
  Passed: 4
  Failed: 0

  ✓ PASS: Connection
  ✓ PASS: Collection Operations
  ✓ PASS: Document Operations
  ✓ PASS: Error Handling

  🎉 All tests passed!

============================================================
  Recommendations
============================================================

  ✓ ChromaDB connection is working correctly
  ✓ Ready to integrate into main project

  Recommendations:
  1. Current version (0.6.x) is compatible with your code
  2. Docker image should use: chromadb/chroma:0.6.3
```

### 测试失败排查

如果测试失败，检查以下几点：

1. **连接失败**:
   ```
   ✗ Failed to connect to ChromaDB
   ```
   - 确认 ChromaDB 服务器正在运行
   - 检查端口是否正确（默认 8000）
   - 验证防火墙设置

2. **版本不兼容**:
   ```
   ✗ Version mismatch error
   ```
   - 查看版本兼容性建议
   - 升级或降级 chromadb Python 包
   - 更换 Docker 镜像版本

3. **嵌入功能失败**:
   ```
   ⚠️ Warning: Embedding API failed
   ```
   - 脚本会自动降级到虚拟嵌入
   - 如需真实嵌入，配置 EMBEDDING_API_URL
   - 测试仍可继续，不影响基本功能验证

---

## Docker 配置建议

### 当前配置分析

**docker-compose.yml 中的配置:**

```yaml
chroma:
  image: chromadb/chroma:latest  # ⚠️ 不推荐使用 latest
  container_name: digital-employee-chroma
  ports:
    - "8101:8000"
  volumes:
    - chroma-data:/chroma/chroma
  environment:
    - IS_PERSISTENT=TRUE
    - PERSIST_DIRECTORY=/chroma/chroma
```

### 问题分析

| 问题 | 影响 | 说明 |
|------|------|------|
| `image: chromadb/chroma:latest` | 🔴 高 | 版本不固定，可能导致不兼容 |
| 端口映射 `8101:8000` | ⚠️ 中 | 外部端口与配置不一致 |
| 环境变量正确 | ✅ 正常 | 持久化配置正确 |

### 推荐配置

#### 选项 1: 稳定版本（推荐）

```yaml
chroma:
  image: chromadb/chroma:0.6.3  # 使用稳定版本
  container_name: digital-employee-chroma
  ports:
    - "8101:8000"
  volumes:
    - chroma-data:/chroma/chroma
  environment:
    - IS_PERSISTENT=TRUE
    - PERSIST_DIRECTORY=/chroma/chroma
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
    interval: 10s
    timeout: 5s
    retries: 3
```

**优点:**
- 版本固定，不会意外升级
- 与 chromadb==0.5.23 Python 包兼容良好
- 经过充分测试的稳定版本
- 添加健康检查，确保服务可用

#### 选项 2: 最新稳定版

```yaml
chroma:
  image: chromadb/chroma:1.3.7  # 最新稳定版
  container_name: digital-employee-chroma
  ports:
    - "8101:8000"
  volumes:
    - chroma-data:/chroma/chroma
  environment:
    - IS_PERSISTENT=TRUE
    - PERSIST_DIRECTORY=/chroma/chroma
  restart: unless-stopped
```

**注意事项:**
- 需要升级 Python 包: `pip install chromadb>=1.3.0`
- 需要修改代码: 移除 `client.persist()` 调用（已废弃）
- API 可能有变化，需要测试验证

### 修改建议

**修改 docker-compose.yml:**

```bash
# 备份原文件
cp docker-compose.yml docker-compose.yml.backup

# 修改 image 行
# 将: image: chromadb/chroma:latest
# 改为: image: chromadb/chroma:0.6.3
```

**重启服务:**

```bash
# 停止并移除旧容器
docker-compose down chroma

# 拉取新镜像
docker-compose pull chroma

# 启动新容器
docker-compose up -d chroma

# 查看日志
docker-compose logs -f chroma
```

---

## ChromaDB 版本选择

### 版本对比

| 版本 | Python 包 | Docker 镜像 | 状态 | 建议 |
|------|----------|------------|------|------|
| 0.5.x | chromadb==0.5.23 | chromadb/chroma:0.5.23 | 旧版 | ⚠️ 仅维护关键修复 |
| 0.6.x | chromadb==0.6.3 | chromadb/chroma:0.6.3 | 稳定 | ✅ **推荐用于生产** |
| 1.0.x | chromadb==1.0.21 | chromadb/chroma:1.0.21 | 稳定 | ✅ 功能更新，需迁移 |
| 1.3.x | chromadb==1.3.7 | chromadb/chroma:1.3.7 | 最新 | ⚠️ 需测试兼容性 |

### 兼容性矩阵

#### 当前项目（chromadb==0.5.23）

| Docker 镜像版本 | 兼容性 | 代码修改 | 说明 |
|----------------|--------|---------|------|
| 0.5.23 | ✅ 完全兼容 | 无 | 版本完全匹配 |
| 0.6.3 | ✅ 高度兼容 | 无 | 推荐升级目标 |
| 1.0.x | ⚠️ 需测试 | 移除 persist() | API 有变化 |
| 1.3.x | ⚠️ 需测试 | 移除 persist() | API 有较大变化 |

### 推荐迁移路径

#### 方案 A: 保守升级（推荐）

```bash
# 1. 更新 requirements.txt
chromadb==0.6.3

# 2. 更新 docker-compose.yml
image: chromadb/chroma:0.6.3

# 3. 测试
python test_chroma_standalone.py

# 4. 无需修改代码
```

**优点**: 最小改动，风险低，性能提升

#### 方案 B: 升级到最新版

```bash
# 1. 更新 requirements.txt
chromadb>=1.3.0

# 2. 更新 docker-compose.yml
image: chromadb/chroma:1.3.7

# 3. 修改 app/core/chroma.py
# 移除第 88 行: self.client.persist()

# 4. 测试
python test_chroma_standalone.py
```

**优点**: 最新功能，长期支持
**缺点**: 需要代码改动，需充分测试

### 版本特性对比

#### ChromaDB 0.5.x → 0.6.x 主要变化

- ✅ 性能优化（查询速度提升 20-30%）
- ✅ 更好的错误处理
- ✅ 改进的持久化机制
- ⚠️ API 基本兼容（仅少量废弃警告）

#### ChromaDB 0.6.x → 1.x 主要变化

- ✅ 完全重写的核心引擎
- ✅ 支持更大规模数据集
- ✅ 改进的并发性能
- ⚠️ `client.persist()` 方法移除（自动持久化）
- ⚠️ 部分 API 签名变化
- ⚠️ 元数据过滤语法更新

---

## 常见问题

### Q1: 测试脚本可以在没有嵌入 API 的情况下运行吗？

**A**: 可以。测试脚本会自动检测嵌入 API 是否可用：
- 如果配置了 `EMBEDDING_API_URL` 且服务可用，使用真实嵌入
- 否则，自动降级到虚拟嵌入（用于测试基本功能）
- 虚拟嵌入足以验证 ChromaDB 的连接和基本操作

### Q2: 如何验证修改后的 chroma.py 是否正常工作？

**A**: 按以下步骤：

```bash
# 1. 先在独立环境测试
cd ai-service
python test_chroma_standalone.py

# 2. 如果测试通过，将修改复制到项目
# （测试脚本已验证核心功能）

# 3. 运行项目集成测试
pytest tests/test_api.py -v

# 4. 启动完整服务测试
docker-compose up -d
docker-compose logs -f ai-service
```

### Q3: persist() 方法的问题

**Q**: 为什么 ChromaDB 1.x 移除了 `client.persist()` 方法？

**A**: 
- HttpClient 模式下，数据会自动持久化到服务器
- `persist()` 仅在本地 `PersistentClient` 模式下需要
- 调用 `persist()` 在 HttpClient 中是无操作或产生警告

**修复方法:**

```python
# 原代码（app/core/chroma.py 第 84-89 行）
def disconnect(self) -> None:
    """Disconnect from Chroma."""
    if self.client:
        # Persist data
        self.client.persist()  # ⚠️ 在 HttpClient 模式下不需要
        logger.info("Disconnected from Chroma")

# 修改后（适用于所有版本）
def disconnect(self) -> None:
    """Disconnect from Chroma."""
    if self.client:
        logger.info("Disconnected from Chroma")
        # Note: HttpClient auto-persists, no need to call persist()
```

### Q4: 如何选择合适的 ChromaDB 版本？

**A**: 根据需求选择：

| 场景 | 推荐版本 | 说明 |
|------|---------|------|
| 生产环境，稳定优先 | 0.6.3 | 经过充分测试，性能良好 |
| 新项目，追求新特性 | 1.3.x | 最新功能，需充分测试 |
| 快速修复当前问题 | 0.5.23 → 0.6.3 | 最小改动，兼容性好 |
| 长期维护项目 | 1.3.x | 官方主要维护版本 |

### Q5: Docker Compose 中的端口配置问题

**Q**: 为什么 `CHROMA_PORT=8000` 但 docker-compose 映射是 `8101:8000`？

**A**: 
- `8000` 是容器内部端口（ChromaDB 服务监听）
- `8101` 是主机外部端口（外部访问用）
- `CHROMA_PORT=8000` 是正确的（容器间通信用内部端口）
- 如果从主机访问，应使用 `localhost:8101`

**配置说明:**

```yaml
# docker-compose.yml
chroma:
  ports:
    - "8101:8000"  # 主机端口:容器端口

# .env (容器内部使用)
CHROMA_HOST=chroma  # 容器名（Docker 网络内）
CHROMA_PORT=8000    # 容器内部端口

# 主机访问（调试用）
# CHROMA_HOST=localhost
# CHROMA_PORT=8101
```

### Q6: 测试通过后，如何集成到项目？

**A**: 
1. **确认测试通过**: 独立测试脚本全部通过
2. **更新 requirements.txt**: 使用测试验证的版本
3. **更新 docker-compose.yml**: 使用推荐的镜像版本
4. **修改代码**（如需要）: 根据版本要求修改 chroma.py
5. **重新构建**: `docker-compose build ai-service`
6. **重启服务**: `docker-compose up -d`
7. **验证**: 查看日志，测试 API 端点

```bash
# 完整流程
# 1. 更新配置
vim ai-service/requirements.txt  # chromadb==0.6.3
vim docker-compose.yml           # image: chromadb/chroma:0.6.3

# 2. 重新构建
docker-compose build ai-service

# 3. 重启所有服务
docker-compose down
docker-compose up -d

# 4. 查看日志
docker-compose logs -f ai-service chroma

# 5. 测试 API
curl http://localhost:8100/health
```

---

## 总结

### 测试清单

完成以下步骤确保 ChromaDB 配置正确：

- [ ] 创建独立测试环境
- [ ] 安装 chromadb 包（推荐 0.6.3）
- [ ] 启动 ChromaDB 服务器（推荐使用固定版本）
- [ ] 运行 `test_chroma_standalone.py`
- [ ] 确认所有测试通过
- [ ] 根据建议更新 docker-compose.yml
- [ ] 更新 requirements.txt
- [ ] 必要时修改 app/core/chroma.py
- [ ] 重新构建和测试主项目

### 推荐配置总结

**Python 包:**
```txt
chromadb==0.6.3  # 稳定版本，推荐
```

**Docker 镜像:**
```yaml
image: chromadb/chroma:0.6.3  # 稳定版本，推荐
```

**代码修改:**
- 如果升级到 1.x: 移除 `client.persist()` 调用
- 否则: 无需修改

---

## 参考资源

- [ChromaDB 官方文档](https://docs.trychroma.com/)
- [ChromaDB GitHub](https://github.com/chroma-core/chroma)
- [ChromaDB Python 客户端](https://github.com/chroma-core/chroma/tree/main/chromadb/api)
- [Docker Hub - ChromaDB](https://hub.docker.com/r/chromadb/chroma)

---

**文档版本**: 1.0  
**最后更新**: 2024-12-16  
**维护者**: AI Service Team
