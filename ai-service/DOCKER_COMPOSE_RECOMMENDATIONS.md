# ChromaDB Docker Compose 配置建议

## 推荐的 docker-compose.yml 修改

### 当前配置（存在问题）

```yaml
chroma:
  image: chromadb/chroma:latest  # ⚠️ 使用 latest 标签不稳定
  container_name: digital-employee-chroma
  ports:
    - "8101:8000"
  volumes:
    - chroma-data:/chroma/chroma
  environment:
    - IS_PERSISTENT=TRUE
    - PERSIST_DIRECTORY=/chroma/chroma
  networks:
    - digital-employee-network
  restart: unless-stopped
```

### 推荐配置选项

#### 选项 1: 稳定版本 0.6.3（强烈推荐）

```yaml
chroma:
  image: chromadb/chroma:0.6.3  # ✅ 使用固定稳定版本
  container_name: digital-employee-chroma
  ports:
    - "8101:8000"
  volumes:
    - chroma-data:/chroma/chroma
  environment:
    - IS_PERSISTENT=TRUE
    - PERSIST_DIRECTORY=/chroma/chroma
    - ANONYMIZED_TELEMETRY=FALSE  # 可选：禁用遥测
  networks:
    - digital-employee-network
  restart: unless-stopped
  healthcheck:  # ✅ 添加健康检查
    test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
    interval: 10s
    timeout: 5s
    retries: 3
    start_period: 10s
```

**优点:**
- 版本固定，不会意外升级
- 与当前 chromadb==0.5.23 完全兼容
- 性能比 0.5.x 提升约 20-30%
- 无需修改现有代码
- 添加健康检查确保服务可用

**适用场景:**
- 生产环境
- 追求稳定性
- 最小改动迁移

#### 选项 2: 最新稳定版 1.3.7

```yaml
chroma:
  image: chromadb/chroma:1.3.7  # ✅ 最新稳定版本
  container_name: digital-employee-chroma
  ports:
    - "8101:8000"
  volumes:
    - chroma-data:/chroma/chroma
  environment:
    - IS_PERSISTENT=TRUE
    - PERSIST_DIRECTORY=/chroma/chroma
    - ANONYMIZED_TELEMETRY=FALSE
    - CHROMA_SERVER_AUTH_CREDENTIALS_PROVIDER=chromadb.auth.token_authn.TokenAuthenticationServerProvider  # 可选：启用认证
    - CHROMA_SERVER_AUTH_CREDENTIALS=test-token  # 可选：设置认证令牌
  networks:
    - digital-employee-network
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
    interval: 10s
    timeout: 5s
    retries: 3
    start_period: 10s
```

**优点:**
- 最新功能和性能优化
- 更好的并发处理能力
- 官方长期支持版本
- 支持认证功能

**注意事项:**
- 需要升级 Python 包: `chromadb>=1.3.0`
- 需要移除代码中的 `client.persist()` 调用
- 需要充分测试兼容性

**代码修改需求:**

```python
# app/core/chroma.py 第 84-89 行
# 原代码
def disconnect(self) -> None:
    """Disconnect from Chroma."""
    if self.client:
        # Persist data
        self.client.persist()  # ❌ 需要删除这行
        logger.info("Disconnected from Chroma")

# 修改后
def disconnect(self) -> None:
    """Disconnect from Chroma."""
    if self.client:
        logger.info("Disconnected from Chroma")
        # HttpClient auto-persists, no manual persist needed
```

#### 选项 3: 保持当前版本（不推荐）

```yaml
chroma:
  image: chromadb/chroma:0.5.23  # ⚠️ 旧版本，仅用于临时兼容
  container_name: digital-employee-chroma
  ports:
    - "8101:8000"
  volumes:
    - chroma-data:/chroma/chroma
  environment:
    - IS_PERSISTENT=TRUE
    - PERSIST_DIRECTORY=/chroma/chroma
  networks:
    - digital-employee-network
  restart: unless-stopped
```

**仅适用于:**
- 临时快速修复
- 确保版本完全匹配
- 避免任何代码改动

**不推荐原因:**
- 性能较差
- 已停止功能更新
- 仅维护关键安全修复

### 版本对比总结

| 特性 | 0.5.23 | 0.6.3（推荐） | 1.3.7 |
|------|--------|--------------|-------|
| 稳定性 | ✅ 稳定 | ✅ 非常稳定 | ✅ 稳定 |
| 性能 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 代码修改 | 无 | 无 | 需要 |
| 功能更新 | ❌ 停止 | ⚠️ 维护中 | ✅ 活跃 |
| 并发性能 | 中等 | 良好 | 优秀 |
| 安全性 | 基础 | 良好 | 优秀 |
| 推荐度 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 迁移步骤

#### 迁移到 0.6.3（推荐）

```bash
# 1. 备份当前配置
cp docker-compose.yml docker-compose.yml.backup

# 2. 编辑 docker-compose.yml
# 将 image: chromadb/chroma:latest
# 改为 image: chromadb/chroma:0.6.3

# 3. 更新 requirements.txt
sed -i 's/chromadb==0.5.23/chromadb==0.6.3/' ai-service/requirements.txt

# 4. 停止并移除旧容器
docker-compose down chroma

# 5. 拉取新镜像
docker-compose pull chroma

# 6. 重新构建 ai-service（如果需要）
docker-compose build ai-service

# 7. 启动服务
docker-compose up -d

# 8. 验证
docker-compose logs -f chroma
curl http://localhost:8101/api/v1/heartbeat
```

#### 迁移到 1.3.7

```bash
# 1. 备份
cp docker-compose.yml docker-compose.yml.backup
cp ai-service/app/core/chroma.py ai-service/app/core/chroma.py.backup

# 2. 修改 docker-compose.yml
# image: chromadb/chroma:1.3.7

# 3. 更新 requirements.txt
sed -i 's/chromadb==0.5.23/chromadb==1.3.7/' ai-service/requirements.txt

# 4. 修改代码（移除 persist() 调用）
# 编辑 ai-service/app/core/chroma.py 第 84-89 行

# 5. 测试修改
cd ai-service
python test_chroma_standalone.py

# 6. 如果测试通过，重新部署
cd ..
docker-compose down
docker-compose build ai-service
docker-compose up -d

# 7. 验证
docker-compose logs -f ai-service chroma
```

### 健康检查说明

添加的健康检查配置:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
  interval: 10s      # 每10秒检查一次
  timeout: 5s        # 5秒超时
  retries: 3         # 连续失败3次才标记为不健康
  start_period: 10s  # 启动后等待10秒再开始检查
```

**好处:**
- Docker 自动监控服务健康状态
- 依赖服务可以等待 ChromaDB 完全启动
- 自动重启不健康的容器
- 与 `depends_on` 配合实现优雅启动

### 环境变量说明

#### 必需环境变量

```yaml
- IS_PERSISTENT=TRUE            # 启用持久化存储
- PERSIST_DIRECTORY=/chroma/chroma  # 持久化目录
```

#### 可选环境变量

```yaml
# 性能调优
- CHROMA_SERVER_CORS_ALLOW_ORIGINS=["*"]  # CORS 配置
- CHROMA_OTEL_COLLECTION_ENDPOINT=""      # 禁用 OpenTelemetry

# 安全性（1.x 版本）
- CHROMA_SERVER_AUTH_CREDENTIALS_PROVIDER=chromadb.auth.token_authn.TokenAuthenticationServerProvider
- CHROMA_SERVER_AUTH_CREDENTIALS=your-secret-token
- CHROMA_SERVER_AUTH_CREDENTIALS_FILE=/path/to/credentials

# 日志
- CHROMA_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# 隐私
- ANONYMIZED_TELEMETRY=FALSE  # 禁用匿名遥测
```

### 验证清单

部署后验证以下内容:

```bash
# 1. 检查容器状态
docker-compose ps chroma
# 应该显示 "healthy" 状态

# 2. 查看日志
docker-compose logs chroma
# 应该没有错误信息

# 3. 测试心跳
curl http://localhost:8101/api/v1/heartbeat
# 应该返回时间戳

# 4. 测试集合列表
curl http://localhost:8101/api/v1/collections
# 应该返回 JSON 数组

# 5. 运行测试脚本
cd ai-service
python test_chroma_standalone.py
# 所有测试应该通过

# 6. 测试主服务
curl http://localhost:8100/health
# 应该返回健康状态
```

### 回滚步骤

如果升级后遇到问题，快速回滚:

```bash
# 1. 停止服务
docker-compose down

# 2. 恢复配置
cp docker-compose.yml.backup docker-compose.yml
cp ai-service/app/core/chroma.py.backup ai-service/app/core/chroma.py

# 3. 恢复依赖
git checkout ai-service/requirements.txt

# 4. 重新启动
docker-compose up -d

# 5. 验证
docker-compose logs -f chroma ai-service
```

### 推荐最终配置

**最佳实践配置（生产环境）:**

```yaml
chroma:
  image: chromadb/chroma:0.6.3  # 稳定版本
  container_name: digital-employee-chroma
  ports:
    - "8101:8000"
  volumes:
    - chroma-data:/chroma/chroma
  environment:
    - IS_PERSISTENT=TRUE
    - PERSIST_DIRECTORY=/chroma/chroma
    - ANONYMIZED_TELEMETRY=FALSE
    - CHROMA_LOG_LEVEL=INFO
  networks:
    - digital-employee-network
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
    interval: 10s
    timeout: 5s
    retries: 3
    start_period: 10s
  deploy:  # 可选：资源限制
    resources:
      limits:
        cpus: '2'
        memory: 2G
      reservations:
        cpus: '1'
        memory: 1G
```

**对应的 requirements.txt:**

```txt
chromadb==0.6.3
```

**无需修改代码**

---

## 快速决策指南

**我应该选择哪个版本？**

回答以下问题:

1. **是否是生产环境？**
   - 是 → 选择 0.6.3
   - 否 → 继续下一题

2. **能否接受代码修改？**
   - 否 → 选择 0.6.3
   - 是 → 继续下一题

3. **需要最新功能？**
   - 是 → 选择 1.3.7
   - 否 → 选择 0.6.3

4. **需要认证功能？**
   - 是 → 选择 1.3.7
   - 否 → 选择 0.6.3

**结论: 90% 的情况推荐 0.6.3**

---

**文档版本**: 1.0  
**最后更新**: 2024-12-16
