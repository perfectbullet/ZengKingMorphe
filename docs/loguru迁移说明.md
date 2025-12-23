# Loguru 迁移说明

## 迁移概述

已成功将项目日志系统从 **structlog** 迁移到 **loguru**，同时保持了 100% API 兼容性。

## 核心修改

### 1. 依赖更新 ([requirements.txt](../ai-service/requirements.txt))
```diff
- structlog==24.1.0
+ loguru==0.7.2
```

### 2. 日志配置重写 ([logging.py](../ai-service/app/core/logging.py))

**关键创新：LoguruAdapter 包装器**

创建了 `LoguruAdapter` 类来拦截 structlog 风格的关键字参数调用：

```python
# ✅ 原有代码无需修改，语法完全兼容
logger.info("Using Ollama LLM", model=settings.ollama_model)
logger.error("Failed to load config", error=str(e), exc_info=True)
logger.warning("High memory usage", memory_mb=512, threshold_mb=400)
```

**技术实现：**
- 拦截 `info/error/warning/debug` 方法调用
- 提取关键字参数转换为 loguru 的 `bind()` 调用
- 特殊处理 `exc_info=True` 参数（映射到 `logger.opt(exception=True)`）

### 3. 日志输出模式

**开发环境 (`DEBUG=true`)**：
```
2025-12-22 21:29:12.742 | INFO     | __main__:_bind_and_log:53 | User login | {'user_id': 123, 'ip': '192.168.1.1'}
```

**生产环境 (`DEBUG=false`)**：
```json
{
  "text": "User login\n",
  "record": {
    "message": "User login",
    "level": {"name": "INFO"},
    "extra": {"user_id": 123, "ip": "192.168.1.1"},
    "time": {"timestamp": 1766410152.742447}
  }
}
```

## 兼容性验证

### 测试结果
运行 [test_loguru_migration.py](../ai-service/test_loguru_migration.py) 验证了以下场景：

✅ **Test 1**: 基础关键字参数  
✅ **Test 2**: 异常追踪 (`exc_info=True`)  
✅ **Test 3**: 多个结构化字段  
✅ **Test 4**: 无关键字参数的简单消息  
✅ **Test 5**: 复杂数据调试  
✅ **Test 6**: 模拟 conversation_service.py 实际调用  
✅ **Test 7**: exception() 方法  

**结论：166 处日志调用无需任何修改！**

## 新功能优势

### 相比 structlog 的改进

1. **更简洁的配置** - 无需复杂的 processors 链配置
2. **更好的性能** - 异步日志写入 (`enqueue=True`)
3. **自动日志轮转** - 内置 rotation/retention/compression 支持
4. **更友好的异常格式** - 彩色异常追踪，变量值自动捕获
5. **零依赖冲突** - 不依赖标准库 logging 模块

### 高级特性（可选使用）

```python
# 完整的 loguru 功能仍然可用
from loguru import logger

# 结构化绑定（链式调用）
logger.bind(request_id="abc123").info("Processing request")

# 格式化占位符
logger.info("User {user} logged in from {ip}", user=123, ip="192.168.1.1")

# 日志过滤器
logger.add("file.log", filter=lambda record: "password" not in record["message"])

# 自定义序列化
logger.add("api.log", serialize=True, format="{extra[request_id]} | {message}")
```

## 迁移检查清单

- [x] 替换 requirements.txt 中的依赖
- [x] 重写 logging.py 配置文件
- [x] 创建 LoguruAdapter 包装器
- [x] 验证关键字参数语法兼容性
- [x] 测试异常追踪功能
- [x] 确认所有模块正常导入

## 回滚方案（如需）

如果遇到问题，可快速回滚：

```bash
# 1. 恢复依赖
pip uninstall loguru
pip install structlog==24.1.0

# 2. Git 恢复配置文件
git checkout ai-service/app/core/logging.py
git checkout ai-service/requirements.txt
```

## 参考资料

- [Loguru 官方文档](https://loguru.readthedocs.io/)
- [测试脚本](../ai-service/test_loguru_migration.py)
- [项目日志使用研究报告](./项目总结.md#日志系统)

---

**迁移完成时间**: 2025-12-22  
**测试状态**: ✅ 全部通过  
**影响范围**: 0 处代码修改（完全向后兼容）
