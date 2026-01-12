from loguru import logger
import sys
import time

# -------------------------- 1. 基础配置（核心） --------------------------
# 移除默认的控制台输出（避免重复日志）
logger.remove()

# 添加控制台输出配置
logger.add(
    sys.stdout,  # 输出到标准输出（控制台）
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",  # 日志格式
    level="DEBUG",  # 控制台只输出 DEBUG 及以上级别日志
    colorize=True,  # 开启控制台日志颜色
)

# 添加文件输出配置（按文件大小轮转）
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",  # 日志文件命名（按日期）
    rotation="500 MB",  # 单个文件达到 500MB 自动轮转
    retention="7 days",  # 保留 7 天的日志文件
    compression="zip",  # 过期日志自动压缩为 zip
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{line} - {message}",
    level="INFO",  # 文件只记录 INFO 及以上级别日志
    encoding="utf-8",  # 解决中文乱码
)

# 添加错误日志单独输出（按时间轮转）
logger.add(
    "logs/error_{time:YYYY-MM-DD}.log",
    rotation="00:00",  # 每天 0 点自动轮转新文件
    retention="14 days",
    compression="zip",
    level="ERROR",  # 只记录 ERROR 及以上级别
    encoding="utf-8",
)

# -------------------------- 2. 不同级别日志使用 --------------------------
def demo_log_levels():
    """演示不同级别日志的输出"""
    logger.trace("这是 TRACE 级别（最详细，控制台/文件都不输出）")
    logger.debug("这是 DEBUG 级别（仅控制台输出）")
    logger.info("这是 INFO 级别（控制台+文件都输出）")
    logger.success("这是 SUCCESS 级别（loguru 特有，控制台+文件都输出）")
    logger.warning("这是 WARNING 级别（控制台+文件都输出）")
    logger.error("这是 ERROR 级别（控制台+文件+错误日志都输出）")
    logger.critical("这是 CRITICAL 级别（最高级，所有输出都包含）")

# -------------------------- 3. 异常捕获（loguru 亮点） --------------------------
def demo_exception_catch():
    """演示 loguru 便捷的异常捕获"""
    # 方式1：直接记录异常（无需手动捕获）
    try:
        1 / 0  # 触发除零异常
    except Exception as e:
        # 自动记录完整堆栈信息
        logger.exception("发生除零异常：{}", e)

    # 方式2：装饰器自动捕获函数内异常
    @logger.catch
    def risky_function():
        lst = [1, 2, 3]
        print(lst[10])  # 触发索引越界异常

    risky_function()  # 调用函数，异常会被自动记录

# -------------------------- 4. 自定义日志上下文（可选） --------------------------
def demo_custom_context():
    """演示添加自定义上下文信息（如用户ID）"""
    # 绑定上下文，后续日志会包含该信息
    user_logger = logger.bind(user_id="10086", module="payment")
    user_logger.info("用户发起支付请求")
    user_logger.warning("用户支付超时，已重试")

# -------------------------- 执行 demo --------------------------
if __name__ == "__main__":
    logger.info("===== 开始执行 Loguru Demo =====")
    
    # 执行各功能演示
    demo_log_levels()
    time.sleep(1)  # 避免日志时间重叠
    demo_exception_catch()
    time.sleep(1)
    demo_custom_context()
    
    logger.info("===== Loguru Demo 执行完成 =====")
