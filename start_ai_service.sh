#!/bin/bash
# =============================================================================
# AI Service 启动脚本
# =============================================================================
# 用途：后台启动 ai-service，支持 start/stop/restart/status
# =============================================================================

# 配置变量
# 自动获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR"
VENV_PATH="$WORK_DIR/venv"
SERVICE_DIR="$WORK_DIR/ai-service"
PYTHONPATH="$SERVICE_DIR"
HOST="0.0.0.0"
PORT=8100
PID_FILE="$SERVICE_DIR/.ai_service.pid"
LOG_FILE="$SERVICE_DIR/logs/ai_service.log"
# 仅监视 Python 源码，避免配置目录存在非 UTF-8 文件名时 watchfiles 崩溃。
UVICORN_ARGS="--reload --reload-dir app --log-level info"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# 辅助函数
# =============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 TCP 端口是否可达（超时 3 秒）
check_port() {
    local host="$1"
    local port="$2"
    local name="$3"
    if timeout 3 bash -c "echo > /dev/tcp/$host/$port" 2>/dev/null; then
        log_info "  ✅ $name ($host:$port) 可达"
        return 0
    else
        log_error "  ❌ $name ($host:$port) 不可达"
        return 1
    fi
}

# 检查进程是否运行（通过 PID 文件）
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# 检查端口是否被占用，返回占用该端口的 PID 列表
get_port_pids() {
    local pids
    pids=$(lsof -t -i :$PORT 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "$pids"
        return 0
    fi
    return 1
}

# 通过 lsof 杀掉占用端口的进程
kill_port_occupant() {
    local pids
    pids=$(get_port_pids)
    if [ -n "$pids" ]; then
        log_warn "端口 $PORT 被占用，占用进程 PID: $(echo $pids | tr '\n' ' ')"
        for pid in $pids; do
            log_info "正在终止进程 $pid ..."
            kill "$pid" 2>/dev/null
        done
        # 等待进程退出
        for i in {1..10}; do
            pids=$(get_port_pids)
            if [ -z "$pids" ]; then
                log_info "端口 $PORT 已释放"
                return 0
            fi
            sleep 1
        done
        # 强制杀掉
        pids=$(get_port_pids)
        if [ -n "$pids" ]; then
            log_warn "进程未退出，强制终止..."
            for pid in $pids; do
                kill -9 "$pid" 2>/dev/null
            done
            sleep 1
            pids=$(get_port_pids)
            if [ -z "$pids" ]; then
                log_info "端口 $PORT 已释放"
                return 0
            else
                log_error "无法释放端口 $PORT"
                return 1
            fi
        fi
    fi
    return 0
}

# =============================================================================
# 启动服务
# =============================================================================
start_service() {
    if is_running; then
        log_warn "服务已在运行中 (PID: $(cat $PID_FILE))"
        return 1
    fi

    # 检查端口是否被其他进程占用
    local port_pids
    port_pids=$(get_port_pids)
    if [ -n "$port_pids" ]; then
        log_warn "端口 $PORT 被其他进程占用，尝试释放..."
        kill_port_occupant || return 1
    fi

    log_info "启动 ai-service..."

    # ── 依赖服务健康检查 ──
    log_info "检查依赖服务..."
    local failed=0
    check_port 192.168.100.233 27017 "MongoDB"       || failed=$((failed+1))
    check_port 192.168.100.233 9200  "ElasticSearch" || failed=$((failed+1))
    check_port 192.168.100.233 19530 "Milvus"        || failed=$((failed+1))
    check_port 192.168.100.233 7687  "Neo4j"         || failed=$((failed+1))
    # check_port 192.168.8.231 11434 "Ollama"        || failed=$((failed+1))

    if [ $failed -gt 0 ]; then
        log_error "$failed 个依赖服务不可达，是否继续启动？(y/N)"
        read -r -t 10 answer
        if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
            log_error "启动已取消"
            return 1
        fi
        log_warn "跳过检查，继续启动..."
    fi

    # 检查目录是否存在
    if [ ! -d "$SERVICE_DIR" ]; then
        log_error "服务目录不存在: $SERVICE_DIR"
        return 1
    fi

    # 检查虚拟环境
    if [ ! -f "$VENV_PATH/bin/activate" ]; then
        log_error "虚拟环境不存在: $VENV_PATH"
        return 1
    fi

    # 创建日志目录（确保权限正确）
    LOG_DIR="$(dirname "$LOG_FILE")"
    # 如果目录存在但权限不对，重新创建
    if [ -d "$LOG_DIR" ]; then
        # 检查是否可写
        if [ ! -w "$LOG_DIR" ]; then
            log_warn "日志目录权限异常，尝试修复..."
            rm -rf "$LOG_DIR" 2>/dev/null || true
        fi
    fi
    mkdir -p "$LOG_DIR"

    # 切换到服务目录并启动
    cd "$SERVICE_DIR" || exit 1

    # 启动服务（后台运行）
    nohup bash -c "
        source '$VENV_PATH/bin/activate'
        export PYTHONPATH='$PYTHONPATH'
        exec uvicorn main:app --host $HOST --port $PORT $UVICORN_ARGS
    " >> "$LOG_FILE" 2>&1 &

    PID=$!
    echo $PID > "$PID_FILE"

    # 等待启动
    sleep 2

    if is_running; then
        log_info "服务启动成功！"
        log_info "  PID: $PID"
        log_info "  地址: http://$HOST:$PORT"
        log_info "  日志: $LOG_FILE"
        log_info "  查看日志: tail -f $LOG_FILE"
        return 0
    else
        log_error "服务启动失败，请查看日志: $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

# =============================================================================
# 停止服务
# =============================================================================
stop_service() {
    if ! is_running; then
        # PID 文件不存在，但仍检查端口是否被占用
        local port_pids
        port_pids=$(get_port_pids)
        if [ -n "$port_pids" ]; then
            log_warn "PID 文件不存在但端口 $PORT 仍被占用，尝试清理..."
            kill_port_occupant || return 1
            return 0
        fi
        log_warn "服务未运行"
        return 1
    fi

    PID=$(cat "$PID_FILE")
    log_info "停止服务 (PID: $PID)..."

    kill "$PID" 2>/dev/null

    # 等待进程结束
    for i in {1..30}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            rm -f "$PID_FILE"
            log_info "服务已停止"
            # 确认端口已释放
            local port_pids
            port_pids=$(get_port_pids)
            if [ -n "$port_pids" ]; then
                log_warn "端口 $PORT 仍被占用，清理残留进程..."
                kill_port_occupant
            fi
            return 0
        fi
        sleep 1
    done

    # 强制结束
    log_warn "强制停止服务..."
    kill -9 "$PID" 2>/dev/null
    rm -f "$PID_FILE"
    # 最终确认端口释放
    kill_port_occupant
    log_info "服务已强制停止"
}

# =============================================================================
# 重启服务
# =============================================================================
restart_service() {
    log_info "重启服务..."
    stop_service
    sleep 2
    start_service
}

# =============================================================================
# 查看状态
# =============================================================================
status_service() {
    if is_running; then
        PID=$(cat "$PID_FILE")
        log_info "服务正在运行"
        log_info "  PID: $PID"
        log_info "  地址: http://$HOST:$PORT"

        # 显示进程信息
        ps -p "$PID" -o pid,ppid,cmd,etime,stat --no-headers 2>/dev/null | while read line; do
            echo "  进程: $line"
        done

        # 显示端口占用信息
        local port_pids
        port_pids=$(get_port_pids)
        if [ -n "$port_pids" ]; then
            log_info "  端口 $PORT 占用详情:"
            lsof -i :$PORT 2>/dev/null | while read line; do
                echo "    $line"
            done
        fi
        return 0
    else
        log_warn "服务未运行"
        # 检查端口是否被其他进程占用
        local port_pids
        port_pids=$(get_port_pids)
        if [ -n "$port_pids" ]; then
            log_warn "端口 $PORT 被其他进程占用:"
            lsof -i :$PORT 2>/dev/null | while read line; do
                echo "    $line"
            done
        fi
        return 1
    fi
}

# =============================================================================
# 查看日志
# =============================================================================
logs_service() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        log_error "日志文件不存在: $LOG_FILE"
    fi
}

# =============================================================================
# 主函数
# =============================================================================

case "${1:-start}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        status_service
        ;;
    logs)
        logs_service
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "命令说明:"
        echo "  start   - 启动服务（后台运行）"
        echo "  stop    - 停止服务"
        echo "  restart - 重启服务"
        echo "  status  - 查看服务状态"
        echo "  logs    - 实时查看日志"
        exit 1
        ;;
esac
