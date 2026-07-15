#!/bin/bash

# =============================================
# rsync 同步部署脚本
# =============================================

# 配置变量
REMOTE_HOST="192.168.100.233"
# REMOTE_HOST="10.1.0.100"
REMOTE_USER="zenking"

# 自动适配不同用户的路径
if [ "$(whoami)" = "summer" ]; then
    SSH_KEY="/Users/summer/.ssh/id_rsa"
    LOCAL_DIR="/Users/summer/Documents/metahuman_work/ZengKingMorphe"
else
    SSH_KEY="/home/zj/.ssh/id_rsa"
    LOCAL_DIR="/home/zj/ZengKingMorphe"
fi

REMOTE_DIR="/data/metahuman_work/ZengKingMorphe"

# 解析参数：-r 才重启
RESTART=false
while getopts "r" opt; do
    case $opt in
        r) RESTART=true ;;
    esac
done

# rsync 选项
# --no-group: 跳过组权限设置（避免权限不足警告）
# --prune-empty-dirs: 删除空目录
# 注意: 不使用 --delete，避免删除远程的缓存、配置等文件
RSYNC_OPTS="-avz --progress --no-group --prune-empty-dirs"

echo "=========================================="
echo "开始同步到 ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
echo "=========================================="


# 同步 docker-compose.yml
echo ""
echo "[1/2] 同步 docker-compose.yml..."
rsync ${RSYNC_OPTS} -e "ssh -i ${SSH_KEY}" \
    --include="docker-compose.yml" \
    --include="start_ai_service.sh" \
    --include="run_stream_test.sh" \
    --exclude="*" \
    "${LOCAL_DIR}/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

# 同步 ai-service 目录（Python 代码 + Dockerfile + requirements.txt + .dockerignore）
echo ""
echo "[2/2] 同步 ai-service 目录..."
rsync ${RSYNC_OPTS} -e "ssh -i ${SSH_KEY}" \
    --exclude="*.md" \
    --exclude="llama-rag-sdk" \
    --exclude="__pycache__/" \
    --exclude="chroma_db/" \
    --exclude="logs/" \
    --exclude="test_files/" \
    --exclude=".pytest_cache/" \
    --exclude="*.pyc" \
    --exclude="*.pyo" \
    --include="*/" \
    --include="**/*.py" \
    --include="Dockerfile" \
    --include="requirements.txt" \
    --include=".dockerignore" \
    --include="DEFAULT_SENSITIVE_WORDS.txt" \
    --include="run_stream_test.sh" \
    --exclude="*" \
    "${LOCAL_DIR}/ai-service/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/ai-service/"

rsync ${RSYNC_OPTS} -e "ssh -i ${SSH_KEY}" \
    --exclude="*.md" \
    --exclude=".claude" \
    --exclude="__pycache__/" \
    --exclude="chroma_db/" \
    --exclude="logs/" \
    --exclude="test_files/" \
    --exclude=".pytest_cache/" \
    --exclude="*.pyc" \
    --exclude="*.pyo" \
    --include="*/*" \
    --include="**/*.py" \
    --include="docker-compose.yml" \
    --include="requirements.txt" \
    --include=".dockerignore" \
    "${LOCAL_DIR}/bge-athenaeum/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/bge-athenaeum/"
    
# 同步 requirements.txt
echo ""
echo "[3/3] 同步并更新依赖..."
rsync ${RSYNC_OPTS} -e "ssh -i ${SSH_KEY}" \
    --include="requirements.txt" \
    --exclude="*" \
    "${LOCAL_DIR}/ai-service/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/ai-service/"


echo ""
echo "[4/4] 同步 zj-aha-vllm-v100"
rsync ${RSYNC_OPTS} -e "ssh -i ${SSH_KEY}" \
    "${LOCAL_DIR}/zj-aha-vllm-v100/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/zj-aha-vllm-v100/"


# 远程重启 AI 服务（仅 -r 时执行）
if $RESTART; then
    echo ""
    echo "=========================================="
    echo "正在远程重启 AI 服务..."
    echo "=========================================="
    ssh -i "${SSH_KEY}" "${REMOTE_USER}@${REMOTE_HOST}" "
        cd ${REMOTE_DIR} &&
        ./start_ai_service.sh restart
    "
else
    echo ""
    echo "提示: 使用 -r 参数可同步后自动重启 (例如: $0 -r)"
fi

echo ""
echo "=========================================="
echo "同步完成！"
echo "=========================================="

