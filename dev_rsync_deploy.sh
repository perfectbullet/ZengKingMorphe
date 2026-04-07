#!/bin/bash

# =============================================
# rsync 同步部署脚本
# =============================================

# 配置变量
REMOTE_HOST="192.168.8.233"
REMOTE_USER="zenking"
SSH_KEY="/home/zj/.ssh/id_rsa"
LOCAL_DIR="/home/zj/ZengKingMorphe"
REMOTE_DIR="/data/metahuman_work/ZengKingMorphe"

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
    --include=".env" \
    --include=".env.example" \
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
    --include=".env" \
    --include=".env.example" \
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

# 远程更新依赖
echo "正在远程服务器上更新依赖..."
ssh -i "${SSH_KEY}" "${REMOTE_USER}@${REMOTE_HOST}" "
    cd ${REMOTE_DIR} &&
    source venv/bin/activate &&
    pip install --upgrade pip &&
    pip install -r ai-service/requirements.txt
"

echo ""
echo "=========================================="
echo "同步完成！"
echo "=========================================="
