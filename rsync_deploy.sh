#!/bin/bash

# =============================================
# rsync 同步部署脚本
# =============================================

# 配置变量
REMOTE_HOST="192.168.8.233"
REMOTE_USER="zenking"
SSH_KEY="/home/zj/.ssh/id_rsa"
LOCAL_DIR="/mnt/d/zenking_work/metahuman_work/ZengKingMorphe"
REMOTE_DIR="/data/metahuman_work/ZengKingMorphe"

# rsync 选项
# --no-group: 跳过组权限设置（避免权限不足警告）
# --prune-empty-dirs: 删除空目录
RSYNC_OPTS="-avz --progress --delete --no-group --prune-empty-dirs"

echo "=========================================="
echo "开始同步到 ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
echo "=========================================="

# 同步 docker-compose.yml
echo ""
echo "[1/2] 同步 docker-compose.yml..."
rsync ${RSYNC_OPTS} -e "ssh -i ${SSH_KEY}" \
    --include="docker-compose.yml" \
    --exclude="*" \
    "${LOCAL_DIR}/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

# 同步 ai-service 目录（Python 代码 + Dockerfile + requirements.txt + .dockerignore）
echo ""
echo "[2/2] 同步 ai-service 目录..."
rsync ${RSYNC_OPTS} -e "ssh -i ${SSH_KEY}" \
    --exclude="__pycache__/" \
    --exclude="chroma_db/" \
    --exclude="logs/" \
    --exclude="test_files/" \
    --exclude=".pytest_cache/" \
    --include="*.py" \
    --include="Dockerfile" \
    --include="requirements.txt" \
    --include=".dockerignore" \
    --exclude="*" \
    "${LOCAL_DIR}/ai-service/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/ai-service/"

echo ""
echo "=========================================="
echo "同步完成！"
echo "=========================================="
