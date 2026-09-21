#!/usr/bin/env bash
# 同步数学项目的运行必需源码到开发机 192.168.100.233。
#
# 只同步：应用 Python 源码、运行必需提示词/默认敏感词、requirements、Dockerfile、
# docker-compose 文件和 233 启动脚本。不会同步 .env、数据、日志、测试、工具、模型部署资产或 BGE。
# 不使用 --delete，远端已有的环境变量、虚拟环境、日志和数据不会被删除。

set -euo pipefail

REMOTE_HOST="192.168.100.233"
REMOTE_USER="zenking"
REMOTE_DIR="/data/metahuman_work/ZengKingMorphe-math"
LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ $# -gt 0 ]]; then
    echo "用法: $0 [--dry-run]" >&2
    exit 2
fi

RSYNC_OPTIONS=(
    -az
    --itemize-changes
    --human-readable
    --no-group
    --prune-empty-dirs
)
if "$DRY_RUN"; then
    RSYNC_OPTIONS+=(--dry-run)
fi

REMOTE_TARGET="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

sync_root_files() {
    rsync "${RSYNC_OPTIONS[@]}" \
        --include='docker-compose*.yml' \
        --include='Dockerfile*' \
        --include='start_ai_service_233.sh' \
        --exclude='*' \
        "${LOCAL_DIR}/" \
        "${REMOTE_TARGET}/"
}

sync_ai_service() {
    rsync "${RSYNC_OPTIONS[@]}" \
        --include='main.py' \
        --include='requirements.txt' \
        --include='Dockerfile*' \
        --include='.dockerignore' \
        --include='DEFAULT_SENSITIVE_WORDS.txt' \
        --include='app/' \
        --include='app/**/' \
        --include='app/**/*.py' \
        --include='prompts/' \
        --include='prompts/**/*.py' \
        --include='prompts/*.txt' \
        --exclude='*' \
        "${LOCAL_DIR}/ai-service/" \
        "${REMOTE_TARGET}/ai-service/"
}

echo "同步目标: ${REMOTE_TARGET}"
echo "同步范围: 应用源码、提示词、默认敏感词、requirements、Dockerfile、docker-compose、233 启动脚本"
echo "不同步: .env、数据、日志、tests、tools、bge-athenaeum、模型服务部署资产、其他脚本"
if "$DRY_RUN"; then
    echo "模式: 演练（不会写入远端）"
fi

if ! "$DRY_RUN"; then
    # 目标目录由用户指定；仅确保目录存在，不触碰目录中的既有内容。
    ssh -o BatchMode=yes "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}/ai-service'"
fi

echo "[1/2] 同步根目录 Compose 与 Dockerfile 文件"
sync_root_files

echo "[2/2] 同步 ai-service 运行必需文件"
sync_ai_service

echo "同步完成。未重启远端服务。"
