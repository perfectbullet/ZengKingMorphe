#!/usr/bin/env bash
# 启动 LightRAG Server + WebUI（文件后端，复用 ai-service 既有 working_dir 数据）
# 用法: ./run_server.sh    浏览器访问 http://localhost:9621
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"   # 让 server 的 load_dotenv(".env") 读到子项目清洁 .env，而非 ai-service/.env
# 防御：若外部 shell 环境带有 LIGHTRAG_*_STORAGE（如 ai-service/.env source 过），强制走文件后端
unset LIGHTRAG_KV_STORAGE LIGHTRAG_VECTOR_STORAGE LIGHTRAG_GRAPH_STORAGE LIGHTRAG_DOC_STATUS_STORAGE 2>/dev/null || true
set -a; . "$HERE/.env"; set +a
: "${WORKING_DIR:=/home/zj/ZengKingMorphe/ai-service/data/lightrag_manual_concepts}"
: "${HOST:=0.0.0.0}"
: "${PORT:=9621}"
exec lightrag-server --working-dir "$WORKING_DIR" --host "$HOST" --port "$PORT"
