#!/usr/bin/env bash
# 导入手动数学概念到 LightRAG（默认带 --replace）
# 用法: ./run_import.sh [--replace]   覆盖: CONFIG=... WORKING_DIR=... ./run_import.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
set -a; . "$HERE/.env"; set +a
: "${CONFIG:=/home/zj/ZengKingMorphe/ai-service/data/math_concepts/05_selective3_math_concepts_definition_blocks_with_concept_name_20260630.jsonl}"
: "${WORKING_DIR:=/home/zj/ZengKingMorphe/ai-service/data/lightrag_manual_concepts}"
exec python "$HERE/import_manual_concepts_lightrag.py" \
  --config "$CONFIG" --working-dir "$WORKING_DIR" --replace "$@"
