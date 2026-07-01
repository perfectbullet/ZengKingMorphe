#!/usr/bin/env bash
# 导入手动数学概念到 LightRAG（默认带 --replace）
# 用法: ./run_import.sh [--replace]   覆盖: CONFIG=... WORKING_DIR=... ./run_import.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
config_override="${CONFIG:-}"
working_dir_override="${WORKING_DIR:-}"
whitelist_override="${ENTITY_WHITELIST:-}"
set -a; . "$HERE/.env"; set +a
: "${CONFIG:=/home/zj/ZengKingMorphe/ai-service/data/math_concepts/05_selective3_math_concepts_definition_blocks_with_concept_name_20260630.jsonl}"
: "${WORKING_DIR:=/home/zj/ZengKingMorphe/ai-service/data/lightrag_manual_concepts}"
: "${ENTITY_WHITELIST:=$HERE/entity_whitelist_draft.txt}"
if [[ -n "$config_override" ]]; then CONFIG="$config_override"; fi
if [[ -n "$working_dir_override" ]]; then WORKING_DIR="$working_dir_override"; fi
if [[ -n "$whitelist_override" ]]; then ENTITY_WHITELIST="$whitelist_override"; fi
exec python "$HERE/import_manual_concepts_lightrag.py" \
  --config "$CONFIG" \
  --working-dir "$WORKING_DIR" \
  --entity-whitelist "$ENTITY_WHITELIST" \
  --replace "$@"
