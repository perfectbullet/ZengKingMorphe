#!/usr/bin/env bash
# 测试手动数学概念结构化召回 / 严格回答
# 用法: QUERY="请帮我讲解二项式定理" MODE=local ./run_test.sh [--answer]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
config_override="${CONFIG:-}"
working_dir_override="${WORKING_DIR:-}"
whitelist_override="${ENTITY_WHITELIST:-}"
query_override="${QUERY:-}"
mode_override="${MODE:-}"
set -a; . "$HERE/.env"; set +a
: "${CONFIG:=/home/zj/ZengKingMorphe/ai-service/data/math_concepts/05_selective3_math_concepts_definition_blocks_with_concept_name_20260630.jsonl}"
: "${WORKING_DIR:=/home/zj/ZengKingMorphe/ai-service/data/lightrag_manual_concepts}"
: "${ENTITY_WHITELIST:=$HERE/entity_whitelist_draft.txt}"
: "${QUERY:=请帮我讲解二项式定理}"
: "${MODE:=local}"
if [[ -n "$config_override" ]]; then CONFIG="$config_override"; fi
if [[ -n "$working_dir_override" ]]; then WORKING_DIR="$working_dir_override"; fi
if [[ -n "$whitelist_override" ]]; then ENTITY_WHITELIST="$whitelist_override"; fi
if [[ -n "$query_override" ]]; then QUERY="$query_override"; fi
if [[ -n "$mode_override" ]]; then MODE="$mode_override"; fi
exec python "$HERE/test_manual_concept_lightrag.py" \
  --config "$CONFIG" \
  --working-dir "$WORKING_DIR" \
  --entity-whitelist "$ENTITY_WHITELIST" \
  --query "$QUERY" \
  --mode "$MODE" \
  "$@"
