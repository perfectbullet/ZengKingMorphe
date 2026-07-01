#!/usr/bin/env bash
# 测试手动数学概念结构化召回 / 严格回答
# 用法: QUERY="请帮我讲解二项式定理" MODE=local ./run_test.sh [--answer]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
set -a; . "$HERE/.env"; set +a
: "${CONFIG:=/home/zj/ZengKingMorphe/ai-service/data/math_concepts/05_selective3_math_concepts_definition_blocks_with_concept_name_20260630.jsonl}"
: "${WORKING_DIR:=/home/zj/ZengKingMorphe/ai-service/data/lightrag_manual_concepts}"
: "${QUERY:=请帮我讲解二项式定理}"
: "${MODE:=local}"
exec python "$HERE/test_manual_concept_lightrag.py" \
  --config "$CONFIG" --working-dir "$WORKING_DIR" --query "$QUERY" --mode "$MODE" "$@"
