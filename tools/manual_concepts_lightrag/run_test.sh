#!/usr/bin/env bash
# 测试人工概念在 LightRAG 中的结构化召回 / 严格回答（默认指向 lightrag_manual_concepts）
#
# 用法: QUERY="请帮我讲解二项式定理" MODE=local ./run_test.sh [--answer]
#
# 环境变量覆盖（优先级：调用方传入 > CONCEPT_RETRIEVAL_* > legacy 旧变量 > 默认值）：
#   CONCEPT_RETRIEVAL_CONFIG / CONFIG
#   CONCEPT_RETRIEVAL_LIGHTRAG_WORKING_DIR / WORKING_DIR
#   CONCEPT_RETRIEVAL_WHITELIST / ENTITY_WHITELIST
#   QUERY / MODE
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# 1) 先捕获调用方通过环境变量传入的覆盖值（CONCEPT_RETRIEVAL_* 优先，其次 legacy 旧变量）
config_override="${CONCEPT_RETRIEVAL_CONFIG:-${CONFIG:-}}"
working_dir_override="${CONCEPT_RETRIEVAL_LIGHTRAG_WORKING_DIR:-${WORKING_DIR:-}}"
whitelist_override="${CONCEPT_RETRIEVAL_WHITELIST:-${ENTITY_WHITELIST:-}}"
query_override="${QUERY:-}"
mode_override="${MODE:-}"

# 2) 载入 .env（主要拿 LLM / Embedding / Rerank 等配置）
if [[ -f "$HERE/.env" ]]; then
  set -a; . "$HERE/.env"; set +a
fi

# 3) 设定默认值；WORKING_DIR 默认使用主路径目录
DEFAULT_CONFIG="/home/zj/ZengKingMorphe/ai-service/data/math_concepts/05_selective3_math_concepts_definition_blocks_with_concept_name_20260630.jsonl"
WORKING_DIR="${working_dir_override:-/home/zj/ZengKingMorphe/ai-service/data/lightrag_manual_concepts}"
: "${CONFIG:=$DEFAULT_CONFIG}"
: "${ENTITY_WHITELIST:=$HERE/entity_whitelist_draft.txt}"
: "${QUERY:=请帮我讲解二项式定理}"
: "${MODE:=local}"

# 4) 回填调用方显式覆盖值（优先级最高）
if [[ -n "$config_override" ]]; then CONFIG="$config_override"; fi
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
