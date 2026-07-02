#!/usr/bin/env bash
# 导入人工概念到 LightRAG（旧 ainsert 路径，带 --replace；对比验证用）
#
# 注意：此路径会触发默认 LLM 实体抽取，可能产生公式 / 变量 / 符号等脏实体，
#       仅作对比验证。生产请优先使用 run_import_custom_kg.sh。
#
# 用法: ./run_import.sh [--replace]
# 环境变量覆盖（优先级：调用方传入 > CONCEPT_RETRIEVAL_* > legacy 旧变量 > 默认值）：
#   CONCEPT_RETRIEVAL_CONFIG / CONFIG
#   CONCEPT_RETRIEVAL_LIGHTRAG_WORKING_DIR / WORKING_DIR
#   CONCEPT_RETRIEVAL_WHITELIST / ENTITY_WHITELIST
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# 1) 先捕获调用方通过环境变量传入的覆盖值（CONCEPT_RETRIEVAL_* 优先，其次 legacy 旧变量）
config_override="${CONCEPT_RETRIEVAL_CONFIG:-${CONFIG:-}}"
working_dir_override="${CONCEPT_RETRIEVAL_LIGHTRAG_WORKING_DIR:-${WORKING_DIR:-}}"
whitelist_override="${CONCEPT_RETRIEVAL_WHITELIST:-${ENTITY_WHITELIST:-}}"

# 2) 载入 .env（主要拿 LLM / Embedding 等配置）
if [[ -f "$HERE/.env" ]]; then
  set -a; . "$HERE/.env"; set +a
fi

# 3) 设定默认值（旧 ainsert 默认目录：lightrag_manual_concepts）
DEFAULT_CONFIG="/home/zj/ZengKingMorphe/ai-service/data/math_concepts/05_selective3_math_concepts_definition_blocks_with_concept_name_20260630.jsonl"
DEFAULT_WORKING_DIR="/home/zj/ZengKingMorphe/ai-service/data/lightrag_manual_concepts"
DEFAULT_WHITELIST="$HERE/entity_whitelist_draft.txt"
: "${CONFIG:=$DEFAULT_CONFIG}"
: "${WORKING_DIR:=$DEFAULT_WORKING_DIR}"
: "${ENTITY_WHITELIST:=$DEFAULT_WHITELIST}"

# 4) 回填调用方显式覆盖值（优先级最高）
if [[ -n "$config_override" ]]; then CONFIG="$config_override"; fi
if [[ -n "$working_dir_override" ]]; then WORKING_DIR="$working_dir_override"; fi
if [[ -n "$whitelist_override" ]]; then ENTITY_WHITELIST="$whitelist_override"; fi

exec python "$HERE/import_manual_concepts_lightrag.py" \
  --config "$CONFIG" \
  --working-dir "$WORKING_DIR" \
  --entity-whitelist "$ENTITY_WHITELIST" \
  --replace "$@"
