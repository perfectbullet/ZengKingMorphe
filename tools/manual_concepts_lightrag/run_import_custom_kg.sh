#!/usr/bin/env bash
# 导入人工概念到 LightRAG（custom KG 模式，不走 LLM 实体抽取，历史实验性路径）
#
# 用法: ./run_import_custom_kg.sh [--dry-run]
#
# 注意：.env 中可能也定义了 WORKING_DIR（指向旧目录 lightrag_manual_concepts）。
# 若直接 `: "${WORKING_DIR:=...}"`，.env 的旧值会顶掉默认值，导致写入旧目录。
# 因此先记录调用方覆盖值，source .env 后再强制把 WORKING_DIR 默认指向新的
# custom KG 目录，仅当调用方显式传入 WORKING_DIR 时才覆盖。
#
# 环境变量覆盖（优先级：调用方传入 > CONCEPT_RETRIEVAL_* > legacy 旧变量 > 默认值）：
#   CONCEPT_RETRIEVAL_CONFIG / CONFIG
#   CONCEPT_RETRIEVAL_LIGHTRAG_WORKING_DIR / WORKING_DIR
#   CONCEPT_RETRIEVAL_WHITELIST / ENTITY_WHITELIST
#   CONCEPT_RETRIEVAL_DOMAIN / DOMAIN
#   CONCEPT_RETRIEVAL_ENTITY_TYPE / ENTITY_TYPE
#   BATCH_SIZE
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# 1) 先捕获调用方通过环境变量传入的覆盖值（CONCEPT_RETRIEVAL_* 优先，其次 legacy 旧变量）
config_override="${CONCEPT_RETRIEVAL_CONFIG:-${CONFIG:-}}"
working_dir_override="${CONCEPT_RETRIEVAL_LIGHTRAG_WORKING_DIR:-${WORKING_DIR:-}}"
whitelist_override="${CONCEPT_RETRIEVAL_WHITELIST:-${ENTITY_WHITELIST:-}}"
batch_size_override="${BATCH_SIZE:-}"
domain_override="${CONCEPT_RETRIEVAL_DOMAIN:-${DOMAIN:-}}"
entity_type_override="${CONCEPT_RETRIEVAL_ENTITY_TYPE:-${ENTITY_TYPE:-}}"

# 2) 载入 .env（主要拿 LLM / Embedding 等配置；WORKING_DIR 会被 .env 旧值覆盖，下一步再修正）
if [[ -f "$HERE/.env" ]]; then
  set -a; . "$HERE/.env"; set +a
fi

# 3) 设定默认值；WORKING_DIR 强制使用历史实验性 custom KG 目录（不沿用 .env 的旧值）
DEFAULT_CONFIG="/home/zj/ZengKingMorphe/ai-service/data/math_concepts/05_selective3_math_concepts_definition_blocks_with_concept_name_20260630.jsonl"
WORKING_DIR="${working_dir_override:-/home/zj/ZengKingMorphe/ai-service/data/lightrag_manual_concepts_custom_kg}"
: "${CONFIG:=$DEFAULT_CONFIG}"
: "${ENTITY_WHITELIST:=$HERE/entity_whitelist_draft.txt}"
: "${BATCH_SIZE:=100}"

# 4) 回填调用方显式覆盖值（优先级最高）
if [[ -n "$config_override" ]]; then CONFIG="$config_override"; fi
if [[ -n "$whitelist_override" ]]; then ENTITY_WHITELIST="$whitelist_override"; fi
if [[ -n "$batch_size_override" ]]; then BATCH_SIZE="$batch_size_override"; fi

# 5) 构造 --domain / --entity-type（仅在调用方显式传入时透传给脚本，否则脚本内部用 env / 默认）
DOMAIN_ARGS=()
if [[ -n "$domain_override" ]]; then DOMAIN_ARGS+=(--domain "$domain_override"); fi
if [[ -n "$entity_type_override" ]]; then DOMAIN_ARGS+=(--entity-type "$entity_type_override"); fi

if [[ ${#DOMAIN_ARGS[@]} -gt 0 ]]; then
  exec python "$HERE/import_manual_concepts_custom_kg.py" \
    --config "$CONFIG" \
    --working-dir "$WORKING_DIR" \
    --entity-whitelist "$ENTITY_WHITELIST" \
    --batch-size "$BATCH_SIZE" \
    "${DOMAIN_ARGS[@]}" \
    "$@"
else
  exec python "$HERE/import_manual_concepts_custom_kg.py" \
    --config "$CONFIG" \
    --working-dir "$WORKING_DIR" \
    --entity-whitelist "$ENTITY_WHITELIST" \
    --batch-size "$BATCH_SIZE" \
    "$@"
fi
