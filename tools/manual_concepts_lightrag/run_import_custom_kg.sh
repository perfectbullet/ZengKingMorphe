#!/usr/bin/env bash
# 导入手动数学概念到 LightRAG（custom KG 模式，不走 LLM 实体抽取）
# 用法: ./run_import_custom_kg.sh [--dry-run]   覆盖: CONFIG=... WORKING_DIR=... ./run_import_custom_kg.sh
#
# 注意：.env 中也定义了 WORKING_DIR（指向旧目录 lightrag_manual_concepts）。
# 若直接 `: "${WORKING_DIR:=...}"`，.env 的旧值会顶掉默认值，导致写入旧目录。
# 因此先记录调用方覆盖值，source .env 后再强制把 WORKING_DIR 默认指向新的
# custom KG 目录，仅当调用方显式传入 WORKING_DIR 时才覆盖。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# 1) 先捕获调用方通过环境变量传入的覆盖值
config_override="${CONFIG:-}"
working_dir_override="${WORKING_DIR:-}"
whitelist_override="${ENTITY_WHITELIST:-}"
batch_size_override="${BATCH_SIZE:-}"

# 2) 载入 .env（主要拿 LLM / Embedding 等配置；WORKING_DIR 会被 .env 的旧值覆盖，下一步再修正）
set -a
. "$HERE/.env"
set +a

# 3) 设定默认值；WORKING_DIR 强制使用 custom KG 新目录（不沿用 .env 的旧值）
: "${CONFIG:=/home/zj/ZengKingMorphe/ai-service/data/math_concepts/05_selective3_math_concepts_definition_blocks_with_concept_name_20260630.jsonl}"
WORKING_DIR="${working_dir_override:-/home/zj/ZengKingMorphe/ai-service/data/lightrag_manual_concepts_custom_kg}"
: "${ENTITY_WHITELIST:=$HERE/entity_whitelist_draft.txt}"
: "${BATCH_SIZE:=100}"

# 4) 回填调用方显式覆盖值
if [[ -n "$config_override" ]]; then CONFIG="$config_override"; fi
if [[ -n "$whitelist_override" ]]; then ENTITY_WHITELIST="$whitelist_override"; fi
if [[ -n "$batch_size_override" ]]; then BATCH_SIZE="$batch_size_override"; fi

exec python "$HERE/import_manual_concepts_custom_kg.py" \
  --config "$CONFIG" \
  --working-dir "$WORKING_DIR" \
  --entity-whitelist "$ENTITY_WHITELIST" \
  --batch-size "$BATCH_SIZE" \
  "$@"
