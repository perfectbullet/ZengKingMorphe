#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: bash run_pipeline.sh /path/to/book.md [options] [-step 1-8]

Required:
  --working-dir PATH          Required for steps 6, 7, or all
  --domain VALUE              Required
  --subject VALUE             Required

Step 5:
  --content-list-v2 PATH      Required for step 5 or all

General options:
  --front-lines INT                         Default: 1000
  --toc-start-line INT --toc-end-line INT --body-start-line INT
  --max-block-lines-warn INT                Default: 500
  --max-block-chars-warn INT                Default: 12000
  --import-method custom_chunks             Default: custom_chunks
  --query-max-tokens INT                    Default: 4096

Step 6 debug selection:
  --entity-types-guidance-file PATH         Optional Step 6 LightRAG guidance file
  --max-kg-chunks INT
  --start-chunk-index INT                   Default: 0
  --chunk-title-contains TEXT
  --chunk-id ID                             Repeatable
  --chunk-id-file PATH
  --dry-run-selected-chunks
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

need_arg() {
  local option="$1"
  local value="${2:-}"
  [[ -n "$value" && "$value" != --* ]] || die "$option requires a value"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MD_FILE="$(realpath "$1")"
INPUT_DIR="$(dirname "$MD_FILE")"
shift

SELECTED_STEP="all"
CLI_CONTENT_LIST_V2=""
CLI_WORKING_DIR=""
CLI_DOMAIN=""
CLI_SUBJECT=""
CLI_FRONT_LINES="1000"
CLI_TOC_START_LINE=""
CLI_TOC_END_LINE=""
CLI_BODY_START_LINE=""
CLI_MAX_LINES="500"
CLI_MAX_CHARS="12000"
CLI_IMPORT_METHOD="custom_chunks"
CLI_QUERY_MAX_TOKENS="4096"
CLI_ENTITY_TYPES_GUIDANCE_FILE=""
CLI_MAX_KG_CHUNKS=""
CLI_START_CHUNK_INDEX="0"
CLI_CHUNK_TITLE_CONTAINS=""
CLI_CHUNK_ID_FILE=""
CLI_DRY_RUN_SELECTED_CHUNKS="false"
CLI_CHUNK_IDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -step)
      need_arg "$1" "${2:-}"
      [[ "$2" =~ ^[1-8]$ ]] || die "-step must be 1-8"
      SELECTED_STEP="$2"
      shift 2
      ;;
    --content-list-v2)
      need_arg "$1" "${2:-}"
      CLI_CONTENT_LIST_V2="$2"
      shift 2
      ;;
    --working-dir)
      need_arg "$1" "${2:-}"
      CLI_WORKING_DIR="$2"
      shift 2
      ;;
    --domain)
      need_arg "$1" "${2:-}"
      CLI_DOMAIN="$2"
      shift 2
      ;;
    --subject)
      need_arg "$1" "${2:-}"
      CLI_SUBJECT="$2"
      shift 2
      ;;
    --front-lines)
      need_arg "$1" "${2:-}"
      CLI_FRONT_LINES="$2"
      shift 2
      ;;
    --toc-start-line)
      need_arg "$1" "${2:-}"
      CLI_TOC_START_LINE="$2"
      shift 2
      ;;
    --toc-end-line)
      need_arg "$1" "${2:-}"
      CLI_TOC_END_LINE="$2"
      shift 2
      ;;
    --body-start-line)
      need_arg "$1" "${2:-}"
      CLI_BODY_START_LINE="$2"
      shift 2
      ;;
    --max-block-lines-warn)
      need_arg "$1" "${2:-}"
      CLI_MAX_LINES="$2"
      shift 2
      ;;
    --max-block-chars-warn)
      need_arg "$1" "${2:-}"
      CLI_MAX_CHARS="$2"
      shift 2
      ;;
    --import-method)
      need_arg "$1" "${2:-}"
      CLI_IMPORT_METHOD="$2"
      shift 2
      ;;
    --query-max-tokens)
      need_arg "$1" "${2:-}"
      CLI_QUERY_MAX_TOKENS="$2"
      shift 2
      ;;
    --entity-types-guidance-file)
      need_arg "$1" "${2:-}"
      CLI_ENTITY_TYPES_GUIDANCE_FILE="$2"
      shift 2
      ;;
    --max-kg-chunks)
      need_arg "$1" "${2:-}"
      CLI_MAX_KG_CHUNKS="$2"
      shift 2
      ;;
    --start-chunk-index)
      need_arg "$1" "${2:-}"
      CLI_START_CHUNK_INDEX="$2"
      shift 2
      ;;
    --chunk-title-contains)
      need_arg "$1" "${2:-}"
      CLI_CHUNK_TITLE_CONTAINS="$2"
      shift 2
      ;;
    --chunk-id)
      need_arg "$1" "${2:-}"
      CLI_CHUNK_IDS+=("$2")
      shift 2
      ;;
    --chunk-id-file)
      need_arg "$1" "${2:-}"
      CLI_CHUNK_ID_FILE="$2"
      shift 2
      ;;
    --dry-run-selected-chunks)
      CLI_DRY_RUN_SELECTED_CHUNKS="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

ENV_FILE="$SCRIPT_DIR/../.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-/home/zj/miniconda3/envs/morphe/bin/python}"
VALIDATE_QUERY="${MARKDOWN_GRAPH_VALIDATE_QUERY:-请概述本教材的核心概念和主要工艺流程}"
TOP_K_CANDIDATES="${TOP_K_CANDIDATE_TITLE_LINES:-12}"
MIN_ANCHOR_CONFIDENCE="${MIN_ANCHOR_CONFIDENCE:-0.65}"
MAX_UNMATCHED="${MAX_UNMATCHED_CATALOG_ITEMS:-0}"
USE_EXISTING_ANCHOR_PLAN="${MARKDOWN_GRAPH_USE_EXISTING_ANCHOR_PLAN:-false}"
SCHEMA_VERSION="${MARKDOWN_GRAPH_SCHEMA_VERSION:-industrial_training_kg_schema.v1}"
SCHEMA_JSON="${MARKDOWN_GRAPH_SCHEMA_JSON:-$SCRIPT_DIR/prompts/schemas/industrial_training_kg_schema.v1.json}"

[[ -n "$CLI_DOMAIN" ]] || die "--domain is required"
[[ -n "$CLI_SUBJECT" ]] || die "--subject is required"

if [[ "$SELECTED_STEP" == "5" || "$SELECTED_STEP" == "all" ]]; then
  if [[ -z "$CLI_CONTENT_LIST_V2" ]]; then
    die "step 5 requires --content-list-v2. Example: ./run_pipeline.sh \"\$INPUT\" --content-list-v2 \"\$V2\" --working-dir ... --domain ... --subject ... -step 5"
  fi
  [[ -f "$CLI_CONTENT_LIST_V2" ]] || die "--content-list-v2 must be an existing file: $CLI_CONTENT_LIST_V2"
  CLI_CONTENT_LIST_V2="$(realpath "$CLI_CONTENT_LIST_V2")"
fi

if [[ "$SELECTED_STEP" == "6" || "$SELECTED_STEP" == "7" || "$SELECTED_STEP" == "all" ]]; then
  [[ -n "$CLI_WORKING_DIR" ]] || die "step $SELECTED_STEP requires --working-dir"
fi

toc_count=0
[[ -n "$CLI_TOC_START_LINE" ]] && toc_count=$((toc_count + 1))
[[ -n "$CLI_TOC_END_LINE" ]] && toc_count=$((toc_count + 1))
[[ -n "$CLI_BODY_START_LINE" ]] && toc_count=$((toc_count + 1))
if (( toc_count > 0 && toc_count < 3 )); then
  die "--toc-start-line, --toc-end-line, and --body-start-line must be provided together"
fi

if [[ -n "$CLI_CHUNK_ID_FILE" ]]; then
  [[ -f "$CLI_CHUNK_ID_FILE" ]] || die "--chunk-id-file must be an existing file: $CLI_CHUNK_ID_FILE"
  CLI_CHUNK_ID_FILE="$(realpath "$CLI_CHUNK_ID_FILE")"
fi

if [[ -n "$CLI_ENTITY_TYPES_GUIDANCE_FILE" ]]; then
  [[ -f "$CLI_ENTITY_TYPES_GUIDANCE_FILE" ]] || die "--entity-types-guidance-file must be an existing file: $CLI_ENTITY_TYPES_GUIDANCE_FILE"
  CLI_ENTITY_TYPES_GUIDANCE_FILE="$(realpath "$CLI_ENTITY_TYPES_GUIDANCE_FILE")"
fi

BOOK_FILE="$(basename "$MD_FILE")"
BOOK_STEM="${BOOK_FILE%.md}"

PREPARED="$SCRIPT_DIR/outputs/01_prepared/$BOOK_STEM.prepared.json"
OUTLINE="$SCRIPT_DIR/outputs/02_outline/$BOOK_STEM.book_outline.json"
RAW_PLAN="$SCRIPT_DIR/outputs/03_structure_plan/$BOOK_STEM.structure_plan.raw.jsonl"
ANCHOR_PLAN="$SCRIPT_DIR/outputs/03_structure_plan/$BOOK_STEM.catalog_anchor_plan.jsonl"
UNMATCHED_REPORT="$SCRIPT_DIR/outputs/03_structure_plan/$BOOK_STEM.unmatched_report.md"
VALIDATED_PLAN="$SCRIPT_DIR/outputs/04_blocks/$BOOK_STEM.structure_plan.validated.jsonl"
BLOCKS="$SCRIPT_DIR/outputs/04_blocks/$BOOK_STEM.blocks.jsonl"
VALIDATION_REPORT="$SCRIPT_DIR/outputs/04_blocks/$BOOK_STEM.validation_report.md"
CHUNKS="$SCRIPT_DIR/outputs/05_chunks/$BOOK_STEM.lightrag_chunks.jsonl"
AUDIT_MD="$SCRIPT_DIR/outputs/08_audit/$BOOK_STEM.graph_audit.md"
AUDIT_JSON="$SCRIPT_DIR/outputs/08_audit/$BOOK_STEM.graph_audit.json"

run_step() {
  case "$1" in
    1)
      echo "[1/8] Prepare Markdown and image index"
      "$PYTHON_BIN" "$SCRIPT_DIR/01_prepare_markdown.py" --md-file "$MD_FILE" --output "$PREPARED"
      ;;
    2)
      echo "[2/8] Detect front matter and table of contents"
      outline_args=(--prepared "$PREPARED" --output "$OUTLINE" --front-lines "$CLI_FRONT_LINES")
      if (( toc_count == 3 )); then
        outline_args+=(--toc-start-line "$CLI_TOC_START_LINE" --toc-end-line "$CLI_TOC_END_LINE" --body-start-line "$CLI_BODY_START_LINE")
      fi
      "$PYTHON_BIN" "$SCRIPT_DIR/02_detect_front_matter_and_toc.py" "${outline_args[@]}"
      ;;
    3)
      echo "[3/8] Build catalog-driven structure plan"
      if [[ "$USE_EXISTING_ANCHOR_PLAN" == "true" && -f "$ANCHOR_PLAN" ]]; then
        "$PYTHON_BIN" "$SCRIPT_DIR/03_catalog_structure_plan.py" --prepared "$PREPARED" --outline "$OUTLINE" --output "$RAW_PLAN" --anchor-plan "$ANCHOR_PLAN" --unmatched-report "$UNMATCHED_REPORT" --use-existing-anchor-plan "$ANCHOR_PLAN" --top-k-candidates "$TOP_K_CANDIDATES" --min-anchor-confidence "$MIN_ANCHOR_CONFIDENCE" --max-unmatched "$MAX_UNMATCHED"
      else
        "$PYTHON_BIN" "$SCRIPT_DIR/03_catalog_structure_plan.py" --prepared "$PREPARED" --outline "$OUTLINE" --output "$RAW_PLAN" --anchor-plan "$ANCHOR_PLAN" --unmatched-report "$UNMATCHED_REPORT" --top-k-candidates "$TOP_K_CANDIDATES" --min-anchor-confidence "$MIN_ANCHOR_CONFIDENCE" --max-unmatched "$MAX_UNMATCHED"
      fi
      ;;
    4)
      echo "[4/8] Validate and apply structure plan"
      "$PYTHON_BIN" "$SCRIPT_DIR/04_apply_structure_plan.py" --prepared "$PREPARED" --outline "$OUTLINE" --structure-plan "$RAW_PLAN" --output-blocks "$BLOCKS" --output-plan "$VALIDATED_PLAN" --report "$VALIDATION_REPORT" --domain "$CLI_DOMAIN" --subject "$CLI_SUBJECT" --max-block-lines-warn "$CLI_MAX_LINES" --max-block-chars-warn "$CLI_MAX_CHARS"
      ;;
    5)
      echo "[5/8] Build LightRAG custom chunks"
      [[ -n "$CLI_CONTENT_LIST_V2" && -f "$CLI_CONTENT_LIST_V2" ]] || die "step 5 requires --content-list-v2. Example: ./run_pipeline.sh \"\$INPUT\" --content-list-v2 \"\$V2\" --working-dir ... --domain ... --subject ... -step 5"
      "$PYTHON_BIN" "$SCRIPT_DIR/05_build_lightrag_chunks.py" --blocks "$BLOCKS" --output "$CHUNKS" --domain "$CLI_DOMAIN" --subject "$CLI_SUBJECT" --mineru-dir "$INPUT_DIR" --content-list-v2 "$CLI_CONTENT_LIST_V2"
      ;;
    6)
      echo "[6/8] Import custom chunks into LightRAG"
      debug_args=()
      [[ -n "$CLI_ENTITY_TYPES_GUIDANCE_FILE" ]] && debug_args+=(--entity-types-guidance-file "$CLI_ENTITY_TYPES_GUIDANCE_FILE")
      [[ -n "$CLI_MAX_KG_CHUNKS" ]] && debug_args+=(--max-kg-chunks "$CLI_MAX_KG_CHUNKS")
      debug_args+=(--start-chunk-index "$CLI_START_CHUNK_INDEX")
      [[ -n "$CLI_CHUNK_TITLE_CONTAINS" ]] && debug_args+=(--chunk-title-contains "$CLI_CHUNK_TITLE_CONTAINS")
      for chunk_id in "${CLI_CHUNK_IDS[@]}"; do
        debug_args+=(--chunk-id "$chunk_id")
      done
      [[ -n "$CLI_CHUNK_ID_FILE" ]] && debug_args+=(--chunk-id-file "$CLI_CHUNK_ID_FILE")
      [[ "$CLI_DRY_RUN_SELECTED_CHUNKS" == "true" ]] && debug_args+=(--dry-run-selected-chunks)
      "$PYTHON_BIN" "$SCRIPT_DIR/06_import_custom_chunks_to_lightrag.py" --chunks "$CHUNKS" --working-dir "$CLI_WORKING_DIR" --domain "$CLI_DOMAIN" --subject "$CLI_SUBJECT" --import-method "$CLI_IMPORT_METHOD" --schema-version "$SCHEMA_VERSION" --schema-json "$SCHEMA_JSON" --replace "${debug_args[@]}"
      ;;
    7)
      echo "[7/8] Validate graph"
      "$PYTHON_BIN" "$SCRIPT_DIR/07_validate_graph.py" --working-dir "$CLI_WORKING_DIR" --domain "$CLI_DOMAIN" --subject "$CLI_SUBJECT" --book-stem "$BOOK_STEM" --query "$VALIDATE_QUERY" --query-max-tokens "$CLI_QUERY_MAX_TOKENS"
      ;;
    8)
      echo "[8/8] Audit graph quality"
      "$PYTHON_BIN" "$SCRIPT_DIR/08_audit_graph_quality.py" --book-stem "$BOOK_STEM" --domain "$CLI_DOMAIN" --subject "$CLI_SUBJECT" --schema-version "$SCHEMA_VERSION" --schema-json "$SCHEMA_JSON" --output-md "$AUDIT_MD" --output-json "$AUDIT_JSON"
      ;;
  esac
}

if [[ "$SELECTED_STEP" == "all" ]]; then
  for step in {1..8}; do
    run_step "$step"
  done
  echo "Pipeline completed. Intermediate artifacts remain under $SCRIPT_DIR/outputs"
else
  run_step "$SELECTED_STEP"
  echo "Step $SELECTED_STEP completed. Intermediate artifacts remain under $SCRIPT_DIR/outputs"
fi
