#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash run_pipeline.sh /path/to/book.md [-step 1-8]" >&2
}

if [[ $# -ne 1 && $# -ne 3 ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MD_FILE="$(realpath "$1")"
SELECTED_STEP="all"
if [[ $# -eq 3 ]]; then
  if [[ "$2" != "-step" || ! "$3" =~ ^[1-8]$ ]]; then
    usage
    exit 2
  fi
  SELECTED_STEP="$3"
fi

ENV_FILE="$SCRIPT_DIR/../.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

BOOK_FILE="$(basename "$MD_FILE")"
BOOK_STEM="${BOOK_FILE%.md}"
SUBJECT="${MARKDOWN_GRAPH_SUBJECT:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/zj/miniconda3/envs/morphe/bin/python}"
WORKING_DIR="${MARKDOWN_GRAPH_WORKING_DIR:-/home/zj/ZengKingMorphe/ai-service/data/lightrag_industrial_training}"
DOMAIN="${MARKDOWN_GRAPH_DOMAIN:-industrial_training}"
FRONT_LINES="${STRUCTURE_FRONT_LINES:-1000}"
TOC_START_LINE="${STRUCTURE_TOC_START_LINE:-}"
TOC_END_LINE="${STRUCTURE_TOC_END_LINE:-}"
BODY_START_LINE="${STRUCTURE_BODY_START_LINE:-}"
MAX_LINES="${MAX_BLOCK_LINES_WARN:-500}"
MAX_CHARS="${MAX_BLOCK_CHARS_WARN:-12000}"
IMPORT_METHOD="${MARKDOWN_GRAPH_IMPORT_METHOD:-custom_chunks}"
VALIDATE_QUERY="${MARKDOWN_GRAPH_VALIDATE_QUERY:-请概述本教材的核心概念和主要工艺流程}"
TOP_K_CANDIDATES="${TOP_K_CANDIDATE_TITLE_LINES:-12}"
MIN_ANCHOR_CONFIDENCE="${MIN_ANCHOR_CONFIDENCE:-0.65}"
MAX_UNMATCHED="${MAX_UNMATCHED_CATALOG_ITEMS:-0}"
USE_EXISTING_ANCHOR_PLAN="${MARKDOWN_GRAPH_USE_EXISTING_ANCHOR_PLAN:-false}"

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
      outline_args=(--prepared "$PREPARED" --output "$OUTLINE" --front-lines "$FRONT_LINES")
      if [[ -n "$TOC_START_LINE" || -n "$TOC_END_LINE" || -n "$BODY_START_LINE" ]]; then
        if [[ -z "$TOC_START_LINE" || -z "$TOC_END_LINE" || -z "$BODY_START_LINE" ]]; then
          echo "STRUCTURE_TOC_START_LINE, STRUCTURE_TOC_END_LINE and STRUCTURE_BODY_START_LINE must be set together" >&2
          exit 2
        fi
        outline_args+=(--toc-start-line "$TOC_START_LINE" --toc-end-line "$TOC_END_LINE" --body-start-line "$BODY_START_LINE")
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
      "$PYTHON_BIN" "$SCRIPT_DIR/04_apply_structure_plan.py" --prepared "$PREPARED" --outline "$OUTLINE" --structure-plan "$RAW_PLAN" --output-blocks "$BLOCKS" --output-plan "$VALIDATED_PLAN" --report "$VALIDATION_REPORT" --domain "$DOMAIN" --subject "$SUBJECT" --max-block-lines-warn "$MAX_LINES" --max-block-chars-warn "$MAX_CHARS"
      ;;
    5)
      echo "[5/8] Build LightRAG custom chunks"
      "$PYTHON_BIN" "$SCRIPT_DIR/05_build_lightrag_chunks.py" --blocks "$BLOCKS" --output "$CHUNKS" --domain "$DOMAIN" --subject "$SUBJECT"
      ;;
    6)
      echo "[6/8] Import custom chunks into LightRAG"
      "$PYTHON_BIN" "$SCRIPT_DIR/06_import_custom_chunks_to_lightrag.py" --chunks "$CHUNKS" --working-dir "$WORKING_DIR" --domain "$DOMAIN" --subject "$SUBJECT" --import-method "$IMPORT_METHOD" --replace
      ;;
    7)
      echo "[7/8] Validate graph"
      "$PYTHON_BIN" "$SCRIPT_DIR/07_validate_graph.py" --working-dir "$WORKING_DIR" --domain "$DOMAIN" --subject "$SUBJECT" --book-stem "$BOOK_STEM" --query "$VALIDATE_QUERY"
      ;;
    8)
      echo "[8/8] Audit graph quality"
      "$PYTHON_BIN" "$SCRIPT_DIR/08_audit_graph_quality.py" --book-stem "$BOOK_STEM" --domain "$DOMAIN" --subject "$SUBJECT" --output-md "$AUDIT_MD" --output-json "$AUDIT_JSON"
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
