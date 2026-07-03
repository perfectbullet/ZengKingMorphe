#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash run_pipeline.sh /path/to/book.md [subject]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MD_FILE="$(realpath "$1")"
SUBJECT="${2:-${MARKDOWN_GRAPH_SUBJECT:-}}"
BOOK_FILE="$(basename "$MD_FILE")"
BOOK_STEM="${BOOK_FILE%.md}"
PYTHON_BIN="${PYTHON_BIN:-/home/zj/miniconda3/envs/morphe/bin/python}"
WORKING_DIR="${MARKDOWN_GRAPH_WORKING_DIR:-/home/zj/ZengKingMorphe/ai-service/data/lightrag_industrial_training}"
DOMAIN="${MARKDOWN_GRAPH_DOMAIN:-industrial_training}"
FRONT_LINES="${STRUCTURE_FRONT_LINES:-1000}"
WINDOW_LINES="${STRUCTURE_WINDOW_LINES:-320}"
OVERLAP_LINES="${STRUCTURE_OVERLAP_LINES:-60}"
MAX_LINES="${MAX_BLOCK_LINES_WARN:-500}"
MAX_CHARS="${MAX_BLOCK_CHARS_WARN:-12000}"
IMPORT_METHOD="${MARKDOWN_GRAPH_IMPORT_METHOD:-custom_chunks}"
VALIDATE_QUERY="${MARKDOWN_GRAPH_VALIDATE_QUERY:-请概述本教材的核心概念和主要工艺流程}"

PREPARED="$SCRIPT_DIR/outputs/01_prepared/$BOOK_STEM.prepared.json"
OUTLINE="$SCRIPT_DIR/outputs/02_outline/$BOOK_STEM.book_outline.json"
RAW_PLAN="$SCRIPT_DIR/outputs/03_structure_plan/$BOOK_STEM.structure_plan.raw.jsonl"
VALIDATED_PLAN="$SCRIPT_DIR/outputs/04_blocks/$BOOK_STEM.structure_plan.validated.jsonl"
BLOCKS="$SCRIPT_DIR/outputs/04_blocks/$BOOK_STEM.blocks.jsonl"
VALIDATION_REPORT="$SCRIPT_DIR/outputs/04_blocks/$BOOK_STEM.validation_report.md"
CHUNKS="$SCRIPT_DIR/outputs/05_chunks/$BOOK_STEM.lightrag_chunks.jsonl"

echo "[1/7] Prepare Markdown and image index"
"$PYTHON_BIN" "$SCRIPT_DIR/01_prepare_markdown.py" --md-file "$MD_FILE" --output "$PREPARED"

echo "[2/7] Detect front matter and table of contents"
"$PYTHON_BIN" "$SCRIPT_DIR/02_detect_front_matter_and_toc.py" --prepared "$PREPARED" --output "$OUTLINE" --front-lines "$FRONT_LINES"

echo "[3/7] Build LLM structure plan"
"$PYTHON_BIN" "$SCRIPT_DIR/03_llm_structure_plan.py" --prepared "$PREPARED" --outline "$OUTLINE" --output "$RAW_PLAN" --window-lines "$WINDOW_LINES" --overlap-lines "$OVERLAP_LINES"

echo "[4/7] Validate and apply structure plan"
"$PYTHON_BIN" "$SCRIPT_DIR/04_apply_structure_plan.py" --prepared "$PREPARED" --outline "$OUTLINE" --structure-plan "$RAW_PLAN" --output-blocks "$BLOCKS" --output-plan "$VALIDATED_PLAN" --report "$VALIDATION_REPORT" --domain "$DOMAIN" --subject "$SUBJECT" --max-block-lines-warn "$MAX_LINES" --max-block-chars-warn "$MAX_CHARS"

echo "[5/7] Build LightRAG custom chunks"
"$PYTHON_BIN" "$SCRIPT_DIR/05_build_lightrag_chunks.py" --blocks "$BLOCKS" --output "$CHUNKS" --domain "$DOMAIN" --subject "$SUBJECT"

echo "[6/7] Import custom chunks into LightRAG"
"$PYTHON_BIN" "$SCRIPT_DIR/06_import_custom_chunks_to_lightrag.py" --chunks "$CHUNKS" --working-dir "$WORKING_DIR" --domain "$DOMAIN" --subject "$SUBJECT" --import-method "$IMPORT_METHOD" --replace

echo "[7/7] Validate graph"
"$PYTHON_BIN" "$SCRIPT_DIR/07_validate_graph.py" --working-dir "$WORKING_DIR" --domain "$DOMAIN" --subject "$SUBJECT" --book-stem "$BOOK_STEM" --query "$VALIDATE_QUERY"

echo "Pipeline completed. Intermediate artifacts remain under $SCRIPT_DIR/outputs"
