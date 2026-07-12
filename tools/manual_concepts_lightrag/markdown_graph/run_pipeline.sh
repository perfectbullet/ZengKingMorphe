#!/usr/bin/env bash
# Inspectable Step 0-12 Markdown -> LightRAG pipeline. Business configuration
# is resolved by book meta (CLI > meta > env > defaults), never by shell cwd.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zj/miniconda3/envs/morphe/bin/python}"
die(){ echo "ERROR: $*" >&2; exit 2; }
usage(){ cat >&2 <<'EOF'
Usage:
  bash run_pipeline.sh --init-meta MARKDOWN [--domain D] [--subject S] [--force-meta] [--force-entity-types]
  bash run_pipeline.sh --meta BOOK_meta.json [-step 1..12|all] [--working-dir PATH]
  bash run_pipeline.sh --meta BOOK_meta.json --confirm-step 4
EOF
}
INIT_MD=""; META=""; STEP="all"; CONFIRM=""; WORKING_DIR=""; DOMAIN=""; SUBJECT=""; CONTENT_V2=""; ENTITY_PROFILE=""; DRY_RUN="false"
FORWARD=()
while [[ $# -gt 0 ]]; do
 case "$1" in
  --init-meta) INIT_MD="${2:-}"; shift 2;; --meta) META="${2:-}"; shift 2;; -step) STEP="${2:-}"; shift 2;; --confirm-step) CONFIRM="${2:-}"; shift 2;;
  --working-dir) WORKING_DIR="${2:-}"; shift 2;; --domain) DOMAIN="${2:-}"; shift 2;; --subject) SUBJECT="${2:-}"; shift 2;; --content-list-v2) CONTENT_V2="${2:-}"; shift 2;; --entity-type-prompt-file) ENTITY_PROFILE="${2:-}"; shift 2;;
  --force-meta|--force-entity-types) FORWARD+=("$1"); shift;; --dry-run) DRY_RUN="true"; shift;; -h|--help) usage; exit 0;; *) FORWARD+=("$1"); shift;; esac
done
if [[ -n "$INIT_MD" ]]; then
 args=("$INIT_MD"); [[ -n "$DOMAIN" ]]&&args+=(--domain "$DOMAIN"); [[ -n "$SUBJECT" ]]&&args+=(--subject "$SUBJECT"); "$PYTHON_BIN" "$SCRIPT_DIR/00_init_book_meta.py" "${args[@]}" "${FORWARD[@]}"; exit 0
fi
[[ -n "$META" && -f "$META" ]] || die "必须传 --meta 已存在的教材 meta，或使用 --init-meta MARKDOWN"
META="$(realpath "$META")"
readarray -t META_VALUES < <("$PYTHON_BIN" - "$META" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parents[0]))
from book_meta import load_book_meta, resolve_book_paths, get_business_config
m=load_book_meta(sys.argv[1]); p=resolve_book_paths(m); b=get_business_config(m)
print(p['markdown']); print(m['artifact_stem']); print(b['domain']); print(b['subject'])
PY
)
MD_FILE="${META_VALUES[0]}"; STEM="${META_VALUES[1]}"; META_DOMAIN="${META_VALUES[2]}"; META_SUBJECT="${META_VALUES[3]}"
DOMAIN="${DOMAIN:-$META_DOMAIN}"; SUBJECT="${SUBJECT:-$META_SUBJECT}"
[[ -n "$DOMAIN" && -n "$SUBJECT" ]] || die "meta 或 CLI 必须提供 domain 和 subject"
OUT="$SCRIPT_DIR/outputs"; PREPARED="$OUT/01_prepared/$STEM.prepared.json"; OUTLINE="$OUT/02_outline/$STEM.book_outline.json"; RAW="$OUT/03_structure_plan/$STEM.structure_plan.raw.jsonl"; ANCHOR="$OUT/03_structure_plan/$STEM.catalog_anchor_plan.jsonl"; UNMATCHED="$OUT/03_structure_plan/$STEM.unmatched_report.md"; VALIDATED="$OUT/04_blocks/$STEM.structure_plan.validated.jsonl"; BLOCKS="$OUT/04_blocks/$STEM.blocks.jsonl"; REPORT="$OUT/04_blocks/$STEM.validation_report.md"; CONFIRMATION="$OUT/04_blocks/$STEM.validation_confirmation.json"; CHUNKS="$OUT/05_chunks/$STEM.lightrag_chunks.jsonl"
if [[ -n "$CONFIRM" ]]; then [[ "$CONFIRM" == 4 ]] || die "当前仅支持 --confirm-step 4"; "$PYTHON_BIN" "$SCRIPT_DIR/confirm_step4.py" --confirmation "$CONFIRMATION"; exit 0; fi
run(){ case "$1" in
  1) echo '[1/12] Prepare Markdown'; "$PYTHON_BIN" "$SCRIPT_DIR/01_prepare_markdown.py" --md-file "$MD_FILE" --output "$PREPARED";;
  2) echo '[2/12] Extract catalog in confirmed range'; "$PYTHON_BIN" "$SCRIPT_DIR/02_detect_front_matter_and_toc.py" --meta "$META" --prepared "$PREPARED" --output "$OUTLINE";;
  3) echo '[3/12] Plan catalog anchors'; args=(--prepared "$PREPARED" --outline "$OUTLINE" --output "$RAW" --anchor-plan "$ANCHOR" --unmatched-report "$UNMATCHED"); [[ "${MARKDOWN_GRAPH_USE_EXISTING_ANCHOR_PLAN:-false}" == true && -f "$ANCHOR" ]] && args+=(--use-existing-anchor-plan "$ANCHOR"); "$PYTHON_BIN" "$SCRIPT_DIR/03_catalog_structure_plan.py" "${args[@]}";;
  4) echo '[4/12] Validate blocks'; "$PYTHON_BIN" "$SCRIPT_DIR/04_apply_structure_plan.py" --prepared "$PREPARED" --outline "$OUTLINE" --structure-plan "$RAW" --output-blocks "$BLOCKS" --output-plan "$VALIDATED" --report "$REPORT" --confirmation "$CONFIRMATION" --domain "$DOMAIN" --subject "$SUBJECT";;
  5) echo '[5/12] Build chunks'; args=(--blocks "$BLOCKS" --output "$CHUNKS" --meta "$META"); [[ -n "$CONTENT_V2" ]]&&args+=(--content-list-v2 "$CONTENT_V2"); "$PYTHON_BIN" "$SCRIPT_DIR/05_build_lightrag_chunks.py" "${args[@]}";;
  6) [[ -n "$WORKING_DIR" ]]||die 'Step 6 requires --working-dir'; echo '[6/12] Import custom chunks'; args=(--chunks "$CHUNKS" --meta "$META" --working-dir "$WORKING_DIR" --domain "$DOMAIN" --subject "$SUBJECT" --replace); [[ -n "$ENTITY_PROFILE" ]]&&args+=(--entity-type-prompt-file "$ENTITY_PROFILE"); "$PYTHON_BIN" "$SCRIPT_DIR/06_import_custom_chunks_to_lightrag.py" "${args[@]}" "${FORWARD[@]}";;
  7) [[ -n "$WORKING_DIR" ]]||die 'Step 7 requires --working-dir'; echo '[7/12] Validate graph'; "$PYTHON_BIN" "$SCRIPT_DIR/07_validate_graph.py" --working-dir "$WORKING_DIR" --domain "$DOMAIN" --subject "$SUBJECT" --book-stem "$STEM";;
  8) [[ -n "$WORKING_DIR" ]]||die 'Step 8 requires --working-dir'; echo '[8/12] Audit graph'; "$PYTHON_BIN" "$SCRIPT_DIR/08_audit_graph_quality.py" --meta "$META" --book-stem "$STEM" --domain "$DOMAIN" --subject "$SUBJECT";;
  9) [[ -n "$WORKING_DIR" ]]||die 'Step 9 requires --working-dir'; echo '[9/12] Export records'; "$PYTHON_BIN" "$SCRIPT_DIR/09_export_graph_records.py" --meta "$META" --working-dir "$WORKING_DIR" --chunks "$CHUNKS";;
  10) echo '[10/12] Adversarial model review'; args=(--meta "$META"); [[ "$DRY_RUN" == true ]]&&args+=(--dry-run); "$PYTHON_BIN" "$SCRIPT_DIR/10_adversarial_entity_review.py" "${args[@]}";;
  11) echo '[11/12] Prepare human entity review'; "$PYTHON_BIN" "$SCRIPT_DIR/11_prepare_entity_review.py" --meta "$META";;
  12) echo '[12/12] Build ASR hotwords'; "$PYTHON_BIN" "$SCRIPT_DIR/12_build_asr_hotwords.py" --meta "$META";;
 esac; }
if [[ "$STEP" != all ]]; then
 [[ "$STEP" =~ ^([1-9]|1[0-2])$ ]]||die '-step must be 1..12 or all'
 if [[ "$STEP" == 5 ]]; then "$PYTHON_BIN" - "$CONFIRMATION" <<'PY'
import json,sys
p=sys.argv[1]
assert __import__('pathlib').Path(p).is_file() and json.load(open(p)).get('confirmed'), 'Step 5 前请查看 validation_report 后执行 --confirm-step 4'
PY
 fi
 run "$STEP"; exit 0
fi
if ! "$PYTHON_BIN" - "$CONFIRMATION" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); raise SystemExit(0 if p.is_file() and json.load(p.open()).get('confirmed') else 1)
PY
then
  for n in 1 2 3 4; do run "$n"; done
  echo 'Step 4 已完成。请查看 validation_report，确认后执行 --confirm-step 4，再次运行 -step all。'
  exit 0
fi
for n in 5 6 7 8 9 10 11; do run "$n"; done
echo 'Step 11 已准备人工审核 JSONL。完成实体人工审核后，再运行 -step 12。'
