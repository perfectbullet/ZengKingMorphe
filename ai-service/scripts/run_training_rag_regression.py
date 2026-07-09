from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QAItem:
    qa_id: str
    question: str
    qa_type: str
    expected_keywords: list[str]
    must_include: list[str]
    must_not_include: list[str]
    source_hint: str


def _split_csv_like(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[，,]", text or "") if part.strip()]


def _parse_qa_file(path: Path) -> list[QAItem]:
    if not path.is_file():
        raise FileNotFoundError(f"TRAINING_RAG_QA_FILE 不存在: {path}")

    items: list[QAItem] = []
    current: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("# ") or line.startswith("#\t"):
            continue
        if line.startswith("## "):
            if current:
                items.append(
                    QAItem(
                        qa_id=current.get("qa_id", "QA-UNKNOWN"),
                        question=current.get("question", ""),
                        qa_type=current.get("type", ""),
                        expected_keywords=_split_csv_like(current.get("expected_keywords", "")),
                        must_include=_split_csv_like(current.get("must_include", "")),
                        must_not_include=_split_csv_like(current.get("must_not_include", "")),
                        source_hint=current.get("source_hint", ""),
                    )
                )
            current = {"qa_id": line.replace("##", "", 1).strip()}
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        current[key.strip()] = value.strip()
    if current:
        items.append(
            QAItem(
                qa_id=current.get("qa_id", "QA-UNKNOWN"),
                question=current.get("question", ""),
                qa_type=current.get("type", ""),
                expected_keywords=_split_csv_like(current.get("expected_keywords", "")),
                must_include=_split_csv_like(current.get("must_include", "")),
                must_not_include=_split_csv_like(current.get("must_not_include", "")),
                source_hint=current.get("source_hint", ""),
            )
        )
    return items


def _latest_file_after(directory: Path, pattern: str, started_at: float) -> Path | None:
    candidates = [p for p in directory.glob(pattern) if p.stat().st_mtime >= started_at - 1]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_raw_debug(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out: dict[str, Any] = {}
    for row in rows:
        stage = row.get("stage")
        if stage:
            out[stage] = row
    return out


def _short_text(text: str | None, max_len: int = 240) -> str:
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    return cleaned[:max_len] + ("..." if len(cleaned) > max_len else "")


def _contains_any(text: str, values: list[str]) -> list[str]:
    return [value for value in values if value and value in text]


def _detect_outside_knowledge(answer: str) -> bool:
    markers = ["根据常识", "通常来说", "一般来说", "广泛用于", "许多国家", "全球"]
    return any(marker in answer for marker in markers)


def _run_single_question(question: str) -> dict[str, Any]:
    started_at = time.time()
    cmd = [sys.executable, "-m", "tests.test_chat_stream_v1", "-q", question]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout = proc.stdout
    stderr = proc.stderr

    finish_path = _latest_file_after(ROOT / "finish_chunk_data", "*.json", started_at)
    debug_context_path = _latest_file_after(ROOT / "debug_contexts", "*.md", started_at)
    raw_debug_path = _latest_file_after(ROOT / "logs" / "rag_stream_debug", "*.raw.jsonl", started_at)
    finish = _load_json(finish_path)
    raw_debug = _load_raw_debug(raw_debug_path)
    metadata = finish.get("metadata") or {}
    sources = metadata.get("sources") or []
    chunk_source = next((src for src in sources if src.get("type") == "chunk"), {})
    citations = chunk_source.get("citations") or []
    before = raw_debug.get("before_aquery_llm") or {}
    after = raw_debug.get("after_aquery_llm") or {}
    info_match = re.search(r"📊 RAG召回: backend=([^,]+), entities=(\d+), relationships=(\d+), chunks=(\d+), references=(\d+), candidate_chunks=(\d+|None), final_chunks=(\d+|None)", stdout)
    mode_match = re.search(r"Using RAG stream \| backend=([^|]+)\| include_history=(true|false) \| query=.*? \| mode=([^\n]+)", stdout)
    label_match = re.search(r"LLM classification: label=([^,]+)", stdout)

    return {
        "question": question,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": proc.returncode,
        "route_intent": label_match.group(1).strip() if label_match else None,
        "entered_lightrag_file": bool(mode_match and "lightrag_file" in mode_match.group(1)),
        "mode": mode_match.group(3).strip() if mode_match else None,
        "original_query": before.get("original_query"),
        "effective_query": before.get("effective_query"),
        "aquery_llm_query_is_original": before.get("aquery_llm_query_is_original"),
        "keyword_injection_mode": before.get("keyword_injection_mode"),
        "injected_hl_keywords": before.get("injected_hl_keywords") or [],
        "injected_ll_keywords": before.get("injected_ll_keywords") or [],
        "low_level_keywords": (after.get("keyword_debug") or {}).get("low_level_keywords") or [],
        "high_level_keywords": (after.get("keyword_debug") or {}).get("high_level_keywords") or [],
        "entities_count": int(info_match.group(2)) if info_match else 0,
        "relationships_count": int(info_match.group(3)) if info_match else 0,
        "chunks_count": int(info_match.group(4)) if info_match else 0,
        "references_count": int(info_match.group(5)) if info_match else 0,
        "candidate_chunks_count": info_match.group(6) if info_match else None,
        "final_chunks_count": info_match.group(7) if info_match else None,
        "chunk_file_paths": [c.get("file_path") for c in citations if c.get("file_path")],
        "chunk_citations": citations,
        "answer": _extract_answer(stdout),
        "finish_path": str(finish_path) if finish_path else "",
        "debug_context_path": str(debug_context_path) if debug_context_path else "",
        "raw_debug_path": str(raw_debug_path) if raw_debug_path else "",
    }


def _extract_answer(stdout: str) -> str:
    if "响应预览:" in stdout:
        return stdout.split("响应预览:", 1)[1].split("============================================================", 1)[0].strip()
    return stdout[-800:].strip()


def _build_report_rows(qa_items: list[QAItem], results: list[dict[str, Any]]) -> str:
    lines = ["# 工训教材通用问答回归报告", ""]
    lines.append(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- TRAINING_RAG_BACKEND: {os.getenv('TRAINING_RAG_BACKEND', '')}")
    lines.append(f"- TRAINING_RAG_KEYWORD_INJECTION_MODE: {os.getenv('TRAINING_RAG_KEYWORD_INJECTION_MODE', '')}")
    lines.append(f"- TRAINING_RAG_ENTITY_TERMS_FILE: {os.getenv('TRAINING_RAG_ENTITY_TERMS_FILE', '')}")
    lines.append(f"- TRAINING_RAG_QA_FILE: {os.getenv('TRAINING_RAG_QA_FILE', '')}")
    lines.append("")
    for qa, result in zip(qa_items, results):
        answer = result["answer"]
        must_include_hit = _contains_any(answer, qa.must_include)
        must_not_include_hit = _contains_any(answer, qa.must_not_include)
        lines.extend(
            [
                f"## {qa.qa_id} {qa.question}",
                "",
                f"- type: {qa.qa_type}",
                f"- route / intent: {result.get('route_intent')}",
                f"- entered_lightrag_file: {result.get('entered_lightrag_file')}",
                f"- mode: {result.get('mode')}",
                f"- original_query: {result.get('original_query')}",
                f"- effective_query: {result.get('effective_query')}",
                f"- aquery_llm_query_is_original: {result.get('aquery_llm_query_is_original')}",
                f"- keyword_injection_mode: {result.get('keyword_injection_mode')}",
                f"- injected_ll_keywords: {result.get('injected_ll_keywords')}",
                f"- injected_hl_keywords: {result.get('injected_hl_keywords')}",
                f"- low_level_keywords: {result.get('low_level_keywords')}",
                f"- high_level_keywords: {result.get('high_level_keywords')}",
                f"- entities_count: {result.get('entities_count')}",
                f"- relationships_count: {result.get('relationships_count')}",
                f"- chunks_count: {result.get('chunks_count')}",
                f"- references_count: {result.get('references_count')}",
                f"- candidate_chunks_count: {result.get('candidate_chunks_count')}",
                f"- final_chunks_count: {result.get('final_chunks_count')}",
                f"- final chunk file paths: {result.get('chunk_file_paths')}",
                f"- expected_keywords: {qa.expected_keywords}",
                f"- must_include hit: {must_include_hit}",
                f"- must_not_include hit: {must_not_include_hit}",
                f"- obvious outside knowledge: {_detect_outside_knowledge(answer)}",
                f"- context insufficient triggered: {'当前知识库资料不足' in answer}",
                f"- answer preview: {_short_text(answer, 500)}",
                f"- latest debug_context path: {result.get('debug_context_path')}",
                f"- latest finish_chunk_data path: {result.get('finish_path')}",
                f"- latest raw debug path: {result.get('raw_debug_path')}",
                "",
            ]
        )
    return "\n".join(lines)


async def main() -> None:
    qa_file = (os.getenv("TRAINING_RAG_QA_FILE", "") or "").strip()
    if not qa_file:
        raise SystemExit("TRAINING_RAG_QA_FILE 未设置")
    qa_path = Path(qa_file).expanduser()
    if not qa_path.is_absolute():
        qa_path = (ROOT / qa_path).resolve()

    qa_items = _parse_qa_file(qa_path)
    results = []
    for qa in qa_items:
        logger.info(f"Running regression QA | {qa.qa_id} | {qa.question}")
        results.append(_run_single_question(qa.question))

    report_name = f"{qa_path.stem.replace('_QA.example', '')}_generalization_regression.md"
    report_path = ROOT / "logs" / "rag_stream_debug" / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_build_report_rows(qa_items, results), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    asyncio.run(main())
