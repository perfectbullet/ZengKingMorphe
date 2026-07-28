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


def _split_csv_like(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[，,]", text or "") if part.strip()]


def _split_semicolon_like(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;；]", text or "") if part.strip()]


def _resolve_books_config_path() -> Path:
    value = (
        os.getenv(
            "TRAINING_RAG_BOOKS_CONFIG",
            str(ROOT / "config" / "training_books.json"),
        )
        or ""
    ).strip()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _load_books_config() -> dict[str, dict[str, Any]]:
    path = _resolve_books_config_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed to load books config: {path} | error={exc}")
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in data.get("books") or []:
        if isinstance(item, dict) and item.get("book_id"):
            out[str(item["book_id"]).strip()] = item
    return out


def _parse_header_metadata(lines: list[str]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("#"):
            continue
        text = line.lstrip("#").strip()
        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta


def _normalize_book_metadata(meta: dict[str, Any], books_config: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = dict(meta or {})
    book_id = str(result.get("book_id") or "").strip()
    config_meta = books_config.get(book_id, {}) if book_id else {}
    return {
        "book_id": book_id or str(config_meta.get("book_id") or "").strip() or None,
        "book_name": str(result.get("book_name") or config_meta.get("book_name") or "").strip() or None,
        "subject": str(result.get("subject") or config_meta.get("subject") or "").strip() or None,
        "domain": str(result.get("domain") or config_meta.get("domain") or "").strip() or None,
        "category": str(result.get("category") or config_meta.get("category") or "").strip() or None,
        "version": str(result.get("version") or "").strip() or None,
    }


def _inherit_metadata(defaults: dict[str, Any], overrides: dict[str, Any], books_config: dict[str, dict[str, Any]]) -> dict[str, Any]:
    merged = dict(defaults or {})
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        merged[key] = value
    return _normalize_book_metadata(merged, books_config)


@dataclass
class QAItem:
    qa_id: str
    question: str
    qa_type: str
    book_id: str | None
    book_name: str | None
    subject: str | None
    domain: str | None
    category: str | None
    version: str | None
    expected_route: str | None
    expected_books: list[str]
    allow_cross_book: bool
    difficulty: str | None
    expected_keywords: list[str]
    must_include: list[str]
    must_not_include: list[str]
    source_hint: str


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_qa_file(path: Path) -> list[QAItem]:
    if not path.is_file():
        raise FileNotFoundError(f"TRAINING_RAG_QA_FILE 不存在: {path}")
    books_config = _load_books_config()
    lines = path.read_text(encoding="utf-8").splitlines()
    defaults = _normalize_book_metadata(_parse_header_metadata(lines), books_config)
    if defaults.get("version") is None:
        defaults["version"] = _parse_header_metadata(lines).get("version")

    items: list[QAItem] = []
    current: dict[str, Any] | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("# ") or line.startswith("#\t"):
            continue
        if line.startswith("## "):
            if current:
                meta = _inherit_metadata(defaults, current, books_config)
                items.append(
                    QAItem(
                        qa_id=current.get("qa_id", "QA-UNKNOWN"),
                        question=current.get("question", ""),
                        qa_type=current.get("type", ""),
                        book_id=meta.get("book_id"),
                        book_name=meta.get("book_name"),
                        subject=meta.get("subject"),
                        domain=meta.get("domain"),
                        category=meta.get("category"),
                        version=meta.get("version"),
                        expected_route=current.get("expected_route") or None,
                        expected_books=_split_csv_like(current.get("expected_books", "")),
                        allow_cross_book=_parse_bool(current.get("allow_cross_book"), False),
                        difficulty=current.get("difficulty") or None,
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
        meta = _inherit_metadata(defaults, current, books_config)
        items.append(
            QAItem(
                qa_id=current.get("qa_id", "QA-UNKNOWN"),
                question=current.get("question", ""),
                qa_type=current.get("type", ""),
                book_id=meta.get("book_id"),
                book_name=meta.get("book_name"),
                subject=meta.get("subject"),
                domain=meta.get("domain"),
                category=meta.get("category"),
                version=meta.get("version"),
                expected_route=current.get("expected_route") or None,
                expected_books=_split_csv_like(current.get("expected_books", "")),
                allow_cross_book=_parse_bool(current.get("allow_cross_book"), False),
                difficulty=current.get("difficulty") or None,
                expected_keywords=_split_csv_like(current.get("expected_keywords", "")),
                must_include=_split_csv_like(current.get("must_include", "")),
                must_not_include=_split_csv_like(current.get("must_not_include", "")),
                source_hint=current.get("source_hint", ""),
            )
        )
    return items


def _latest_file_after(directory: Path, pattern: str, started_at: float) -> Path | None:
    if not directory.exists():
        return None
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


def _extract_answer(stdout: str) -> str:
    if "响应预览:" in stdout:
        return stdout.split("响应预览:", 1)[1].split("============================================================", 1)[0].strip()
    return stdout[-800:].strip()


def _guess_book_ids_from_text(text: str, books_config: dict[str, dict[str, Any]]) -> list[str]:
    found: list[str] = []
    if not text:
        return found
    for book_id, meta in books_config.items():
        name = str(meta.get("book_name") or "").strip()
        aliases = [str(x).strip() for x in (meta.get("aliases") or []) if str(x).strip()]
        for token in [name, *aliases]:
            if token and token in text and book_id not in found:
                found.append(book_id)
                break
    return found


def _extract_book_id_from_file_path(file_path: str, books_config: dict[str, dict[str, Any]]) -> str | None:
    if not file_path:
        return None
    match = re.search(r"(^|/)(\d{2})[^\d]", file_path)
    if match and match.group(2) in books_config:
        return match.group(2)
    guessed = _guess_book_ids_from_text(file_path, books_config)
    return guessed[0] if guessed else None


def _summarize_chunk_books(citations: list[dict[str, Any]], books_config: dict[str, dict[str, Any]]) -> tuple[list[str], list[str], bool]:
    book_ids: list[str] = []
    subjects: list[str] = []
    unknown = False
    for citation in citations:
        file_path = citation.get("file_path") or citation.get("source") or ""
        book_id = _extract_book_id_from_file_path(file_path, books_config)
        if not book_id:
            unknown = True
            continue
        if book_id not in book_ids:
            book_ids.append(book_id)
        subject = str((books_config.get(book_id) or {}).get("subject") or "").strip()
        if subject and subject not in subjects:
            subjects.append(subject)
    return book_ids, subjects, unknown


def _run_single_question(question: str, books_config: dict[str, dict[str, Any]]) -> dict[str, Any]:
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
    info_match = re.search(
        r"📊 RAG召回: backend=([^,]+), entities=(\d+), relationships=(\d+), chunks=(\d+), "
        r"references=(\d+), candidate_chunks=(\d+|None), final_chunks=(\d+|None)",
        stdout,
    )
    mode_match = re.search(
        r"Using RAG stream \| backend=([^|]+)\| include_history=(true|false) \| query=.*? \| mode=([^\n]+)",
        stdout,
    )
    label_match = re.search(r"LLM classification: label=([^,]+)", stdout)
    retrieved_chunk_books, retrieved_chunk_subjects, retrieved_book_unknown = _summarize_chunk_books(citations, books_config)

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
        "retrieved_chunk_books": retrieved_chunk_books,
        "retrieved_chunk_subjects": retrieved_chunk_subjects,
        "retrieved_book_unknown": retrieved_book_unknown,
        "answer": _extract_answer(stdout),
        "finish_path": str(finish_path) if finish_path else "",
        "debug_context_path": str(debug_context_path) if debug_context_path else "",
        "raw_debug_path": str(raw_debug_path) if raw_debug_path else "",
    }


def _build_report_rows(qa_items: list[QAItem], results: list[dict[str, Any]]) -> str:
    lines = ["# 工训教材通用问答回归报告", ""]
    lines.append(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- TRAINING_RAG_QA_FILE: {os.getenv('TRAINING_RAG_QA_FILE', '')}")
    lines.append("")
    for qa, result in zip(qa_items, results):
        answer = result["answer"]
        must_include_hit = _contains_any(answer, qa.must_include)
        must_not_include_hit = _contains_any(answer, qa.must_not_include)
        expected_books = qa.expected_books
        retrieved_chunk_books = result.get("retrieved_chunk_books") or []
        cross_book_hit_count = len([book for book in retrieved_chunk_books if book not in expected_books]) if expected_books else 0
        cross_book_suspicious = bool(expected_books and not qa.allow_cross_book and cross_book_hit_count > 0)
        route_failed = (
            (qa.expected_route == "general_knowledge" and result.get("entered_lightrag_file"))
            or (qa.expected_route == "lightrag_file" and not result.get("entered_lightrag_file"))
        )
        lines.extend(
            [
                f"## {qa.qa_id} {qa.question}",
                "",
                f"- type: {qa.qa_type}",
                f"- qa_book_id: {qa.book_id}",
                f"- qa_book_name: {qa.book_name}",
                f"- qa_subject: {qa.subject}",
                f"- expected_route: {qa.expected_route}",
                f"- route / intent: {result.get('route_intent')}",
                f"- route_failed: {route_failed}",
                f"- expected_books: {expected_books}",
                f"- allow_cross_book: {qa.allow_cross_book}",
                f"- entered_lightrag_file: {result.get('entered_lightrag_file')}",
                f"- mode: {result.get('mode')}",
                f"- original_query: {result.get('original_query')}",
                f"- effective_query: {result.get('effective_query')}",
                f"- aquery_llm_query_is_original: {result.get('aquery_llm_query_is_original')}",
                f"- low_level_keywords: {result.get('low_level_keywords')}",
                f"- high_level_keywords: {result.get('high_level_keywords')}",
                f"- entities_count: {result.get('entities_count')}",
                f"- relationships_count: {result.get('relationships_count')}",
                f"- chunks_count: {result.get('chunks_count')}",
                f"- references_count: {result.get('references_count')}",
                f"- candidate_chunks_count: {result.get('candidate_chunks_count')}",
                f"- final_chunks_count: {result.get('final_chunks_count')}",
                f"- final chunk file paths: {result.get('chunk_file_paths')}",
                f"- retrieved_chunk_books: {retrieved_chunk_books}",
                f"- retrieved_chunk_subjects: {result.get('retrieved_chunk_subjects')}",
                f"- retrieved_book_unknown: {result.get('retrieved_book_unknown')}",
                f"- cross_book_hit_count: {cross_book_hit_count}",
                f"- cross_book_suspicious: {cross_book_suspicious}",
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

    books_config = _load_books_config()
    qa_items = _parse_qa_file(qa_path)
    results = []
    for qa in qa_items:
        logger.info(f"Running regression QA | {qa.qa_id} | {qa.question}")
        results.append(_run_single_question(qa.question, books_config))

    report_name = f"{qa_path.stem.replace('_QA.example', '')}_generalization_regression.md"
    report_path = ROOT / "logs" / "rag_stream_debug" / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_build_report_rows(qa_items, results), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    asyncio.run(main())
