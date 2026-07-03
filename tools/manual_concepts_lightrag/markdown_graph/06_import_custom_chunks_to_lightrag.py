#!/usr/bin/env python3
"""Import pre-cut textbook chunks through the current LightRAG storage APIs."""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from lightrag import LightRAG

from common import (
    DEFAULT_WORKING_DIR,
    PROJECT_DIR,
    build_embedding_func,
    build_llm_model_func,
    load_prompt,
    read_jsonl,
    resolve_embedding_config,
    resolve_llm_config,
    write_json,
    write_jsonl,
)

logger = logging.getLogger(__name__)


class EntityExtractionError(RuntimeError):
    def __init__(self, message: str, chunks: list[dict]):
        super().__init__(message)
        self.chunks = chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用自建 custom chunks 导入 LightRAG")
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path(os.getenv("MARKDOWN_GRAPH_WORKING_DIR", str(DEFAULT_WORKING_DIR))),
    )
    parser.add_argument(
        "--domain", default=os.getenv("MARKDOWN_GRAPH_DOMAIN", "industrial_training")
    )
    parser.add_argument("--subject", default=os.getenv("MARKDOWN_GRAPH_SUBJECT", ""))
    parser.add_argument(
        "--import-method",
        default=os.getenv("MARKDOWN_GRAPH_IMPORT_METHOD", "custom_chunks"),
        choices=["custom_chunks"],
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--failed-chunks", type=Path)
    parser.add_argument("--env-file", type=Path, default=PROJECT_DIR.parent / ".env")
    return parser.parse_args()


def build_rag(working_dir: Path, domain: str, subject: str) -> LightRAG:
    guidance = load_prompt(PROJECT_DIR / "prompts/graph_extraction_guidance.md")
    guidance = (
        f"{guidance.strip()}\n\n当前领域：{domain}\n当前科目：{subject or '未指定'}"
    )
    working_dir.mkdir(parents=True, exist_ok=True)
    return LightRAG(
        working_dir=str(working_dir),
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        addon_params={
            "language": "Chinese",
            "entity_types_guidance": guidance,
        },
        llm_model_func=build_llm_model_func(),
        embedding_func=build_embedding_func(),
    )


def validate_chunks(chunks: list[dict]) -> None:
    if not chunks:
        raise ValueError("chunks JSONL 为空")
    required = {
        "chunk_id",
        "doc_id",
        "chunk_order_index",
        "file_path",
        "source_md_path",
        "content",
    }
    seen: set[str] = set()
    for index, chunk in enumerate(chunks, 1):
        missing = required - chunk.keys()
        if missing:
            raise ValueError(f"chunk {index} 缺少字段: {sorted(missing)}")
        chunk_id = str(chunk["chunk_id"])
        if chunk_id in seen:
            raise ValueError(f"chunk_id 重复: {chunk_id}")
        seen.add(chunk_id)
        if not str(chunk["content"]).strip():
            raise ValueError(f"chunk 内容为空: {chunk_id}")


async def import_custom_chunks(
    rag: LightRAG,
    chunks: list[dict],
    *,
    replace: bool = False,
) -> dict:
    """Write caller-owned chunks, then run entity/relation extraction once."""
    validate_chunks(chunks)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        grouped[str(chunk["doc_id"])].append(chunk)

    new_docs: dict[str, dict[str, Any]] = {}
    inserting_chunks: dict[str, dict[str, Any]] = {}
    for doc_id, doc_chunks in grouped.items():
        doc_chunks.sort(key=lambda item: int(item["chunk_order_index"]))
        if replace:
            try:
                result = await rag.adelete_by_doc_id(doc_id)
                status = getattr(result, "status", None)
                if status not in {None, "success"}:
                    logger.warning(
                        "replace 删除未成功 | doc_id=%s result=%s", doc_id, result
                    )
            except Exception as exc:
                logger.warning(
                    "replace 删除失败，继续导入 | doc_id=%s error=%s", doc_id, exc
                )
        full_text = "\n\n".join(str(chunk["content"]) for chunk in doc_chunks)
        source_paths = {str(chunk["source_md_path"]) for chunk in doc_chunks}
        if len(source_paths) != 1:
            raise ValueError(
                f"同一 doc_id 对应多个 source_md_path: {doc_id}: {sorted(source_paths)}"
            )
        new_docs[doc_id] = {"content": full_text, "file_path": next(iter(source_paths))}
        for chunk in doc_chunks:
            chunk_id = str(chunk["chunk_id"])
            content = str(chunk["content"])
            inserting_chunks[chunk_id] = {
                "content": content,
                "full_doc_id": doc_id,
                "tokens": len(rag.tokenizer.encode(content)),
                "chunk_order_index": int(chunk["chunk_order_index"]),
                "file_path": str(chunk["file_path"]),
            }

    flush_needed = False
    active_error: BaseException | None = None
    try:
        flush_needed = True
        await rag.chunks_vdb.upsert(inserting_chunks)
        try:
            extraction_results = await rag._process_extract_entities(inserting_chunks)
        except Exception as exc:
            raise EntityExtractionError(
                f"实体关系抽取失败: {type(exc).__name__}: {exc}",
                chunks,
            ) from exc
        await rag.full_docs.upsert(new_docs)
        await rag.text_chunks.upsert(inserting_chunks)
        return {
            "doc_count": len(new_docs),
            "chunk_count": len(inserting_chunks),
            "extraction_result_count": len(extraction_results or []),
        }
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        if flush_needed:
            try:
                await rag._insert_done_with_cleanup()
            except Exception as cleanup_error:
                if active_error is None:
                    raise
                logger.exception(
                    "导入失败后的 storage cleanup 同时失败；保留原始导入异常 | error=%s",
                    cleanup_error,
                )


async def run(args: argparse.Namespace) -> int:
    chunks_path = args.chunks.expanduser().resolve()
    chunks = read_jsonl(chunks_path)
    working_dir = args.working_dir.expanduser().resolve()
    book_stem = (
        Path(str(chunks[0]["source_md_path"])).stem
        if chunks
        else chunks_path.stem.split(".")[0]
    )
    report_path = (
        (
            args.report
            or PROJECT_DIR / "outputs/06_import" / f"{book_stem}.import_report.json"
        )
        .expanduser()
        .resolve()
    )
    failed_path = (
        (
            args.failed_chunks
            or PROJECT_DIR / "outputs/06_import" / f"{book_stem}.failed_chunks.jsonl"
        )
        .expanduser()
        .resolve()
    )
    started = datetime.now(timezone.utc)
    failures: list[dict] = []
    error_message = ""
    result = {"doc_count": 0, "chunk_count": len(chunks), "extraction_result_count": 0}

    llm = resolve_llm_config()
    embedding = resolve_embedding_config()
    logger.info(
        "配置 | llm_model=%s llm_base_url=%s embedding_model=%s embedding_dim=%s working_dir=%s",
        llm["model"],
        llm["base_url"],
        embedding["model"],
        embedding["dim"],
        working_dir,
    )
    rag = build_rag(working_dir, args.domain, args.subject)
    await rag.initialize_storages()
    logger.info("LightRAG storages initialized")
    try:
        try:
            result = await import_custom_chunks(rag, chunks, replace=args.replace)
        except EntityExtractionError as exc:
            error_message = str(exc)
            failures = [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "doc_id": chunk.get("doc_id"),
                    "file_path": chunk.get("file_path"),
                    "error": error_message,
                }
                for chunk in exc.chunks
            ]
        labels = await rag.get_graph_labels()
    finally:
        await rag.finalize_storages()
        logger.info("LightRAG storages finalized")

    write_jsonl(failed_path, failures)
    report = {
        "working_dir": str(working_dir),
        "doc_count": result["doc_count"],
        "chunk_count": result["chunk_count"],
        "entity_count_after": len(labels),
        "replace": bool(args.replace),
        "failures": len(failures),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if error_message:
        report["error"] = error_message
    write_json(report_path, report)
    logger.info("导入报告 | %s", report_path)
    if failures:
        logger.error("导入存在失败 chunks=%d | %s", len(failures), failed_path)
        return 1
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    env_path = args.env_file.expanduser().resolve()
    if env_path.is_file():
        load_dotenv(env_path, override=False)
        logger.info("loaded env: %s", env_path)
    try:
        return asyncio.run(run(args))
    except Exception as exc:
        logger.exception("custom chunks 导入失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
