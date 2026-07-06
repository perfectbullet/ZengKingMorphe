#!/usr/bin/env python3
"""Import pre-cut textbook chunks through the current LightRAG storage APIs."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.operate import merge_nodes_and_edges

from common import (
    DEFAULT_WORKING_DIR,
    build_embedding_func,
    build_llm_model_func,
    read_jsonl,
    resolve_embedding_config,
    resolve_llm_config,
    write_json,
    write_jsonl,
)

PROJECT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PROJECT_DIR / "prompts"
PROFILES_DIR = PROMPTS_DIR / "profiles"
logger = logging.getLogger(__name__)
CUSTOM_CHUNK_METADATA_FIELDS = [
    "chunk_id",
    "block_id",
    "doc_id",
    "catalog_index",
    "catalog_level",
    "catalog_title",
    "matched_title_line",
    "matched_title_text",
    "content_scope",
    "should_extract_kg",
    "structural_children",
    "book_title",
    "chapter_title",
    "section_title",
    "heading_path",
    "start_line",
    "end_line",
    "image_lines",
    "images",
    "image_assets",
    "confidence",
    "reason",
    "file_path",
    "source_md_path",
    "domain",
    "subject",
]


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
    parser.add_argument("--base-prompt", type=Path)
    parser.add_argument("--profile-prompt", type=Path)
    parser.add_argument("--no-profile-prompt", action="store_true")
    parser.add_argument("--print-prompt-preview", action="store_true")
    parser.add_argument("--env-file", type=Path, default=PROJECT_DIR.parent / ".env")
    return parser.parse_args()


def preload_env() -> Path:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--env-file", type=Path, default=PROJECT_DIR.parent / ".env"
    )
    pre_args, _ = pre_parser.parse_known_args()
    env_path = pre_args.env_file.expanduser().resolve()
    if env_path.is_file():
        load_dotenv(env_path, override=False)
        logger.info("loaded env: %s", env_path)
    return env_path


def read_text_file(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"prompt 文件不存在: {resolved}")
    return resolved.read_text(encoding="utf-8").strip()


def resolve_base_prompt_path(args: argparse.Namespace) -> Path:
    if args.base_prompt is not None:
        return args.base_prompt.expanduser().resolve()
    env_path = os.getenv("MARKDOWN_GRAPH_BASE_PROMPT", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    for candidate in (
        PROMPTS_DIR / "graph_extraction_base.md",
        PROMPTS_DIR / "graph_extraction_guidance.md",
    ):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "未找到 KG 抽取 base prompt；请提供 --base-prompt 或创建 "
        f"{PROMPTS_DIR / 'graph_extraction_base.md'}"
    )


def resolve_profile_prompt_path(args: argparse.Namespace) -> Path | None:
    if args.no_profile_prompt:
        return None
    if args.profile_prompt is not None:
        return args.profile_prompt.expanduser().resolve()
    env_path = os.getenv("MARKDOWN_GRAPH_PROFILE_PROMPT", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()

    profile = PROFILES_DIR / "01_enamel.profile.md"
    chunks_name = args.chunks.name.casefold()
    subject = str(args.subject or "").casefold()
    if subject == "enamel" or any(
        marker.casefold() in chunks_name for marker in ("01珐琅工艺", "enamel", "珐琅")
    ):
        return profile.resolve() if profile.is_file() else None
    return None


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_extraction_prompt(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    base_path = resolve_base_prompt_path(args)
    base_prompt = read_text_file(base_path)
    profile_path = resolve_profile_prompt_path(args)
    profile_prompt = read_text_file(profile_path) if profile_path is not None else ""

    final_prompt = base_prompt.strip()
    if profile_prompt:
        final_prompt += "\n\n---\n\n# 教材级抽取 Profile\n\n"
        final_prompt += profile_prompt.strip()
    prompt_meta = {
        "base_prompt_path": str(base_path),
        "base_prompt_chars": len(base_prompt),
        "base_prompt_sha256": prompt_sha256(base_prompt),
        "profile_prompt_enabled": profile_path is not None,
        "profile_prompt_path": str(profile_path) if profile_path is not None else None,
        "profile_prompt_chars": len(profile_prompt),
        "profile_prompt_sha256": (
            prompt_sha256(profile_prompt) if profile_path is not None else None
        ),
        "final_prompt_chars": len(final_prompt),
        "final_prompt_sha256": prompt_sha256(final_prompt),
    }
    return final_prompt, prompt_meta


def build_rag(working_dir: Path, extraction_prompt: str) -> LightRAG:
    working_dir.mkdir(parents=True, exist_ok=True)
    return LightRAG(
        working_dir=str(working_dir),
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        addon_params={
            "language": "Chinese",
            "entity_types_guidance": extraction_prompt,
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
        "content_scope",
        "should_extract_kg",
        "catalog_title",
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
        if not isinstance(chunk["should_extract_kg"], bool):
            raise ValueError(f"chunk {chunk_id} 的 should_extract_kg 必须是 JSON boolean")


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
    inserting_chunks_by_doc: dict[str, dict[str, dict[str, Any]]] = {}
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
        doc_inserting_chunks: dict[str, dict[str, Any]] = {}
        for chunk in doc_chunks:
            chunk_id = str(chunk["chunk_id"])
            content = str(chunk["content"])
            chunk_record = {
                "content": content,
                "full_doc_id": doc_id,
                "tokens": len(rag.tokenizer.encode(content)),
                "chunk_order_index": int(chunk["chunk_order_index"]),
                "file_path": str(chunk["file_path"]),
            }
            for field in CUSTOM_CHUNK_METADATA_FIELDS:
                if field in chunk:
                    chunk_record[field] = chunk.get(field)
            inserting_chunks[chunk_id] = chunk_record
            doc_inserting_chunks[chunk_id] = chunk_record
        inserting_chunks_by_doc[doc_id] = doc_inserting_chunks

    content_scope_stats = dict(
        sorted(Counter(chunk["content_scope"] for chunk in chunks).items())
    )
    should_extract_kg_stats = dict(
        sorted(Counter(chunk["should_extract_kg"] for chunk in chunks).items())
    )
    kg_chunk_count = sum(
        chunk.get("should_extract_kg", True) is not False for chunk in chunks
    )
    skipped_kg_chunk_count = len(chunks) - kg_chunk_count
    logger.info(
        "custom chunks | total=%d kg=%d skipped_kg=%d content_scope=%s "
        "should_extract_kg=%s",
        len(chunks),
        kg_chunk_count,
        skipped_kg_chunk_count,
        content_scope_stats,
        should_extract_kg_stats,
    )

    flush_needed = False
    active_error: BaseException | None = None
    try:
        flush_needed = True
        await rag.chunks_vdb.upsert(inserting_chunks)
        await rag.full_docs.upsert(new_docs)
        await rag.text_chunks.upsert(inserting_chunks)

        extraction_result_count = 0
        extracted_entity_mentions = 0
        extracted_relation_mentions = 0
        total_kg_docs = sum(
            any(
                record.get("should_extract_kg", True) is not False
                for record in doc_records.values()
            )
            for doc_records in inserting_chunks_by_doc.values()
        )
        current_kg_doc = 0
        for doc_id, doc_chunks in inserting_chunks_by_doc.items():
            kg_doc_chunks = {
                chunk_id: record
                for chunk_id, record in doc_chunks.items()
                if record.get("should_extract_kg", True) is not False
            }
            if not kg_doc_chunks:
                continue
            current_kg_doc += 1
            pipeline_status = {
                "latest_message": "",
                "history_messages": [],
                "cancellation_requested": False,
            }
            pipeline_status_lock = asyncio.Lock()
            try:
                extraction_results = await rag._process_extract_entities(
                    kg_doc_chunks,
                    pipeline_status,
                    pipeline_status_lock,
                )
                extraction_result_count += len(extraction_results or [])
                for maybe_nodes, maybe_edges in extraction_results or []:
                    extracted_entity_mentions += len(maybe_nodes)
                    extracted_relation_mentions += len(maybe_edges)
                await merge_nodes_and_edges(
                    chunk_results=extraction_results,
                    knowledge_graph_inst=rag.chunk_entity_relation_graph,
                    entity_vdb=rag.entities_vdb,
                    relationships_vdb=rag.relationships_vdb,
                    global_config=rag._build_global_config(),
                    full_entities_storage=rag.full_entities,
                    full_relations_storage=rag.full_relations,
                    doc_id=doc_id,
                    pipeline_status=pipeline_status,
                    pipeline_status_lock=pipeline_status_lock,
                    llm_response_cache=rag.llm_response_cache,
                    entity_chunks_storage=rag.entity_chunks,
                    relation_chunks_storage=rag.relation_chunks,
                    current_file_number=current_kg_doc,
                    total_files=total_kg_docs,
                    file_path=new_docs[doc_id]["file_path"],
                )
            except Exception as exc:
                failed_doc_chunks = grouped[doc_id]
                raise EntityExtractionError(
                    f"实体关系抽取或合并失败: {type(exc).__name__}: {exc}",
                    failed_doc_chunks,
                ) from exc
        return {
            "doc_count": len(new_docs),
            "chunk_count": len(inserting_chunks),
            "kg_chunk_count": kg_chunk_count,
            "skipped_kg_chunk_count": skipped_kg_chunk_count,
            "content_scope_stats": content_scope_stats,
            "should_extract_kg_stats": should_extract_kg_stats,
            "extraction_result_count": extraction_result_count,
            "extracted_entity_mentions": extracted_entity_mentions,
            "extracted_relation_mentions": extracted_relation_mentions,
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
    validate_chunks(chunks)
    working_dir = args.working_dir.expanduser().resolve()
    logger.info("最终 working_dir | %s", working_dir)
    if (
        args.domain == "industrial_training"
        and "lightrag_manual_concepts" in str(working_dir)
    ):
        raise ValueError(
            f"工业教材不能导入 manual concepts working_dir: {working_dir}"
        )
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
    result = {
        "doc_count": 0,
        "chunk_count": len(chunks),
        "kg_chunk_count": sum(
            chunk.get("should_extract_kg", True) is not False for chunk in chunks
        ),
        "skipped_kg_chunk_count": sum(
            chunk.get("should_extract_kg") is False for chunk in chunks
        ),
        "content_scope_stats": dict(
            sorted(Counter(chunk.get("content_scope") for chunk in chunks).items())
        ),
        "should_extract_kg_stats": dict(
            sorted(
                Counter(chunk.get("should_extract_kg") for chunk in chunks).items()
            )
        ),
        "extraction_result_count": 0,
        "extracted_entity_mentions": 0,
        "extracted_relation_mentions": 0,
    }

    extraction_prompt, prompt_meta = build_extraction_prompt(args)
    logger.info(
        "extraction prompt | base=%s profile=%s final_sha256=%s final_chars=%d",
        prompt_meta["base_prompt_path"],
        prompt_meta["profile_prompt_path"],
        prompt_meta["final_prompt_sha256"],
        prompt_meta["final_prompt_chars"],
    )
    if args.print_prompt_preview:
        logger.info("extraction prompt preview:\n%s", extraction_prompt[:1200])

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
    rag = build_rag(working_dir, extraction_prompt)
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
        relations = await rag.chunk_entity_relation_graph.get_all_edges()
    finally:
        await rag.finalize_storages()
        logger.info("LightRAG storages finalized")

    write_jsonl(failed_path, failures)
    report = {
        "working_dir": str(working_dir),
        "doc_count": result["doc_count"],
        "chunk_count": result["chunk_count"],
        "kg_chunk_count": result["kg_chunk_count"],
        "skipped_kg_chunk_count": result["skipped_kg_chunk_count"],
        "content_scope_stats": result["content_scope_stats"],
        "should_extract_kg_stats": result["should_extract_kg_stats"],
        "prompt": prompt_meta,
        "entity_count_after": len(labels),
        "relation_count_after": len(relations),
        "extracted_entity_mentions": result["extracted_entity_mentions"],
        "extracted_relation_mentions": result["extracted_relation_mentions"],
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
    preload_env()
    args = parse_args()
    try:
        return asyncio.run(run(args))
    except Exception as exc:
        logger.exception("custom chunks 导入失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
