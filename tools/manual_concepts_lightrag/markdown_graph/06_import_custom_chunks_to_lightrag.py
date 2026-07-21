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
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from dotenv import load_dotenv

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
from book_meta import get_business_config, get_entity_extraction_config, load_book_meta, resolve_book_paths, resolve_config_value

PROJECT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PROJECT_DIR / "prompts"
# Legacy long prompts under prompts/ are kept for history, but Step 6 now only
# injects a short LightRAG entity_types_guidance string by default.
ENTITY_TYPES_GUIDANCE_DIR = PROMPTS_DIR / "entity_types_guidance"
ENTITY_TYPE_DIR = PROMPTS_DIR / "entity_type"
SCHEMAS_DIR = PROMPTS_DIR / "schemas"
DEFAULT_SCHEMA_VERSION = "industrial_training_kg_schema.v1"
DEFAULT_SCHEMA_JSON = SCHEMAS_DIR / "industrial_training_kg_schema.v1.json"
DEFAULT_ENAMEL_GUIDANCE = ENTITY_TYPES_GUIDANCE_DIR / "01_enamel.guidance.md"
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
    "kg_content",
]


class EntityExtractionError(RuntimeError):
    def __init__(self, message: str, chunks: list[dict]):
        super().__init__(message)
        self.chunks = chunks


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().casefold() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数: {value!r}") from exc


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字: {value!r}") from exc


def entity_extraction_settings() -> dict[str, int | float | bool]:
    """Read Step 6-only extraction settings without changing global LLM defaults."""
    settings: dict[str, int | float | bool] = {
        "entity_extraction_use_json": env_bool("ENTITY_EXTRACTION_USE_JSON", True),
        "entity_extraction_temperature": env_float(
            "ENTITY_EXTRACTION_TEMPERATURE", 0.0
        ),
        "entity_extraction_max_tokens": env_int(
            "ENTITY_EXTRACTION_MAX_TOKENS", 4096
        ),
        "max_gleaning": env_int("MAX_GLEANING", 0),
        "max_extraction_records": env_int("MAX_EXTRACTION_RECORDS", 30),
        "max_extraction_entities": env_int("MAX_EXTRACTION_ENTITIES", 12),
    }
    if not 0.0 <= float(settings["entity_extraction_temperature"]) <= 2.0:
        raise ValueError("ENTITY_EXTRACTION_TEMPERATURE 必须在 0 到 2 之间")
    for key in ("entity_extraction_max_tokens", "max_extraction_records", "max_extraction_entities"):
        if int(settings[key]) <= 0:
            raise ValueError(f"{key} 必须大于 0")
    if int(settings["max_gleaning"]) < 0:
        raise ValueError("MAX_GLEANING 不能小于 0")
    return settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用自建 custom chunks 导入 LightRAG")
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--entity-type-prompt-file", help="LightRAG YAML profile file name")
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--domain")
    parser.add_argument("--subject")
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
    parser.add_argument("--entity-types-guidance-file", type=Path)
    parser.add_argument(
        "--extract-batch-size",
        type=int,
        default=env_int("MARKDOWN_GRAPH_EXTRACT_BATCH_SIZE", 8),
    )
    parser.add_argument(
        "--single-chunk-retry",
        type=int,
        default=env_int("MARKDOWN_GRAPH_SINGLE_CHUNK_RETRY", 1),
    )
    parser.add_argument(
        "--continue-on-chunk-error",
        action="store_true",
        default=env_bool("MARKDOWN_GRAPH_CONTINUE_ON_CHUNK_ERROR", True),
    )
    parser.add_argument(
        "--failed-chunk-policy",
        choices=["return_nonzero", "return_zero"],
        default=os.getenv("MARKDOWN_GRAPH_FAILED_CHUNK_POLICY", "return_nonzero"),
    )
    parser.add_argument("--max-kg-chunks", type=int)
    parser.add_argument("--start-chunk-index", type=int, default=0)
    parser.add_argument("--chunk-id", action="append", default=[])
    parser.add_argument("--chunk-id-file", type=Path)
    parser.add_argument("--chunk-title-contains")
    parser.add_argument("--dry-run-selected-chunks", action="store_true")
    parser.add_argument(
        "--schema-version",
        default=os.getenv("MARKDOWN_GRAPH_SCHEMA_VERSION", DEFAULT_SCHEMA_VERSION),
    )
    parser.add_argument(
        "--schema-json",
        type=Path,
        default=Path(
            os.getenv("MARKDOWN_GRAPH_SCHEMA_JSON", str(DEFAULT_SCHEMA_JSON))
        ),
    )
    parser.add_argument("--env-file", type=Path, default=PROJECT_DIR.parent / ".env")
    args = parser.parse_args()
    if args.base_prompt or args.profile_prompt or args.no_profile_prompt:
        parser.error(
            "Step 6 now uses LightRAG entity_types_guidance. "
            "Use --entity-types-guidance-file instead."
        )
    return args


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
        raise FileNotFoundError(f"guidance 文件不存在: {resolved}")
    return resolved.read_text(encoding="utf-8").strip()


def resolve_entity_types_guidance_path(args: argparse.Namespace) -> Path:
    if args.entity_types_guidance_file is not None:
        guidance_path = args.entity_types_guidance_file.expanduser().resolve()
        if not guidance_path.is_file():
            raise FileNotFoundError(
                f"--entity-types-guidance-file 不存在: {guidance_path}"
            )
        return guidance_path

    chunks_name = args.chunks.name.casefold()
    subject = str(args.subject or "").casefold()
    if subject == "enamel" or any(
        marker.casefold() in chunks_name for marker in ("01珐琅工艺", "enamel", "珐琅")
    ):
        if DEFAULT_ENAMEL_GUIDANCE.is_file():
            return DEFAULT_ENAMEL_GUIDANCE.resolve()
        raise FileNotFoundError(f"默认 enamel guidance 文件不存在: {DEFAULT_ENAMEL_GUIDANCE}")
    raise ValueError(
        "无法根据 subject/chunks 文件名推断 entity_types_guidance；"
        "请显式传入 --entity-types-guidance-file"
    )


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_entity_types_guidance(
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    guidance_path = resolve_entity_types_guidance_path(args)
    guidance = read_text_file(guidance_path)
    guidance_meta = {
        "guidance_mode": "entity_types_guidance",
        "guidance_path": str(guidance_path),
        "guidance_chars": len(guidance),
        "guidance_sha256": prompt_sha256(guidance),
    }
    return guidance, guidance_meta


def load_entity_type_profile(path: Path) -> dict[str, Any]:
    try:
        profile = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"实体类型 YAML 格式错误: {path}: {exc}") from exc
    if not isinstance(profile, dict):
        raise ValueError(f"实体类型 YAML 必须是对象: {path}")
    for key in ("entity_types_guidance", "entity_extraction_examples", "entity_extraction_json_examples", "allowed_entity_types", "allowed_relation_keywords"):
        if key not in profile:
            raise ValueError(f"实体类型 YAML 缺少必要字段: {key}")
    if not isinstance(profile["entity_types_guidance"], str) or not profile["entity_types_guidance"].strip():
        raise ValueError("实体类型 YAML 的 entity_types_guidance 必须为非空字符串")
    for key in ("entity_extraction_examples", "entity_extraction_json_examples"):
        if not isinstance(profile[key], list) or not profile[key] or not all(isinstance(item, str) and item.strip() for item in profile[key]):
            raise ValueError(f"实体类型 YAML 的 {key} 必须为非空字符串列表")
    for key in ("allowed_entity_types", "allowed_relation_keywords"):
        if not isinstance(profile[key], list) or not all(isinstance(item, str) and item.strip() for item in profile[key]):
            raise ValueError(f"实体类型 YAML 的 {key} 必须为字符串列表")
    for key in (
        "excluded_entity_names",
        "excluded_entity_suffixes",
        "excluded_entity_patterns",
        "generic_entity_names",
        "entity_name_whitelist",
    ):
        value = profile.get(key, [])
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ValueError(f"实体类型 YAML 的 {key} 必须为字符串列表")
        profile[key] = value
    for pattern in profile["excluded_entity_patterns"]:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"实体类型 YAML 的 excluded_entity_patterns 正则无效: {pattern!r}") from exc
    return profile


def resolve_entity_type_profile(args: argparse.Namespace) -> tuple[Path | None, dict[str, Any] | None, dict[str, Any]]:
    if args.entity_type_prompt_file and args.entity_types_guidance_file:
        raise ValueError("--entity-type-prompt-file 与已废弃的 --entity-types-guidance-file 不能同时使用")
    meta_profile = ""
    if args.meta:
        meta = load_book_meta(args.meta)
        paths = resolve_book_paths(meta)
        meta_profile = str(paths["prompt_file"] or "")
    file_name = resolve_config_value(args.entity_type_prompt_file, meta_profile, "ENTITY_TYPE_PROMPT_FILE", "")
    if not file_name:
        return None, None, {}
    candidate = Path(str(file_name)).expanduser()
    if candidate.is_absolute():
        path = candidate.resolve()
        file_name = path.name
    else:
        # meta stores a file name, never a shell-relative path.
        path = (ENTITY_TYPE_DIR / candidate.name).resolve()
        file_name = candidate.name
    if path.parent != ENTITY_TYPE_DIR.resolve() or not path.is_file():
        raise FileNotFoundError(f"教材专属实体类型 YAML 不存在于 prompts/entity_type: {path}")
    profile = load_entity_type_profile(path)
    extraction_settings = entity_extraction_settings()
    meta = {
        "guidance_mode": "entity_type_prompt_file",
        "profile_path": str(path),
        "profile_file": file_name,
        "profile_sha256": prompt_sha256(path.read_text(encoding="utf-8")),
        "allowed_entity_types": profile["allowed_entity_types"],
        "allowed_relation_keywords": profile["allowed_relation_keywords"],
        **extraction_settings,
    }
    return path, profile, meta


def build_rag(working_dir: Path, *, profile_file_name: str | None = None, entity_types_guidance: str | None = None) -> LightRAG:
    from lightrag import LightRAG

    working_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PROMPT_DIR"] = str(PROMPTS_DIR)
    addon_params: dict[str, Any] = {"language": "Chinese"}
    if profile_file_name:
        addon_params["entity_type_prompt_file"] = profile_file_name
    elif entity_types_guidance:
        addon_params["entity_types_guidance"] = entity_types_guidance
    extraction_settings = entity_extraction_settings()
    return LightRAG(
        working_dir=str(working_dir),
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        addon_params=addon_params,
        llm_model_func=build_llm_model_func(
            max_tokens=int(extraction_settings["entity_extraction_max_tokens"]),
            temperature=float(extraction_settings["entity_extraction_temperature"]),
        ),
        embedding_func=build_embedding_func(),
        entity_extraction_use_json=bool(extraction_settings["entity_extraction_use_json"]),
        entity_extract_max_gleaning=int(extraction_settings["max_gleaning"]),
        entity_extract_max_records=int(extraction_settings["max_extraction_records"]),
        entity_extract_max_entities=int(extraction_settings["max_extraction_entities"]),
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


def read_chunk_id_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"chunk-id-file 不存在: {resolved}")
    chunk_ids: list[str] = []
    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        chunk_ids.append(line)
    return chunk_ids


def chunk_search_text(chunk: dict[str, Any]) -> str:
    heading_path = chunk.get("heading_path") or ""
    if isinstance(heading_path, list):
        heading_text = " ".join(str(item) for item in heading_path)
    else:
        heading_text = str(heading_path)
    return "\n".join(
        [
            str(chunk.get("chunk_id") or ""),
            str(chunk.get("catalog_title") or ""),
            str(chunk.get("chapter_title") or ""),
            str(chunk.get("section_title") or ""),
            heading_text,
        ]
    )


def selection_enabled(args: argparse.Namespace, chunk_ids: set[str]) -> bool:
    return bool(
        chunk_ids
        or args.chunk_title_contains
        or args.max_kg_chunks is not None
        or args.start_chunk_index != 0
        or args.dry_run_selected_chunks
    )


def select_kg_chunks(
    chunks: list[dict],
    args: argparse.Namespace,
) -> tuple[list[dict], dict[str, Any]]:
    total_chunks = len(chunks)
    kg_chunks = [
        chunk for chunk in chunks if chunk.get("should_extract_kg", True) is not False
    ]
    chunk_id_values = list(args.chunk_id or []) + read_chunk_id_file(args.chunk_id_file)
    chunk_id_set = set(chunk_id_values)
    enabled = selection_enabled(args, chunk_id_set)
    selected = list(kg_chunks)

    if chunk_id_set:
        selected = [
            chunk for chunk in selected if str(chunk.get("chunk_id")) in chunk_id_set
        ]

    if args.chunk_title_contains:
        needle = str(args.chunk_title_contains)
        needle_folded = needle.casefold()
        selected = [
            chunk
            for chunk in selected
            if needle in chunk_search_text(chunk)
            or needle_folded in chunk_search_text(chunk).casefold()
        ]

    if args.start_chunk_index < 0:
        raise ValueError("--start-chunk-index 不能小于 0")
    if args.max_kg_chunks is not None and args.max_kg_chunks <= 0:
        raise ValueError("--max-kg-chunks 必须大于 0")
    selected = selected[args.start_chunk_index :]
    if args.max_kg_chunks is not None:
        selected = selected[: args.max_kg_chunks]

    meta = {
        "enabled": enabled,
        "total_chunks_before_filter": total_chunks,
        "kg_chunks_before_filter": len(kg_chunks),
        "selected_chunks_after_filter": len(selected),
        "skipped_by_filter": len(kg_chunks) - len(selected),
        "start_chunk_index": args.start_chunk_index,
        "max_kg_chunks": args.max_kg_chunks,
        "chunk_title_contains": args.chunk_title_contains,
        "chunk_id_count": len(chunk_id_set),
        "selected_chunk_ids": [str(chunk.get("chunk_id")) for chunk in selected],
    }
    if not selected:
        raise ValueError(
            "筛选后没有可抽取的 KG chunks: "
            f"total={total_chunks} kg={len(kg_chunks)} criteria={meta}"
        )
    if enabled:
        logger.info(
            "debug selection | total=%d kg_before=%d selected=%d start=%d "
            "max=%s title_contains=%s chunk_ids=%d",
            total_chunks,
            len(kg_chunks),
            len(selected),
            args.start_chunk_index,
            args.max_kg_chunks,
            args.chunk_title_contains,
            len(chunk_id_set),
        )
    else:
        logger.info(
            "debug selection disabled | total=%d kg=%d", total_chunks, len(kg_chunks)
        )
        return chunks, meta
    return selected, meta


def print_dry_run_selection(chunks: list[dict], meta: dict[str, Any]) -> None:
    print(f"total_chunks: {meta['total_chunks_before_filter']}")
    print(f"kg_chunks_before_filter: {meta['kg_chunks_before_filter']}")
    print(f"selected_kg_chunks: {meta['selected_chunks_after_filter']}")
    print(f"skipped_by_filter: {meta['skipped_by_filter']}")
    print(f"max_kg_chunks: {meta['max_kg_chunks']}")
    print(f"start_chunk_index: {meta['start_chunk_index']}")
    print(f"chunk_title_contains: {meta['chunk_title_contains']}")
    print(f"chunk_id_count: {meta['chunk_id_count']}")
    print("selected chunks:")
    for order, chunk in enumerate(chunks, start=1):
        content = str(chunk.get("content") or "")
        print(
            "  "
            f"order={order} "
            f"chunk_order_index={chunk.get('chunk_order_index')} "
            f"chunk_id={chunk.get('chunk_id')} "
            f"catalog_title={chunk.get('catalog_title')} "
            f"content_chars={len(content)} "
            f"file_path={chunk.get('file_path')} "
            f"should_extract_kg={chunk.get('should_extract_kg')}"
        )


def chunk_batches(
    chunks: dict[str, dict[str, Any]], batch_size: int
) -> list[dict[str, dict[str, Any]]]:
    items = list(chunks.items())
    return [
        dict(items[index : index + batch_size])
        for index in range(0, len(items), batch_size)
    ]


def count_extraction_mentions(
    extraction_results: Any,
) -> tuple[int, int, int]:
    result_count = len(extraction_results or [])
    entity_mentions = 0
    relation_mentions = 0
    for maybe_nodes, maybe_edges in extraction_results or []:
        entity_mentions += len(maybe_nodes)
        relation_mentions += len(maybe_edges)
    return result_count, entity_mentions, relation_mentions


_FIGURE_RE = re.compile(r"^图\s*\d+\s*[-－]\s*\d+$")
_CHAPTER_RE = re.compile(r"^(?:第\s*\d+\s*[章节]|\d+(?:\.\d+){0,4})$")
_NUMBER_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:%|℃|°C|:\d+)?$", re.I)
_PATH_RE = re.compile(r"(?:https?://|(?:^|/)images/|\.(?:png|jpe?g|webp)$)", re.I)


def build_entity_filter_config(profile: dict[str, Any]) -> dict[str, Any]:
    """Normalize project-level semantic filters declared by a YAML profile."""
    return {
        "excluded_names": {
            item.strip().casefold()
            for item in profile.get("excluded_entity_names", [])
        },
        "excluded_suffixes": tuple(
            item.strip().casefold()
            for item in profile.get("excluded_entity_suffixes", [])
        ),
        "excluded_patterns": tuple(
            re.compile(item) for item in profile.get("excluded_entity_patterns", [])
        ),
        "generic_names": {
            item.strip().casefold() for item in profile.get("generic_entity_names", [])
        },
        "whitelist": {
            item.strip().casefold() for item in profile.get("entity_name_whitelist", [])
        },
    }


def suspicious_entity_name(name: str) -> bool:
    """Reject menu-like concatenations, while preserving normal technical notation."""
    for separator in ("|", "｜", "&"):
        if separator not in name:
            continue
        pieces = [piece.strip() for piece in name.split(separator)]
        if len(pieces) > 1 and all(piece for piece in pieces):
            return True
    if "/" not in name:
        return False
    pieces = [piece.strip() for piece in name.split("/")]
    return (
        len(pieces) > 1
        and all(len(piece) >= 2 for piece in pieces)
        and all(re.search(r"[\u4e00-\u9fffA-Za-z]", piece) for piece in pieces)
    )


def _reject_entity(
    name: str,
    data: dict[str, Any],
    allowed: set[str],
    filter_config: dict[str, Any],
) -> str | None:
    name = name.strip()
    entity_type = str(data.get("entity_type") or "").strip()
    description = str(data.get("description") or "").strip()
    if not name: return "empty_entity_name"
    if not description: return "empty_entity_description"
    if entity_type.casefold() in {"other", "unknown"}: return "forbidden_entity_type"
    if entity_type.casefold() not in {item.casefold() for item in allowed}: return "invalid_entity_type"
    if _FIGURE_RE.fullmatch(name) or _CHAPTER_RE.fullmatch(name): return "numbering_entity"
    if _NUMBER_RE.fullmatch(name): return "numeric_entity"
    if _PATH_RE.search(name): return "image_or_url_entity"
    folded_name = name.casefold()
    # Whitelist intentionally only bypasses semantic name filtering. It never
    # bypasses missing data, invalid types, numbering, URL, or image checks.
    if folded_name in filter_config["whitelist"]:
        return None
    if folded_name in filter_config["excluded_names"]:
        return "excluded_entity_name"
    if any(folded_name.endswith(suffix) for suffix in filter_config["excluded_suffixes"]):
        return "excluded_entity_suffix"
    if any(pattern.search(name) for pattern in filter_config["excluded_patterns"]):
        return "excluded_entity_pattern"
    if folded_name in filter_config["generic_names"]:
        return "generic_entity_name"
    if suspicious_entity_name(name):
        return "suspicious_entity_name"
    return None


_INCLUDES = {("Technique", "Process"), ("Process", "Process"), ("Concept", "Concept"), ("Object", "Structure")}
_COMPOSES = {("Process", "Technique"), ("Process", "Process"), ("Structure", "Object"), ("Structure", "Structure"), ("Material", "Object")}
_USES = {(source, target) for source in ("Technique", "Method", "Process") for target in ("Tool", "Material")}
_CREATES = {(source, target) for source in ("Technique", "Process", "Material", "Parameter") for target in ("Property", "Object", "QualityIssue")}
_HAS_PARAMETER = {(source, "Parameter") for source in ("Technique", "Process", "Tool", "Object")}
_HAS_PROPERTY = {(source, "Property") for source in ("Material", "Object", "Structure", "Tool", "Concept")}
_APPLIES_TO = {(source, target) for source in ("Tool", "Material", "Method", "Technique") for target in ("Object", "Process", "Concept")}


def normalize_relation_direction(
    source: str,
    target: str,
    keywords: list[str],
    accepted_entity_types: dict[str, str],
) -> tuple[str, str, bool] | None:
    """Return a canonical relation direction, or None when type evidence is weak."""
    source_type = accepted_entity_types.get(source)
    target_type = accepted_entity_types.get(target)
    if not source_type or not target_type:
        return None
    pair = (source_type, target_type)
    reverse_pair = (target_type, source_type)
    expected_pairs: set[tuple[str, str]] | None = None
    for keyword in keywords:
        if keyword == "包括": expected_pairs = _INCLUDES
        elif keyword == "组成": expected_pairs = _COMPOSES
        elif keyword == "使用": expected_pairs = _USES
        elif keyword in {"产生", "形成", "导致"}: expected_pairs = _CREATES
        elif keyword == "具有参数": expected_pairs = _HAS_PARAMETER
        elif keyword == "具有属性": expected_pairs = _HAS_PROPERTY
        elif keyword in {"用于", "适用于"}: expected_pairs = _APPLIES_TO
        elif keyword in {"先于", "后于"}:
            expected_pairs = {("Process", "Process")}
        else:
            # The relation keyword itself defines the direction, but only retain
            # it when both endpoints carry a recognized, typed extraction.
            continue
        if pair in expected_pairs:
            continue
        if reverse_pair in expected_pairs:
            if source == target:
                return None
            source, target = target, source
            source_type, target_type = target_type, source_type
            pair = (source_type, target_type)
            continue
        return None
    return source, target, False


def filter_extraction_results(extraction_results: Any, profile: dict[str, Any] | None) -> tuple[list, dict[str, int], list[dict[str, Any]]]:
    """Apply project validation before LightRAG's merge surface."""
    if not profile:
        return list(extraction_results or []), {"raw_entity_mentions": 0, "accepted_entity_mentions": 0, "rejected_entity_mentions": 0, "raw_relation_mentions": 0, "accepted_relation_mentions": 0, "rejected_relation_mentions": 0, "normalized_relation_direction_count": 0}, []
    allowed_types = {str(item).strip() for item in profile["allowed_entity_types"]}
    allowed_relations = {str(item).strip().casefold() for item in profile["allowed_relation_keywords"]}
    filter_config = build_entity_filter_config(profile)
    counters = Counter()
    rejected: list[dict[str, Any]] = []
    filtered: list = []
    for nodes, edges in extraction_results or []:
        accepted_nodes: dict[str, list] = {}
        accepted_names: set[str] = set()
        accepted_entity_types: dict[str, str] = {}
        seen_entities: set[tuple[str, str]] = set()
        for name, candidates in (nodes or {}).items():
            for data in candidates or []:
                counters["raw_entity_mentions"] += 1
                entity_name = str(name).strip()
                reason = _reject_entity(entity_name, data, allowed_types, filter_config)
                identity = (str(name), str(data.get("entity_type") or ""))
                if reason is None and identity in seen_entities: reason = "duplicate_entity"
                if reason:
                    counters["rejected_entity_mentions"] += 1
                    rejected.append({"kind": "entity", "reason": reason, "entity_name": name, "entity": data})
                    continue
                seen_entities.add(identity); accepted_names.add(entity_name)
                accepted_entity_types.setdefault(
                    entity_name, str(data.get("entity_type") or "").strip()
                )
                accepted_nodes.setdefault(entity_name, []).append(data)
                counters["accepted_entity_mentions"] += 1
        accepted_edges: dict[tuple[str, str], list] = {}
        seen_edges: set[tuple[str, str, str]] = set()
        for edge_key, candidates in (edges or {}).items():
            for data in candidates or []:
                counters["raw_relation_mentions"] += 1
                fallback_source = edge_key[0] if isinstance(edge_key, tuple) and len(edge_key) > 0 else ""
                fallback_target = edge_key[1] if isinstance(edge_key, tuple) and len(edge_key) > 1 else ""
                source = str(data.get("src_id") or fallback_source)
                target = str(data.get("tgt_id") or fallback_target)
                keywords = [item.strip() for item in re.split(r"[,，]", str(data.get("keywords") or "")) if item.strip()]
                reason = None
                if source not in accepted_names or target not in accepted_names: reason = "invalid_relation_endpoint"
                elif not keywords or any(item.casefold() not in allowed_relations for item in keywords): reason = "invalid_relation_keyword"
                elif not str(data.get("description") or "").strip(): reason = "empty_relation_description"
                normalized = None
                if reason is None:
                    normalized = normalize_relation_direction(
                        source, target, keywords, accepted_entity_types
                    )
                    if normalized is None:
                        reason = "ambiguous_relation_direction"
                    else:
                        normalized_source, normalized_target, _ = normalized
                        if (normalized_source, normalized_target) != (source, target):
                            counters["normalized_relation_direction_count"] += 1
                            source, target = normalized_source, normalized_target
                            data = {**data, "src_id": source, "tgt_id": target}
                identity = (source, target, ",".join(keywords))
                if reason is None and identity in seen_edges: reason = "duplicate_relation"
                if reason:
                    counters["rejected_relation_mentions"] += 1
                    rejected.append({"kind": "relation", "reason": reason, "source_entity": source, "target_entity": target, "relation": data})
                    continue
                seen_edges.add(identity); accepted_edges.setdefault((source, target), []).append(data)
                counters["accepted_relation_mentions"] += 1
        filtered.append((accepted_nodes, accepted_edges))
    for key in ("raw_entity_mentions", "accepted_entity_mentions", "rejected_entity_mentions", "raw_relation_mentions", "accepted_relation_mentions", "rejected_relation_mentions", "normalized_relation_direction_count"):
        counters.setdefault(key, 0)
    return filtered, dict(counters), rejected


def failed_chunk_record(
    chunk_id: str,
    chunk: dict[str, Any],
    exc: BaseException,
    *,
    retry_count: int,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": chunk.get("full_doc_id") or chunk.get("doc_id"),
        "catalog_index": chunk.get("catalog_index"),
        "catalog_title": chunk.get("catalog_title"),
        "file_path": chunk.get("file_path"),
        "content_chars": len(str(chunk.get("content") or "")),
        "image_count": len(chunk.get("images") or []),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "retry_count": retry_count,
    }


async def extract_and_merge_chunks(
    rag: LightRAG,
    chunk_records: dict[str, dict[str, Any]],
    *,
    doc_id: str,
    file_path: str,
    current_file_number: int,
    total_files: int,
    profile: dict[str, Any] | None = None,
) -> tuple[int, int, int, dict[str, int], list[dict[str, Any]]]:
    from lightrag.operate import merge_nodes_and_edges

    pipeline_status = {
        "latest_message": "",
        "history_messages": [],
        "cancellation_requested": False,
    }
    pipeline_status_lock = asyncio.Lock()
    extraction_chunks = {
        chunk_id: {**record, "content": str(record.get("kg_content") or record.get("content") or "")}
        for chunk_id, record in chunk_records.items()
    }
    extraction_results = await rag._process_extract_entities(
        extraction_chunks,
        pipeline_status,
        pipeline_status_lock,
    )
    extraction_results, validation_stats, rejected = filter_extraction_results(extraction_results, profile)
    result_count, entity_mentions, relation_mentions = count_extraction_mentions(
        extraction_results
    )
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
        current_file_number=current_file_number,
        total_files=total_files,
        file_path=file_path,
    )
    return result_count, entity_mentions, relation_mentions, validation_stats, rejected


async def extract_single_chunk_with_retry(
    rag: LightRAG,
    chunk_id: str,
    chunk: dict[str, Any],
    *,
    doc_id: str,
    file_path: str,
    current_file_number: int,
    total_files: int,
    single_chunk_retry: int,
    profile: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any] | None, int, int, int, dict[str, int], list[dict[str, Any]]]:
    attempts = max(single_chunk_retry, 0) + 1
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            result_count, entity_mentions, relation_mentions, validation_stats, rejected = await extract_and_merge_chunks(
                rag,
                {chunk_id: chunk},
                doc_id=doc_id,
                file_path=file_path,
                current_file_number=current_file_number,
                total_files=total_files,
                profile=profile,
            )
            logger.info(
                "extract single done | chunk_id=%s ent=%d rel=%d",
                chunk_id,
                entity_mentions,
                relation_mentions,
            )
            return True, None, result_count, entity_mentions, relation_mentions, validation_stats, rejected
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                logger.warning(
                    "extract single retry | chunk_id=%s attempt=%d/%d error=%s",
                    chunk_id,
                    attempt,
                    attempts,
                    exc,
                )
    assert last_error is not None
    logger.error("extract single failed | chunk_id=%s error=%s", chunk_id, last_error)
    return (
        False,
        failed_chunk_record(
            chunk_id,
            chunk,
            last_error,
            retry_count=max(single_chunk_retry, 0),
        ),
        0, 0, 0, {}, [],
    )


async def import_custom_chunks(
    rag: LightRAG,
    chunks: list[dict],
    *,
    replace: bool = False,
    extract_batch_size: int = 8,
    single_chunk_retry: int = 1,
    continue_on_chunk_error: bool = True,
    profile: dict[str, Any] | None = None,
) -> dict:
    """Write caller-owned chunks, then extract KG with chunk-level failure isolation."""
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
            kg_content = str(chunk.get("kg_content") or content)
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
            chunk_record["kg_content"] = kg_content
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
        succeeded_kg_chunk_count = 0
        failed_chunks: list[dict[str, Any]] = []
        validation_totals: Counter = Counter()
        rejected_extractions: list[dict[str, Any]] = []
        total_batches = sum(
            len(
                chunk_batches(
                    {
                        chunk_id: record
                        for chunk_id, record in doc_records.items()
                        if record.get("should_extract_kg", True) is not False
                    },
                    extract_batch_size,
                )
            )
            for doc_records in inserting_chunks_by_doc.values()
        )
        current_batch = 0
        for doc_id, doc_chunks in inserting_chunks_by_doc.items():
            kg_doc_chunks = {
                chunk_id: record
                for chunk_id, record in doc_chunks.items()
                if record.get("should_extract_kg", True) is not False
            }
            if not kg_doc_chunks:
                continue
            batches = chunk_batches(kg_doc_chunks, extract_batch_size)
            for batch_index, batch_chunks in enumerate(batches, start=1):
                current_batch += 1
                logger.info(
                    "extract batch | doc_id=%s batch=%d/%d chunks=%d",
                    doc_id,
                    batch_index,
                    len(batches),
                    len(batch_chunks),
                )
                try:
                    (
                        result_count,
                        entity_mentions,
                        relation_mentions, validation_stats, rejected,
                    ) = await extract_and_merge_chunks(
                        rag,
                        batch_chunks,
                        doc_id=doc_id,
                        file_path=new_docs[doc_id]["file_path"],
                        current_file_number=current_batch,
                        total_files=total_batches,
                        profile=profile,
                    )
                    extraction_result_count += result_count
                    extracted_entity_mentions += entity_mentions
                    extracted_relation_mentions += relation_mentions
                    validation_totals.update(validation_stats); rejected_extractions.extend(rejected)
                    succeeded_kg_chunk_count += len(batch_chunks)
                    logger.info(
                        "extract batch done | doc_id=%s batch=%d/%d chunks=%d "
                        "ent=%d rel=%d",
                        doc_id,
                        batch_index,
                        len(batches),
                        len(batch_chunks),
                        entity_mentions,
                        relation_mentions,
                    )
                    continue
                except Exception as exc:
                    if len(batch_chunks) <= 1:
                        chunk_id, chunk = next(iter(batch_chunks.items()))
                        logger.warning(
                            "extract batch failed, retry as single chunk | "
                            "doc_id=%s batch=%d/%d chunk_id=%s error=%s",
                            doc_id,
                            batch_index,
                            len(batches),
                            chunk_id,
                            exc,
                        )
                        (
                            success,
                            failed_chunk,
                            result_count,
                            entity_mentions,
                            relation_mentions, validation_stats, rejected,
                        ) = await extract_single_chunk_with_retry(
                            rag,
                            chunk_id,
                            chunk,
                            doc_id=doc_id,
                            file_path=new_docs[doc_id]["file_path"],
                            current_file_number=current_batch,
                            total_files=total_batches,
                            single_chunk_retry=single_chunk_retry,
                            profile=profile,
                        )
                        if success:
                            succeeded_kg_chunk_count += 1
                            extraction_result_count += result_count
                            extracted_entity_mentions += entity_mentions
                            extracted_relation_mentions += relation_mentions
                            validation_totals.update(validation_stats); rejected_extractions.extend(rejected)
                        elif failed_chunk is not None:
                            failed_chunks.append(failed_chunk)
                            if not continue_on_chunk_error:
                                raise EntityExtractionError(
                                    "实体关系抽取或合并失败: "
                                    f"{failed_chunk['error_type']}: "
                                    f"{failed_chunk['error']}",
                                    failed_chunks,
                                ) from exc
                        continue
                    logger.warning(
                        "extract batch failed, fallback to single chunks | "
                        "doc_id=%s batch=%d/%d error=%s",
                        doc_id,
                        batch_index,
                        len(batches),
                        exc,
                    )

                for chunk_id, chunk in batch_chunks.items():
                    (
                        success,
                        failed_chunk,
                        result_count,
                        entity_mentions,
                        relation_mentions, validation_stats, rejected,
                    ) = await extract_single_chunk_with_retry(
                        rag,
                        chunk_id,
                        chunk,
                        doc_id=doc_id,
                        file_path=new_docs[doc_id]["file_path"],
                        current_file_number=current_batch,
                        total_files=total_batches,
                        single_chunk_retry=single_chunk_retry,
                        profile=profile,
                    )
                    if success:
                        succeeded_kg_chunk_count += 1
                        extraction_result_count += result_count
                        extracted_entity_mentions += entity_mentions
                        extracted_relation_mentions += relation_mentions
                        validation_totals.update(validation_stats); rejected_extractions.extend(rejected)
                    elif failed_chunk is not None:
                        failed_chunks.append(failed_chunk)
                        if not continue_on_chunk_error:
                            raise EntityExtractionError(
                                "实体关系抽取或合并失败: "
                                f"{failed_chunk['error_type']}: {failed_chunk['error']}",
                                failed_chunks,
                            )
        failed_kg_chunk_count = len(failed_chunks)
        logger.info(
            "custom chunks result | total=%d kg=%d succeeded=%d failed=%d skipped=%d",
            len(chunks),
            kg_chunk_count,
            succeeded_kg_chunk_count,
            failed_kg_chunk_count,
            skipped_kg_chunk_count,
        )
        return {
            "doc_count": len(new_docs),
            "chunk_count": len(inserting_chunks),
            "kg_chunk_count": kg_chunk_count,
            "succeeded_kg_chunk_count": succeeded_kg_chunk_count,
            "failed_kg_chunk_count": failed_kg_chunk_count,
            "skipped_kg_chunk_count": skipped_kg_chunk_count,
            "failed_chunks": failed_chunks,
            "content_scope_stats": content_scope_stats,
            "should_extract_kg_stats": should_extract_kg_stats,
            "extraction_result_count": extraction_result_count,
            "extracted_entity_mentions": extracted_entity_mentions,
            "extracted_relation_mentions": extracted_relation_mentions,
            "validation_stats": dict(validation_totals),
            "rejected_extractions": rejected_extractions,
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
    meta = load_book_meta(args.meta) if args.meta else None
    if meta is not None:
        business = get_business_config(meta)
        args.domain = resolve_config_value(args.domain, business["domain"], "MARKDOWN_GRAPH_DOMAIN", "industrial_training")
        args.subject = resolve_config_value(args.subject, business["subject"], "MARKDOWN_GRAPH_SUBJECT", "")
    else:
        args.domain = args.domain or os.getenv("MARKDOWN_GRAPH_DOMAIN", "industrial_training")
        args.subject = args.subject or os.getenv("MARKDOWN_GRAPH_SUBJECT", "")
    if args.working_dir is None:
        args.working_dir = Path(os.getenv("MARKDOWN_GRAPH_WORKING_DIR", str(DEFAULT_WORKING_DIR)))
    chunks_path = args.chunks.expanduser().resolve()
    chunks = read_jsonl(chunks_path)
    selected_chunks, debug_selection = select_kg_chunks(chunks, args)
    validate_chunks(selected_chunks)
    if args.dry_run_selected_chunks:
        print_dry_run_selection(selected_chunks, debug_selection)
        return 0

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
    rejected_path = (PROJECT_DIR / "outputs/06_import" / f"{book_stem}.rejected_extractions.jsonl").resolve()
    started = datetime.now(timezone.utc)
    failures: list[dict] = []
    error_message = ""
    result = {
        "doc_count": 0,
        "chunk_count": len(selected_chunks),
        "kg_chunk_count": sum(
            chunk.get("should_extract_kg", True) is not False
            for chunk in selected_chunks
        ),
        "succeeded_kg_chunk_count": 0,
        "failed_kg_chunk_count": 0,
        "skipped_kg_chunk_count": sum(
            chunk.get("should_extract_kg") is False for chunk in selected_chunks
        ),
        "failed_chunks": [],
        "content_scope_stats": dict(
            sorted(
                Counter(chunk.get("content_scope") for chunk in selected_chunks).items()
            )
        ),
        "should_extract_kg_stats": dict(
            sorted(
                Counter(
                    chunk.get("should_extract_kg") for chunk in selected_chunks
                ).items()
            )
        ),
        "extraction_result_count": 0,
        "extracted_entity_mentions": 0,
        "extracted_relation_mentions": 0,
    }
    if args.extract_batch_size <= 0:
        raise ValueError("--extract-batch-size 必须大于 0")
    if args.single_chunk_retry < 0:
        raise ValueError("--single-chunk-retry 不能小于 0")

    profile_path, profile, prompt_meta = resolve_entity_type_profile(args)
    entity_types_guidance = None
    if profile_path is not None:
        logger.info("entity type profile | file=%s sha256=%s allowed_types=%s", prompt_meta["profile_file"], prompt_meta["profile_sha256"], prompt_meta["allowed_entity_types"])
    else:
        # Explicit compatibility only: legacy debug callers may still supply a
        # guidance file. New meta-driven runs require a per-book YAML profile.
        if args.entity_types_guidance_file is None:
            raise ValueError("Step 6 缺少教材专属实体类型 YAML；请在 meta.entity_extraction.prompt_file 配置，或显式传 --entity-type-prompt-file")
        entity_types_guidance, prompt_meta = build_entity_types_guidance(args)
        prompt_meta["deprecated"] = True
        logger.warning("使用已废弃的 --entity-types-guidance-file；新流水线应使用 YAML profile")

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
    rag = build_rag(working_dir, profile_file_name=profile_path.name if profile_path else None, entity_types_guidance=entity_types_guidance)
    await rag.initialize_storages()
    logger.info("LightRAG storages initialized")
    try:
        try:
            result = await import_custom_chunks(
                rag,
                selected_chunks,
                replace=args.replace,
                extract_batch_size=args.extract_batch_size,
                single_chunk_retry=args.single_chunk_retry,
                continue_on_chunk_error=args.continue_on_chunk_error,
                profile=profile,
            )
        except EntityExtractionError as exc:
            error_message = str(exc)
            failures = exc.chunks
            result["failed_chunks"] = failures
            result["failed_kg_chunk_count"] = len(failures)
        labels = await rag.get_graph_labels()
        relations = await rag.chunk_entity_relation_graph.get_all_edges()
    finally:
        await rag.finalize_storages()
        logger.info("LightRAG storages finalized")

    failures = list(result.get("failed_chunks") or failures)
    failed_kg_chunk_count = int(result.get("failed_kg_chunk_count") or len(failures))
    succeeded_kg_chunk_count = int(result.get("succeeded_kg_chunk_count") or 0)
    write_jsonl(failed_path, failures)
    write_jsonl(rejected_path, result.get("rejected_extractions") or [])
    report = {
        "working_dir": str(working_dir),
        "doc_count": result["doc_count"],
        "chunk_count": result["chunk_count"],
        "kg_chunk_count": result["kg_chunk_count"],
        "succeeded_kg_chunk_count": succeeded_kg_chunk_count,
        "failed_kg_chunk_count": failed_kg_chunk_count,
        "skipped_kg_chunk_count": result["skipped_kg_chunk_count"],
        "content_scope_stats": result["content_scope_stats"],
        "should_extract_kg_stats": result["should_extract_kg_stats"],
        "prompt": prompt_meta,
        "prompt_meta": prompt_meta,
        "entity_count_after": len(labels),
        "relation_count_after": len(relations),
        "extracted_entity_mentions": result["extracted_entity_mentions"],
        "extracted_relation_mentions": result["extracted_relation_mentions"],
        "raw_entity_mentions": result.get("validation_stats", {}).get("raw_entity_mentions", 0),
        "accepted_entity_mentions": result.get("validation_stats", {}).get("accepted_entity_mentions", 0),
        "rejected_entity_mentions": result.get("validation_stats", {}).get("rejected_entity_mentions", 0),
        "raw_relation_mentions": result.get("validation_stats", {}).get("raw_relation_mentions", 0),
        "accepted_relation_mentions": result.get("validation_stats", {}).get("accepted_relation_mentions", 0),
        "rejected_relation_mentions": result.get("validation_stats", {}).get("rejected_relation_mentions", 0),
        "normalized_relation_direction_count": result.get("validation_stats", {}).get("normalized_relation_direction_count", 0),
        "rejected_by_reason": dict(Counter(item.get("reason") for item in result.get("rejected_extractions", []) if item.get("reason"))),
        "rejected_extractions_path": str(rejected_path),
        "prompt_profile": prompt_meta,
        "replace": bool(args.replace),
        "failures": failed_kg_chunk_count,
        "failed_chunk_ids": [chunk.get("chunk_id") for chunk in failures],
        "debug_selection": debug_selection,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if error_message:
        report["error"] = error_message
    write_json(report_path, report)
    logger.info(
        "extraction validation | raw_entity_mentions=%d accepted_entity_mentions=%d "
        "rejected_entity_mentions=%d raw_relation_mentions=%d "
        "accepted_relation_mentions=%d rejected_relation_mentions=%d "
        "normalized_relation_direction_count=%d rejected_by_reason=%s",
        report["raw_entity_mentions"],
        report["accepted_entity_mentions"],
        report["rejected_entity_mentions"],
        report["raw_relation_mentions"],
        report["accepted_relation_mentions"],
        report["rejected_relation_mentions"],
        report["normalized_relation_direction_count"],
        dict(list(report["rejected_by_reason"].items())[:8]),
    )
    logger.info("导入报告 | %s", report_path)
    if failed_kg_chunk_count:
        logger.error("导入存在失败 chunks=%d | %s", failed_kg_chunk_count, failed_path)
        return 1 if args.failed_chunk_policy == "return_nonzero" else 0
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
