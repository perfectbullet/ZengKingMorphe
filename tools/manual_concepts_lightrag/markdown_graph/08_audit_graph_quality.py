#!/usr/bin/env python3
"""Audit an existing file-backed LightRAG graph without mutating it."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable

from dotenv import load_dotenv
import networkx as nx

from common import PROJECT_DIR

logger = logging.getLogger(__name__)

SCHEMAS_DIR = PROJECT_DIR / "prompts/schemas"
DEFAULT_SCHEMA_VERSION = "industrial_training_kg_schema.v1"
DEFAULT_SCHEMA_JSON = SCHEMAS_DIR / "industrial_training_kg_schema.v1.json"
DEFAULT_AUDIT_WORKING_DIR = Path(
    "/home/zj/ZengKingMorphe/ai-service/data/lightrag_industrial_training"
)
GRAPH_FILENAME = "graph_chunk_entity_relation.graphml"
OPTIONAL_STORE_FILENAMES = [
    "vdb_entities.json",
    "vdb_relationships.json",
    "kv_store_text_chunks.json",
    "kv_store_full_docs.json",
]
PURE_FIGURE_RE = re.compile(r"^图\s*\d+[-－]\d+$", re.I)
STEP_RE = re.compile(r"^(?:STEP\s*0?\d+|操作步骤\s*\d+)$", re.I)
IMAGE_PATH_RE = re.compile(
    r"(?:images/|https?://|\.(?:jpe?g|png|webp)(?:\b|$))", re.I
)
NUMERIC_ONLY_RES = [
    re.compile(r"^\d+$"),
    re.compile(r"^(?:第\s*)?\d+\s*页$", re.I),
    re.compile(r"^[Pp]\.?\s*\d+$"),
    re.compile(r"^\d+\s*/\s*\d+$"),
    re.compile(r"^\d+(?:\.\d+)?\s*(?:摄氏度|℃|°C)$", re.I),
    re.compile(r"^\d+(?:\.\d+)?%$"),
    re.compile(r"^\d+\s*:\s*\d+$"),
]
CHAPTER_LIKE_RES = [
    re.compile(r"^第\s*\d+\s*章$", re.I),
    re.compile(r"^CHAPTER\s*\d+$", re.I),
    re.compile(r"^\d+(?:\.\d+)+$"),
]
ENGLISH_PERSON_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$")
PERSON_HINT_RE = re.compile(r"(?:作者\s*[:：]|\bAuthor\b|\bby\b)", re.I)
FIGURE_LIKE_RE = re.compile(r"图\s*\d+[-－]\d+", re.I)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
URL_RE = re.compile(r"https?://[^\s<>()\[\]]+", re.I)
LOCAL_IMAGE_PATH_RE = re.compile(r"(?:\.\.?/)*images/[^\s<>()\[\]]+", re.I)
IMAGE_FILENAME_RE = re.compile(r"\b[^\s/]+\.(?:jpe?g|png|webp)\b", re.I)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读审计 LightRAG 工业教材知识图谱质量"
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path(
            os.getenv("MARKDOWN_GRAPH_WORKING_DIR", str(DEFAULT_AUDIT_WORKING_DIR))
        ),
    )
    parser.add_argument("--book-stem", default="")
    parser.add_argument(
        "--domain", default=os.getenv("MARKDOWN_GRAPH_DOMAIN", "industrial_training")
    )
    parser.add_argument(
        "--subject", default=os.getenv("MARKDOWN_GRAPH_SUBJECT", "enamel")
    )
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--anchor-plan", type=Path)
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
    return parser.parse_args()


def normalize_json_records(data: Any) -> list[dict[str, Any]]:
    """Normalize LightRAG VDB JSON variants into a list of records."""
    if isinstance(data, dict) and "data" in data:
        return normalize_json_records(data["data"])
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        if not data:
            return []
        if data and all(isinstance(value, dict) for value in data.values()):
            records: list[dict[str, Any]] = []
            for key, value in data.items():
                record = dict(value)
                record.setdefault("_key", str(key))
                records.append(record)
            return records
        return [dict(data)]
    return []


def load_optional_json(path: Path, missing_files: list[str]) -> Any:
    if not path.is_file():
        missing_files.append(str(path))
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取 JSON 失败，按缺失处理 | path=%s error=%s", path, exc)
        missing_files.append(str(path))
        return {}


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def truncate(value: Any, limit: int = 160) -> str:
    text = stringify(value).replace("\r", " ").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def entity_name(node_id: Any, attrs: dict[str, Any]) -> str:
    return stringify(attrs.get("entity_id") or node_id).strip()


def normalized_entity_type(attrs: dict[str, Any]) -> str:
    value = stringify(attrs.get("entity_type")).strip().lower()
    return "UNKNOWN" if value in {"", "unknown", "none", "null"} else value


def node_sample(
    graph: nx.Graph, node_id: Any, attrs: dict[str, Any]
) -> dict[str, Any]:
    return {
        "node_id": entity_name(node_id, attrs),
        "entity_type": normalized_entity_type(attrs),
        "degree": int(graph.degree(node_id)),
        "description": truncate(attrs.get("description")),
        "source_id": truncate(attrs.get("source_id")),
        "file_path": truncate(attrs.get("file_path")),
    }


def audit_matching_nodes(
    graph: nx.Graph,
    nodes: list[tuple[Any, dict[str, Any]]],
    predicate: Callable[[str, dict[str, Any]], bool],
    *,
    sample_limit: int = 50,
) -> dict[str, Any]:
    matched = [
        (node_id, attrs)
        for node_id, attrs in nodes
        if predicate(entity_name(node_id, attrs), attrs)
    ]
    return {
        "count": len(matched),
        "samples": [
            node_sample(graph, node_id, attrs)
            for node_id, attrs in matched[:sample_limit]
        ],
    }


def contains_book_stem(values: list[Any], book_stem: str) -> bool:
    needle = book_stem.casefold()
    return any(needle in stringify(value).casefold() for value in values if value)


def normalize_chunk_store(data: Any) -> dict[str, dict[str, Any]]:
    if isinstance(data, dict) and "data" in data:
        return normalize_chunk_store(data["data"])
    if isinstance(data, dict):
        result: dict[str, dict[str, Any]] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                record = dict(value)
                record.setdefault("chunk_id", str(key))
                result[str(key)] = record
        return result
    if isinstance(data, list):
        return {
            stringify(item.get("chunk_id") or item.get("_id") or index): dict(item)
            for index, item in enumerate(data)
            if isinstance(item, dict)
        }
    return {}


def select_book_data(
    graph: nx.Graph,
    chunks: dict[str, dict[str, Any]],
    book_stem: str,
) -> tuple[
    list[tuple[Any, dict[str, Any]]],
    list[tuple[Any, Any, dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    all_nodes = list(graph.nodes(data=True))
    all_edges = list(graph.edges(data=True))
    if not book_stem:
        return all_nodes, all_edges, chunks, {
            "book_filter_enabled": False,
            "filtered": False,
            "matched_node_count": len(all_nodes),
            "matched_edge_count": len(all_edges),
            "matched_chunk_count": len(chunks),
            "filter_warning": "",
        }

    matched_nodes = [
        (node_id, attrs)
        for node_id, attrs in all_nodes
        if contains_book_stem(
            [
                node_id,
                attrs.get("entity_id"),
                attrs.get("source_id"),
                attrs.get("file_path"),
                attrs.get("description"),
            ],
            book_stem,
        )
    ]
    matched_node_ids = {node_id for node_id, _ in matched_nodes}
    matched_edges = [
        (source, target, attrs)
        for source, target, attrs in all_edges
        if source in matched_node_ids
        or target in matched_node_ids
        or contains_book_stem(
            [attrs.get("source_id"), attrs.get("file_path")], book_stem
        )
    ]
    matched_chunks = {
        chunk_id: chunk
        for chunk_id, chunk in chunks.items()
        if contains_book_stem(
            [
                chunk_id,
                chunk.get("chunk_id"),
                chunk.get("doc_id"),
                chunk.get("full_doc_id"),
                chunk.get("file_path"),
                chunk.get("source_md_path"),
                chunk.get("catalog_title"),
            ],
            book_stem,
        )
    }
    counts = {
        "matched_node_count": len(matched_nodes),
        "matched_edge_count": len(matched_edges),
        "matched_chunk_count": len(matched_chunks),
    }
    if not matched_nodes and not matched_edges and not matched_chunks:
        return all_nodes, all_edges, chunks, {
            "book_filter_enabled": True,
            "filtered": False,
            **counts,
            "filter_warning": (
                f"book_stem={book_stem!r} 未匹配任何节点、边或 chunk，已退回全量统计"
            ),
        }
    warning_parts = []
    if not matched_nodes:
        warning_parts.append("matched_node_count=0")
    if not matched_chunks:
        warning_parts.append("matched_chunk_count=0")
    return matched_nodes, matched_edges, matched_chunks, {
        "book_filter_enabled": True,
        "filtered": True,
        **counts,
        "filter_warning": "；".join(warning_parts),
    }


def image_items_from_chunk(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for field in ("images", "image_assets", "image_lines"):
        value = chunk.get(field)
        if isinstance(value, list):
            items = [
                dict(item) if isinstance(item, dict) else {"line_no": item}
                for item in value
            ]
        elif isinstance(value, dict):
            items = normalize_json_records(value)
        else:
            items = []
        for item in items:
            path = stringify(
                item.get("relative_path")
                or item.get("local_path")
                or item.get("img_path")
                or item.get("url")
                or item.get("original_url")
                or item.get("absolute_path")
            ).strip()
            line_no = item.get("line_no")
            existing = next(
                (
                    candidate
                    for candidate in merged
                    if (path and path == candidate.get("_audit_path"))
                    or (
                        line_no is not None
                        and line_no == candidate.get("line_no")
                    )
                ),
                None,
            )
            if existing is None:
                merged.append({**item, "_audit_path": path})
            else:
                for key, item_value in item.items():
                    existing_value = existing.get(key)
                    existing_empty = (
                        existing_value is None
                        or existing_value == ""
                        or existing_value == "[]"
                        or existing_value == []
                    )
                    item_has_value = not (
                        item_value is None
                        or item_value == ""
                        or item_value == "[]"
                        or item_value == []
                    )
                    if existing_empty and item_has_value:
                        existing[key] = item_value
    for item in merged:
        item.pop("_audit_path", None)
    return merged


def audit_image_metadata(chunks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for chunk_id, chunk in chunks.items():
        items = image_items_from_chunk(chunk)
        if items:
            stats["chunks_with_images"] += 1
        for item in items:
            stats["total_images"] += 1
            caption = item.get("caption") or item.get("image_caption")
            raw_caption = item.get("raw_caption")
            if stringify(caption).strip():
                stats["images_with_caption"] += 1
            else:
                stats["images_missing_caption_count"] += 1
            if stringify(raw_caption).strip():
                stats["images_with_raw_caption"] += 1
            if stringify(item.get("figure_no")).strip():
                stats["images_with_figure_no"] += 1
            if item.get("page_idx") is not None:
                stats["images_with_page_idx"] += 1
            bbox = item.get("bbox")
            if bbox is not None and bbox != "" and bbox != "[]" and bbox != []:
                stats["images_with_bbox"] += 1
            local_path = item.get("local_path") or item.get("img_path")
            relative_path = stringify(item.get("relative_path")).strip()
            if local_path or (relative_path and not relative_path.startswith(("http://", "https://"))):
                stats["images_with_local_path"] += 1
            if stringify(item.get("absolute_path")).strip():
                stats["images_with_absolute_path"] += 1
            exists = item.get("exists")
            if exists is True:
                stats["images_exists_true"] += 1
            elif exists is False:
                stats["images_exists_false"] += 1
            else:
                stats["images_exists_none"] += 1
            url = stringify(item.get("url") or item.get("original_url") or relative_path)
            reference_type = stringify(item.get("reference_type")).lower()
            if url.startswith(("http://", "https://")) or reference_type == "remote_url":
                stats["images_remote_url_count"] += 1
            else:
                stats["images_local_reference_count"] += 1
            if len(samples) < 30:
                samples.append(
                    {
                        "chunk_id": chunk_id,
                        "line_no": item.get("line_no"),
                        "relative_path": truncate(item.get("relative_path")),
                        "local_path": truncate(local_path),
                        "url": truncate(item.get("url") or item.get("original_url")),
                        "reference_type": item.get("reference_type"),
                        "exists": exists,
                        "caption": truncate(caption),
                        "raw_caption": truncate(raw_caption),
                        "figure_no": item.get("figure_no"),
                        "page_idx": item.get("page_idx"),
                        "bbox": item.get("bbox"),
                    }
                )
    fields = [
        "chunks_with_images",
        "total_images",
        "images_with_caption",
        "images_with_raw_caption",
        "images_with_figure_no",
        "images_with_page_idx",
        "images_with_bbox",
        "images_with_local_path",
        "images_with_absolute_path",
        "images_exists_true",
        "images_exists_false",
        "images_exists_none",
        "images_remote_url_count",
        "images_local_reference_count",
        "images_missing_caption_count",
    ]
    return {**{field: stats[field] for field in fields}, "sample_images": samples}


def matched_snippet(content: str, match: re.Match[str], radius: int = 80) -> str:
    start = max(0, match.start() - radius)
    end = min(len(content), match.end() + radius)
    return truncate(content[start:end], 220)


def audit_chunk_noise(chunks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    url_count = 0
    local_count = 0
    markdown_count = 0
    filename_count = 0
    samples: list[dict[str, Any]] = []
    for chunk_id, chunk in chunks.items():
        content = stringify(chunk.get("content"))
        matches = {
            "url": URL_RE.search(content),
            "local_image_path": LOCAL_IMAGE_PATH_RE.search(content),
            "markdown_image": MARKDOWN_IMAGE_RE.search(content),
            "image_filename": IMAGE_FILENAME_RE.search(content),
        }
        url_count += matches["url"] is not None
        local_count += matches["local_image_path"] is not None
        markdown_count += matches["markdown_image"] is not None
        filename_count += matches["image_filename"] is not None
        present = [(kind, match) for kind, match in matches.items() if match]
        if present and len(samples) < 30:
            first_kind, first_match = min(present, key=lambda item: item[1].start())
            samples.append(
                {
                    "chunk_id": chunk_id,
                    "catalog_title": truncate(chunk.get("catalog_title")),
                    "file_path": truncate(chunk.get("file_path")),
                    "matched_types": [kind for kind, _ in present],
                    "matched_snippet": matched_snippet(content, first_match),
                    "first_match_type": first_kind,
                }
            )
    return {
        "chunks_with_url_in_content": url_count,
        "chunks_with_local_image_path_in_content": local_count,
        "chunks_with_markdown_image_syntax": markdown_count,
        "chunks_with_image_filename_in_content": filename_count,
        "samples": samples,
    }


def audit_source_missing(anchor_plan: Path | None) -> dict[str, Any]:
    if anchor_plan is None or not anchor_plan.is_file():
        return {
            "anchor_plan": str(anchor_plan) if anchor_plan else "",
            "anchor_plan_found": False,
            "source_missing_count": 0,
            "source_missing_items": [],
        }
    records = []
    for line_number, line in enumerate(
        anchor_plan.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"anchor plan JSONL 解析失败: {anchor_plan}:{line_number}: {exc}"
            ) from exc
        if isinstance(record, dict):
            records.append(record)
    items = [
        {
            "catalog_index": record.get("catalog_index"),
            "catalog_level": record.get("catalog_level"),
            "catalog_title": record.get("catalog_title") or record.get("title"),
            "number_key": record.get("catalog_number_key") or record.get("number_key"),
            "reason": record.get("reason"),
            "manual_override": record.get("manual_override"),
        }
        for record in records
        if record.get("review_status") == "source_missing"
    ]
    return {
        "anchor_plan": str(anchor_plan),
        "anchor_plan_found": True,
        "source_missing_count": len(items),
        "source_missing_items": items,
    }


def load_schema_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        logger.warning("schema json 不存在，跳过 schema 审计: %s", path)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("schema json 读取失败，跳过 schema 审计 | path=%s error=%s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("schema json 顶层不是对象，跳过 schema 审计: %s", path)
        return None
    return data


def edge_sample(
    source: Any,
    target: Any,
    attrs: dict[str, Any],
    relation_type: str | None,
) -> dict[str, Any]:
    return {
        "source": truncate(source),
        "target": truncate(target),
        "relation_type": relation_type,
        "keywords": truncate(attrs.get("keywords")),
        "description": truncate(attrs.get("description")),
        "source_id": truncate(attrs.get("source_id")),
        "file_path": truncate(attrs.get("file_path")),
    }


def resolve_edge_relation_type(
    attrs: dict[str, Any], allowed_relation_types: set[str]
) -> str | None:
    for field in ("relation_type", "edge_type", "type"):
        value = stringify(attrs.get(field)).strip()
        if value:
            return value
    keywords = stringify(attrs.get("keywords")).strip()
    if keywords:
        tokens = [
            token.strip()
            for token in re.split(r"<SEP>|[,，;；|/]", keywords)
            if token.strip()
        ]
        for token in tokens:
            if token in allowed_relation_types:
                return token
        return tokens[0] if tokens else keywords
    description = stringify(attrs.get("description"))
    for relation_type in sorted(allowed_relation_types, key=len, reverse=True):
        if relation_type in description:
            return relation_type
    return None


def audit_schema(
    graph: nx.Graph,
    nodes: list[tuple[Any, dict[str, Any]]],
    edges: list[tuple[Any, Any, dict[str, Any]]],
    schema_path: Path,
    configured_version: str,
) -> dict[str, Any]:
    schema = load_schema_json(schema_path)
    base = {
        "schema_version": configured_version,
        "schema_json_path": str(schema_path),
        "schema_loaded": schema is not None,
        "allowed_entity_types": [],
        "forbidden_entity_types": [],
        "allowed_relation_types": [],
        "invalid_entity_type_count": 0,
        "forbidden_entity_type_count": 0,
        "unknown_entity_type_count": 0,
        "invalid_relation_type_count": 0,
        "unknown_relation_type_count": 0,
        "invalid_entity_type_samples": [],
        "forbidden_entity_type_samples": [],
        "invalid_relation_type_samples": [],
        "unknown_relation_type_samples": [],
        "forbidden_name_pattern_hits": {},
    }
    if schema is None:
        return base

    allowed_entities = {
        stringify(value).strip().upper()
        for value in schema.get("allowed_entity_types", [])
        if stringify(value).strip()
    }
    forbidden_entities = {
        stringify(value).strip().upper()
        for value in schema.get("forbidden_entity_types", [])
        if stringify(value).strip()
    }
    allowed_relations = {
        stringify(value).strip()
        for value in schema.get("allowed_relation_types", [])
        if stringify(value).strip()
    }
    base.update(
        {
            "schema_version": stringify(schema.get("schema_version"))
            or configured_version,
            "allowed_entity_types": sorted(allowed_entities),
            "forbidden_entity_types": sorted(forbidden_entities),
            "allowed_relation_types": sorted(allowed_relations),
        }
    )

    for node_id, attrs in nodes:
        raw_type = stringify(attrs.get("entity_type")).strip().upper()
        if raw_type in {"", "UNKNOWN", "NONE", "NULL"}:
            base["unknown_entity_type_count"] += 1
        elif raw_type in allowed_entities:
            pass
        elif raw_type in forbidden_entities:
            base["forbidden_entity_type_count"] += 1
            if len(base["forbidden_entity_type_samples"]) < 50:
                base["forbidden_entity_type_samples"].append(
                    node_sample(graph, node_id, attrs)
                )
        else:
            base["invalid_entity_type_count"] += 1
            if len(base["invalid_entity_type_samples"]) < 50:
                base["invalid_entity_type_samples"].append(
                    node_sample(graph, node_id, attrs)
                )

    for source, target, attrs in edges:
        relation_type = resolve_edge_relation_type(attrs, allowed_relations)
        if relation_type is None:
            base["unknown_relation_type_count"] += 1
            if len(base["unknown_relation_type_samples"]) < 50:
                base["unknown_relation_type_samples"].append(
                    edge_sample(source, target, attrs, None)
                )
        elif relation_type not in allowed_relations:
            base["invalid_relation_type_count"] += 1
            if len(base["invalid_relation_type_samples"]) < 50:
                base["invalid_relation_type_samples"].append(
                    edge_sample(source, target, attrs, relation_type)
                )

    pattern_hits: dict[str, dict[str, Any]] = {}
    for index, rule in enumerate(schema.get("forbidden_entity_name_patterns", []), 1):
        if not isinstance(rule, dict):
            continue
        name = stringify(rule.get("name") or f"pattern_{index}")
        pattern_text = stringify(rule.get("pattern"))
        try:
            pattern = re.compile(pattern_text, re.I)
        except re.error as exc:
            logger.warning("schema 禁抽正则无效 | name=%s error=%s", name, exc)
            pattern_hits[name] = {
                "count": 0,
                "samples": [],
                "severity": rule.get("severity"),
                "message": rule.get("message"),
                "pattern": pattern_text,
                "regex_error": str(exc),
            }
            continue
        matched = []
        for node_id, attrs in nodes:
            names = {stringify(node_id), stringify(attrs.get("entity_id"))}
            if any(value and pattern.search(value) for value in names):
                matched.append((node_id, attrs))
        pattern_hits[name] = {
            "count": len(matched),
            "samples": [
                node_sample(graph, node_id, attrs) for node_id, attrs in matched[:50]
            ],
            "severity": rule.get("severity"),
            "message": rule.get("message"),
            "pattern": pattern_text,
        }
    base["forbidden_name_pattern_hits"] = pattern_hits
    return base


def calculate_quality_score(audit: dict[str, Any]) -> tuple[float, str]:
    forbidden = audit["forbidden_entity_audit"]
    unknown = audit["unknown_type_audit"]
    evidence = audit["evidence_completeness"]
    noise = audit["chunk_content_noise_audit"]
    schema = audit["schema_audit"]
    forbidden_pattern_error_count = sum(
        item.get("count", 0)
        for item in schema["forbidden_name_pattern_hits"].values()
        if stringify(item.get("severity")).lower() == "error"
    )
    penalties = [
        min(forbidden["pure_figure_nodes"]["count"] * 0.2, 20),
        min(forbidden["image_path_nodes"]["count"] * 1.0, 20),
        min(forbidden["step_nodes"]["count"] * 0.5, 10),
        min(forbidden["numeric_only_nodes"]["count"] * 0.5, 10),
        min(forbidden["chapter_like_nodes"]["count"] * 0.2, 5),
        min(unknown["unknown_count"] * 0.2, 10),
        min(evidence["missing_source_id_count"] * 0.1, 10),
        min(evidence["missing_file_path_count"] * 0.1, 10),
        min(noise["chunks_with_url_in_content"] * 1.0, 10),
        min(noise["chunks_with_local_image_path_in_content"] * 1.0, 10),
        min(schema["invalid_entity_type_count"] * 1.0, 15),
        min(schema["forbidden_entity_type_count"] * 1.0, 20),
        min(schema["invalid_relation_type_count"] * 0.5, 10),
        min(forbidden_pattern_error_count * 0.5, 20),
    ]
    score = round(max(0.0, 100.0 - sum(penalties)), 2)
    if score >= 90:
        level = "excellent"
    elif score >= 80:
        level = "good"
    elif score >= 70:
        level = "usable"
    elif score >= 60:
        level = "noisy"
    else:
        level = "poor"
    return score, level


def build_recommendations(audit: dict[str, Any]) -> list[str]:
    forbidden = audit["forbidden_entity_audit"]
    figure = audit["figure_node_audit"]
    noise = audit["chunk_content_noise_audit"]
    images = audit["image_metadata_audit"]
    source_missing = audit["source_missing_audit"]
    evidence = audit["evidence_completeness"]
    schema = audit["schema_audit"]
    recommendations: list[str] = []
    if forbidden["pure_figure_nodes"]["count"] or figure["figure_type_count"]:
        recommendations.append(
            "发现纯图号/FIGURE 节点。建议在 01_enamel.profile.md 中禁用 FIGURE 实体，"
            "图号只作为 evidence.figure_no 保存。"
        )
    if forbidden["image_path_nodes"]["count"]:
        recommendations.append(
            "发现图片路径或 URL 被抽成实体。建议加强抽取 prompt 禁止图片路径、hash、URL 作为实体。"
        )
    if noise["chunks_with_url_in_content"]:
        recommendations.append(
            "chunk content 中仍有 URL，需修复 05_build_lightrag_chunks.py，"
            "确保图片路径只保留在 metadata。"
        )
    total_images = images["total_images"]
    if total_images and images["images_with_caption"] / total_images < 0.5:
        recommendations.append(
            "图片 caption 覆盖率低，建议接入 MinerU content_list_v2.json "
            "补充 image_caption/page_idx/bbox。"
        )
    if source_missing["source_missing_count"]:
        recommendations.append(
            "检测到 source_missing 目录项，建议在查询层注入 source_missing evidence，"
            "防止相邻章节串章。"
        )
    if evidence["missing_source_id_count"] or evidence["missing_file_path_count"]:
        recommendations.append(
            "部分实体缺少 source_id 或 file_path，建议修复导入 metadata 以保证稳定溯源。"
        )
    if schema["forbidden_entity_type_count"]:
        recommendations.append(
            "发现 schema 明确禁止的实体类型，建议收紧 base prompt/profile，禁止 forbidden entity type。"
        )
    if schema["invalid_entity_type_count"]:
        recommendations.append(
            "发现统一 schema 之外的实体类型，建议检查 LLM 是否输出未允许的 entity_type。"
        )
    if schema["invalid_relation_type_count"]:
        recommendations.append(
            "发现统一 schema 之外的关系类型，建议检查关系抽取是否超出 allowed_relation_types。"
        )
    if schema["forbidden_name_pattern_hits"].get("pure_figure_no", {}).get("count"):
        recommendations.append(
            "schema 检测到纯图号实体，建议图号只进入 evidence.figure_no。"
        )
    return recommendations or ["未发现需要立即处理的规则型问题。"]


def markdown_cell(value: Any) -> str:
    text = stringify(value).replace("\r", " ").replace("\n", "<br>")
    return text.replace("|", "\\|")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(markdown_cell(value) for value in row) + " |"
        for row in rows
    )
    if not rows:
        lines.append("| " + " | ".join(["（无）"] + [""] * (len(headers) - 1)) + " |")
    return lines


def sample_table(samples: list[dict[str, Any]]) -> list[str]:
    headers = ["node_id", "entity_type", "degree", "description", "source_id", "file_path"]
    rows = [[sample.get(header, "") for header in headers] for sample in samples]
    return markdown_table(headers, rows)


def render_markdown(audit: dict[str, Any]) -> str:
    meta = audit["meta"]
    overall = audit["overall_stats"]
    forbidden = audit["forbidden_entity_audit"]
    lines = ["# KG Quality Audit Report", "", "## Meta", ""]
    meta_rows = [
        ["generated_at", meta["generated_at"]],
        ["working_dir", meta["working_dir"]],
        ["book_stem", meta["book_stem"]],
        ["domain", meta["domain"]],
        ["subject", meta["subject"]],
        ["quality_score", audit["quality_score"]],
        ["quality_level", audit["quality_level"]],
        ["book_filter_enabled", meta["book_filter_enabled"]],
        ["filtered", meta["filtered"]],
        ["matched_node_count", meta["matched_node_count"]],
        ["matched_edge_count", meta["matched_edge_count"]],
        ["matched_chunk_count", meta["matched_chunk_count"]],
        ["filter_warning", meta.get("filter_warning", "")],
        ["missing_files", meta["missing_files"]],
    ]
    lines.extend(markdown_table(["field", "value"], meta_rows))
    lines.extend(["", "## Overall Stats", ""])
    lines.extend(
        markdown_table(
            ["metric", "count"],
            [
                ["nodes", overall["node_count"]],
                ["edges", overall["edge_count"]],
                ["chunks", overall["chunk_count"]],
                ["entity_vdb_count", overall["entity_vdb_count"]],
                ["relationship_vdb_count", overall["relationship_vdb_count"]],
            ],
        )
    )
    schema = audit["schema_audit"]
    lines.extend(["", "## Schema Audit", ""])
    lines.extend(
        markdown_table(
            ["field", "value"],
            [
                ["schema_version", schema["schema_version"]],
                ["schema_json_path", schema["schema_json_path"]],
                ["schema_loaded", schema["schema_loaded"]],
                ["invalid_entity_type_count", schema["invalid_entity_type_count"]],
                [
                    "forbidden_entity_type_count",
                    schema["forbidden_entity_type_count"],
                ],
                ["unknown_entity_type_count", schema["unknown_entity_type_count"]],
                [
                    "invalid_relation_type_count",
                    schema["invalid_relation_type_count"],
                ],
                [
                    "unknown_relation_type_count",
                    schema["unknown_relation_type_count"],
                ],
            ],
        )
    )
    lines.extend(["", "### Forbidden Name Pattern Hits", ""])
    lines.extend(
        markdown_table(
            ["name", "severity", "count", "message"],
            [
                [
                    name,
                    item.get("severity", ""),
                    item.get("count", 0),
                    item.get("message", ""),
                ]
                for name, item in schema["forbidden_name_pattern_hits"].items()
            ],
        )
    )
    lines.extend(["", "## Entity Type Distribution", ""])
    lines.extend(
        markdown_table(
            ["type", "count", "ratio"],
            [
                [item["type"], item["count"], f"{item['ratio']:.4f}"]
                for item in audit["entity_type_distribution"]
            ],
        )
    )
    lines.extend(["", "## Forbidden Entity Audit", ""])
    labels = [
        ("pure_figure_nodes", "Pure figure nodes"),
        ("step_nodes", "STEP nodes"),
        ("image_path_nodes", "Image path nodes"),
        ("numeric_only_nodes", "Numeric-only nodes"),
        ("chapter_like_nodes", "Chapter-like nodes"),
        ("possible_person_nodes", "Possible person nodes"),
    ]
    for key, label in labels:
        item = forbidden[key]
        lines.extend([f"### {label}", "", f"- count: {item['count']}", ""])
        lines.extend(sample_table(item["samples"]))
        lines.append("")

    figure = audit["figure_node_audit"]
    lines.extend(
        [
            "## FIGURE Node Audit",
            "",
            f"- figure_type_count: {figure['figure_type_count']}",
            f"- pure_figure_type_count: {figure['pure_figure_type_count']}",
            f"- figure_like_count: {figure['figure_like_count']}",
            "",
        ]
    )
    lines.extend(sample_table(figure["samples"]))
    unknown = audit["unknown_type_audit"]
    lines.extend(
        [
            "",
            "## UNKNOWN Type Audit",
            "",
            f"- unknown_count: {unknown['unknown_count']}",
            "",
        ]
    )
    lines.extend(sample_table(unknown["samples"]))
    lines.extend(["", "## High Degree Nodes", ""])
    lines.extend(sample_table(audit["high_degree_nodes"]))

    evidence = audit["evidence_completeness"]
    lines.extend(["", "## Evidence Completeness", ""])
    lines.extend(
        markdown_table(
            ["metric", "count"],
            [
                ["missing_source_id_count", evidence["missing_source_id_count"]],
                ["missing_file_path_count", evidence["missing_file_path_count"]],
                ["empty_description_count", evidence["empty_description_count"]],
            ],
        )
    )
    for key, label in [
        ("missing_source_id_samples", "Missing source_id samples"),
        ("missing_file_path_samples", "Missing file_path samples"),
        ("empty_description_samples", "Empty description samples"),
    ]:
        lines.extend(["", f"### {label}", ""])
        lines.extend(sample_table(evidence[key]))

    images = audit["image_metadata_audit"]
    lines.extend(["", "## Image Metadata Audit", ""])
    lines.extend(
        markdown_table(
            ["metric", "count"],
            [[key, value] for key, value in images.items() if key != "sample_images"],
        )
    )
    image_headers = [
        "chunk_id",
        "line_no",
        "relative_path",
        "local_path",
        "url",
        "reference_type",
        "exists",
        "caption",
        "raw_caption",
        "figure_no",
        "page_idx",
        "bbox",
    ]
    lines.extend(["", "### Image Samples", ""])
    lines.extend(
        markdown_table(
            image_headers,
            [[item.get(key, "") for key in image_headers] for item in images["sample_images"]],
        )
    )

    noise = audit["chunk_content_noise_audit"]
    lines.extend(["", "## Chunk Content Noise Audit", ""])
    lines.extend(
        markdown_table(
            ["metric", "count"],
            [[key, value] for key, value in noise.items() if key != "samples"],
        )
    )
    noise_headers = [
        "chunk_id",
        "catalog_title",
        "file_path",
        "matched_types",
        "matched_snippet",
    ]
    lines.extend(["", "### Noise Samples", ""])
    lines.extend(
        markdown_table(
            noise_headers,
            [[item.get(key, "") for key in noise_headers] for item in noise["samples"]],
        )
    )

    source_missing = audit["source_missing_audit"]
    lines.extend(
        [
            "",
            "## Source Missing Audit",
            "",
            f"- anchor_plan: `{source_missing['anchor_plan']}`",
            f"- anchor_plan_found: {source_missing['anchor_plan_found']}",
            f"- source_missing_count: {source_missing['source_missing_count']}",
            "",
        ]
    )
    source_headers = [
        "catalog_index",
        "catalog_level",
        "catalog_title",
        "number_key",
        "reason",
        "manual_override",
    ]
    lines.extend(
        markdown_table(
            source_headers,
            [
                [item.get(key, "") for key in source_headers]
                for item in source_missing["source_missing_items"]
            ],
        )
    )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in audit["recommendations"])
    return "\n".join(lines).rstrip() + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    working_dir = args.working_dir.expanduser().resolve()
    if (
        args.domain == "industrial_training"
        and "lightrag_manual_concepts" in str(working_dir)
    ):
        raise ValueError(
            f"工业教材不能使用 manual concepts working_dir: {working_dir}"
        )
    graph_path = working_dir / GRAPH_FILENAME
    if not graph_path.is_file():
        raise FileNotFoundError(f"GraphML 不存在: {graph_path}")

    book_stem = args.book_stem.strip()
    report_name = f"{book_stem}.graph_audit" if book_stem else "graph_audit"
    output_dir = PROJECT_DIR / "outputs/08_audit"
    output_md = (args.output_md or output_dir / f"{report_name}.md").expanduser().resolve()
    output_json = (
        args.output_json or output_dir / f"{report_name}.json"
    ).expanduser().resolve()
    anchor_plan = args.anchor_plan
    if anchor_plan is None and book_stem:
        anchor_plan = (
            PROJECT_DIR
            / "outputs/03_structure_plan"
            / f"{book_stem}.catalog_anchor_plan.jsonl"
        )
    if anchor_plan is not None:
        anchor_plan = anchor_plan.expanduser().resolve()

    missing_files: list[str] = []
    stores = {
        filename: load_optional_json(working_dir / filename, missing_files)
        for filename in OPTIONAL_STORE_FILENAMES
    }
    graph = nx.read_graphml(graph_path)
    chunks = normalize_chunk_store(stores["kv_store_text_chunks.json"])
    entity_vdb = normalize_json_records(stores["vdb_entities.json"])
    relationship_vdb = normalize_json_records(stores["vdb_relationships.json"])
    normalize_json_records(stores["kv_store_full_docs.json"])
    nodes, edges, selected_chunks, filter_meta = select_book_data(
        graph, chunks, book_stem
    )
    if filter_meta["filter_warning"]:
        logger.warning("book filter | %s", filter_meta["filter_warning"])
    schema_path = args.schema_json.expanduser().resolve()
    schema_audit = audit_schema(
        graph,
        nodes,
        edges,
        schema_path,
        args.schema_version,
    )

    type_counts = Counter(normalized_entity_type(attrs) for _, attrs in nodes)
    node_count = len(nodes)
    entity_type_distribution = [
        {
            "type": entity_type,
            "count": count,
            "ratio": count / node_count if node_count else 0.0,
        }
        for entity_type, count in sorted(
            type_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    forbidden = {
        "pure_figure_nodes": audit_matching_nodes(
            graph, nodes, lambda name, _: PURE_FIGURE_RE.fullmatch(name) is not None
        ),
        "step_nodes": audit_matching_nodes(
            graph, nodes, lambda name, _: STEP_RE.fullmatch(name) is not None
        ),
        "image_path_nodes": audit_matching_nodes(
            graph, nodes, lambda name, _: IMAGE_PATH_RE.search(name) is not None
        ),
        "numeric_only_nodes": audit_matching_nodes(
            graph,
            nodes,
            lambda name, _: any(pattern.fullmatch(name) for pattern in NUMERIC_ONLY_RES),
        ),
        "chapter_like_nodes": audit_matching_nodes(
            graph,
            nodes,
            lambda name, _: any(pattern.fullmatch(name) for pattern in CHAPTER_LIKE_RES),
        ),
        "possible_person_nodes": audit_matching_nodes(
            graph,
            nodes,
            lambda name, _: ENGLISH_PERSON_RE.fullmatch(name) is not None
            or PERSON_HINT_RE.search(name) is not None,
        ),
    }
    figure_nodes = [
        (node_id, attrs)
        for node_id, attrs in nodes
        if normalized_entity_type(attrs) == "figure"
    ]
    pure_figure_type_nodes = [
        (node_id, attrs)
        for node_id, attrs in figure_nodes
        if PURE_FIGURE_RE.fullmatch(entity_name(node_id, attrs))
    ]
    figure_like_nodes = [
        (node_id, attrs)
        for node_id, attrs in nodes
        if FIGURE_LIKE_RE.search(entity_name(node_id, attrs))
    ]
    figure_sample_nodes = list(
        dict.fromkeys(node_id for node_id, _ in figure_nodes + figure_like_nodes)
    )[:50]
    node_attrs = dict(nodes)
    figure_audit = {
        "figure_type_count": len(figure_nodes),
        "pure_figure_type_count": len(pure_figure_type_nodes),
        "figure_like_count": len(figure_like_nodes),
        "samples": [
            node_sample(graph, node_id, node_attrs[node_id])
            for node_id in figure_sample_nodes
        ],
    }
    unknown_nodes = [
        (node_id, attrs)
        for node_id, attrs in nodes
        if normalized_entity_type(attrs) == "UNKNOWN"
    ]
    unknown_audit = {
        "unknown_count": len(unknown_nodes),
        "samples": [
            node_sample(graph, node_id, attrs) for node_id, attrs in unknown_nodes[:50]
        ],
    }
    high_degree = sorted(nodes, key=lambda item: graph.degree(item[0]), reverse=True)[:50]
    high_degree_samples = [
        node_sample(graph, node_id, attrs) for node_id, attrs in high_degree
    ]

    missing_source = [
        (node_id, attrs)
        for node_id, attrs in nodes
        if not stringify(attrs.get("source_id")).strip()
    ]
    missing_path = [
        (node_id, attrs)
        for node_id, attrs in nodes
        if not stringify(attrs.get("file_path")).strip()
    ]
    empty_description = [
        (node_id, attrs)
        for node_id, attrs in nodes
        if not stringify(attrs.get("description")).strip()
    ]
    evidence = {
        "missing_source_id_count": len(missing_source),
        "missing_file_path_count": len(missing_path),
        "empty_description_count": len(empty_description),
        "missing_source_id_samples": [
            node_sample(graph, node_id, attrs) for node_id, attrs in missing_source[:30]
        ],
        "missing_file_path_samples": [
            node_sample(graph, node_id, attrs) for node_id, attrs in missing_path[:30]
        ],
        "empty_description_samples": [
            node_sample(graph, node_id, attrs)
            for node_id, attrs in empty_description[:30]
        ],
    }
    image_metadata = audit_image_metadata(selected_chunks)
    chunk_noise = audit_chunk_noise(selected_chunks)
    source_missing = audit_source_missing(anchor_plan)
    generated_at = datetime.now(timezone.utc).isoformat()
    audit: dict[str, Any] = {
        "meta": {
            "working_dir": str(working_dir),
            "book_stem": book_stem,
            "domain": args.domain,
            "subject": args.subject,
            "generated_at": generated_at,
            "graph_path": str(graph_path),
            "output_md": str(output_md),
            "output_json": str(output_json),
            "missing_files": missing_files,
            **filter_meta,
        },
        "overall_stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "chunk_count": len(selected_chunks),
            "entity_vdb_count": len(entity_vdb),
            "relationship_vdb_count": len(relationship_vdb),
        },
        "entity_type_distribution": entity_type_distribution,
        "schema_audit": schema_audit,
        "forbidden_entity_audit": forbidden,
        "figure_node_audit": figure_audit,
        "unknown_type_audit": unknown_audit,
        "high_degree_nodes": high_degree_samples,
        "evidence_completeness": evidence,
        "image_metadata_audit": image_metadata,
        "chunk_content_noise_audit": chunk_noise,
        "source_missing_audit": source_missing,
    }
    score, level = calculate_quality_score(audit)
    audit["quality_score"] = score
    audit["quality_level"] = level
    audit["recommendations"] = build_recommendations(audit)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(audit), encoding="utf-8")
    output_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "graph audit 完成 | output_md=%s output_json=%s quality_score=%s",
        output_md,
        output_json,
        score,
    )
    return audit


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preload_env()
    args = parse_args()
    try:
        run(args)
        return 0
    except Exception as exc:
        logger.exception("图谱质量审计失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
