#!/usr/bin/env python3
"""通过 LightRAG 存储抽象接口导出图谱记录或全部实体节点名称。"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from book_meta import get_business_config, load_book_meta, resolve_book_paths
from common import (
    PROJECT_DIR,
    build_embedding_func,
    build_llm_model_func,
    read_jsonl,
    write_json,
    write_jsonl,
    write_text_atomic,
)

logger = logging.getLogger(__name__)

DEFAULT_ALL_NODE_NAMES_OUTPUT = PROJECT_DIR / "outputs/09_export/all_entity_node_names.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 LightRAG 实体、知识点或全部实体节点名称")
    parser.add_argument("--meta", type=Path, help="教材级导出所需的教材 meta JSON")
    parser.add_argument("--working-dir", required=True, type=Path)
    parser.add_argument("--chunks", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--env-file", type=Path, default=PROJECT_DIR.parent / ".env")
    parser.add_argument(
        "--all-node-names",
        action="store_true",
        help="仅导出共享 LightRAG working directory 中全部实体节点名称",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="--all-node-names 模式下的 TXT 输出文件",
    )
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def preload_env() -> Path:
    """在构造 LightRAG 前加载项目环境，保持与 Step 6--8 一致。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env-file", type=Path, default=PROJECT_DIR.parent / ".env")
    args, _ = parser.parse_known_args()
    env_path = args.env_file.expanduser().resolve()
    if env_path.is_file():
        load_dotenv(env_path, override=False)
        logger.info("loaded env: %s", env_path)
    return env_path


def validate_args(args: argparse.Namespace) -> None:
    """校验两种互斥业务语义的最小参数集合。"""
    if not getattr(args, "all_node_names", False) and not getattr(args, "meta", None):
        raise ValueError("教材级 Step 9 导出必须提供 --meta；如只导出全部节点名请传 --all-node-names")


def normalize_graph_nodes(raw_nodes: Any) -> list[dict[str, Any]]:
    """归一化不同 LightRAG 图存储实现可能返回的节点容器。"""
    if raw_nodes is None:
        return []
    if isinstance(raw_nodes, list):
        return [dict(node) for node in raw_nodes if isinstance(node, dict)]
    if isinstance(raw_nodes, dict):
        nodes: list[dict[str, Any]] = []
        for node_id, attributes in raw_nodes.items():
            if isinstance(attributes, dict):
                node = dict(attributes)
                node.setdefault("id", node_id)
            else:
                node = {"id": node_id, "data": attributes}
            nodes.append(node)
        return nodes
    logger.warning("无法识别 get_all_nodes() 返回格式，按空节点列表处理: %s", type(raw_nodes).__name__)
    return []


def extract_node_name(node: dict[str, Any]) -> str:
    """从节点记录中提取稳定的人类可读实体名称。"""
    data = node.get("data")
    candidates: list[Any] = [
        node.get("id"),
        node.get("entity_name"),
        node.get("name"),
    ]
    if isinstance(data, dict):
        candidates.extend(
            [data.get("entity_id"), data.get("entity_name"), data.get("name")]
        )
    for value in candidates:
        if value is not None:
            name = str(value).strip()
            if name:
                return name
    return ""


def collect_sorted_node_names(raw_nodes: Any) -> tuple[list[str], int]:
    """返回去重、稳定排序后的实体名称及原始节点数。"""
    nodes = normalize_graph_nodes(raw_nodes)
    names = {name for node in nodes if (name := extract_node_name(node))}
    return sorted(names, key=lambda value: value.casefold()), len(nodes)


def write_node_names_txt(output_file: Path, names: list[str]) -> Path:
    output_file = output_file.expanduser().resolve()
    # 即使图中没有节点也写入空文本；非空文件必定以换行结尾。
    content = "".join(f"{name}\n" for name in names)
    write_text_atomic(output_file, content)
    return output_file


async def maybe_get_by_ids(storage: Any, ids: list[str]) -> dict[str, Any]:
    if not ids or not hasattr(storage, "get_by_ids"):
        return {}
    result = await storage.get_by_ids(ids)
    return result if isinstance(result, dict) else {}


def source_ids(data: dict[str, Any]) -> list[str]:
    raw = data.get("source_id") or data.get("source_ids") or ""
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _build_rag(working_dir: Path) -> Any:
    from lightrag import LightRAG

    return LightRAG(
        working_dir=str(working_dir.expanduser().resolve()),
        addon_params={"language": "Chinese"},
        llm_model_func=build_llm_model_func(),
        embedding_func=build_embedding_func(),
    )


async def _export_all_node_names(graph: Any, output_file: Path, working_dir: Path) -> None:
    if not hasattr(graph, "get_all_nodes"):
        raise RuntimeError("当前 LightRAG 图存储未实现 get_all_nodes，无法导出全部实体节点名称")
    raw_nodes = await graph.get_all_nodes()
    names, raw_node_count = collect_sorted_node_names(raw_nodes)
    destination = write_node_names_txt(output_file, names)
    logger.info(
        "全图实体节点名称导出完成 | working_dir=%s | raw_nodes=%d | unique_names=%d | output=%s",
        working_dir.expanduser().resolve(),
        raw_node_count,
        len(names),
        destination,
    )


async def _export_book_records(args: argparse.Namespace, graph: Any) -> None:
    # 仅教材级导出才读取 meta、chunks 和教材业务字段。
    meta = load_book_meta(args.meta)
    resolve_book_paths(meta)
    business = get_business_config(meta)
    chunks_path = args.chunks or PROJECT_DIR / "outputs/05_chunks" / f"{meta['artifact_stem']}.lightrag_chunks.jsonl"
    chunk_rows = {str(row.get("chunk_id")): row for row in read_jsonl(chunks_path)}

    if not hasattr(graph, "get_all_nodes") or not hasattr(graph, "get_all_edges"):
        raise RuntimeError("当前 LightRAG 图存储未实现 get_all_nodes/get_all_edges，无法进行后端无关导出")
    nodes = normalize_graph_nodes(await graph.get_all_nodes())
    edges = await graph.get_all_edges()

    def evidence(ids: list[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for chunk_id in ids:
            row = chunk_rows.get(chunk_id)
            if not row:
                continue
            item = {
                key: row.get(key)
                for key in (
                    "chunk_id", "file_path", "source_md_path", "catalog_title",
                    "heading_path", "start_line", "end_line", "images",
                )
            }
            item["text_preview"] = str(row.get("kg_content") or row.get("content") or "")[:500]
            result.append(item)
        return result

    entities: list[dict[str, Any]] = []
    for node in nodes:
        data = node.get("data") if isinstance(node.get("data"), dict) else node
        ids = source_ids(data)
        entities.append(
            {
                "entity_name": extract_node_name(node),
                "entity_type": data.get("entity_type", ""),
                "description": data.get("description", ""),
                "source_chunk_ids": ids,
                "file_paths": data.get("file_path", []),
                "domain": business["domain"],
                "subject": business["subject"],
                "book_id": meta["book_id"],
                "book_title": meta["book_title"],
                "evidence": evidence(ids),
            }
        )

    points: list[dict[str, Any]] = []
    for edge in edges or []:
        if not isinstance(edge, dict):
            continue
        data = edge.get("data") if isinstance(edge.get("data"), dict) else edge
        source = str(edge.get("source") or edge.get("src_id") or data.get("src_id") or "")
        target = str(edge.get("target") or edge.get("tgt_id") or data.get("tgt_id") or "")
        ids = source_ids(data)
        keywords = [
            item.strip()
            for item in str(data.get("keywords") or "").replace("，", ",").split(",")
            if item.strip()
        ]
        points.append(
            {
                "source_entity": source,
                "target_entity": target,
                "keywords": keywords,
                "description": data.get("description", ""),
                "weight": float(data.get("weight") or 1.0),
                "source_chunk_ids": ids,
                "domain": business["domain"],
                "subject": business["subject"],
                "book_id": meta["book_id"],
                "evidence": evidence(ids),
            }
        )

    output_dir = (args.output_dir or PROJECT_DIR / "outputs/09_export").expanduser().resolve()
    stem = meta["artifact_stem"]
    write_jsonl(output_dir / f"{stem}.entities.jsonl", entities)
    write_jsonl(output_dir / f"{stem}.knowledge_points.jsonl", points)
    write_json(
        output_dir / f"{stem}.export_report.json",
        {
            "entity_count": len(entities),
            "knowledge_point_count": len(points),
            "storage_interface": "get_all_nodes/get_all_edges",
            "chunks_source": str(chunks_path),
        },
    )
    logger.info("Step 9 导出完成 | entities=%d points=%d", len(entities), len(points))


async def run(args: argparse.Namespace) -> None:
    validate_args(args)
    if args.all_node_names and args.meta:
        logger.warning("--all-node-names 模式忽略 --meta，不加载教材级配置: %s", args.meta)

    rag = _build_rag(args.working_dir)
    try:
        await rag.initialize_storages()
        graph = rag.chunk_entity_relation_graph
        if args.all_node_names:
            await _export_all_node_names(
                graph,
                args.output_file or DEFAULT_ALL_NODE_NAMES_OUTPUT,
                args.working_dir,
            )
            return
        await _export_book_records(args, graph)
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preload_env()
    asyncio.run(run(parse_args()))
