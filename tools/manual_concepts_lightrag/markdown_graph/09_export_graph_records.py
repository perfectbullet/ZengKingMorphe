#!/usr/bin/env python3
"""Export graph records through LightRAG storage abstractions only."""
from __future__ import annotations

import argparse, asyncio, inspect, logging
from pathlib import Path
from typing import Any

from book_meta import get_business_config, load_book_meta, resolve_book_paths
from common import PROJECT_DIR, build_embedding_func, build_llm_model_func, read_jsonl, write_json, write_jsonl

logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="导出 LightRAG 实体和知识点 JSONL")
    p.add_argument("--meta", required=True, type=Path); p.add_argument("--working-dir", required=True, type=Path)
    p.add_argument("--chunks", type=Path); p.add_argument("--output-dir", type=Path)
    return p.parse_args()

async def maybe_get_by_ids(storage: Any, ids: list[str]) -> dict[str, Any]:
    if not ids or not hasattr(storage, "get_by_ids"): return {}
    result = await storage.get_by_ids(ids)
    return result if isinstance(result, dict) else {}

def source_ids(data: dict[str, Any]) -> list[str]:
    raw = data.get("source_id") or data.get("source_ids") or ""
    return [item.strip() for item in str(raw).split(",") if item.strip()]

async def run(args):
    meta = load_book_meta(args.meta); paths = resolve_book_paths(meta); business = get_business_config(meta)
    chunks_path = args.chunks or PROJECT_DIR / "outputs/05_chunks" / f"{meta['artifact_stem']}.lightrag_chunks.jsonl"
    chunk_rows = {str(row.get("chunk_id")): row for row in read_jsonl(chunks_path)}
    from lightrag import LightRAG
    rag = LightRAG(working_dir=str(args.working_dir.expanduser().resolve()), addon_params={"language":"Chinese"}, llm_model_func=build_llm_model_func(), embedding_func=build_embedding_func())
    await rag.initialize_storages()
    try:
        graph = rag.chunk_entity_relation_graph
        if not hasattr(graph, "get_all_nodes") or not hasattr(graph, "get_all_edges"):
            raise RuntimeError("当前 LightRAG 图存储未实现 get_all_nodes/get_all_edges，无法进行后端无关导出")
        nodes = await graph.get_all_nodes(); edges = await graph.get_all_edges()
        def evidence(ids: list[str]) -> list[dict]:
            result=[]
            for cid in ids:
                row=chunk_rows.get(cid)
                if not row: continue
                result.append({k: row.get(k) for k in ("chunk_id","file_path","source_md_path","catalog_title","heading_path","start_line","end_line","images")})
                result[-1]["text_preview"] = str(row.get("kg_content") or row.get("content") or "")[:500]
            return result
        entities=[]
        for node in nodes or []:
            name=str(node.get("id") or node.get("entity_name") or node.get("name") or "")
            data=node.get("data") if isinstance(node.get("data"),dict) else node
            ids=source_ids(data)
            entities.append({"entity_name":name,"entity_type":data.get("entity_type", ""),"description":data.get("description", ""),"source_chunk_ids":ids,"file_paths":data.get("file_path", []),"domain":business["domain"],"subject":business["subject"],"book_id":meta["book_id"],"book_title":meta["book_title"],"evidence":evidence(ids)})
        points=[]
        for edge in edges or []:
            data=edge.get("data") if isinstance(edge.get("data"),dict) else edge
            src=str(edge.get("source") or edge.get("src_id") or data.get("src_id") or ""); tgt=str(edge.get("target") or edge.get("tgt_id") or data.get("tgt_id") or "")
            ids=source_ids(data); keywords=[x.strip() for x in str(data.get("keywords") or "").replace("，",",").split(",") if x.strip()]
            points.append({"source_entity":src,"target_entity":tgt,"keywords":keywords,"description":data.get("description", ""),"weight":float(data.get("weight") or 1.0),"source_chunk_ids":ids,"domain":business["domain"],"subject":business["subject"],"book_id":meta["book_id"],"evidence":evidence(ids)})
    finally:
        await rag.finalize_storages()
    out=(args.output_dir or PROJECT_DIR/"outputs/09_export").expanduser().resolve(); stem=meta["artifact_stem"]
    write_jsonl(out/f"{stem}.entities.jsonl",entities); write_jsonl(out/f"{stem}.knowledge_points.jsonl",points)
    write_json(out/f"{stem}.export_report.json",{"entity_count":len(entities),"knowledge_point_count":len(points),"storage_interface":"get_all_nodes/get_all_edges","chunks_source":str(chunks_path)})
    logger.info("Step 9 导出完成 | entities=%d points=%d",len(entities),len(points))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s"); asyncio.run(run(parse_args()))
