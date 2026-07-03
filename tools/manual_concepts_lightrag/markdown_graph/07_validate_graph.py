#!/usr/bin/env python3
"""Run lightweight graph/entity/query checks against an imported working dir."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.base import QueryParam

from common import (
    DEFAULT_WORKING_DIR,
    PROJECT_DIR,
    build_embedding_func,
    build_llm_model_func,
    load_prompt,
)

logger = logging.getLogger(__name__)


def build_rag(working_dir: Path, domain: str, subject: str) -> LightRAG:
    guidance = load_prompt(PROJECT_DIR / "prompts/graph_extraction_guidance.md")
    guidance = (
        f"{guidance.strip()}\n\n当前领域：{domain}\n当前科目：{subject or '未指定'}"
    )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="验证 LightRAG 教材知识图谱的基本可用性"
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path(os.getenv("MARKDOWN_GRAPH_WORKING_DIR", str(DEFAULT_WORKING_DIR))),
    )
    parser.add_argument(
        "--domain", default=os.getenv("MARKDOWN_GRAPH_DOMAIN", "industrial_training")
    )
    parser.add_argument("--subject", default=os.getenv("MARKDOWN_GRAPH_SUBJECT", ""))
    parser.add_argument("--book-stem", default="graph")
    parser.add_argument(
        "--entity", action="append", default=[], help="可重复，检查实体并输出局部图"
    )
    parser.add_argument(
        "--query", action="append", default=[], help="可重复，执行 LightRAG 查询"
    )
    parser.add_argument(
        "--query-mode",
        choices=["local", "global", "hybrid", "naive", "mix"],
        default="hybrid",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-file", type=Path, default=PROJECT_DIR.parent / ".env")
    return parser.parse_args()


def json_text(value: object) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


async def run(args: argparse.Namespace) -> Path:
    working_dir = args.working_dir.expanduser().resolve()
    output = (
        (
            args.output
            or PROJECT_DIR
            / "outputs/07_validate"
            / f"{args.book_stem}.validate_report.md"
        )
        .expanduser()
        .resolve()
    )
    rag = build_rag(working_dir, args.domain, args.subject)
    await rag.initialize_storages()
    report = [
        "# LightRAG 图谱验证报告",
        "",
        f"- 时间: {datetime.now(timezone.utc).isoformat()}",
        f"- working_dir: `{working_dir}`",
    ]
    try:
        labels = await rag.get_graph_labels()
        report.extend(
            [f"- graph labels 数量: {len(labels)}", "", "## 关键实体检查", ""]
        )
        if args.entity:
            label_map = {str(label).casefold(): str(label) for label in labels}
            for entity in args.entity:
                actual = label_map.get(entity.casefold())
                report.append(f"### {entity}")
                report.append("")
                if actual is None:
                    report.append("- 未找到")
                else:
                    info = await rag.get_entity_info(actual)
                    local_graph = await rag.get_knowledge_graph(actual, max_depth=2)
                    if is_dataclass(local_graph):
                        local_graph = asdict(local_graph)
                    report.extend(
                        [
                            "- 已找到",
                            "",
                            "```json",
                            json_text(
                                {"entity_info": info, "local_graph": local_graph}
                            ),
                            "```",
                        ]
                    )
                report.append("")
        else:
            preview = "、".join(str(label) for label in labels[:30]) or "（空图）"
            report.extend([f"- 未指定 --entity；labels 预览: {preview}", ""])

        report.extend(["## 查询验证", ""])
        if args.query:
            param = QueryParam(mode=args.query_mode, stream=False, enable_rerank=False)
            for query in args.query:
                response = await rag.aquery(query, param=param)
                report.extend([f"### {query}", "", str(response), ""])
        else:
            report.extend(["- 未指定 --query，仅完成图和实体检查。", ""])
    finally:
        await rag.finalize_storages()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    logger.info("验证完成 | output=%s", output)
    return output


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    env_path = args.env_file.expanduser().resolve()
    if env_path.is_file():
        load_dotenv(env_path, override=False)
        logger.info("loaded env: %s", env_path)
    try:
        asyncio.run(run(args))
        return 0
    except Exception as exc:
        logger.exception("图谱验证失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
