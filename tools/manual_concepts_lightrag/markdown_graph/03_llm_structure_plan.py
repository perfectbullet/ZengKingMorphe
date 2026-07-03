#!/usr/bin/env python3
"""Generate one combined JSONL structure plan from overlapping LLM windows."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from common import PROJECT_DIR, ensure_dir, load_json, load_prompt, write_jsonl
from llm_client import call_llm_json

logger = logging.getLogger(__name__)

BLOCK_TYPES = [
    "preface",
    "catalog",
    "chapter",
    "section",
    "subsection",
    "procedure",
    "table",
    "figure_group",
    "exercise",
    "appendix",
    "unknown",
]

STRUCTURE_PLAN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "block_id": {"type": "string"},
                    "block_type": {"type": "string", "enum": BLOCK_TYPES},
                    "book_title": {"type": "string"},
                    "chapter_title": {"type": "string"},
                    "section_title": {"type": "string"},
                    "heading_path": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "image_lines": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "block_id",
                    "block_type",
                    "book_title",
                    "chapter_title",
                    "section_title",
                    "heading_path",
                    "start_line",
                    "end_line",
                    "image_lines",
                    "confidence",
                    "reason",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["blocks"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分窗口生成教材正文结构切分计划")
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--outline", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--window-lines",
        type=int,
        default=int(os.getenv("STRUCTURE_WINDOW_LINES", "320")),
    )
    parser.add_argument("--env-file", type=Path, default=PROJECT_DIR.parent / ".env")
    parser.add_argument(
        "--overlap-lines",
        type=int,
        default=int(os.getenv("STRUCTURE_OVERLAP_LINES", "60")),
    )
    return parser.parse_args()


def numbered_lines(lines: list[dict]) -> str:
    return "\n".join(f"{item['line_no']:>6} | {item['text']}" for item in lines)


async def run(args: argparse.Namespace) -> Path:
    prepared = load_json(args.prepared.expanduser().resolve())
    outline = load_json(args.outline.expanduser().resolve())
    if not isinstance(prepared, dict) or not isinstance(outline, dict):
        raise ValueError("prepared 和 outline 顶层必须是 JSON 对象")
    if args.window_lines <= 0 or args.overlap_lines < 0:
        raise ValueError("window-lines 必须大于 0，overlap-lines 不能为负数")
    if args.overlap_lines >= args.window_lines:
        raise ValueError("overlap-lines 必须小于 window-lines")

    book_stem = str(prepared["book_stem"])
    output = (
        (
            args.output
            or PROJECT_DIR
            / "outputs/03_structure_plan"
            / f"{book_stem}.structure_plan.raw.jsonl"
        )
        .expanduser()
        .resolve()
    )
    raw_dir = ensure_dir(output.parent / "raw_windows")
    template = load_prompt(PROJECT_DIR / "prompts/structure_plan_prompt.md")
    lines: list[dict] = prepared["lines"]
    body_start = int(outline["body_start_line"])
    start_index = body_start - 1
    step = args.window_lines - args.overlap_lines
    all_records: list[dict] = []
    window_number = 0

    while start_index < len(lines):
        window_number += 1
        window_id = f"window_{window_number:04d}"
        window = lines[start_index : start_index + args.window_lines]
        if not window:
            break
        window_start = int(window[0]["line_no"])
        window_end = int(window[-1]["line_no"])
        prompt = (
            f"{template}\n\n"
            f"【book_outline】\n{json.dumps(outline, ensure_ascii=False, indent=2)}\n\n"
            f"【当前窗口】{window_id}，行 {window_start}-{window_end}\n"
            f"{numbered_lines(window)}"
        )
        raw_path = raw_dir / f"{book_stem}.{window_id}.raw.txt"
        result = await call_llm_json(
            prompt,
            raw_output_path=raw_path,
            expect_jsonl=True,
            guided_json_schema=STRUCTURE_PLAN_JSON_SCHEMA,
        )
        if not isinstance(result, list):
            raise ValueError(f"{window_id} 必须返回 JSONL 记录")
        for record in result:
            record = dict(record)
            record["source_window_id"] = window_id
            record["source_window_start_line"] = window_start
            record["source_window_end_line"] = window_end
            all_records.append(record)
        logger.info(
            "窗口完成 | id=%s range=%d-%d blocks=%d",
            window_id,
            window_start,
            window_end,
            len(result),
        )
        if window_end >= len(lines):
            break
        start_index += step

    write_jsonl(output, all_records)
    logger.info(
        "structure plan 完成 | windows=%d blocks=%d output=%s",
        window_number,
        len(all_records),
        output,
    )
    return output


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        args = parse_args()
        env_path = args.env_file.expanduser().resolve()
        if env_path.is_file():
            load_dotenv(env_path, override=False)
            logger.info("loaded env: %s", env_path)
        asyncio.run(run(args))
        return 0
    except Exception as exc:
        logger.exception("structure plan 失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
