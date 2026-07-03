#!/usr/bin/env python3
"""Use an LLM to identify front matter, TOC, and the body start."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from common import PROJECT_DIR, load_json, load_prompt, write_json
from llm_client import call_llm_json

logger = logging.getLogger(__name__)

OUTLINE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "book_title": {"type": "string"},
        "front_matter_range": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                {"type": "null"},
            ]
        },
        "toc_range": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                {"type": "null"},
            ]
        },
        "body_start_line": {"type": "integer"},
        "chapter_catalog": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "level": {"type": "integer"},
                    "page_hint": {"type": "string"},
                    "normalized_title": {"type": "string"},
                },
                "required": [
                    "title",
                    "level",
                    "page_hint",
                    "normalized_title",
                ],
                "additionalProperties": False,
            },
        },
        "heading_patterns": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "book_title",
        "front_matter_range",
        "toc_range",
        "body_start_line",
        "chapter_catalog",
        "heading_patterns",
        "notes",
        "confidence",
    ],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="识别教材前言、目录和正文起始位置")
    parser.add_argument(
        "--prepared", required=True, type=Path, help="01 阶段 prepared.json"
    )
    parser.add_argument("--output", type=Path, help="book_outline.json 输出路径")
    parser.add_argument(
        "--front-lines",
        type=int,
        default=int(os.getenv("STRUCTURE_FRONT_LINES", "1000")),
        help="发送给 LLM 的文档前部行数",
    )
    parser.add_argument("--env-file", type=Path, default=PROJECT_DIR.parent / ".env")
    return parser.parse_args()


def format_numbered_lines(lines: list[dict]) -> str:
    return "\n".join(f"{item['line_no']:>6} | {item['text']}" for item in lines)


async def run(args: argparse.Namespace) -> Path:
    prepared_path = args.prepared.expanduser().resolve()
    prepared = load_json(prepared_path)
    if not isinstance(prepared, dict) or not isinstance(prepared.get("lines"), list):
        raise ValueError(f"prepared.json 格式错误: {prepared_path}")
    if args.front_lines <= 0:
        raise ValueError("--front-lines 必须大于 0")
    book_stem = str(prepared["book_stem"])
    output = (
        (
            args.output
            or PROJECT_DIR / "outputs/02_outline" / f"{book_stem}.book_outline.json"
        )
        .expanduser()
        .resolve()
    )
    prompt_template = load_prompt(PROJECT_DIR / "prompts/front_matter_toc_prompt.md")
    prompt = (
        f"{prompt_template}\n\n"
        f"【文件名】{book_stem}\n"
        f"【总行数】{prepared['line_count']}\n"
        f"【输入前部行号文本】\n{format_numbered_lines(prepared['lines'][: args.front_lines])}"
    )
    raw_path = output.with_suffix(".raw.txt")
    result = await call_llm_json(
        prompt,
        raw_output_path=raw_path,
        guided_json_schema=OUTLINE_JSON_SCHEMA,
    )
    if not isinstance(result, dict):
        raise ValueError("目录识别 LLM 必须返回一个 JSON 对象")
    required = {"book_title", "body_start_line", "chapter_catalog"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"目录识别结果缺少字段: {sorted(missing)}")
    visible_end = min(args.front_lines, int(prepared["line_count"]))
    for field in ("front_matter_range", "toc_range"):
        value = result.get(field)
        if value is None:
            continue
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not all(isinstance(line_no, int) for line_no in value)
            or not 1 <= value[0] <= value[1] <= visible_end
        ):
            raise ValueError(
                f"{field} 必须是输入窗口内的 [start_line, end_line]: {value}"
            )
    body_start = result["body_start_line"]
    if not isinstance(body_start, int) or not 1 <= body_start <= visible_end:
        raise ValueError(f"body_start_line 越界: {body_start}")
    if not isinstance(result["chapter_catalog"], list):
        raise ValueError("chapter_catalog 必须是数组")
    try:
        confidence = float(result.get("confidence", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence 必须是 0 到 1 之间的数字") from exc
    if not 0 <= confidence <= 1:
        raise ValueError("confidence 必须是 0 到 1 之间的数字")
    result["confidence"] = confidence
    write_json(output, result)
    logger.info(
        "outline 完成 | body_start=%d chapters=%d output=%s",
        body_start,
        len(result.get("chapter_catalog", [])),
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
        logger.exception("outline 失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
