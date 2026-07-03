#!/usr/bin/env python3
"""Use an LLM to identify front matter, TOC, and the body start."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from common import PROJECT_DIR, load_json, load_prompt, write_json
from llm_client import call_llm_json

logger = logging.getLogger(__name__)

RANGE_JSON_SCHEMA = {
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
        "notes": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "book_title",
        "front_matter_range",
        "toc_range",
        "body_start_line",
        "notes",
        "confidence",
    ],
    "additionalProperties": False,
}

CATALOG_JSON_SCHEMA = {
    "type": "object",
    "properties": {
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
        "chapter_catalog",
        "heading_patterns",
        "notes",
        "confidence",
    ],
    "additionalProperties": False,
}

CATALOG_ITEM_FIELDS = {"title", "level", "page_hint", "normalized_title"}


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


def parse_confidence(result: dict, label: str) -> float:
    try:
        confidence = float(result.get("confidence", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} confidence 必须是 0 到 1 之间的数字") from exc
    if not 0 <= confidence <= 1:
        raise ValueError(f"{label} confidence 必须是 0 到 1 之间的数字")
    return confidence


def normalize_body_start_line(
    lines: list[dict], line_no: int, visible_end: int, *, max_distance: int = 5
) -> int:
    """Move an empty predicted boundary to the nearest visible non-empty line."""
    if str(lines[line_no - 1]["text"]).strip():
        return line_no
    for distance in range(1, max_distance + 1):
        # Prefer the previous line on ties so a heading immediately before a
        # blank line is not discarded from the body.
        for candidate in (line_no - distance, line_no + distance):
            if 1 <= candidate <= visible_end and str(
                lines[candidate - 1]["text"]
            ).strip():
                return candidate
    raise ValueError(
        f"body_start_line 指向空行，且前后 {max_distance} 行内无非空内容: {line_no}"
    )


def is_numbering_only_title(title: str) -> bool:
    """Return True when a catalog title contains a number but no title text."""
    compact = "".join(title.split()).replace("．", ".").rstrip(".")
    if compact and all(part.isdigit() for part in compact.split(".")):
        return True
    if compact.startswith("第") and compact.endswith(("章", "节", "篇")):
        number = compact[1:-1]
        chinese_digits = set("零〇一二三四五六七八九十百千万两")
        return number.isdigit() or (
            bool(number) and all(char in chinese_digits for char in number)
        )
    return False


def validate_and_normalize_catalog_items(items: list) -> list[dict]:
    """Validate catalog item fields and keep a printed page out of title."""
    normalized_items: list[dict] = []
    for index, item in enumerate(items, start=1):
        label = f"chapter_catalog[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} 必须是 JSON 对象")
        missing = CATALOG_ITEM_FIELDS - item.keys()
        unexpected = item.keys() - CATALOG_ITEM_FIELDS
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"缺少字段 {sorted(missing)}")
            if unexpected:
                details.append(f"包含未知字段 {sorted(unexpected)}")
            raise ValueError(f"{label} 字段错误: {'；'.join(details)}")

        title = item["title"]
        level = item["level"]
        page_hint = item["page_hint"]
        normalized_title = item["normalized_title"]
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"{label}.title 必须是非空字符串")
        if isinstance(level, bool) or not isinstance(level, int) or level <= 0:
            raise ValueError(f"{label}.level 必须是正整数")
        if not isinstance(page_hint, str):
            raise ValueError(f"{label}.page_hint 必须是字符串")
        if not isinstance(normalized_title, str):
            raise ValueError(f"{label}.normalized_title 必须是字符串")

        title = title.strip()
        page_hint = page_hint.strip()
        # The LLM may copy the printed page into both fields despite the
        # schema. Remove it only when it is an exact, whitespace-separated
        # final token; this is mechanical normalization, not title inference.
        if page_hint:
            title_parts = title.rsplit(maxsplit=1)
            if len(title_parts) == 2 and title_parts[1] == page_hint:
                title = title_parts[0]
        if is_numbering_only_title(title):
            logger.warning("%s 的 title 只有编号、缺少标题文本: %s", label, title)

        normalized_items.append(
            {
                "title": title,
                "level": level,
                "page_hint": page_hint,
                "normalized_title": normalized_title.strip(),
            }
        )
    return normalized_items


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
    range_prompt_template = load_prompt(
        PROJECT_DIR / "prompts/front_matter_toc_prompt.md"
    )
    range_prompt = (
        f"{range_prompt_template}\n\n"
        f"【文件名】{book_stem}\n"
        f"【总行数】{prepared['line_count']}\n"
        f"【输入前部行号文本】\n{format_numbered_lines(prepared['lines'][: args.front_lines])}"
    )
    ranges_raw_path = output.with_suffix(".ranges.raw.txt")
    ranges = await call_llm_json(
        range_prompt,
        raw_output_path=ranges_raw_path,
        guided_json_schema=RANGE_JSON_SCHEMA,
        max_tokens=1024,
    )
    if not isinstance(ranges, dict):
        raise ValueError("边界识别 LLM 必须返回一个 JSON 对象")
    required = {"book_title", "body_start_line", "toc_range"}
    missing = required - ranges.keys()
    if missing:
        raise ValueError(f"边界识别结果缺少字段: {sorted(missing)}")
    visible_end = min(args.front_lines, int(prepared["line_count"]))
    for field in ("front_matter_range", "toc_range"):
        value = ranges.get(field)
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
    body_start = ranges["body_start_line"]
    if not isinstance(body_start, int) or not 1 <= body_start <= visible_end:
        raise ValueError(f"body_start_line 越界: {body_start}")
    normalization_notes: list[str] = []
    normalized_body_start = normalize_body_start_line(
        prepared["lines"], body_start, visible_end
    )
    if normalized_body_start != body_start:
        note = (
            f"body_start_line 从空行 {body_start} 校正到最近的非空行 "
            f"{normalized_body_start}"
        )
        logger.warning(note)
        normalization_notes.append(note)
        body_start = normalized_body_start
        ranges["body_start_line"] = body_start
    toc_range = ranges.get("toc_range")
    if toc_range is None:
        raise ValueError("未识别到 toc_range，无法执行第二次目录骨架提取")
    if toc_range[1] >= body_start:
        adjusted_toc_end = body_start - 1
        if adjusted_toc_end < toc_range[0]:
            raise ValueError(
                f"toc_range 与正文严重冲突: toc={toc_range} body_start={body_start}"
            )
        note = (
            f"toc_range 末行从 {toc_range[1]} 校正到正文起始行之前的 "
            f"{adjusted_toc_end}"
        )
        logger.warning(note)
        normalization_notes.append(note)
        toc_range = [toc_range[0], adjusted_toc_end]
        ranges["toc_range"] = toc_range
    ranges_confidence = parse_confidence(ranges, "边界识别")
    logger.info(
        "边界识别完成 | front_matter=%s toc=%s body_start=%d raw=%s",
        ranges.get("front_matter_range"),
        toc_range,
        body_start,
        ranges_raw_path,
    )

    toc_start, toc_end = toc_range
    toc_lines = prepared["lines"][toc_start - 1 : toc_end]
    catalog_prompt_template = load_prompt(
        PROJECT_DIR / "prompts/chapter_catalog_prompt.md"
    )
    catalog_prompt = (
        f"{catalog_prompt_template}\n\n"
        f"【文件名】{book_stem}\n"
        f"【第一步边界结果】\n{json.dumps(ranges, ensure_ascii=False, indent=2)}\n\n"
        f"【仅目录区间行号文本】\n{format_numbered_lines(toc_lines)}"
    )
    catalog_raw_path = output.with_suffix(".catalog.raw.txt")
    catalog = await call_llm_json(
        catalog_prompt,
        raw_output_path=catalog_raw_path,
        guided_json_schema=CATALOG_JSON_SCHEMA,
        max_tokens=8192,
    )
    if not isinstance(catalog, dict):
        raise ValueError("目录骨架 LLM 必须返回一个 JSON 对象")
    if not isinstance(catalog.get("chapter_catalog"), list):
        raise ValueError("chapter_catalog 必须是数组")
    if not catalog["chapter_catalog"]:
        raise ValueError("chapter_catalog 为空")
    catalog_items = validate_and_normalize_catalog_items(
        catalog["chapter_catalog"]
    )
    if not isinstance(catalog.get("heading_patterns"), list):
        raise ValueError("heading_patterns 必须是数组")
    if not all(isinstance(pattern, str) for pattern in catalog["heading_patterns"]):
        raise ValueError("heading_patterns 的每一项都必须是字符串")
    catalog_confidence = parse_confidence(catalog, "目录骨架")

    range_note = "" if normalization_notes else str(ranges.get("notes") or "").strip()
    notes = "\n".join(
        note
        for note in (
            range_note,
            str(catalog.get("notes") or "").strip(),
            *normalization_notes,
        )
        if note
    )
    result = {
        "book_title": ranges["book_title"],
        "front_matter_range": ranges.get("front_matter_range"),
        "toc_range": toc_range,
        "body_start_line": body_start,
        "chapter_catalog": catalog_items,
        "heading_patterns": catalog["heading_patterns"],
        "notes": notes,
        "confidence": min(ranges_confidence, catalog_confidence),
    }
    write_json(output, result)
    logger.info(
        "outline 完成 | body_start=%d catalog_items=%d output=%s catalog_raw=%s",
        body_start,
        len(result["chapter_catalog"]),
        output,
        catalog_raw_path,
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
