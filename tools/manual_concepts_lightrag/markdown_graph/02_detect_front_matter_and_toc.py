#!/usr/bin/env python3
"""Use an LLM to identify front matter, TOC, and the body start."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from common import PROJECT_DIR, load_json, load_prompt, write_json
from book_meta import (
    get_business_config,
    get_structure_config,
    load_book_meta,
    resolve_book_paths,
    validate_book_meta,
)
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
CATALOG_ITEM_OPTIONAL_FIELDS = {
    "aliases",
    "chapter_no",
    "source_titles",
    "normalization",
}
CHINESE_CHAPTER_RE = re.compile(r"^第\s*[一二三四五六七八九十百零〇两0-9]+\s*章$")
ENGLISH_CHAPTER_RE = re.compile(r"^CHAPTER\s*0*\d+$", re.IGNORECASE)
PAGE_ONLY_RE = re.compile(r"^\s*(?:p\.?\s*)?\d{1,4}\s*$", re.IGNORECASE)
FIGURE_LINE_RE = re.compile(r"^\s*图\s*\d+\s*[-－]\s*\d+")
URL_OR_IMAGE_RE = re.compile(r"https?://|(?:^|/)images/|\.(?:png|jpe?g|webp)\b", re.I)
SUBSECTION_PREFIX_RE = re.compile(r"^\s*(?:[◆◇●○■□▪▫★☆]|[（(]\s*\d+\s*[）)])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="识别教材前言、目录和正文起始位置")
    parser.add_argument("--prepared", type=Path, help="01 阶段 prepared.json")
    parser.add_argument("--meta", type=Path, help="单本教材 meta；meta 模式只在人工确认目录范围内提取目录")
    parser.add_argument("--output", type=Path, help="book_outline.json 输出路径")
    parser.add_argument(
        "--front-lines",
        type=int,
        default=int(os.getenv("STRUCTURE_FRONT_LINES", "1000")),
        help="发送给 LLM 的文档前部行数",
    )
    parser.add_argument(
        "--toc-start-line",
        type=int,
        default=(
            int(os.environ["STRUCTURE_TOC_START_LINE"])
            if os.getenv("STRUCTURE_TOC_START_LINE")
            else None
        ),
        help="人工覆盖目录起始行，必须与 toc-end/body-start 一起使用",
    )
    parser.add_argument(
        "--toc-end-line",
        type=int,
        default=(
            int(os.environ["STRUCTURE_TOC_END_LINE"])
            if os.getenv("STRUCTURE_TOC_END_LINE")
            else None
        ),
        help="人工覆盖目录结束行，必须与 toc-start/body-start 一起使用",
    )
    parser.add_argument(
        "--body-start-line",
        type=int,
        default=(
            int(os.environ["STRUCTURE_BODY_START_LINE"])
            if os.getenv("STRUCTURE_BODY_START_LINE")
            else None
        ),
        help="人工覆盖正文起始行，必须与 toc-start/toc-end 一起使用",
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


def is_chapter_number_only(title: str) -> bool:
    compact = " ".join(str(title or "").split())
    return bool(CHINESE_CHAPTER_RE.fullmatch(compact) or ENGLISH_CHAPTER_RE.fullmatch(compact))


def is_numbering_only_title(title: str) -> bool:
    """Return True when a catalog title contains a number but no title text."""
    compact = "".join(title.split()).replace("．", ".").rstrip(".")
    if compact and all(part.isdigit() for part in compact.split(".")):
        return True
    if is_chapter_number_only(title):
        return True
    if compact.startswith("第") and compact.endswith(("章", "节", "篇")):
        number = compact[1:-1]
        chinese_digits = set("零〇一二三四五六七八九十百千万两")
        return number.isdigit() or (
            bool(number) and all(char in chinese_digits for char in number)
        )
    return False


def _catalog_title_source_line(title: str, toc_lines: list[dict]) -> int | None:
    """Find the first source TOC line containing an LLM title, if recoverable."""
    compact = re.sub(r"\s+", "", str(title or ""))
    for line in toc_lines:
        source_text = re.sub(r"^\s*#{1,6}\s*", "", str(line.get("text") or ""))
        source = re.sub(r"\s+", "", source_text)
        if compact and (source == compact or source.startswith(compact)):
            return int(line["line_no"])
    return None


def _looks_like_chapter_title(item: dict) -> bool:
    title = str(item.get("title") or "").strip()
    if not title or is_chapter_number_only(title):
        return False
    if str(item.get("page_hint") or "").strip() or PAGE_ONLY_RE.fullmatch(title):
        return False
    if FIGURE_LINE_RE.match(title) or URL_OR_IMAGE_RE.search(title):
        return False
    if SUBSECTION_PREFIX_RE.match(title):
        return False
    if len(title) > 42 or title.endswith(("。", "！", "？", ".", "!", "?", "；", ";")):
        return False
    return True


def merge_split_chapter_titles(
    items: list[dict], toc_lines: list[dict] | None = None
) -> tuple[list[dict], dict[str, int], list[str]]:
    """Merge adjacent `第X章` / title pairs before catalog schema validation.

    The LLM schema has no source line fields, so raw catalog adjacency is the
    fallback. When source lines can be recovered we require no non-empty line
    between the pair except the title itself.
    """
    merged: list[dict] = []
    warnings: list[str] = []
    merged_count = 0
    unresolved_count = 0
    index = 0
    while index < len(items):
        current = items[index]
        current_title = str(current.get("title") or "").strip()
        if not is_chapter_number_only(current_title):
            merged.append(current)
            index += 1
            continue
        next_item = items[index + 1] if index + 1 < len(items) else None
        next_title = str((next_item or {}).get("title") or "").strip()
        can_merge = bool(next_item and _looks_like_chapter_title(next_item))
        if can_merge and toc_lines:
            current_line = _catalog_title_source_line(current_title, toc_lines)
            next_line = _catalog_title_source_line(next_title, toc_lines)
            if current_line is not None and next_line is not None:
                can_merge = 0 < next_line - current_line <= 3
        if can_merge:
            display_title = f"{current_title} {next_title}"
            aliases = []
            for value in (current_title, next_title, display_title):
                if value not in aliases:
                    aliases.append(value)
            merged.append(
                {
                    **next_item,
                    "title": display_title,
                    "level": 1,
                    "page_hint": "",
                    "normalized_title": str(
                        next_item.get("normalized_title") or next_title
                    ).strip(),
                    "chapter_no": current_title,
                    "aliases": aliases,
                    "source_titles": [current_title, next_title],
                    "normalization": "merge_chapter_number_and_title",
                }
            )
            merged_count += 1
            index += 2
            continue
        unresolved_count += 1
        warnings.append(
            "WARNING 未找到可归并的章节标题: "
            f"chapter_no={current_title} next_title={next_title or '<none>'} "
            f"source_index={index + 1}"
        )
        merged.append(current)
        index += 1
    return merged, {
        "merged_chapter_title_count": merged_count,
        "unresolved_chapter_number_count": unresolved_count,
    }, warnings


def validate_and_normalize_catalog_items(
    items: list, *, warnings: list[str] | None = None
) -> list[dict]:
    """Validate catalog item fields and keep a printed page out of title."""
    normalized_items: list[dict] = []
    for index, item in enumerate(items, start=1):
        label = f"chapter_catalog[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} 必须是 JSON 对象")
        missing = CATALOG_ITEM_FIELDS - item.keys()
        unexpected = item.keys() - CATALOG_ITEM_FIELDS - CATALOG_ITEM_OPTIONAL_FIELDS
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
        if is_chapter_number_only(title):
            message = f"WARNING 未找到可归并的章节标题: chapter_no={title} source_index={index}"
            logger.warning(message)
            if warnings is not None:
                warnings.append(message)

        output_item = {
            "title": title,
            "level": level,
            "page_hint": page_hint,
            "normalized_title": normalized_title.strip(),
        }
        for field in CATALOG_ITEM_OPTIONAL_FIELDS:
            if field in item:
                output_item[field] = item[field]
        normalized_items.append(output_item)
    return normalized_items


async def run(args: argparse.Namespace) -> Path:
    meta = None
    if args.meta:
        meta = load_book_meta(args.meta)
        validate_book_meta(meta, stage="step2")
        paths = resolve_book_paths(meta)
        prepared_path = (PROJECT_DIR / "outputs/01_prepared" / f"{meta['artifact_stem']}.prepared.json").resolve()
        structure = get_structure_config(meta)
        business = get_business_config(meta)
        args.front_lines = structure["front_lines"]
        args.toc_start_line = structure["toc_start_line"]
        args.toc_end_line = structure["toc_end_line"]
        args.body_start_line = structure["body_start_line"]
        logger.info("Step 2 meta 模式 | markdown=%s domain=%s subject=%s", paths["markdown"], business["domain"], business["subject"])
    elif args.prepared:
        prepared_path = args.prepared.expanduser().resolve()
    else:
        raise ValueError("Step 2 必须传 --meta 或 --prepared")
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
    ranges_raw_path = output.with_suffix(".ranges.raw.txt")
    if meta is not None:
        ranges = {
            "book_title": str(meta["book_title"]),
            "front_matter_range": [1, max(1, args.toc_start_line - 1)],
            "toc_range": [args.toc_start_line, args.toc_end_line],
            "body_start_line": args.body_start_line,
            "notes": "目录范围由教材 meta 人工确认。",
            "confidence": 1.0,
        }
    else:
        range_prompt_template = load_prompt(PROJECT_DIR / "prompts/front_matter_toc_prompt.md")
        range_prompt = (
            f"{range_prompt_template}\n\n【文件名】{book_stem}\n【总行数】{prepared['line_count']}\n"
            f"【输入前部行号文本】\n{format_numbered_lines(prepared['lines'][: args.front_lines])}"
        )
        ranges = await call_llm_json(range_prompt, raw_output_path=ranges_raw_path, guided_json_schema=RANGE_JSON_SCHEMA, max_tokens=1024)
        if not isinstance(ranges, dict):
            raise ValueError("边界识别 LLM 必须返回一个 JSON 对象")
    normalization_notes: list[str] = []
    manual_boundaries = (
        args.toc_start_line,
        args.toc_end_line,
        args.body_start_line,
    )
    if any(value is not None for value in manual_boundaries):
        if not all(value is not None for value in manual_boundaries):
            raise ValueError(
                "人工边界覆盖必须同时提供 --toc-start-line、"
                "--toc-end-line 和 --body-start-line"
            )
        previous_toc = ranges.get("toc_range")
        previous_body = ranges.get("body_start_line")
        ranges["toc_range"] = [args.toc_start_line, args.toc_end_line]
        ranges["body_start_line"] = args.body_start_line
        note = (
            f"人工覆盖边界: toc_range {previous_toc} -> {ranges['toc_range']}，"
            f"body_start_line {previous_body} -> {args.body_start_line}"
        )
        logger.warning(note)
        normalization_notes.append(note)
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
        max_tokens=16384,
    )
    if not isinstance(catalog, dict):
        raise ValueError("目录骨架 LLM 必须返回一个 JSON 对象")
    if not isinstance(catalog.get("chapter_catalog"), list):
        raise ValueError("chapter_catalog 必须是数组")
    if not catalog["chapter_catalog"]:
        raise ValueError("chapter_catalog 为空")
    merged_raw_catalog, chapter_merge_summary, chapter_merge_warnings = (
        merge_split_chapter_titles(catalog["chapter_catalog"], toc_lines)
    )
    catalog_items = validate_and_normalize_catalog_items(
        merged_raw_catalog, warnings=chapter_merge_warnings
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
            *chapter_merge_warnings,
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
        "chapter_title_merge_summary": chapter_merge_summary,
        "chapter_title_merge_warnings": chapter_merge_warnings,
    }
    write_json(output, result)
    logger.info(
        "outline 完成 | body_start=%d catalog_items=%d merged_chapters=%d unresolved_chapters=%d output=%s catalog_raw=%s",
        body_start,
        len(result["chapter_catalog"]),
        chapter_merge_summary["merged_chapter_title_count"],
        chapter_merge_summary["unresolved_chapter_number_count"],
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
