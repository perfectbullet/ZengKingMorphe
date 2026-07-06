#!/usr/bin/env python3
"""Validate/deduplicate an LLM plan and cut immutable source line ranges."""

from __future__ import annotations

import argparse
import difflib
import logging
import os
import re
import sys
from pathlib import Path
from common import PROJECT_DIR, load_json, read_jsonl, slugify_for_id, write_jsonl

logger = logging.getLogger(__name__)
ALLOWED_BLOCK_TYPES = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="校验结构计划并按原始行号生成正文 blocks"
    )
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--outline", required=True, type=Path)
    parser.add_argument("--structure-plan", required=True, type=Path)
    parser.add_argument("--output-blocks", type=Path)
    parser.add_argument("--output-plan", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--domain", default=os.getenv("MARKDOWN_GRAPH_DOMAIN", "industrial_training")
    )
    parser.add_argument("--subject", default=os.getenv("MARKDOWN_GRAPH_SUBJECT", ""))
    parser.add_argument(
        "--max-block-lines-warn",
        type=int,
        default=int(os.getenv("MAX_BLOCK_LINES_WARN", "500")),
    )
    parser.add_argument(
        "--max-block-chars-warn",
        type=int,
        default=int(os.getenv("MAX_BLOCK_CHARS_WARN", "12000")),
    )
    return parser.parse_args()


def block_title(record: dict) -> str:
    heading_path = record.get("heading_path")
    if isinstance(heading_path, list) and heading_path:
        return str(heading_path[-1])
    return str(record.get("section_title") or record.get("chapter_title") or "")


def normalized_title(record: dict) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", block_title(record).lower())


def overlap_ratio(left: dict, right: dict) -> float:
    overlap = max(
        0,
        min(left["end_line"], right["end_line"])
        - max(left["start_line"], right["start_line"])
        + 1,
    )
    shorter = min(
        left["end_line"] - left["start_line"] + 1,
        right["end_line"] - right["start_line"] + 1,
    )
    return overlap / shorter if shorter else 0.0


def titles_similar(left: dict, right: dict) -> bool:
    left_title = normalized_title(left)
    right_title = normalized_title(right)
    if not left_title or not right_title:
        return False
    return (
        left_title == right_title
        or left_title in right_title
        or right_title in left_title
        or difflib.SequenceMatcher(None, left_title, right_title).ratio() >= 0.78
    )


def preferred(left: dict, right: dict) -> dict:
    left_rank = (
        float(left.get("confidence") or 0),
        left["end_line"] - left["start_line"] + 1,
    )
    right_rank = (
        float(right.get("confidence") or 0),
        right["end_line"] - right["start_line"] + 1,
    )
    return left if left_rank >= right_rank else right


def document_id(book_stem: str, subject: str) -> str:
    ascii_book = re.sub(r"[^a-zA-Z0-9]+", "_", book_stem).strip("_").lower()
    parts = [
        part
        for part in (ascii_book, slugify_for_id(subject) if subject else "")
        if part
    ]
    return "doc-" + ("_".join(parts) if parts else slugify_for_id(book_stem))


def validate_plan(
    records: list[dict], line_count: int, report: list[str]
) -> list[dict]:
    valid: list[dict] = []
    for index, source in enumerate(records, 1):
        record = dict(source)
        errors: list[str] = []
        try:
            record["start_line"] = int(record.get("start_line"))
            record["end_line"] = int(record.get("end_line"))
        except (TypeError, ValueError):
            errors.append("start_line/end_line 不是整数")
        if not errors:
            if record["start_line"] < 1 or record["end_line"] > line_count:
                errors.append(f"行号越界 1-{line_count}")
            if record["start_line"] > record["end_line"]:
                errors.append("start_line > end_line")
            window_start = record.get("source_window_start_line")
            window_end = record.get("source_window_end_line")
            if isinstance(window_start, int) and record["start_line"] < window_start:
                errors.append("start_line 超出来源窗口")
            if isinstance(window_end, int) and record["end_line"] > window_end:
                errors.append("end_line 超出来源窗口")
        if not str(record.get("block_id") or "").strip():
            errors.append("block_id 为空")
        try:
            record["confidence"] = float(record.get("confidence") or 0)
        except (TypeError, ValueError):
            errors.append("confidence 不是数字")
        else:
            if not 0 <= record["confidence"] <= 1:
                errors.append("confidence 必须在 0 到 1 之间")
        block_type = str(record.get("block_type") or "unknown")
        if block_type not in ALLOWED_BLOCK_TYPES:
            report.append(
                f"- WARNING record {index}: 未知 block_type={block_type!r}，归一为 unknown"
            )
            record["block_type"] = "unknown"
        image_lines = record.get("image_lines", [])
        if not isinstance(image_lines, list):
            errors.append("image_lines 必须是数组")
        elif not errors:
            outside = [
                line
                for line in image_lines
                if not isinstance(line, int)
                or not record["start_line"] <= line <= record["end_line"]
            ]
            if outside:
                report.append(
                    f"- WARNING {record.get('block_id')}: image_lines 不在 block 内，已忽略 {outside}"
                )
                record["image_lines"] = [
                    line
                    for line in image_lines
                    if isinstance(line, int)
                    and record["start_line"] <= line <= record["end_line"]
                ]
        if errors:
            report.append(
                f"- DROPPED record {index} ({record.get('block_id', '<empty>')}): {'; '.join(errors)}"
            )
            continue
        valid.append(record)
    return valid


def deduplicate(records: list[dict], report: list[str]) -> list[dict]:
    kept: list[dict] = []
    for record in sorted(
        records, key=lambda item: (item["start_line"], item["end_line"])
    ):
        duplicate_index: int | None = None
        reason = ""
        for index, existing in enumerate(kept):
            if record["block_id"] == existing["block_id"]:
                duplicate_index, reason = index, "block_id 重复"
                break
            if overlap_ratio(record, existing) >= 0.70 and titles_similar(
                record, existing
            ):
                duplicate_index, reason = index, "高度重叠且标题相近"
                break
        if duplicate_index is None:
            kept.append(record)
            continue
        existing = kept[duplicate_index]
        winner = preferred(existing, record)
        loser = record if winner is existing else existing
        kept[duplicate_index] = winner
        report.append(
            f"- DEDUP {reason}: 丢弃 {loser['block_id']} "
            f"L{loser['start_line']}-L{loser['end_line']}，保留 {winner['block_id']} "
            f"L{winner['start_line']}-L{winner['end_line']}"
        )
    return sorted(kept, key=lambda item: (item["start_line"], item["end_line"]))


def build_gap_report(
    records: list[dict], lines: list[dict], body_start: int, report: list[str]
) -> None:
    cursor = body_start
    for record in records:
        if record["start_line"] > cursor:
            _report_gap(cursor, record["start_line"] - 1, lines, report)
        cursor = max(cursor, record["end_line"] + 1)
    if cursor <= len(lines):
        _report_gap(cursor, len(lines), lines, report)


def _report_gap(start: int, end: int, lines: list[dict], report: list[str]) -> None:
    non_empty = sum(
        bool(lines[index - 1]["text"].strip()) for index in range(start, end + 1)
    )
    if end - start + 1 >= 50 or non_empty >= 20:
        report.append(f"- WARNING 可能漏切正文: L{start}-L{end}，非空行 {non_empty}")


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    prepared = load_json(args.prepared.expanduser().resolve())
    outline = load_json(args.outline.expanduser().resolve())
    raw_records = read_jsonl(args.structure_plan.expanduser().resolve())
    if not isinstance(prepared, dict) or not isinstance(outline, dict):
        raise ValueError("prepared 和 outline 顶层必须是对象")
    book_stem = str(prepared["book_stem"])
    output_dir = PROJECT_DIR / "outputs/04_blocks"
    output_blocks = (
        (args.output_blocks or output_dir / f"{book_stem}.blocks.jsonl")
        .expanduser()
        .resolve()
    )
    output_plan = (
        (args.output_plan or output_dir / f"{book_stem}.structure_plan.validated.jsonl")
        .expanduser()
        .resolve()
    )
    report_path = (
        (args.report or output_dir / f"{book_stem}.validation_report.md")
        .expanduser()
        .resolve()
    )
    discoveries: list[str] = []
    valid = validate_plan(raw_records, int(prepared["line_count"]), discoveries)
    deduplicated = deduplicate(valid, discoveries)
    source_lines: list[dict] = prepared["lines"]
    images: list[dict] = prepared.get("image_refs", [])
    subject = args.subject.strip()
    doc_id = document_id(book_stem, subject)
    blocks: list[dict] = []
    for record in deduplicated:
        start, end = record["start_line"], record["end_line"]
        content = "\n".join(item["text"] for item in source_lines[start - 1 : end])
        if not content.strip():
            discoveries.append(f"- DROPPED {record['block_id']}: content 为空")
            continue
        block_images: list[dict] = []
        for image in images:
            if start <= int(image["line_no"]) <= end:
                captions = image.get("caption_candidates") or []
                block_images.append(
                    {
                        "line_no": image["line_no"],
                        "relative_path": image["relative_path"],
                        "absolute_path": image["absolute_path"],
                        "exists": image["exists"],
                        "reference_type": image.get("reference_type", "local_file"),
                        "url": image.get("url", ""),
                        "caption": captions[0]["text"] if captions else "",
                    }
                )
        block = {
            "doc_id": doc_id,
            "block_id": record["block_id"],
            "block_type": record.get("block_type", "unknown"),
            "domain": args.domain,
            "subject": subject,
            "book_title": record.get("book_title")
            or outline.get("book_title")
            or book_stem,
            "chapter_title": record.get("chapter_title") or "",
            "section_title": record.get("section_title") or "",
            "heading_path": record.get("heading_path")
            if isinstance(record.get("heading_path"), list)
            else [],
            "source_md_path": prepared["source_md_path"],
            "start_line": start,
            "end_line": end,
            "content": content,
            "images": block_images,
            "confidence": float(record.get("confidence") or 0),
            "source_window_id": record.get("source_window_id", ""),
        }
        line_count = end - start + 1
        if line_count > args.max_block_lines_warn:
            discoveries.append(
                f"- WARNING {record['block_id']}: {line_count} 行超过阈值 "
                f"{args.max_block_lines_warn}，未自动拆分"
            )
        if len(content) > args.max_block_chars_warn:
            discoveries.append(
                f"- WARNING {record['block_id']}: {len(content)} 字符超过阈值 "
                f"{args.max_block_chars_warn}，未自动拆分"
            )
        blocks.append(block)

    for previous, current in zip(blocks, blocks[1:]):
        if current["start_line"] <= previous["end_line"]:
            discoveries.append(
                f"- WARNING 保留 block 重叠: {previous['block_id']} 与 {current['block_id']}，"
                f"L{current['start_line']}-L{min(previous['end_line'], current['end_line'])}"
            )
    output_ids = {block["block_id"] for block in blocks}
    final_plan = [record for record in deduplicated if record["block_id"] in output_ids]
    build_gap_report(
        final_plan, source_lines, int(outline["body_start_line"]), discoveries
    )
    report = [
        "# 结构计划校验报告",
        "",
        f"- 原始记录数: {len(raw_records)}",
        f"- 基础校验后: {len(valid)}",
        f"- 去重且内容有效: {len(final_plan)}",
        f"- 输出 blocks: {len(blocks)}",
        "",
        "## 发现项",
        "",
    ]
    report.extend(discoveries or ["- 未发现问题"])
    write_jsonl(output_plan, final_plan)
    write_jsonl(output_blocks, blocks)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    logger.info(
        "apply 完成 | raw=%d validated=%d blocks=%d",
        len(raw_records),
        len(final_plan),
        len(blocks),
    )
    return output_blocks, output_plan, report_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run(parse_args())
        return 0
    except Exception as exc:
        logger.exception("apply structure plan 失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
