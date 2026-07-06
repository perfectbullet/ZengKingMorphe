#!/usr/bin/env python3
"""Prepare immutable, line-numbered Markdown and image metadata."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from common import PROJECT_DIR, ensure_dir, write_json

logger = logging.getLogger(__name__)
IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+[\"'][^\"']*[\"'])?\s*\)"
)
CAPTION_RE = re.compile(r"^(?:[（(]?图\s*\d|图版|附图|Figure\s*\d|Fig\.?\s*\d)", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为教材 Markdown 添加行号并提取图片索引"
    )
    parser.add_argument(
        "--md-file", required=True, type=Path, help="原始 Markdown 文件"
    )
    parser.add_argument("--output", type=Path, help="prepared.json 输出路径")
    return parser.parse_args()


def caption_candidates(lines: list[str], image_index: int) -> list[dict]:
    candidates: list[dict] = []
    for index in range(image_index + 1, min(len(lines), image_index + 4)):
        text = lines[index].strip()
        if text and CAPTION_RE.search(text.lstrip("#>*- ")):
            candidates.append({"line_no": index + 1, "text": text})
    return candidates


def resolve_image_reference(reference: str, source_dir: Path) -> dict:
    """Describe a local image path or preserve an HTTP(S) image URL."""
    parsed = urlsplit(reference)
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        return {
            "reference_type": "remote_url",
            "relative_path": reference,
            "absolute_path": "",
            "url": reference,
            "exists": None,
        }

    relative_path = unquote(reference)
    local_path = relative_path.split("?", 1)[0].split("#", 1)[0]
    absolute_path = (source_dir / local_path).resolve()
    return {
        "reference_type": "local_file",
        "relative_path": relative_path,
        "absolute_path": str(absolute_path),
        "url": "",
        "exists": absolute_path.is_file(),
    }


def prepare_markdown(md_file: Path) -> dict:
    source = md_file.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Markdown 文件不存在: {source}")
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines()
    image_refs: list[dict] = []
    for index, line in enumerate(lines):
        for match in IMAGE_RE.finditer(line):
            reference = (match.group(1) or match.group(2)).strip()
            image_refs.append(
                {
                    "line_no": index + 1,
                    **resolve_image_reference(reference, source.parent),
                    "caption_candidates": caption_candidates(lines, index),
                }
            )
    return {
        "source_md_path": str(source),
        "source_md_dir": str(source.parent),
        "book_stem": source.stem,
        "line_count": len(lines),
        "lines": [
            {"line_no": index + 1, "text": line} for index, line in enumerate(lines)
        ],
        "image_refs": image_refs,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        args = parse_args()
        data = prepare_markdown(args.md_file)
        output = (
            args.output
            or PROJECT_DIR
            / "outputs/01_prepared"
            / f"{data['book_stem']}.prepared.json"
        )
        output = output.expanduser().resolve()
        ensure_dir(output.parent)
        write_json(output, data)
        remote = sum(
            item["reference_type"] == "remote_url" for item in data["image_refs"]
        )
        missing = sum(
            item["reference_type"] == "local_file" and not item["exists"]
            for item in data["image_refs"]
        )
        logger.info(
            "prepared 完成 | lines=%d images=%d remote=%d local_missing=%d output=%s",
            data["line_count"],
            len(data["image_refs"]),
            remote,
            missing,
            output,
        )
        return 0
    except Exception as exc:
        logger.exception("prepared 失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
