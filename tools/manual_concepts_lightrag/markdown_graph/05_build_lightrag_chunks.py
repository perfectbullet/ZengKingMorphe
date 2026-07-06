#!/usr/bin/env python3
"""Convert each validated textbook block into exactly one custom chunk."""

from __future__ import annotations

import argparse
from collections import Counter
import logging
import os
import re
import sys
from pathlib import Path

from common import PROJECT_DIR, read_jsonl, write_jsonl

logger = logging.getLogger(__name__)
BLOCK_METADATA_FIELDS = [
    "catalog_index",
    "catalog_level",
    "catalog_title",
    "matched_title_line",
    "matched_title_text",
    "content_scope",
    "should_extract_kg",
    "structural_children",
    "book_title",
    "chapter_title",
    "section_title",
    "heading_path",
    "start_line",
    "end_line",
    "image_lines",
    "image_assets",
    "confidence",
    "reason",
]
MARKDOWN_IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]\(\s*(?:<([^>]+)>|([^\s)]+))"
    r"(?:\s+[\"'][^\"']*[\"'])?\s*\)"
)
REMOTE_URL_RE = re.compile(r"https?://[^\s<>()\[\]，。；、]+", re.I)
LOCAL_IMAGE_PATH_RE = re.compile(
    r"(?<![\w])(?:\.\.?/)*images/[^\s<>()\[\]，。；、]+?"
    r"\.(?:jpe?g|png|webp)(?:[?#][^\s<>()\[\]，。；、]*)?",
    re.I,
)
IMAGE_FILENAME_RE = re.compile(
    r"(?<![\w./-])[\w.-]+\.(?:jpe?g|png|webp)(?![\w])"
    r"(?:[?#][^\s<>()\[\]，。；、]*)?",
    re.I,
)
MEANINGLESS_IMAGE_ALTS = {
    "无图注",
    "image",
    "img",
    "图片",
    "图像",
    "photo",
    "picture",
    "screenshot",
    "截图",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 blocks.jsonl 转换为 LightRAG custom chunks"
    )
    parser.add_argument("--blocks", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--domain", default=os.getenv("MARKDOWN_GRAPH_DOMAIN", "industrial_training")
    )
    parser.add_argument("--subject", default=os.getenv("MARKDOWN_GRAPH_SUBJECT", ""))
    return parser.parse_args()


def meaningful_image_alt(alt: str) -> str | None:
    value = alt.strip()
    lowered = value.casefold()
    if not value or lowered in MEANINGLESS_IMAGE_ALTS:
        return None
    if re.fullmatch(r"[0-9a-f]{8,}", value, re.I):
        return None
    if (
        "images/" in lowered
        or "http://" in lowered
        or "https://" in lowered
        or re.search(r"\.(?:jpe?g|png|webp)(?:\b|$)", lowered)
    ):
        return None
    return value


def clean_body_content(content: str) -> tuple[str, list[str]]:
    image_descriptions: list[str] = []

    def replace_markdown_image(match: re.Match[str]) -> str:
        description = meaningful_image_alt(match.group(1))
        if description and description not in image_descriptions:
            image_descriptions.append(description)
        return ""

    cleaned = MARKDOWN_IMAGE_RE.sub(replace_markdown_image, content)
    cleaned = REMOTE_URL_RE.sub("", cleaned)
    cleaned = LOCAL_IMAGE_PATH_RE.sub("", cleaned)
    cleaned = IMAGE_FILENAME_RE.sub("", cleaned)
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, image_descriptions


def build_content(block: dict, domain: str, subject: str, file_path: str) -> str:
    heading_path = [
        str(item) for item in block.get("heading_path", []) if str(item).strip()
    ]
    body_content, image_descriptions = clean_body_content(str(block.get("content") or ""))
    headers = [
        "【文档类型】工业实训教材",
        f"【领域】{domain}",
        f"【科目】{subject}",
        f"【书名】{block.get('book_title', '')}",
        f"【章节路径】{' > '.join(heading_path)}",
        f"【目录项】{block.get('catalog_title', '')}",
        f"【源码位置】{file_path}",
    ]
    sections = ["\n".join(headers)]
    if image_descriptions:
        descriptions = "\n".join(f"- {item}" for item in image_descriptions)
        sections.append(f"【图片说明】\n{descriptions}")
    sections.append(f"【正文】\n{body_content}")
    return "\n\n".join(sections)


def run(args: argparse.Namespace) -> Path:
    blocks_path = args.blocks.expanduser().resolve()
    blocks = read_jsonl(blocks_path)
    if not blocks:
        raise ValueError(f"blocks.jsonl 为空: {blocks_path}")
    blocks.sort(key=lambda item: (int(item["start_line"]), int(item["end_line"])))
    book_stem = Path(str(blocks[0]["source_md_path"])).stem
    output = (
        (
            args.output
            or PROJECT_DIR / "outputs/05_chunks" / f"{book_stem}.lightrag_chunks.jsonl"
        )
        .expanduser()
        .resolve()
    )
    records: list[dict] = []
    seen_ids: set[str] = set()
    for order, block in enumerate(blocks):
        block_id = str(block["block_id"])
        chunk_id = block_id if block_id.startswith("chunk-") else f"chunk-{block_id}"
        if chunk_id in seen_ids:
            raise ValueError(f"chunk_id 重复: {chunk_id}")
        seen_ids.add(chunk_id)
        domain = args.domain or str(block.get("domain") or "")
        subject = args.subject or str(block.get("subject") or "")
        source_path = Path(str(block["source_md_path"]))
        file_path = f"{source_path.name}#L{block['start_line']}-L{block['end_line']}"
        images = [dict(image) for image in block.get("images", [])]
        record = {
            "chunk_id": chunk_id,
            "doc_id": block["doc_id"],
            "block_id": block_id,
            "chunk_order_index": order,
            "file_path": file_path,
            "source_md_path": str(source_path),
            "domain": domain,
            "subject": subject,
            "book_title": block.get("book_title", ""),
            "chapter_title": block.get("chapter_title", ""),
            "section_title": block.get("section_title", ""),
            "heading_path": block.get("heading_path", []),
            "start_line": block["start_line"],
            "end_line": block["end_line"],
            "images": images,
        }
        for field in BLOCK_METADATA_FIELDS:
            if field in block:
                record[field] = block.get(field)

        body_content = str(block.get("content") or "")
        if not record.get("content_scope"):
            structural_children = record.get("structural_children") or []
            if structural_children:
                record["content_scope"] = (
                    "direct" if body_content.strip() else "structural"
                )
            else:
                record["content_scope"] = "leaf"
        if (
            "should_extract_kg" not in record
            or record.get("should_extract_kg") is None
        ):
            record["should_extract_kg"] = record["content_scope"] != "structural"
        elif not isinstance(record["should_extract_kg"], bool):
            raise ValueError(
                f"block {block_id} 的 should_extract_kg 必须是 JSON boolean"
            )

        record["content"] = build_content(
            {**block, "images": images}, domain, subject, file_path
        )
        records.append(record)
    write_jsonl(output, records)
    content_scope_counts = Counter(record["content_scope"] for record in records)
    should_extract_kg_counts = Counter(
        record["should_extract_kg"] for record in records
    )
    logger.info(
        "chunks 完成 | count=%d content_scope=%s should_extract_kg=%s output=%s",
        len(records),
        dict(sorted(content_scope_counts.items())),
        dict(sorted(should_extract_kg_counts.items())),
        output,
    )
    return output


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run(parse_args())
        return 0
    except Exception as exc:
        logger.exception("build chunks 失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
