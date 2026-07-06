#!/usr/bin/env python3
"""Convert each validated textbook block into exactly one custom chunk."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from common import PROJECT_DIR, read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


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


def build_content(block: dict, domain: str, subject: str, file_path: str) -> str:
    heading_path = [
        str(item) for item in block.get("heading_path", []) if str(item).strip()
    ]
    image_descriptions: list[str] = []
    for image in block.get("images", []):
        caption = str(image.get("caption") or "无图注")
        image_descriptions.append(f"{caption} | {image.get('relative_path', '')}")
    headers = [
        "【文档类型】工业实训教材",
        f"【领域】{domain}",
        f"【科目】{subject}",
        f"【书名】{block.get('book_title', '')}",
        f"【章节路径】{' > '.join(heading_path)}",
        f"【源码位置】{file_path}",
    ]
    if image_descriptions:
        headers.append(f"【图片】{'；'.join(image_descriptions)}")
    return "\n".join(headers) + f"\n\n【正文】\n{block['content']}"


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
        images = [
            {
                "relative_path": image.get("relative_path", ""),
                "absolute_path": image.get("absolute_path", ""),
                "reference_type": image.get("reference_type", "local_file"),
                "url": image.get("url", ""),
                "caption": image.get("caption", ""),
            }
            for image in block.get("images", [])
        ]
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
        record["content"] = build_content(
            {**block, "images": images}, domain, subject, file_path
        )
        records.append(record)
    write_jsonl(output, records)
    logger.info("chunks 完成 | count=%d output=%s", len(records), output)
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
