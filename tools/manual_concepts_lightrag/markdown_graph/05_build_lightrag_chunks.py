#!/usr/bin/env python3
"""Convert each validated textbook block into exactly one custom chunk."""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
import logging
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

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
FIGURE_PREFIX_RE = re.compile(r"^\s*(图\s*\d+\s*[-－]\s*\d+)")
PURE_FIGURE_LINE_RE = re.compile(r"^\s*图\s*\d+\s*[-－]\s*\d+\s*$")
AUTHOR_SUFFIX_RE = re.compile(
    r"\s*(?:作者\s*[:：]|(?<![A-Za-z])author\s*[:：]|(?<![A-Za-z])by\s+).*$",
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
CONTENT_IMAGE_CHECKS = {
    "url": re.compile(r"https?://", re.I),
    "local_image_path": re.compile(r"(?:^|[\s(])(?:\.\.?/)*images/", re.I),
    "markdown_image_syntax": re.compile(r"!\[[^\]]*\]\("),
    "image_filename": re.compile(r"\.(?:jpe?g|png|webp)(?:\b|$)", re.I),
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
    parser.add_argument(
        "--content-list-v2",
        type=Path,
        help="显式指定 MinerU *_content_list_v2.json",
    )
    parser.add_argument("--mineru-dir", type=Path, help="MinerU 输出目录")
    parser.add_argument("--image-root", type=Path, help="解析相对图片路径的根目录")
    parser.add_argument(
        "--disable-image-caption-content",
        action="store_true",
        help="仅在 metadata 中保留图片说明，不写入 chunk content",
    )
    return parser.parse_args()


def normalize_image_path(path: str) -> str:
    value = str(path or "").strip().replace("\\", "/")
    if re.match(r"^https?://", value, re.I):
        return value
    while value.startswith("./"):
        value = value[2:]
    return value


def split_figure_caption(raw_caption: str) -> tuple[str | None, str | None]:
    raw_value = str(raw_caption or "")
    figure_match = FIGURE_PREFIX_RE.match(raw_value)
    figure_no = None
    caption = raw_value
    if figure_match:
        figure_no = re.sub(r"\s+", "", figure_match.group(1)).replace("－", "-")
        caption = raw_value[figure_match.end() :]
    caption = AUTHOR_SUFFIX_RE.sub("", caption)
    caption = re.sub(r"\s+", " ", caption).strip(" \t\r\n-—:：;,，。")
    if PURE_FIGURE_LINE_RE.fullmatch(caption):
        caption = ""
    if not caption or caption.casefold() in MEANINGLESS_IMAGE_ALTS:
        return figure_no, None
    return figure_no, caption


def _extract_text(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_extract_text(item))
        return result
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return [content]
        if content is not None:
            return _extract_text(content)
    return []


def _join_text(value: object) -> str:
    return " ".join(item.strip() for item in _extract_text(value) if item.strip()).strip()


def _iter_v2_images(data: object):
    if not isinstance(data, list):
        raise ValueError("content_list_v2 顶层必须是 JSON list")
    if all(isinstance(item, list) for item in data):
        for page_idx, page_elements in enumerate(data):
            for element in page_elements:
                yield page_idx, element
        return
    if data and all(isinstance(item, dict) and "elements" in item for item in data):
        for page_idx, page in enumerate(data):
            elements = page.get("elements") or []
            if not isinstance(elements, list):
                raise ValueError(f"content_list_v2 第 {page_idx} 页 elements 必须是 list")
            for element in elements:
                yield page_idx, element
        return
    for element in data:
        yield None, element


def load_content_list_v2(
    path: Path | None, image_root: Path | None
) -> dict[str, dict]:
    if path is None:
        return {}
    resolved_path = path.expanduser().resolve()
    data = json.loads(resolved_path.read_text(encoding="utf-8"))
    root = image_root.expanduser().resolve() if image_root else resolved_path.parent
    assets: dict[str, dict] = {}
    for page_idx, element in _iter_v2_images(data):
        if not isinstance(element, dict) or element.get("type") != "image":
            continue
        content = element.get("content") or {}
        if not isinstance(content, dict):
            continue
        image_source = content.get("image_source") or {}
        if not isinstance(image_source, dict):
            continue
        relative_path = normalize_image_path(str(image_source.get("path") or ""))
        if not relative_path:
            continue
        raw_caption = _join_text(content.get("image_caption"))
        figure_no, caption = split_figure_caption(raw_caption)
        footnotes = _extract_text(content.get("image_footnote"))
        footnotes = [item.strip() for item in footnotes if item.strip()]
        is_remote = bool(re.match(r"^https?://", relative_path, re.I))
        absolute_path = relative_path if is_remote else str((root / relative_path).resolve())
        asset = {
            "relative_path": relative_path,
            "absolute_path": absolute_path,
            "exists": None if is_remote else Path(absolute_path).exists(),
            "reference_type": "remote_url" if is_remote else "local_image",
            "figure_no": figure_no,
            "caption": caption,
            "raw_caption": raw_caption,
            "page_idx": page_idx,
            "bbox": element.get("bbox"),
            "image_footnote": footnotes,
        }
        if is_remote:
            asset["url"] = relative_path
        assets[relative_path] = asset
    return assets


def _find_content_list_v2(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob("*_content_list_v2.json"))
    if not matches:
        return None
    if len(matches) > 1:
        choices = ", ".join(path.name for path in matches)
        raise ValueError(
            f"发现多个 content_list_v2 文件 ({choices})，请显式传 --content-list-v2"
        )
    return matches[0].resolve()


def resolve_image_evidence_paths(
    args: argparse.Namespace, source_md_path: Path
) -> tuple[Path | None, Path]:
    mineru_dir = args.mineru_dir.expanduser().resolve() if args.mineru_dir else None
    if mineru_dir is not None and not mineru_dir.is_dir():
        raise FileNotFoundError(f"MinerU 目录不存在: {mineru_dir}")
    if args.content_list_v2:
        content_list_v2 = args.content_list_v2.expanduser().resolve()
        if not content_list_v2.is_file():
            raise FileNotFoundError(f"content_list_v2 文件不存在: {content_list_v2}")
    else:
        content_list_v2 = _find_content_list_v2(mineru_dir or source_md_path.parent)
    if args.image_root:
        image_root = args.image_root.expanduser().resolve()
    else:
        image_root = mineru_dir or source_md_path.parent.resolve()
    return content_list_v2, image_root


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
    _, caption = split_figure_caption(value)
    return caption


def _image_match_key(path: str) -> str:
    normalized = normalize_image_path(path)
    if re.match(r"^https?://", normalized, re.I):
        filename = Path(urlsplit(normalized).path).name
        return f"images/{filename}" if filename else normalized
    return normalized


def _extract_markdown_images(content: str, start_line: int) -> list[dict]:
    images: list[dict] = []
    for match in MARKDOWN_IMAGE_RE.finditer(content):
        path = normalize_image_path(match.group(2) or match.group(3) or "")
        if not path:
            continue
        is_remote = bool(re.match(r"^https?://", path, re.I))
        image = {
            "line_no": start_line + content[: match.start()].count("\n"),
            "relative_path": path,
            "absolute_path": "",
            "exists": None,
            "reference_type": "remote_url" if is_remote else "local_image",
            "caption": meaningful_image_alt(match.group(1)),
        }
        if is_remote:
            image["url"] = path
        images.append(image)
    return images


def enhance_block_images(
    block: dict, v2_assets: dict[str, dict], image_root: Path
) -> list[dict]:
    content = str(block.get("content") or "")
    block_images = block.get("images") or []
    if block_images:
        images = [dict(image) for image in block_images]
    else:
        images = _extract_markdown_images(content, int(block.get("start_line") or 1))

    for image in images:
        original_path = normalize_image_path(
            str(image.get("relative_path") or image.get("url") or "")
        )
        is_remote = bool(
            image.get("reference_type") == "remote_url"
            or re.match(r"^https?://", original_path, re.I)
        )
        existing_raw_caption = str(
            image.get("raw_caption") or image.get("caption") or ""
        )
        existing_figure_no, existing_caption = split_figure_caption(existing_raw_caption)
        if existing_raw_caption:
            image["raw_caption"] = existing_raw_caption
        image["figure_no"] = image.get("figure_no") or existing_figure_no
        image["caption"] = existing_caption

        evidence = v2_assets.get(original_path)
        if evidence is None:
            evidence = v2_assets.get(_image_match_key(original_path))
        if evidence is not None:
            for field in ("figure_no", "caption", "raw_caption", "page_idx", "bbox"):
                value = evidence.get(field)
                if value is not None and value != "" and value != []:
                    image[field] = value
            image["image_footnote"] = evidence.get("image_footnote") or []
            if not is_remote:
                for field in ("reference_type", "absolute_path", "exists"):
                    image[field] = evidence.get(field)

        if original_path:
            image["relative_path"] = original_path
        if is_remote:
            image["reference_type"] = "remote_url"
            image["url"] = original_path
        elif original_path:
            image["reference_type"] = image.get("reference_type") or "local_image"
            if not image.get("absolute_path"):
                absolute_path = (image_root / original_path).resolve()
                image["absolute_path"] = str(absolute_path)
                image["exists"] = absolute_path.exists()
    return images


def _caption_compare_key(value: str) -> str:
    _, caption = split_figure_caption(value)
    value = caption or ""
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())


def _caption_matches(line: str, captions: list[str]) -> bool:
    line_key = _caption_compare_key(line)
    if not line_key:
        return False
    for caption in captions:
        caption_key = _caption_compare_key(caption)
        if not caption_key:
            continue
        if line_key == caption_key:
            return True
        if min(len(line_key), len(caption_key)) >= 8:
            if line_key in caption_key or caption_key in line_key:
                return True
            if SequenceMatcher(None, line_key, caption_key).ratio() >= 0.92:
                return True
    return False


def clean_body_content(
    content: str, enhanced_images: list[dict] | None = None
) -> tuple[str, list[str]]:
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
    enhanced_captions = [
        str(image["caption"])
        for image in enhanced_images or []
        if image.get("caption")
    ]
    kept_lines: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if PURE_FIGURE_LINE_RE.fullmatch(stripped):
            continue
        if enhanced_captions and FIGURE_PREFIX_RE.match(stripped):
            if _caption_matches(stripped, enhanced_captions):
                continue
        kept_lines.append(line.rstrip())
    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, image_descriptions


def _image_description(image: dict) -> str | None:
    caption = meaningful_image_alt(str(image.get("caption") or ""))
    if not caption:
        return None
    figure_no = str(image.get("figure_no") or "").strip()
    suffix = f"（{figure_no}）" if figure_no else ""
    caption = caption[: 120 - len(suffix)].rstrip()
    return f"{caption}{suffix}"


def build_content(
    block: dict,
    domain: str,
    subject: str,
    file_path: str,
    images: list[dict] | None = None,
    include_image_captions: bool = True,
) -> str:
    heading_path = [
        str(item) for item in block.get("heading_path", []) if str(item).strip()
    ]
    enhanced_images = images or []
    body_content, alt_descriptions = clean_body_content(
        str(block.get("content") or ""), enhanced_images
    )
    image_descriptions: list[str] = []
    seen_descriptions: set[str] = set()
    known_caption_keys: set[str] = set()
    for image in enhanced_images:
        description = _image_description(image)
        if not description:
            continue
        key = _caption_compare_key(description)
        if key and key not in seen_descriptions:
            image_descriptions.append(description)
            seen_descriptions.add(key)
        caption_key = _caption_compare_key(str(image.get("caption") or ""))
        if caption_key:
            known_caption_keys.add(caption_key)
    for item in alt_descriptions:
        description = item[:120].rstrip()
        key = _caption_compare_key(description)
        if key and key not in known_caption_keys and key not in seen_descriptions:
            image_descriptions.append(description)
            seen_descriptions.add(key)
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
    if include_image_captions and image_descriptions:
        descriptions = "\n".join(f"- {item}" for item in image_descriptions)
        sections.append(f"【图片说明】\n{descriptions}")
    sections.append(f"【正文】\n{body_content}")
    return "\n\n".join(sections)


def _build_evidence(record: dict) -> dict:
    source = {
        "file_path": record["file_path"],
        "source_md_path": record["source_md_path"],
        "catalog_index": record.get("catalog_index"),
        "catalog_title": record.get("catalog_title"),
        "heading_path": record.get("heading_path", []),
    }
    images: list[dict] = []
    for image in record.get("images") or []:
        evidence_image: dict = {}
        for field in ("figure_no", "caption", "relative_path", "page_idx", "bbox"):
            value = image.get(field)
            if value is not None and value != "" and value != []:
                evidence_image[field] = value
        images.append(evidence_image)
    return {"source": source, "images": images}


def run(args: argparse.Namespace) -> Path:
    blocks_path = args.blocks.expanduser().resolve()
    blocks = read_jsonl(blocks_path)
    if not blocks:
        raise ValueError(f"blocks.jsonl 为空: {blocks_path}")
    blocks.sort(key=lambda item: (int(item["start_line"]), int(item["end_line"])))
    source_md_path = Path(str(blocks[0]["source_md_path"])).expanduser().resolve()
    content_list_v2_path, image_root = resolve_image_evidence_paths(
        args, source_md_path
    )
    v2_assets = load_content_list_v2(content_list_v2_path, image_root)
    book_stem = source_md_path.stem
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
    image_caption_content_lines = 0
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
        images = enhance_block_images(block, v2_assets, image_root)
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
            block,
            domain,
            subject,
            file_path,
            images=images,
            include_image_captions=not args.disable_image_caption_content,
        )
        if "【图片说明】\n" in record["content"]:
            caption_section = record["content"].split("【图片说明】\n", 1)[1]
            caption_section = caption_section.split("\n\n【正文】", 1)[0]
            image_caption_content_lines += sum(
                line.startswith("- ") for line in caption_section.splitlines()
            )
        record["evidence"] = _build_evidence(record)
        records.append(record)
    write_jsonl(output, records)

    content_scope_counts = Counter(record["content_scope"] for record in records)
    should_extract_kg_counts = Counter(
        record["should_extract_kg"] for record in records
    )
    all_images = [image for record in records for image in record.get("images") or []]
    content_issue_counts = {
        name: sum(bool(pattern.search(record["content"])) for record in records)
        for name, pattern in CONTENT_IMAGE_CHECKS.items()
    }
    stats = {
        "count": len(records),
        "content_scope": dict(sorted(content_scope_counts.items())),
        "should_extract_kg": dict(sorted(should_extract_kg_counts.items())),
        "content_list_v2_path": str(content_list_v2_path or ""),
        "content_list_v2_images_total": len(v2_assets),
        "chunks_with_images": sum(bool(record.get("images")) for record in records),
        "total_images": len(all_images),
        "images_with_caption": sum(bool(image.get("caption")) for image in all_images),
        "images_with_figure_no": sum(bool(image.get("figure_no")) for image in all_images),
        "images_with_bbox": sum(bool(image.get("bbox")) for image in all_images),
        "images_with_page_idx": sum(
            image.get("page_idx") is not None for image in all_images
        ),
        "image_caption_content_lines": image_caption_content_lines,
        "chunks_with_url_in_content": content_issue_counts["url"],
        "chunks_with_local_image_path_in_content": content_issue_counts[
            "local_image_path"
        ],
        "chunks_with_markdown_image_syntax": content_issue_counts[
            "markdown_image_syntax"
        ],
    }
    logger.info("chunks 完成 | %s output=%s", stats, output)
    if any(content_issue_counts.values()):
        logger.warning("chunk content 仍包含图片引用 | %s", content_issue_counts)
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
