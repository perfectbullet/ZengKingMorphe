#!/usr/bin/env python3
"""Convert validated textbook blocks into LightRAG custom chunks."""

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

from book_meta import get_business_config, load_book_meta, resolve_book_paths, resolve_config_value
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
TARGET_KG_CHUNK_CHARS = 2000
HARD_MAX_KG_CHUNK_CHARS = 3000
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。；;！？!?])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 blocks.jsonl 转换为 LightRAG custom chunks"
    )
    parser.add_argument("--blocks", required=True, type=Path)
    parser.add_argument("--meta", type=Path, help="单本教材 meta")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--domain")
    parser.add_argument("--subject")
    parser.add_argument(
        "--content-list-v2",
        type=Path,
        help="显式指定 MinerU *_content_list_v2.json",
    )
    parser.add_argument("--mineru-dir", type=Path, help="MinerU 输出目录")
    parser.add_argument("--image-root", type=Path, help="解析相对图片路径的根目录")
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


def clean_body_lines(content: str, start_line: int) -> list[dict[str, int | str]]:
    """Clean image references and figure-caption-only lines while preserving line numbers."""
    cleaned_lines: list[dict[str, int | str]] = []
    for offset, line in enumerate(content.splitlines()):
        line_no = start_line + offset
        cleaned = MARKDOWN_IMAGE_RE.sub("", line)
        cleaned = REMOTE_URL_RE.sub("", cleaned)
        cleaned = LOCAL_IMAGE_PATH_RE.sub("", cleaned)
        cleaned = IMAGE_FILENAME_RE.sub("", cleaned)
        cleaned = cleaned.rstrip()
        stripped = cleaned.strip()
        if PURE_FIGURE_LINE_RE.fullmatch(stripped):
            continue
        if FIGURE_PREFIX_RE.match(stripped):
            continue
        cleaned_lines.append({"line_no": line_no, "text": cleaned})
    return cleaned_lines


def body_lines_to_text(lines: list[dict[str, int | str]]) -> str:
    text = "\n".join(str(line.get("text") or "").rstrip() for line in lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_content(
    block: dict,
    domain: str,
    subject: str,
    file_path: str,
    body_content: str,
    split_label: str | None = None,
) -> str:
    heading_path = [
        str(item) for item in block.get("heading_path", []) if str(item).strip()
    ]
    headers = [
        "【文档类型】工业实训教材",
        f"【领域】{domain}",
        f"【科目】{subject}",
        f"【书名】{block.get('book_title', '')}",
        f"【章节路径】{' > '.join(heading_path)}",
        f"【目录项】{block.get('catalog_title', '')}",
        f"【源码位置】{file_path}",
    ]
    if split_label:
        headers.append(f"【分片】{split_label}")
    sections = ["\n".join(headers)]
    sections.append(f"【正文】\n{body_content}")
    return "\n\n".join(sections)


def _is_split_boundary_line(text: str) -> bool:
    stripped = text.strip()
    return bool(
        not stripped
        or re.match(r"^#{1,6}\s+", stripped)
        or re.match(r"^\s*(?:[-*+]\s+|[（(]?\d+[）).、]\s+)", text)
        or stripped.startswith(("注意", "提示", "步骤", "要点"))
    )


def _split_long_text_line(line: dict[str, int | str]) -> list[list[dict[str, int | str]]]:
    text = str(line.get("text") or "")
    line_no = int(line.get("line_no") or 0)
    if len(text) <= TARGET_KG_CHUNK_CHARS:
        return [[line]]
    fragments = [item for item in SENTENCE_SPLIT_RE.split(text) if item]
    units: list[list[dict[str, int | str]]] = []

    def append_fragment(fragment: str) -> None:
        fragment = fragment.strip()
        if not fragment:
            return
        if len(fragment) <= TARGET_KG_CHUNK_CHARS:
            units.append([{"line_no": line_no, "text": fragment}])
            return
        for index in range(0, len(fragment), TARGET_KG_CHUNK_CHARS):
            units.append(
                [
                    {
                        "line_no": line_no,
                        "text": fragment[index : index + TARGET_KG_CHUNK_CHARS],
                    }
                ]
            )

    buffer = ""
    for fragment in fragments:
        if len(fragment) > TARGET_KG_CHUNK_CHARS:
            append_fragment(buffer)
            buffer = ""
            append_fragment(fragment)
            continue
        if buffer and len(buffer) + len(fragment) > TARGET_KG_CHUNK_CHARS:
            append_fragment(buffer)
            buffer = fragment
        else:
            buffer += fragment
    append_fragment(buffer)
    if not units:
        units = [
            [{"line_no": line_no, "text": text[index : index + TARGET_KG_CHUNK_CHARS]}]
            for index in range(0, len(text), TARGET_KG_CHUNK_CHARS)
        ]
    return units


def _body_line_units(
    lines: list[dict[str, int | str]]
) -> list[list[dict[str, int | str]]]:
    units: list[list[dict[str, int | str]]] = []
    current: list[dict[str, int | str]] = []

    def flush_current() -> None:
        nonlocal current
        if current:
            units.extend(_split_unit_if_needed(current))
            current = []

    for line in lines:
        text = str(line.get("text") or "")
        if _is_split_boundary_line(text):
            flush_current()
            units.extend(_split_unit_if_needed([line]))
            continue
        current.append(line)
    flush_current()
    return units


def _split_unit_if_needed(
    unit: list[dict[str, int | str]]
) -> list[list[dict[str, int | str]]]:
    if len(body_lines_to_text(unit)) <= TARGET_KG_CHUNK_CHARS:
        return [unit]
    if len(unit) == 1:
        return _split_long_text_line(unit[0])
    result: list[list[dict[str, int | str]]] = []
    for line in unit:
        result.extend(_split_long_text_line(line))
    return result


def split_body_lines_for_kg(
    body_lines: list[dict[str, int | str]],
) -> list[list[dict[str, int | str]]]:
    units = _body_line_units(body_lines)
    parts: list[list[dict[str, int | str]]] = []
    current: list[dict[str, int | str]] = []
    for unit in units:
        tentative = current + unit
        if current and len(body_lines_to_text(tentative)) > TARGET_KG_CHUNK_CHARS:
            parts.append(current)
            current = list(unit)
        else:
            current = tentative
    if current:
        parts.append(current)
    return [part for part in parts if body_lines_to_text(part)]


def line_range_for_part(
    part_lines: list[dict[str, int | str]], fallback_start: int, fallback_end: int
) -> tuple[int, int]:
    line_numbers = [int(line["line_no"]) for line in part_lines if line.get("line_no")]
    if not line_numbers:
        return fallback_start, fallback_end
    return min(line_numbers), max(line_numbers)


def images_for_line_range(images: list[dict], start_line: int, end_line: int) -> list[dict]:
    result = []
    for image in images:
        line_no = image.get("line_no")
        if isinstance(line_no, bool) or not isinstance(line_no, int):
            continue
        if start_line <= line_no <= end_line:
            result.append(image)
    return result


def file_path_for_range(source_path: Path, start_line: int, end_line: int) -> str:
    return f"{source_path.name}#L{start_line}-L{end_line}"


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
    meta = load_book_meta(args.meta) if args.meta else None
    if meta is not None:
        business = get_business_config(meta)
        paths = resolve_book_paths(meta)
        args.domain = resolve_config_value(args.domain, business["domain"], "MARKDOWN_GRAPH_DOMAIN", "industrial_training")
        args.subject = resolve_config_value(args.subject, business["subject"], "MARKDOWN_GRAPH_SUBJECT", "")
        if args.content_list_v2 is None:
            meta_v2 = paths["content_list_v2"]
            if meta_v2 is None or not meta_v2.is_file():
                raise ValueError("Step 5 需要 meta.inputs.content_list_v2 指向存在的文件，或显式传 --content-list-v2")
            args.content_list_v2 = meta_v2
        if args.mineru_dir is None:
            args.mineru_dir = paths["markdown"].parent
    else:
        args.domain = args.domain or os.getenv("MARKDOWN_GRAPH_DOMAIN", "industrial_training")
        args.subject = args.subject or os.getenv("MARKDOWN_GRAPH_SUBJECT", "")
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
    split_long_chunks_count = 0
    split_generated_chunks_count = 0
    for block in blocks:
        block_id = str(block["block_id"])
        chunk_id = block_id if block_id.startswith("chunk-") else f"chunk-{block_id}"
        domain = args.domain or str(block.get("domain") or "")
        subject = args.subject or str(block.get("subject") or "")
        source_path = Path(str(block["source_md_path"]))
        block_start_line = int(block["start_line"])
        block_end_line = int(block["end_line"])
        file_path = file_path_for_range(source_path, block_start_line, block_end_line)
        images = enhance_block_images(block, v2_assets, image_root)
        base_record = {
            "chunk_id": chunk_id,
            "doc_id": block["doc_id"],
            "block_id": block_id,
            "file_path": file_path,
            "source_md_path": str(source_path),
            "domain": domain,
            "subject": subject,
            "book_title": block.get("book_title", ""),
            "chapter_title": block.get("chapter_title", ""),
            "section_title": block.get("section_title", ""),
            "heading_path": block.get("heading_path", []),
            "start_line": block_start_line,
            "end_line": block_end_line,
            "images": images,
        }
        for field in BLOCK_METADATA_FIELDS:
            if field in block:
                base_record[field] = block.get(field)

        body_content = str(block.get("content") or "")
        if not base_record.get("content_scope"):
            structural_children = base_record.get("structural_children") or []
            if structural_children:
                base_record["content_scope"] = (
                    "direct" if body_content.strip() else "structural"
                )
            else:
                base_record["content_scope"] = "leaf"
        if (
            "should_extract_kg" not in base_record
            or base_record.get("should_extract_kg") is None
        ):
            base_record["should_extract_kg"] = (
                base_record["content_scope"] != "structural"
            )
        elif not isinstance(base_record["should_extract_kg"], bool):
            raise ValueError(
                f"block {block_id} 的 should_extract_kg 必须是 JSON boolean"
            )

        body_lines = clean_body_lines(body_content, block_start_line)
        cleaned_body = body_lines_to_text(body_lines)
        base_content = build_content(
            block,
            domain,
            subject,
            file_path,
            cleaned_body,
        )
        should_split = (
            base_record.get("should_extract_kg") is True
            and len(base_content) > HARD_MAX_KG_CHUNK_CHARS
            and bool(body_lines)
        )
        if should_split:
            parts = split_body_lines_for_kg(body_lines)
            if len(parts) > 1:
                split_long_chunks_count += 1
                split_generated_chunks_count += len(parts)
                split_count = len(parts)
                for split_index, part_lines in enumerate(parts, start=1):
                    part_start, part_end = line_range_for_part(
                        part_lines, block_start_line, block_end_line
                    )
                    split_suffix = f"__part_{split_index:02d}_of_{split_count:02d}"
                    part_chunk_id = f"{chunk_id}{split_suffix}"
                    part_block_id = f"{block_id}{split_suffix}"
                    if part_chunk_id in seen_ids:
                        raise ValueError(f"chunk_id 重复: {part_chunk_id}")
                    seen_ids.add(part_chunk_id)
                    part_file_path = file_path_for_range(
                        source_path, part_start, part_end
                    )
                    part_record = dict(base_record)
                    part_record.update(
                        {
                            "chunk_id": part_chunk_id,
                            "block_id": part_block_id,
                            "parent_chunk_id": chunk_id,
                            "parent_block_id": block_id,
                            "parent_file_path": file_path,
                            "split_index": split_index,
                            "split_count": split_count,
                            "split_reason": "content_too_long_for_kg",
                            "split_target_chars": TARGET_KG_CHUNK_CHARS,
                            "split_hard_max_chars": HARD_MAX_KG_CHUNK_CHARS,
                            "file_path": part_file_path,
                            "start_line": part_start,
                            "end_line": part_end,
                            "images": images_for_line_range(
                                images, part_start, part_end
                            ),
                        }
                    )
                    part_record["chunk_order_index"] = len(records)
                    kg_body = body_lines_to_text(part_lines)
                    part_record["kg_content"] = kg_body
                    part_record["content"] = build_content(
                        block,
                        domain,
                        subject,
                        part_file_path,
                        kg_body,
                        split_label=f"{split_index}/{split_count}",
                    )
                    part_record["evidence"] = _build_evidence(part_record)
                    records.append(part_record)
                continue

        if chunk_id in seen_ids:
            raise ValueError(f"chunk_id 重复: {chunk_id}")
        seen_ids.add(chunk_id)
        base_record["chunk_order_index"] = len(records)
        base_record["kg_content"] = cleaned_body
        base_record["content"] = base_content
        base_record["evidence"] = _build_evidence(base_record)
        records.append(base_record)
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
    chunks_with_image_caption_content = sum(
        "【图片说明】" in record["content"] for record in records
    )
    image_caption_content_lines = 0
    for record in records:
        if "【图片说明】\n" not in record["content"]:
            continue
        caption_section = record["content"].split("【图片说明】\n", 1)[1]
        caption_section = caption_section.split("\n\n【正文】", 1)[0]
        image_caption_content_lines += sum(
            line.startswith("- ") for line in caption_section.splitlines()
        )
    content_lengths = [len(record["content"]) for record in records]
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
        "chunks_with_image_caption_content": chunks_with_image_caption_content,
        "split_long_chunks_count": split_long_chunks_count,
        "split_generated_chunks_count": split_generated_chunks_count,
        "max_content_chars_after_split": max(content_lengths) if content_lengths else 0,
        "chunks_over_hard_max_after_split": sum(
            length > HARD_MAX_KG_CHUNK_CHARS for length in content_lengths
        ),
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
