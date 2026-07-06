#!/usr/bin/env python3
"""Build a complete structure plan by anchoring every catalog item to the body."""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from common import (
    PROJECT_DIR,
    ensure_dir,
    load_json,
    read_jsonl,
    slugify_for_id,
    write_jsonl,
)
from llm_client import call_llm_json

logger = logging.getLogger(__name__)

ANCHOR_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "catalog_index": {"type": "integer"},
        "matched": {"type": "boolean"},
        "matched_candidate_id": {
            "anyOf": [{"type": "string"}, {"type": "null"}]
        },
        "matched_start_line": {
            "anyOf": [{"type": "integer"}, {"type": "null"}]
        },
        "matched_title_text": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": [
        "catalog_index",
        "matched",
        "matched_candidate_id",
        "matched_start_line",
        "matched_title_text",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}

CONFIDENCE_REPAIR_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["confidence", "reason"],
    "additionalProperties": False,
}

ANCHOR_PROMPT = """你是教材目录项与正文标题的锚点匹配器。

任务：只从 candidate_title_lines 中选择当前目录项在正文中的标题锚点。

严格规则：
1. 不允许自由生成行号，只能原样选择一个 candidate_id 及其 line_no。
2. 目录和正文可能有 OCR 差异，应优先依据章节编号，其次依据标题语义和上下文。
3. 页码、图号、STEP 编号、图片文件名不能作为目录标题锚点。
4. 没有可靠候选时 matched=false，相关字段使用 null 或空字符串。
5. catalog_index 必须原样返回。
6. 必须输出下面列出的全部 7 个字段，不能改名、不能省略。
7. 只输出严格 JSON，不要输出 Markdown 或解释。

输出格式示例：
{
  "catalog_index": 13,
  "matched": true,
  "matched_candidate_id": "L550",
  "matched_start_line": 550,
  "matched_title_text": "# 2.1.2 珐琅烧制的辅助工具",
  "confidence": 0.94,
  "reason": "编号一致，标题仅存在 OCR 差异"
}
"""


class UnmatchedAnchorsError(RuntimeError):
    """Raised after artifacts are written when unmatched anchors block output."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按教材目录锚定正文并生成结构计划")
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--outline", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--anchor-plan", type=Path)
    parser.add_argument("--unmatched-report", type=Path)
    parser.add_argument("--use-existing-anchor-plan", type=Path)
    parser.add_argument(
        "--top-k-candidates",
        type=int,
        default=int(os.getenv("TOP_K_CANDIDATE_TITLE_LINES", "12")),
    )
    parser.add_argument(
        "--min-anchor-confidence",
        type=float,
        default=float(os.getenv("MIN_ANCHOR_CONFIDENCE", "0.65")),
    )
    parser.add_argument(
        "--max-unmatched",
        type=int,
        default=int(os.getenv("MAX_UNMATCHED_CATALOG_ITEMS", "0")),
    )
    return parser.parse_args()


def strip_title_markup(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^\s*#{1,6}\s*", "", value)
    value = re.sub(r"^\$\\[A-Za-z]+\$\s*", "", value)
    value = re.sub(r"\$?\\(?:spadesuit|heartsuit|diamondsuit|clubsuit)\$?", " ", value)
    value = re.sub(r"^[\s◆◇●○■□▪▫★☆♠♥♦♣※·•]+", "", value)
    return value.strip()


def normalized_title(text: str) -> str:
    value = strip_title_markup(text).lower()
    value = re.sub(r"\bchapter\s*0*\d+\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"第\s*[0-9一二三四五六七八九十百千万零〇两]+\s*[章节篇]", "", value)
    value = re.sub(r"^\s*\d+(?:\.\d+)*\s*", "", value)
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", value)


def extract_number_key(text: str) -> str:
    value = strip_title_markup(text)
    if re.match(r"^(?:STEP|步骤)\s*0*\d+\b", value, flags=re.IGNORECASE):
        return ""
    if re.match(r"^图\s*\d", value):
        return ""
    chapter = re.match(r"^第\s*0*(\d+)\s*[章节篇]", value)
    if chapter:
        return str(int(chapter.group(1)))
    english = re.match(r"^CHAPTER\s*0*(\d+)\b", value, flags=re.IGNORECASE)
    if english:
        return str(int(english.group(1)))
    numbered = re.match(r"^(\d+(?:\.\d+)*)(?![\d.])", value)
    if not numbered:
        return ""
    parts = numbered.group(1).split(".")
    return ".".join(str(int(part)) for part in parts)


def is_markdown_heading(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("#") and bool(stripped.lstrip("#").strip())


def is_chapter_marker(text: str) -> bool:
    return bool(re.search(r"\bCHAPTER\s*0*\d+\b", text, flags=re.IGNORECASE))


def is_short_title_like(text: str) -> bool:
    value = text.strip()
    if not value or len(value) > 80:
        return False
    if value.startswith(("![", "|", "```", "<")):
        return False
    if re.match(r"^图\s*\d", value) or re.match(r"^STEP\s*\d+\s*$", value, re.I):
        return False
    if value.endswith(("。", "！", "？", ".", "!", "?", "；", ";")):
        return False
    return len(value) <= 36 or bool(extract_number_key(value))


def nearby_nonempty(lines: list[dict], index: int, direction: int) -> list[dict]:
    result: list[dict] = []
    cursor = index + direction
    while 0 <= cursor < len(lines) and len(result) < 2:
        text = str(lines[cursor].get("text") or "").strip()
        if text:
            result.append(
                {"line_no": int(lines[cursor]["line_no"]), "text": text}
            )
        cursor += direction
    if direction < 0:
        result.reverse()
    return result


def build_global_candidates(
    prepared: dict, body_start_line: int
) -> list[dict[str, Any]]:
    lines: list[dict] = prepared["lines"]
    by_line: dict[int, dict[str, Any]] = {}
    priorities = {
        "short_title": 10,
        "numbered_line": 20,
        "markdown_heading": 30,
        "chapter_group": 40,
    }

    def add_candidate(
        index: int,
        candidate_type: str,
        *,
        line_span: tuple[int, int] | None = None,
        combined_text: str | None = None,
        number_key: str | None = None,
        reason: str,
    ) -> None:
        item = lines[index]
        line_no = int(item["line_no"])
        if line_no < body_start_line:
            return
        span = line_span or (line_no, line_no)
        text = combined_text or str(item["text"])
        candidate = {
            "candidate_id": f"L{line_no}",
            "line_no": line_no,
            "line_span": [span[0], span[1]],
            "text": text,
            "normalized_text": normalized_title(text),
            "number_key": number_key
            if number_key is not None
            else extract_number_key(text),
            "candidate_type": candidate_type,
            "score": 0.0,
            "reasons": [reason],
            "context_before": nearby_nonempty(lines, index, -1),
            "context_after": nearby_nonempty(lines, index, 1),
            "_priority": priorities[candidate_type],
        }
        existing = by_line.get(line_no)
        if existing is None or candidate["_priority"] > existing["_priority"]:
            if existing:
                candidate["reasons"] = existing["reasons"] + candidate["reasons"]
            by_line[line_no] = candidate
        elif reason not in existing["reasons"]:
            existing["reasons"].append(reason)

    body_index = body_start_line - 1
    for index in range(body_index, len(lines)):
        text = str(lines[index].get("text") or "")
        if is_markdown_heading(text):
            add_candidate(index, "markdown_heading", reason="Markdown heading")
        key = extract_number_key(text)
        if key and (is_markdown_heading(text) or is_short_title_like(text)):
            add_candidate(
                index,
                "numbered_line",
                number_key=key,
                reason=f"包含章节编号 {key}",
            )
        elif is_short_title_like(text):
            add_candidate(index, "short_title", reason="短标题式文本兜底")

    for index in range(body_index, len(lines)):
        text = str(lines[index].get("text") or "")
        marker_key = extract_number_key(text) if is_chapter_marker(text) else ""
        chapter_key = extract_number_key(text) if re.search(r"第\s*\d+\s*章", text) else ""
        key = marker_key or chapter_key
        if not key:
            continue
        group_indexes = [index]
        if marker_key:
            for previous in range(max(body_index, index - 6), index):
                previous_text = str(lines[previous].get("text") or "")
                if is_markdown_heading(previous_text):
                    group_indexes.append(previous)
        if chapter_key:
            for following in range(index + 1, min(len(lines), index + 7)):
                following_text = str(lines[following].get("text") or "")
                if is_markdown_heading(following_text) or is_chapter_marker(
                    following_text
                ):
                    group_indexes.append(following)
        group_indexes = sorted(set(group_indexes))
        start_index = group_indexes[0]
        end_index = group_indexes[-1]
        group_text = " | ".join(
            str(lines[group_index]["text"]).strip()
            for group_index in group_indexes
            if str(lines[group_index]["text"]).strip()
        )
        add_candidate(
            start_index,
            "chapter_group",
            line_span=(
                int(lines[start_index]["line_no"]),
                int(lines[end_index]["line_no"]),
            ),
            combined_text=group_text,
            number_key=key,
            reason=f"CHAPTER/章标题邻近组，编号 {key}",
        )

    candidates = sorted(by_line.values(), key=lambda item: item["line_no"])
    for candidate in candidates:
        candidate.pop("_priority", None)
    return candidates


def score_candidates(
    catalog_item: dict,
    catalog_index: int,
    catalog_count: int,
    candidates: list[dict[str, Any]],
    body_start_line: int,
    line_count: int,
) -> list[dict[str, Any]]:
    catalog_title = str(catalog_item["title"])
    catalog_normalized = normalized_title(
        str(catalog_item.get("normalized_title") or catalog_title)
    )
    catalog_key = extract_number_key(catalog_title)
    expected_ratio = (catalog_index - 1) / max(1, catalog_count - 1)
    scored: list[dict[str, Any]] = []
    for source in candidates:
        candidate = dict(source)
        reasons = list(candidate.get("reasons") or [])
        score = {
            "chapter_group": 28.0,
            "markdown_heading": 22.0,
            "numbered_line": 16.0,
            "short_title": 5.0,
        }.get(str(candidate.get("candidate_type")), 0.0)
        candidate_key = str(candidate.get("number_key") or "")
        if catalog_key and candidate_key == catalog_key:
            score += 120.0
            reasons.append(f"number_key 精确匹配 {catalog_key}")
        elif catalog_key and candidate_key:
            if candidate_key.startswith(catalog_key + ".") or catalog_key.startswith(
                candidate_key + "."
            ):
                score += 10.0
                reasons.append("number_key 存在父子前缀关系")

        candidate_normalized = str(candidate.get("normalized_text") or "")
        similarity = 0.0
        if catalog_normalized and candidate_normalized:
            similarity = difflib.SequenceMatcher(
                None, catalog_normalized, candidate_normalized
            ).ratio()
            score += similarity * 55.0
            if (
                catalog_normalized in candidate_normalized
                or candidate_normalized in catalog_normalized
            ):
                score += 18.0
                reasons.append("清洗后标题存在包含关系")
            reasons.append(f"标题相似度 {similarity:.3f}")

        actual_ratio = (int(candidate["line_no"]) - body_start_line) / max(
            1, line_count - body_start_line
        )
        position_score = max(0.0, 6.0 - abs(actual_ratio - expected_ratio) * 6.0)
        score += position_score
        candidate["score"] = round(score, 3)
        candidate["reasons"] = reasons
        scored.append(candidate)
    return sorted(scored, key=lambda item: (-item["score"], item["line_no"]))


def top_candidates_for_item(
    catalog_item: dict,
    catalog_index: int,
    catalog_count: int,
    candidates: list[dict[str, Any]],
    top_k: int,
    body_start_line: int,
    line_count: int,
) -> list[dict[str, Any]]:
    scored = score_candidates(
        catalog_item,
        catalog_index,
        catalog_count,
        candidates,
        body_start_line,
        line_count,
    )
    catalog_key = extract_number_key(str(catalog_item["title"]))
    exact = [item for item in scored if catalog_key and item["number_key"] == catalog_key]
    exact_ids = {item["candidate_id"] for item in exact}
    selected = list(exact)
    selected.extend(
        item for item in scored if item["candidate_id"] not in exact_ids
    )
    return selected[: max(top_k, len(exact))]


def unmatched_anchor(
    catalog_index: int,
    catalog_item: dict,
    candidates: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    return {
        "catalog_index": catalog_index,
        "catalog_level": int(catalog_item["level"]),
        "catalog_title": str(catalog_item["title"]),
        "catalog_number_key": extract_number_key(str(catalog_item["title"])),
        "matched": False,
        "matched_candidate_id": None,
        "matched_start_line": None,
        "matched_title_text": "",
        "confidence": 0.0,
        "reason": reason,
        "manual_override": False,
        "review_status": "unmatched",
        "anchor_status": "blocking_unmatched",
        "source_missing": False,
        "invalid": False,
        "top_candidates": candidates,
    }


def invalid_anchor(
    catalog_index: int,
    catalog_item: dict,
    candidates: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    anchor = unmatched_anchor(catalog_index, catalog_item, candidates, reason)
    anchor["review_status"] = "invalid"
    anchor["anchor_status"] = "invalid"
    anchor["invalid"] = True
    return anchor


def validate_source_missing_anchor(
    source: Any,
    catalog_index: int,
    catalog_item: dict,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Validate a manual source_missing record, or return None when not requested."""
    if not isinstance(source, dict) or source.get("review_status") != "source_missing":
        return None
    errors: list[str] = []
    if source.get("catalog_index") != catalog_index:
        errors.append(
            f"catalog_index 不一致: expected={catalog_index} "
            f"actual={source.get('catalog_index')!r}"
        )
    if source.get("matched") is not False:
        errors.append("source_missing 必须 matched=false")
    if source.get("manual_override") is not True:
        errors.append("source_missing 必须 manual_override=true")
    if source.get("missing_source") is not True:
        errors.append("source_missing 必须 missing_source=true")
    reason = str(source.get("reason") or "").strip()
    missing_reason = str(source.get("missing_reason") or "").strip()
    if not reason:
        errors.append("source_missing 的 reason 必须非空")
    if not missing_reason:
        errors.append("source_missing 的 missing_reason 必须非空")
    if errors:
        return invalid_anchor(
            catalog_index,
            catalog_item,
            candidates,
            "；".join(errors),
        )
    return {
        "catalog_index": catalog_index,
        "catalog_level": int(catalog_item["level"]),
        "catalog_title": str(catalog_item["title"]),
        "catalog_number_key": extract_number_key(str(catalog_item["title"])),
        "matched": False,
        "matched_candidate_id": None,
        "matched_start_line": None,
        "matched_title_text": "",
        "confidence": 0.0,
        "reason": reason or missing_reason,
        "missing_reason": missing_reason,
        "missing_source": True,
        "manual_override": True,
        "review_status": "source_missing",
        "anchor_status": "source_missing",
        "source_missing": True,
        "invalid": False,
        "top_candidates": candidates,
    }


def normalize_anchor_response_aliases(
    source: Any, candidates: list[dict[str, Any]]
) -> Any:
    """Normalize common model aliases without relaxing candidate validation."""
    if not isinstance(source, dict):
        return source
    result = dict(source)
    if "matched_candidate_id" not in result:
        result["matched_candidate_id"] = result.get("candidate_id")
    if "matched_start_line" not in result:
        result["matched_start_line"] = result.get(
            "matched_line_no", result.get("line_no")
        )
    if "matched_title_text" not in result:
        result["matched_title_text"] = result.get(
            "matched_title", result.get("title", "")
        )
    candidate = next(
        (
            item
            for item in candidates
            if item["candidate_id"] == result.get("matched_candidate_id")
        ),
        None,
    )
    if candidate and not str(result.get("matched_title_text") or "").strip():
        result["matched_title_text"] = candidate["text"]
    return result


def repairable_selected_candidate(
    source: Any,
    catalog_index: int,
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    normalized = normalize_anchor_response_aliases(source, candidates)
    if not isinstance(normalized, dict):
        return None
    if normalized.get("catalog_index") != catalog_index:
        return None
    if normalized.get("matched") is not True:
        return None
    candidate_id = normalized.get("matched_candidate_id")
    start_line = normalized.get("matched_start_line")
    candidate = next(
        (item for item in candidates if item["candidate_id"] == candidate_id),
        None,
    )
    if candidate is None or isinstance(start_line, bool) or not isinstance(
        start_line, int
    ):
        return None
    if int(candidate["line_no"]) != start_line:
        return None
    return normalized, candidate


def needs_confidence_repair(source: dict[str, Any]) -> bool:
    try:
        confidence = float(source.get("confidence"))
    except (TypeError, ValueError):
        return True
    return not 0 <= confidence <= 1 or not str(source.get("reason") or "").strip()


def build_confidence_repair_prompt(
    catalog_index: int,
    catalog_item: dict,
    candidate: dict[str, Any],
    original_result: dict[str, Any],
) -> str:
    payload = {
        "catalog_index": catalog_index,
        "catalog_item": catalog_item,
        "locked_candidate": candidate,
        "original_result": original_result,
    }
    return (
        "你是 JSON 格式修复器。候选已经锁定，不要重新选择候选，也不要输出行号。\n"
        "请根据目录项、锁定候选的编号/标题相似度和上下文，补充匹配置信度与简短理由。\n"
        "confidence 必须是 0 到 1 之间的数字。只输出下面两个字段的严格 JSON：\n"
        '{"confidence": 0.94, "reason": "编号一致，标题存在 OCR 差异"}\n\n'
        f"【输入】\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def validate_anchor(
    source: Any,
    catalog_index: int,
    catalog_item: dict,
    candidates: list[dict[str, Any]],
    *,
    body_start_line: int,
    line_count: int,
    min_confidence: float,
) -> dict[str, Any]:
    source = normalize_anchor_response_aliases(source, candidates)
    if not isinstance(source, dict):
        return unmatched_anchor(
            catalog_index, catalog_item, candidates, "LLM/人工锚点不是 JSON 对象"
        )
    base = unmatched_anchor(catalog_index, catalog_item, candidates, "")
    if source.get("catalog_index") != catalog_index:
        base["reason"] = (
            f"catalog_index 不一致: expected={catalog_index} "
            f"actual={source.get('catalog_index')!r}"
        )
        return base
    if source.get("matched") is not True:
        base["reason"] = str(source.get("reason") or "matched=false")
        return base
    try:
        confidence = float(source.get("confidence"))
    except (TypeError, ValueError):
        base["reason"] = "confidence 不是数字"
        return base
    if not 0 <= confidence <= 1:
        base["reason"] = "confidence 必须在 0 到 1 之间"
        return base
    if confidence < min_confidence:
        base["confidence"] = confidence
        base["reason"] = (
            f"confidence={confidence:.3f} 低于阈值 {min_confidence:.3f}"
        )
        return base

    candidate_id = source.get("matched_candidate_id")
    start_line = source.get("matched_start_line")
    manual_override = source.get("manual_override") is True
    if isinstance(start_line, bool) or not isinstance(start_line, int):
        base["reason"] = "matched_start_line 不是整数"
        return base
    if manual_override:
        if not body_start_line <= start_line <= line_count:
            base["reason"] = (
                f"人工 matched_start_line 越界: {start_line}，"
                f"正文范围 {body_start_line}-{line_count}"
            )
            return base
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            base["reason"] = "人工 matched_candidate_id 必须是非空字符串"
            return base
        matched_text = str(source.get("matched_title_text") or "")
        line_span = [start_line, start_line]
    else:
        by_id = {item["candidate_id"]: item for item in candidates}
        candidate_lines = {int(item["line_no"]) for item in candidates}
        if candidate_id not in by_id:
            base["reason"] = "matched_candidate_id 不在 top_candidates 中"
            return base
        if start_line not in candidate_lines:
            base["reason"] = "matched_start_line 不在候选行号中"
            return base
        candidate = by_id[candidate_id]
        if int(candidate["line_no"]) != start_line:
            base["reason"] = "matched_candidate_id 对应行号与 matched_start_line 不一致"
            return base
        catalog_number_key = extract_number_key(str(catalog_item["title"]))
        candidate_number_key = str(candidate.get("number_key") or "")
        if (
            catalog_number_key
            and candidate_number_key
            and catalog_number_key != candidate_number_key
        ):
            base["confidence"] = confidence
            base["reason"] = (
                f"目录 number_key={catalog_number_key} 与候选 "
                f"number_key={candidate_number_key} 不一致"
            )
            return base
        matched_text = str(candidate["text"])
        line_span = list(candidate["line_span"])

    base.update(
        {
            "matched": True,
            "matched_candidate_id": candidate_id,
            "matched_start_line": start_line,
            "matched_title_text": matched_text,
            "matched_line_span": line_span,
            "confidence": confidence,
            "reason": str(source.get("reason") or "锚点匹配通过"),
            "manual_override": manual_override,
            "review_status": str(
                source.get("review_status")
                or ("manual_corrected" if manual_override else "llm_matched")
            ),
            "anchor_status": "matched",
            "source_missing": False,
            "invalid": False,
        }
    )
    return base


def validate_anchor_order(anchors: list[dict[str, Any]]) -> None:
    """Keep the largest globally increasing anchor set without cascade drops."""
    matched_positions = [
        position for position, anchor in enumerate(anchors) if anchor["matched"]
    ]
    if len(matched_positions) < 2:
        return

    best_counts = [1] * len(matched_positions)
    best_confidences = [
        float(anchors[position].get("confidence") or 0)
        for position in matched_positions
    ]
    previous_positions = [-1] * len(matched_positions)
    for current_pos, current_anchor_pos in enumerate(matched_positions):
        current_line = int(anchors[current_anchor_pos]["matched_start_line"])
        for earlier_pos in range(current_pos):
            earlier_anchor_pos = matched_positions[earlier_pos]
            earlier_line = int(
                anchors[earlier_anchor_pos]["matched_start_line"]
            )
            if earlier_line >= current_line:
                continue
            candidate_count = best_counts[earlier_pos] + 1
            candidate_confidence = best_confidences[earlier_pos] + float(
                anchors[current_anchor_pos].get("confidence") or 0
            )
            if (candidate_count, candidate_confidence) > (
                best_counts[current_pos],
                best_confidences[current_pos],
            ):
                best_counts[current_pos] = candidate_count
                best_confidences[current_pos] = candidate_confidence
                previous_positions[current_pos] = earlier_pos

    best_end = max(
        range(len(matched_positions)),
        key=lambda position: (best_counts[position], best_confidences[position]),
    )
    kept_anchor_positions: set[int] = set()
    cursor = best_end
    while cursor >= 0:
        kept_anchor_positions.add(matched_positions[cursor])
        cursor = previous_positions[cursor]

    for anchor_position in matched_positions:
        if anchor_position in kept_anchor_positions:
            continue
        anchor = anchors[anchor_position]
        rejected_id = anchor["matched_candidate_id"]
        rejected_line = int(anchor["matched_start_line"])
        previous_kept = next(
            (
                anchors[position]
                for position in range(anchor_position - 1, -1, -1)
                if position in kept_anchor_positions
            ),
            None,
        )
        next_kept = next(
            (
                anchors[position]
                for position in range(anchor_position + 1, len(anchors))
                if position in kept_anchor_positions
            ),
            None,
        )
        neighbor_text = "，".join(
            text
            for text in (
                (
                    f"前一保留锚点 index={previous_kept['catalog_index']} "
                    f"line={previous_kept['matched_start_line']}"
                    if previous_kept
                    else ""
                ),
                (
                    f"后一保留锚点 index={next_kept['catalog_index']} "
                    f"line={next_kept['matched_start_line']}"
                    if next_kept
                    else ""
                ),
            )
            if text
        )
        anchor["matched"] = False
        anchor["review_status"] = "unmatched"
        anchor["anchor_status"] = "blocking_unmatched"
        anchor["source_missing"] = False
        anchor["invalid"] = False
        anchor["reason"] = (
            "锚点不属于目录顺序的全局最长严格递增序列: "
            f"candidate={rejected_id} line={rejected_line}"
            + (f"；{neighbor_text}" if neighbor_text else "")
        )
        anchor["rejected_candidate_id"] = rejected_id
        anchor["rejected_start_line"] = rejected_line
        anchor["matched_candidate_id"] = None
        anchor["matched_start_line"] = None
        anchor["matched_title_text"] = ""


def build_anchor_prompt(
    catalog_index: int, catalog_item: dict, candidates: list[dict[str, Any]]
) -> str:
    payload = {
        "catalog_index": catalog_index,
        "catalog_item": catalog_item,
        "candidate_title_lines": candidates,
    }
    return f"{ANCHOR_PROMPT}\n\n【输入】\n{json.dumps(payload, ensure_ascii=False, indent=2)}"


async def generate_anchors(
    catalog: list[dict],
    candidate_sets: list[list[dict[str, Any]]],
    *,
    raw_dir: Path,
    book_stem: str,
    body_start_line: int,
    line_count: int,
    min_confidence: float,
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for catalog_index, (catalog_item, candidates) in enumerate(
        zip(catalog, candidate_sets), start=1
    ):
        raw_path = raw_dir / f"{book_stem}.catalog_{catalog_index:04d}.raw.txt"
        repaired = False
        try:
            result = await call_llm_json(
                build_anchor_prompt(catalog_index, catalog_item, candidates),
                raw_output_path=raw_path,
                guided_json_schema=ANCHOR_JSON_SCHEMA,
                max_tokens=1024,
            )
            repairable = repairable_selected_candidate(
                result, catalog_index, candidates
            )
            if repairable is not None:
                normalized_result, selected_candidate = repairable
                if needs_confidence_repair(normalized_result):
                    repair_path = raw_dir / (
                        f"{book_stem}.catalog_{catalog_index:04d}.repair.raw.txt"
                    )
                    repair_result = await call_llm_json(
                        build_confidence_repair_prompt(
                            catalog_index,
                            catalog_item,
                            selected_candidate,
                            normalized_result,
                        ),
                        raw_output_path=repair_path,
                        guided_json_schema=CONFIDENCE_REPAIR_JSON_SCHEMA,
                        max_tokens=256,
                    )
                    if not isinstance(repair_result, dict):
                        raise ValueError("格式修复结果不是 JSON 对象")
                    normalized_result["confidence"] = repair_result.get("confidence")
                    normalized_result["reason"] = repair_result.get("reason")
                    result = normalized_result
                    repaired = True
            anchor = validate_anchor(
                result,
                catalog_index,
                catalog_item,
                candidates,
                body_start_line=body_start_line,
                line_count=line_count,
                min_confidence=min_confidence,
            )
        except Exception as exc:
            anchor = unmatched_anchor(
                catalog_index,
                catalog_item,
                candidates,
                f"LLM 调用或 JSON 解析失败: {type(exc).__name__}: {exc}",
            )
        anchors.append(anchor)
        logger.info(
            "目录锚点完成 | index=%d/%d matched=%s line=%s confidence=%.3f "
            "format_repair=%s",
            catalog_index,
            len(catalog),
            anchor["matched"],
            anchor["matched_start_line"],
            float(anchor["confidence"]),
            repaired,
        )
    validate_anchor_order(anchors)
    return anchors


def load_existing_anchors(
    path: Path,
    catalog: list[dict],
    candidate_sets: list[list[dict[str, Any]]],
    *,
    body_start_line: int,
    line_count: int,
    min_confidence: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    records = read_jsonl(path)
    by_index: dict[int, dict] = {}
    duplicates: set[int] = set()
    extra_notes: list[str] = []
    for record in records:
        index = record.get("catalog_index")
        if isinstance(index, bool) or not isinstance(index, int):
            extra_notes.append(f"忽略无效 catalog_index: {index!r}")
            continue
        if index in by_index:
            duplicates.add(index)
        else:
            by_index[index] = record
    anchors: list[dict[str, Any]] = []
    for catalog_index, (catalog_item, candidates) in enumerate(
        zip(catalog, candidate_sets), start=1
    ):
        if catalog_index in duplicates:
            anchor = invalid_anchor(
                catalog_index, catalog_item, candidates, "已有 anchor plan 中该索引重复"
            )
        elif catalog_index not in by_index:
            anchor = invalid_anchor(
                catalog_index, catalog_item, candidates, "已有 anchor plan 缺少该目录项"
            )
        else:
            source = by_index[catalog_index]
            anchor = validate_source_missing_anchor(
                source,
                catalog_index,
                catalog_item,
                candidates,
            )
            if anchor is None:
                anchor = validate_anchor(
                    source,
                    catalog_index,
                    catalog_item,
                    candidates,
                    body_start_line=body_start_line,
                    line_count=line_count,
                    min_confidence=min_confidence,
                )
        anchors.append(anchor)
    out_of_range = sorted(index for index in by_index if not 1 <= index <= len(catalog))
    if out_of_range:
        extra_notes.append(f"忽略目录范围外索引: {out_of_range}")
    validate_anchor_order(anchors)
    return anchors, extra_notes


def catalog_parent_indexes(catalog: list[dict]) -> list[int | None]:
    parents: list[int | None] = []
    level_stack: list[tuple[int, int]] = []
    for index, item in enumerate(catalog, start=1):
        level = int(item["level"])
        while level_stack and level_stack[-1][0] >= level:
            level_stack.pop()
        parents.append(level_stack[-1][1] if level_stack else None)
        level_stack.append((level, index))
    return parents


def is_direct_body_text(text: str) -> bool:
    value = text.strip()
    if not value or is_markdown_heading(value) or is_chapter_marker(value):
        return False
    if value.startswith("!["):
        return False
    return True


def build_structure_plan(
    prepared: dict, outline: dict, anchors: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    catalog: list[dict] = outline["chapter_catalog"]
    lines: list[dict] = prepared["lines"]
    line_count = int(prepared["line_count"])
    book_stem = str(prepared["book_stem"])
    parents = catalog_parent_indexes(catalog)
    anchor_by_index = {
        int(anchor["catalog_index"]): anchor
        for anchor in anchors
        if anchor["matched"]
    }
    block_ids = {
        index: (
            f"{slugify_for_id(book_stem)}__catalog_{index:04d}__"
            f"{slugify_for_id(str(catalog[index - 1]['title']))}"
        )
        for index in anchor_by_index
    }
    image_line_numbers = sorted(
        int(image["line_no"]) for image in prepared.get("image_refs", [])
    )
    records: list[dict[str, Any]] = []

    for catalog_index in sorted(anchor_by_index):
        item = catalog[catalog_index - 1]
        level = int(item["level"])
        anchor = anchor_by_index[catalog_index]
        start_line = int(anchor["matched_start_line"])
        subtree_end_index = len(catalog) + 1
        for later_index in range(catalog_index + 1, len(catalog) + 1):
            if int(catalog[later_index - 1]["level"]) <= level:
                subtree_end_index = later_index
                break
        matched_descendants = [
            later_index
            for later_index in range(catalog_index + 1, subtree_end_index)
            if later_index in anchor_by_index
        ]
        if matched_descendants:
            boundary_index = matched_descendants[0]
        else:
            boundary_index = next(
                (
                    later_index
                    for later_index in range(subtree_end_index, len(catalog) + 1)
                    if later_index in anchor_by_index
                    and int(catalog[later_index - 1]["level"]) <= level
                ),
                None,
            )
        end_line = (
            int(anchor_by_index[boundary_index]["matched_start_line"]) - 1
            if boundary_index is not None
            else line_count
        )
        if end_line < start_line:
            raise ValueError(
                f"目录项 {catalog_index} block 范围无效: {start_line}-{end_line}"
            )

        child_indexes = [
            child_index
            for child_index, parent_index in enumerate(parents, start=1)
            if parent_index == catalog_index and child_index in anchor_by_index
        ]
        has_catalog_children = any(
            parent_index == catalog_index for parent_index in parents
        )
        if not has_catalog_children:
            content_scope = "leaf"
            should_extract_kg = True
        else:
            direct_lines = lines[start_line - 1 : end_line]
            has_direct_content = any(
                is_direct_body_text(str(line.get("text") or ""))
                for line in direct_lines
            )
            content_scope = "direct" if has_direct_content else "structural"
            should_extract_kg = has_direct_content

        ancestry: list[int] = []
        cursor: int | None = catalog_index
        while cursor is not None:
            ancestry.append(cursor)
            cursor = parents[cursor - 1]
        ancestry.reverse()
        heading_path = [str(catalog[index - 1]["title"]) for index in ancestry]
        chapter_index = next(
            (
                index
                for index in ancestry
                if int(catalog[index - 1]["level"]) == 1
            ),
            ancestry[0],
        )
        block_type = "chapter" if level == 1 else "section" if level == 2 else "subsection"
        image_lines = [
            line_no
            for line_no in image_line_numbers
            if start_line <= line_no <= end_line
        ]
        structural_children = [
            {
                "catalog_index": child_index,
                "block_id": block_ids[child_index],
                "title": str(catalog[child_index - 1]["title"]),
            }
            for child_index in child_indexes
        ]
        range_reason = (
            "范围截止到第一个 matched 子项标题之前"
            if matched_descendants
            else "范围截止到下一个 matched 同级或更高层级标题之前"
        )
        records.append(
            {
                "block_id": block_ids[catalog_index],
                "block_type": block_type,
                "book_title": str(outline.get("book_title") or book_stem),
                "chapter_title": str(catalog[chapter_index - 1]["title"]),
                "section_title": str(item["title"]) if level >= 2 else "",
                "heading_path": heading_path,
                "start_line": start_line,
                "end_line": end_line,
                "image_lines": image_lines,
                "confidence": float(anchor["confidence"]),
                "reason": f"{anchor['reason']}；{range_reason}",
                "catalog_index": catalog_index,
                "catalog_level": level,
                "catalog_title": str(item["title"]),
                "matched_title_line": start_line,
                "matched_title_text": str(anchor["matched_title_text"]),
                "content_scope": content_scope,
                "should_extract_kg": should_extract_kg,
                "structural_children": structural_children,
            }
        )
    return records


def write_unmatched_report(
    path: Path,
    anchors: list[dict[str, Any]],
    *,
    max_unmatched: int,
    manual_mode: bool,
    notes: list[str] | None = None,
) -> None:
    matched = [anchor for anchor in anchors if anchor["matched"]]
    source_missing = [
        anchor for anchor in anchors if anchor.get("anchor_status") == "source_missing"
    ]
    invalid = [
        anchor for anchor in anchors if anchor.get("anchor_status") == "invalid"
    ]
    blocking_unmatched = [
        anchor
        for anchor in anchors
        if not anchor["matched"]
        and anchor.get("anchor_status") not in {"source_missing", "invalid"}
    ]
    content = [
        "# 目录锚点未匹配报告",
        "",
        f"- 目录项总数: {len(anchors)}",
        f"- matched_count: {len(matched)}",
        f"- source_missing_count: {len(source_missing)}",
        f"- blocking_unmatched_count: {len(blocking_unmatched)}",
        f"- invalid_count: {len(invalid)}",
        f"- max_unmatched: {max_unmatched}",
        f"- 模式: {'人工复核' if manual_mode else 'LLM'}",
    ]
    if notes:
        content.extend(["", "## 读取提示", ""])
        content.extend(f"- {note}" for note in notes)
    if source_missing:
        content.extend(["", "## Source Missing 目录项", ""])
        for anchor in source_missing:
            content.extend(
                [
                    f"### {anchor['catalog_index']}. {anchor['catalog_title']}",
                    "",
                    f"- catalog_index: {anchor['catalog_index']}",
                    f"- title: {anchor['catalog_title']}",
                    f"- level: {anchor['catalog_level']}",
                    f"- number_key: {anchor['catalog_number_key'] or '<empty>'}",
                    f"- reason: {anchor['reason']}",
                    f"- missing_reason: {anchor.get('missing_reason') or '<empty>'}",
                    "",
                ]
            )
    if invalid:
        content.extend(["", "## Invalid 目录项", ""])
        for anchor in invalid:
            content.extend(
                [
                    f"### {anchor['catalog_index']}. {anchor['catalog_title']}",
                    "",
                    f"- 原因: {anchor['reason']}",
                    f"- level: {anchor['catalog_level']}",
                    f"- number_key: {anchor['catalog_number_key'] or '<empty>'}",
                    "",
                ]
            )
    if blocking_unmatched:
        content.extend(["", "## Unmatched 目录项", ""])
        for anchor in blocking_unmatched:
            content.extend(
                [
                    f"### {anchor['catalog_index']}. {anchor['catalog_title']}",
                    "",
                    f"- 原因: {anchor['reason']}",
                    f"- level: {anchor['catalog_level']}",
                    f"- number_key: {anchor['catalog_number_key'] or '<empty>'}",
                    "- top candidates:",
                ]
            )
            candidates = anchor.get("top_candidates") or []
            if candidates:
                content.extend(
                    f"  - {item['candidate_id']} / L{item['line_no']} / "
                    f"score={item['score']}: {item['text']}"
                    for item in candidates[:5]
                )
            else:
                content.append("  - <none>")
            content.append("")
    if not source_missing and not invalid and not blocking_unmatched:
        content.extend(["", "全部目录项均已匹配。", ""])
    ensure_dir(path.parent)
    path.write_text("\n".join(content).rstrip() + "\n", encoding="utf-8")


def validate_inputs(prepared: Any, outline: Any, args: argparse.Namespace) -> None:
    if not isinstance(prepared, dict) or not isinstance(outline, dict):
        raise ValueError("prepared 和 outline 顶层必须是 JSON 对象")
    if not isinstance(prepared.get("lines"), list) or not prepared["lines"]:
        raise ValueError("prepared.lines 必须是非空数组")
    catalog = outline.get("chapter_catalog")
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("outline.chapter_catalog 必须是非空数组")
    for index, item in enumerate(catalog, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"chapter_catalog[{index}] 必须是对象")
        if not str(item.get("title") or "").strip():
            raise ValueError(f"chapter_catalog[{index}].title 为空")
        level = item.get("level")
        if isinstance(level, bool) or not isinstance(level, int) or level <= 0:
            raise ValueError(f"chapter_catalog[{index}].level 必须是正整数")
    body_start = outline.get("body_start_line")
    if isinstance(body_start, bool) or not isinstance(body_start, int):
        raise ValueError("outline.body_start_line 必须是整数")
    if not 1 <= body_start <= int(prepared["line_count"]):
        raise ValueError("outline.body_start_line 越界")
    if args.top_k_candidates <= 0:
        raise ValueError("--top-k-candidates 必须大于 0")
    if not 0 <= args.min_anchor_confidence <= 1:
        raise ValueError("--min-anchor-confidence 必须在 0 到 1 之间")
    if args.max_unmatched < 0:
        raise ValueError("--max-unmatched 不能小于 0")


async def run(args: argparse.Namespace) -> Path:
    prepared_path = args.prepared.expanduser().resolve()
    outline_path = args.outline.expanduser().resolve()
    prepared = load_json(prepared_path)
    outline = load_json(outline_path)
    validate_inputs(prepared, outline, args)

    book_stem = str(prepared["book_stem"])
    output_dir = PROJECT_DIR / "outputs/03_structure_plan"
    output = (
        args.output
        or output_dir / f"{book_stem}.structure_plan.raw.jsonl"
    ).expanduser().resolve()
    anchor_plan = (
        args.anchor_plan
        or output_dir / f"{book_stem}.catalog_anchor_plan.jsonl"
    ).expanduser().resolve()
    unmatched_report = (
        args.unmatched_report
        or output_dir / f"{book_stem}.unmatched_report.md"
    ).expanduser().resolve()

    catalog: list[dict] = outline["chapter_catalog"]
    body_start_line = int(outline["body_start_line"])
    line_count = int(prepared["line_count"])
    global_candidates = build_global_candidates(prepared, body_start_line)
    candidate_sets = [
        top_candidates_for_item(
            item,
            catalog_index,
            len(catalog),
            global_candidates,
            args.top_k_candidates,
            body_start_line,
            line_count,
        )
        for catalog_index, item in enumerate(catalog, start=1)
    ]
    logger.info(
        "候选标题池完成 | global=%d catalog_items=%d top_k=%d",
        len(global_candidates),
        len(catalog),
        args.top_k_candidates,
    )

    manual_mode = args.use_existing_anchor_plan is not None
    report_notes: list[str] = []
    if manual_mode:
        existing_path = args.use_existing_anchor_plan.expanduser().resolve()
        if not existing_path.is_file():
            raise FileNotFoundError(f"已有 anchor plan 不存在: {existing_path}")
        anchors, report_notes = load_existing_anchors(
            existing_path,
            catalog,
            candidate_sets,
            body_start_line=body_start_line,
            line_count=line_count,
            min_confidence=args.min_anchor_confidence,
        )
        logger.info("人工复核模式 | anchor_plan=%s，已跳过全部 LLM 调用", existing_path)
    else:
        raw_dir = ensure_dir(output.parent / "raw_anchors")
        anchors = await generate_anchors(
            catalog,
            candidate_sets,
            raw_dir=raw_dir,
            book_stem=book_stem,
            body_start_line=body_start_line,
            line_count=line_count,
            min_confidence=args.min_anchor_confidence,
        )
        write_jsonl(anchor_plan, anchors)
        logger.info("anchor plan 已写入 | %s", anchor_plan)

    matched_count = sum(anchor["matched"] for anchor in anchors)
    source_missing_count = sum(
        anchor.get("anchor_status") == "source_missing" for anchor in anchors
    )
    invalid_count = sum(
        anchor.get("anchor_status") == "invalid" for anchor in anchors
    )
    blocking_unmatched_count = sum(
        not anchor["matched"]
        and anchor.get("anchor_status") not in {"source_missing", "invalid"}
        for anchor in anchors
    )
    write_unmatched_report(
        unmatched_report,
        anchors,
        max_unmatched=args.max_unmatched,
        manual_mode=manual_mode,
        notes=report_notes,
    )
    blocking_count = blocking_unmatched_count + invalid_count
    must_fail = blocking_count > args.max_unmatched
    if must_fail:
        output.unlink(missing_ok=True)
        raise UnmatchedAnchorsError(
            f"目录锚点存在 blocking_unmatched={blocking_unmatched_count}、"
            f"invalid={invalid_count}，合计 {blocking_count}；"
            f"允许上限 {args.max_unmatched}；报告: {unmatched_report}"
        )

    structure_plan = build_structure_plan(prepared, outline, anchors)
    write_jsonl(output, structure_plan)
    logger.info(
        "目录驱动 structure plan 完成 | catalog=%d matched=%d "
        "source_missing=%d blocking_unmatched=%d invalid=%d blocks=%d output=%s",
        len(catalog),
        matched_count,
        source_missing_count,
        blocking_unmatched_count,
        invalid_count,
        len(structure_plan),
        output,
    )
    return output


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    env_path = (PROJECT_DIR.parent / ".env").resolve()
    if env_path.is_file():
        load_dotenv(env_path, override=False)
        logger.info("loaded env: %s", env_path)
    try:
        args = parse_args()
        asyncio.run(run(args))
        return 0
    except UnmatchedAnchorsError as exc:
        logger.error("structure plan 未通过: %s", exc)
        return 2
    except Exception as exc:
        logger.exception("catalog structure plan 失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
