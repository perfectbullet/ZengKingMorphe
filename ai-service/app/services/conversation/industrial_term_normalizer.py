"""工训术语的确定性查询归一化。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

RULES_PATH = Path(__file__).with_name("industrial_asr_corrections.json")


@dataclass(frozen=True)
class IndustrialTermMatch:
    rule_id: str
    source: str
    target: str
    start: int
    end: int
    priority: int


@dataclass(frozen=True)
class IndustrialTermNormalizationResult:
    original: str
    normalized: str
    applied: bool
    matches: tuple[IndustrialTermMatch, ...]


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    canonical: str
    variants: tuple[str, ...]
    match_mode: str
    priority: int
    context_any: tuple[str, ...]
    context_all: tuple[str, ...]
    context_none: tuple[str, ...]
    rule_index: int


@dataclass(frozen=True)
class _Candidate:
    match: IndustrialTermMatch
    rule_index: int
    variant_index: int


def _enabled() -> bool:
    return os.getenv("INDUSTRIAL_TERM_NORMALIZATION_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }


def _string_list(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        return None
    return tuple(item.strip() for item in value)


def _parse_rule(raw_rule: Any, index: int, seen_ids: set[str]) -> _Rule | None:
    label = raw_rule.get("id") if isinstance(raw_rule, dict) else index
    if not isinstance(raw_rule, dict):
        logger.warning(f"Invalid industrial term rule skipped | rule={label!r}")
        return None

    rule_id = raw_rule.get("id")
    enabled = raw_rule.get("enabled")
    canonical = raw_rule.get("canonical")
    variants = _string_list(raw_rule.get("variants"))
    match_mode = raw_rule.get("match_mode")
    priority = raw_rule.get("priority")
    if (
        not isinstance(rule_id, str)
        or not rule_id.strip()
        or rule_id in seen_ids
        or not isinstance(enabled, bool)
        or not isinstance(canonical, str)
        or not canonical.strip()
        or variants is None
        or any(variant == canonical.strip() for variant in variants)
        or match_mode not in {"direct", "contextual"}
        or not isinstance(priority, int)
        or isinstance(priority, bool)
    ):
        logger.warning(f"Invalid industrial term rule skipped | rule={label!r}")
        return None
    if not enabled:
        seen_ids.add(rule_id)
        return None

    context_any: tuple[str, ...] = ()
    context_all: tuple[str, ...] = ()
    context_none: tuple[str, ...] = ()
    if match_mode == "contextual":
        context = raw_rule.get("context")
        if not isinstance(context, dict):
            logger.warning(f"Invalid industrial term contextual rule skipped | rule={rule_id!r}")
            return None
        context_any = _string_list(context.get("any", []))
        context_all = _string_list(context.get("all", []))
        context_none = _string_list(context.get("none", []))
        if context_any is None or context_all is None or context_none is None:
            logger.warning(f"Invalid industrial term context skipped | rule={rule_id!r}")
            return None

    seen_ids.add(rule_id)
    return _Rule(
        rule_id=rule_id,
        canonical=canonical.strip(),
        variants=variants,
        match_mode=match_mode,
        priority=priority,
        context_any=context_any,
        context_all=context_all,
        context_none=context_none,
        rule_index=index,
    )


def _load_rules_from_path(path: Path) -> tuple[_Rule, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning(f"Industrial term rules file not found | path={path}")
        return ()
    except (OSError, json.JSONDecodeError):
        logger.exception(f"Industrial term rules load failed | path={path}")
        return ()

    if not isinstance(payload, dict) or payload.get("version") != 1:
        logger.warning(f"Invalid industrial term rules root | path={path}")
        return ()
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list):
        logger.warning(f"Invalid industrial term rules list | path={path}")
        return ()

    seen_ids: set[str] = set()
    rules = tuple(
        rule
        for index, raw_rule in enumerate(raw_rules)
        if (rule := _parse_rule(raw_rule, index, seen_ids)) is not None
    )
    variant_targets: dict[tuple[str, int], str] = {}
    for rule in rules:
        for variant in rule.variants:
            key = (variant, rule.priority)
            previous = variant_targets.setdefault(key, rule.canonical)
            if previous != rule.canonical:
                logger.warning(
                    "Industrial term variant has ambiguous same-priority targets | "
                    f"variant={variant!r} | priority={rule.priority}"
                )
    logger.info(f"Industrial term rules loaded | valid_rules={len(rules)}")
    return rules


@lru_cache(maxsize=1)
def _load_rules() -> tuple[_Rule, ...]:
    return _load_rules_from_path(RULES_PATH)


def _context_matches(rule: _Rule, query: str) -> bool:
    if rule.match_mode == "direct":
        return True
    return (
        (not rule.context_any or any(term in query for term in rule.context_any))
        and all(term in query for term in rule.context_all)
        and not any(term in query for term in rule.context_none)
    )


def _candidates(query: str, rules: tuple[_Rule, ...]) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for rule in rules:
        if not _context_matches(rule, query):
            continue
        for variant_index, variant in enumerate(rule.variants):
            start = query.find(variant)
            while start != -1:
                candidates.append(
                    _Candidate(
                        match=IndustrialTermMatch(
                            rule_id=rule.rule_id,
                            source=variant,
                            target=rule.canonical,
                            start=start,
                            end=start + len(variant),
                            priority=rule.priority,
                        ),
                        rule_index=rule.rule_index,
                        variant_index=variant_index,
                    )
                )
                start = query.find(variant, start + 1)
    return candidates


def normalize_industrial_terms(query: str) -> IndustrialTermNormalizationResult:
    """仅对 JSON 中明确列出的错误词执行一次、非级联的精确替换。"""
    original = query or ""
    if not original or not _enabled():
        return IndustrialTermNormalizationResult(original, original, False, ())

    candidates = _candidates(original, _load_rules())
    candidates.sort(
        key=lambda item: (
            item.match.start,
            -len(item.match.source),
            -item.match.priority,
            item.rule_index,
            item.variant_index,
        )
    )
    selected: list[IndustrialTermMatch] = []
    for candidate in candidates:
        match = candidate.match
        if any(match.start < chosen.end and chosen.start < match.end for chosen in selected):
            continue
        selected.append(match)

    if not selected:
        return IndustrialTermNormalizationResult(original, original, False, ())
    parts: list[str] = []
    cursor = 0
    for match in selected:
        parts.extend((original[cursor:match.start], match.target))
        cursor = match.end
    parts.append(original[cursor:])
    normalized = "".join(parts)
    return IndustrialTermNormalizationResult(
        original=original,
        normalized=normalized,
        applied=normalized != original,
        matches=tuple(selected),
    )
