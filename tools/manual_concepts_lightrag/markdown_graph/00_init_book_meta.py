#!/usr/bin/env python3
"""Initialize one textbook's JSON meta and LightRAG entity-type YAML profile."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import yaml

from book_meta import DEFAULT_DOMAIN, DEFAULT_STRUCTURE, DEFAULT_SUBJECT, load_book_meta
from common import PROJECT_DIR, write_json, write_text_atomic

ENTITY_TYPE_DIR = PROJECT_DIR / "prompts" / "entity_type"
TEMPLATE = PROJECT_DIR / "prompts" / "entity_type_templates" / "industrial_training_entity_types.template.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="初始化单本教材 meta 与实体类型 YAML")
    parser.add_argument("markdown", type=Path)
    parser.add_argument("--domain")
    parser.add_argument("--subject")
    parser.add_argument("--force-meta", action="store_true")
    parser.add_argument("--force-entity-types", action="store_true")
    return parser.parse_args()


def derive_book_values(markdown: Path) -> tuple[str, str, str]:
    artifact_stem = markdown.stem
    book_id = re.sub(r"_full$", "", artifact_stem)
    book_title = re.sub(r"^\s*\d+\s*", "", book_id).strip() or book_id
    return artifact_stem, book_id, book_title


def find_content_list_v2(directory: Path, book_id: str, artifact_stem: str) -> str:
    names = (f"{book_id}_list_v2.json", f"{artifact_stem}_list_v2.json")
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate.name
    candidates = sorted(directory.glob("*_content_list_v2.json"))
    if len(candidates) == 1:
        return candidates[0].name
    if len(candidates) > 1:
        choices = "\n".join(f"- {item}" for item in candidates)
        raise ValueError(f"发现多个 content_list_v2 候选，请人工确认后初始化：\n{choices}")
    print("WARNING: 未找到 content_list_v2；meta 将保留空值，Step 5 会要求补齐。")
    return ""


def render_template(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def validate_existing_yaml(
    yaml_path: Path,
    *,
    book_id: str,
    book_title: str,
    domain: str,
    subject: str,
) -> dict:
    """Validate a human-maintained profile without modifying it."""
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"实体类型 YAML 解析失败: {yaml_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"实体类型 YAML 根节点必须是 mapping: {yaml_path}")

    mismatches: list[str] = []
    for field, expected in {
        "book_id": book_id,
        "book_title": book_title,
        "domain": domain,
        "subject": subject,
    }.items():
        actual = data.get(field)
        if actual != expected:
            mismatches.append(
                f"{field} 不一致：实际值={actual!r}，期望值={expected!r}"
            )
    for field in ("allowed_entity_types", "allowed_relation_keywords"):
        value = data.get(field)
        if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            mismatches.append(f"{field} 必须是非空字符串列表：实际值={value!r}")
    if not isinstance(data.get("entity_types_guidance"), str) or not data[
        "entity_types_guidance"
    ].strip():
        mismatches.append(
            "entity_types_guidance 必须是非空字符串："
            f"实际值={data.get('entity_types_guidance')!r}"
        )
    for field in ("entity_extraction_examples", "entity_extraction_json_examples"):
        value = data.get(field)
        if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            mismatches.append(f"{field} 必须是非空字符串列表：实际值={value!r}")
    if mismatches:
        raise ValueError("已有实体类型 YAML 校验失败：\n- " + "\n- ".join(mismatches))
    return data


def build_meta(
    markdown: Path,
    *,
    artifact_stem: str,
    book_id: str,
    book_title: str,
    domain: str,
    subject: str,
    prompt_file: str,
) -> dict:
    return {
        "schema_version": "markdown_graph_book_meta.v1",
        "book_id": book_id,
        "book_title": book_title,
        "artifact_stem": artifact_stem,
        "inputs": {
            "markdown": markdown.name,
            "content_list_v2": find_content_list_v2(
                markdown.parent, book_id, artifact_stem
            ),
        },
        "business": {"domain": domain, "subject": subject},
        "structure": dict(DEFAULT_STRUCTURE),
        "entity_extraction": {"prompt_file": prompt_file},
    }


def write_profile_from_template(
    yaml_path: Path,
    *,
    book_id: str,
    book_title: str,
    domain: str,
    subject: str,
) -> None:
    if not TEMPLATE.is_file():
        raise FileNotFoundError(f"实体 YAML 模板不存在: {TEMPLATE}")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(
        yaml_path,
        render_template(
            TEMPLATE.read_text(encoding="utf-8"),
            {
                "book_id": book_id,
                "book_title": book_title,
                "domain": domain,
                "subject": subject,
            },
        ),
    )


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    markdown = args.markdown.expanduser().resolve()
    if not markdown.is_file():
        raise FileNotFoundError(f"Markdown 文件不存在: {markdown}")
    artifact_stem, book_id, book_title = derive_book_values(markdown)
    meta_path = markdown.with_name(f"{book_id}_meta.json")
    domain = args.domain or os.getenv("MARKDOWN_GRAPH_DOMAIN") or DEFAULT_DOMAIN
    subject = args.subject or os.getenv("MARKDOWN_GRAPH_SUBJECT") or DEFAULT_SUBJECT
    expected_profile_name = f"{artifact_stem}_entity_types.yml"
    existing_meta = load_book_meta(meta_path) if meta_path.is_file() else None
    meta_profile_name = str(
        (existing_meta or {}).get("entity_extraction", {}).get("prompt_file") or ""
    ).strip()
    # A profile referenced by an existing meta is authoritative when repairing a
    # missing YAML. Otherwise the stable artifact-stem-derived name is used.
    target_profile_name = meta_profile_name or expected_profile_name
    yaml_path = ENTITY_TYPE_DIR / Path(target_profile_name).name

    if args.force_entity_types:
        yaml_path = ENTITY_TYPE_DIR / expected_profile_name
        write_profile_from_template(
            yaml_path,
            book_id=book_id,
            book_title=book_title,
            domain=domain,
            subject=subject,
        )
        if existing_meta is not None:
            current = str(
                existing_meta.get("entity_extraction", {}).get("prompt_file") or ""
            )
            if current != expected_profile_name:
                existing_meta.setdefault("entity_extraction", {})[
                    "prompt_file"
                ] = expected_profile_name
                write_json(meta_path, {k: v for k, v in existing_meta.items() if k != "_meta_path"})
                print("INFO: --force-entity-types 已修正 meta 的 prompt_file 引用")
        else:
            write_json(
                meta_path,
                build_meta(
                    markdown,
                    artifact_stem=artifact_stem,
                    book_id=book_id,
                    book_title=book_title,
                    domain=domain,
                    subject=subject,
                    prompt_file=expected_profile_name,
                ),
            )
        print("INFO: 已覆盖实体类型 YAML；meta 未被重写（除 prompt_file 修正外）")
    elif existing_meta is None and yaml_path.is_file():
        validate_existing_yaml(
            yaml_path,
            book_id=book_id,
            book_title=book_title,
            domain=domain,
            subject=subject,
        )
        write_json(
            meta_path,
            build_meta(
                markdown,
                artifact_stem=artifact_stem,
                book_id=book_id,
                book_title=book_title,
                domain=domain,
                subject=subject,
                prompt_file=yaml_path.name,
            ),
        )
        print("INFO: 复用已有实体类型 YAML")
    elif existing_meta is None and not yaml_path.exists():
        write_profile_from_template(
            yaml_path,
            book_id=book_id,
            book_title=book_title,
            domain=domain,
            subject=subject,
        )
        write_json(
            meta_path,
            build_meta(
                markdown,
                artifact_stem=artifact_stem,
                book_id=book_id,
                book_title=book_title,
                domain=domain,
                subject=subject,
                prompt_file=yaml_path.name,
            ),
        )
        print("INFO: 已创建 meta 和实体类型 YAML")
    elif existing_meta is not None and not yaml_path.exists() and not args.force_meta:
        if not meta_profile_name:
            raise ValueError("已有 meta 缺少 entity_extraction.prompt_file，无法确定需要创建的 YAML")
        write_profile_from_template(
            yaml_path,
            book_id=book_id,
            book_title=book_title,
            domain=domain,
            subject=subject,
        )
        print("INFO: 已根据 meta 的 prompt_file 创建缺失实体类型 YAML")
    elif args.force_meta:
        if yaml_path.is_file():
            validate_existing_yaml(
                yaml_path,
                book_id=book_id,
                book_title=book_title,
                domain=domain,
                subject=subject,
            )
        else:
            print("WARNING: --force-meta 未创建缺失 YAML；后续请重新运行 Step 0 补齐 profile")
        write_json(
            meta_path,
            build_meta(
                markdown,
                artifact_stem=artifact_stem,
                book_id=book_id,
                book_title=book_title,
                domain=domain,
                subject=subject,
                prompt_file=yaml_path.name,
            ),
        )
        print("INFO: 已覆盖 meta；保留已有实体类型 YAML")
    else:
        referenced = meta_profile_name
        if referenced != yaml_path.name:
            raise ValueError(
                "已有 meta 的 entity_extraction.prompt_file 与可用 YAML 不一致："
                f"实际值={referenced!r}，期望值={yaml_path.name!r}"
            )
        validate_existing_yaml(
            yaml_path,
            book_id=book_id,
            book_title=book_title,
            domain=domain,
            subject=subject,
        )
        print("INFO: meta 与实体类型 YAML 已完成初始化，未修改任何文件")
    print(json.dumps({"meta": str(meta_path), "entity_type_prompt": str(yaml_path)}, ensure_ascii=False))
    return meta_path, yaml_path


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
