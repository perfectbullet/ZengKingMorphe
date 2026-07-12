"""Single-book metadata contract for the Markdown graph pipeline.

All paths in a meta file are intentionally resolved relative to that meta file,
never relative to the shell's current directory.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from common import load_json, resolve_path

DEFAULT_DOMAIN = "industrial_training"
DEFAULT_SUBJECT = ""
DEFAULT_STRUCTURE = {
    "front_lines": 250,
    "toc_start_line": 0,
    "toc_end_line": 0,
    "body_start_line": 0,
    "window_lines": 320,
    "overlap_lines": 60,
}


def _nonempty(value: Any) -> str:
    return str(value or "").strip()


def load_book_meta(path: Path | str) -> dict[str, Any]:
    meta_path = Path(path).expanduser().resolve()
    if not meta_path.is_file():
        raise FileNotFoundError(f"教材 meta 文件不存在: {meta_path}")
    data = load_json(meta_path)
    if not isinstance(data, dict):
        raise ValueError(f"教材 meta 必须是 JSON 对象: {meta_path}")
    result = copy.deepcopy(data)
    result["_meta_path"] = str(meta_path)
    validate_book_meta(result, stage="init")
    return result


def validate_book_meta(meta: dict[str, Any], *, stage: str = "run") -> None:
    if not isinstance(meta, dict):
        raise ValueError("教材 meta 必须是对象")
    if _nonempty(meta.get("schema_version")) != "markdown_graph_book_meta.v1":
        raise ValueError("教材 meta 的 schema_version 必须是 markdown_graph_book_meta.v1")
    for key in ("book_id", "book_title", "artifact_stem"):
        if not _nonempty(meta.get(key)):
            raise ValueError(f"教材 meta 缺少 {key}")
    inputs = meta.get("inputs")
    business = meta.get("business")
    structure = meta.get("structure")
    extraction = meta.get("entity_extraction")
    if not isinstance(inputs, dict):
        raise ValueError("教材 meta.inputs 必须是对象")
    if not isinstance(business, dict):
        raise ValueError("教材 meta.business 必须是对象")
    if not isinstance(structure, dict):
        raise ValueError("教材 meta.structure 必须是对象")
    if not isinstance(extraction, dict):
        raise ValueError("教材 meta.entity_extraction 必须是对象")
    if not _nonempty(inputs.get("markdown")):
        raise ValueError("教材 meta.inputs.markdown 不能为空")
    for key in DEFAULT_STRUCTURE:
        try:
            value = int(structure.get(key, DEFAULT_STRUCTURE[key]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"教材 meta.structure.{key} 必须是整数") from exc
        if value < 0:
            raise ValueError(f"教材 meta.structure.{key} 不能小于 0")
    if stage == "step2":
        validate_step2_structure(meta)


def validate_step2_structure(meta: dict[str, Any]) -> None:
    business = get_business_config(meta)
    structure = get_structure_config(meta)
    missing = []
    if not business["domain"]:
        missing.append("business.domain")
    if not business["subject"]:
        missing.append("business.subject")
    if missing:
        raise ValueError("Step 2 缺少必要业务配置: " + ", ".join(missing))
    values = {key: int(structure[key]) for key in DEFAULT_STRUCTURE}
    required = ("front_lines", "toc_start_line", "toc_end_line", "body_start_line")
    if any(values[key] <= 0 for key in required):
        stem = str(meta.get("artifact_stem") or "教材")
        raise ValueError(
            "Step 2 需要人工确认目录范围。\n"
            f"请查看 {stem}.md，并在 meta 中填写：\n"
            "structure.toc_start_line\nstructure.toc_end_line\nstructure.body_start_line"
        )
    if values["toc_start_line"] > values["toc_end_line"]:
        raise ValueError("Step 2 目录边界无效: toc_start_line 必须小于等于 toc_end_line")
    if values["toc_end_line"] >= values["body_start_line"]:
        raise ValueError("Step 2 目录边界无效: toc_end_line 必须小于 body_start_line")
    if values["body_start_line"] > values["front_lines"]:
        raise ValueError("Step 2 目录边界无效: body_start_line 必须小于等于 front_lines")


def resolve_book_paths(meta: dict[str, Any]) -> dict[str, Path | None]:
    meta_path = Path(_nonempty(meta.get("_meta_path"))).expanduser().resolve()
    base = meta_path.parent
    inputs = meta["inputs"]
    markdown = resolve_path(_nonempty(inputs.get("markdown")), [base])
    v2_value = _nonempty(inputs.get("content_list_v2"))
    v2 = resolve_path(v2_value, [base]) if v2_value else None
    prompt_value = _nonempty(meta.get("entity_extraction", {}).get("prompt_file"))
    # Entity profiles are deliberately centralized under this subproject, not
    # duplicated next to each source textbook meta file.
    prompt = (Path(__file__).resolve().parent / "prompts" / "entity_type" / Path(prompt_value).name).resolve() if prompt_value else None
    return {"meta": meta_path, "markdown": markdown, "content_list_v2": v2, "prompt_file": prompt}


def resolve_config_value(
    cli_value: Any,
    meta_value: Any,
    env_name: str | None,
    default: Any,
) -> Any:
    """Resolve CLI > meta > environment > code default without falsey ambiguity."""
    if cli_value is not None and str(cli_value).strip() != "":
        return cli_value
    if meta_value is not None and str(meta_value).strip() != "":
        return meta_value
    if env_name:
        env_value = os.getenv(env_name)
        if env_value is not None and env_value.strip() != "":
            return env_value
    return default


def get_business_config(meta: dict[str, Any]) -> dict[str, str]:
    business = meta.get("business") or {}
    return {
        "domain": _nonempty(business.get("domain")),
        "subject": _nonempty(business.get("subject")),
    }


def get_structure_config(meta: dict[str, Any]) -> dict[str, int]:
    structure = meta.get("structure") or {}
    return {key: int(structure.get(key, default)) for key, default in DEFAULT_STRUCTURE.items()}


def get_entity_extraction_config(meta: dict[str, Any]) -> dict[str, str]:
    extraction = meta.get("entity_extraction") or {}
    return {"prompt_file": _nonempty(extraction.get("prompt_file"))}
