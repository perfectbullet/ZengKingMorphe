#!/usr/bin/env python3
"""
导入人工数学概念 Markdown 到 RAGAnything。

固定流程：
1. 读取 math_concepts_content_list.json
2. 校验每条必须有 md_path
3. 读取 Markdown
4. 包装为 RAGAnything content_list
5. 调用 rag.insert_content_list()

RAGAnything.insert_content_list 真实签名（raganything 1.2.9）：
    async def insert_content_list(
        self,
        content_list: List[Dict[str, Any]],
        file_path: str = "unknown_document",
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        display_stats: bool = None,
    )
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

# 加载 ai-service/.env（override=False：不覆盖已注入的环境变量），
# 保证脚本可按 `python scripts/import_math_concepts_content_list.py` 直接运行。
load_dotenv(ROOT_DIR / ".env", override=False)

from app.services.raganything_wrapper import get_raganything_instance


DEFAULT_CONFIG_PATH = ROOT_DIR / "data" / "math_concepts" / "math_concepts_content_list.json"


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    # 优先按项目根目录解析
    project_relative = ROOT_DIR.parent / path
    if project_relative.exists():
        return project_relative

    # 再按 ai-service 根目录解析
    service_relative = ROOT_DIR / path
    if service_relative.exists():
        return service_relative

    # 最后返回项目根目录拼接路径，便于错误信息展示
    return project_relative


def load_config(config_path: Path) -> list[dict[str, Any]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Config root must be a JSON array")

    return data


def validate_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    seen_file_paths: set[str] = set()

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Item #{index} must be an object")

        if "content_list" in item:
            raise ValueError(
                f"Item #{index} contains unsupported field content_list. "
                "This importer only supports md_path."
            )

        missing = [key for key in ("doc_id", "md_path", "file_path") if not item.get(key)]
        if missing:
            raise ValueError(f"Item #{index} missing required fields: {missing}")

        doc_id = str(item["doc_id"]).strip()
        md_path_value = str(item["md_path"]).strip()
        file_path = str(item["file_path"]).strip()

        if doc_id in seen_doc_ids:
            raise ValueError(f"Duplicate doc_id: {doc_id}")
        seen_doc_ids.add(doc_id)

        if file_path in seen_file_paths:
            print(f"[WARN] Duplicate file_path in config: {file_path}")
        seen_file_paths.add(file_path)

        md_path = resolve_path(md_path_value)
        if not md_path.exists():
            raise FileNotFoundError(f"Markdown file not found for doc_id={doc_id}: {md_path}")

        if not md_path.is_file():
            raise ValueError(f"md_path is not a file for doc_id={doc_id}: {md_path}")

        md_content = md_path.read_text(encoding="utf-8").strip()
        if not md_content:
            raise ValueError(f"Markdown file is empty for doc_id={doc_id}: {md_path}")

        validated.append(
            {
                "doc_id": doc_id,
                "md_path": md_path,
                "file_path": file_path,
                "md_content": md_content,
            }
        )

    return validated


async def import_one(rag: Any, item: dict[str, Any], dry_run: bool = False) -> bool:
    doc_id = item["doc_id"]
    file_path = item["file_path"]
    md_path = item["md_path"]
    md_content = item["md_content"]

    # content_list 固定结构：只放纯文本数学概念，不含 entity_name / entity_type
    content_list = [
        {
            "type": "text",
            "text": md_content,
            "page_idx": 0,
        }
    ]

    print(f"[IMPORT] doc_id={doc_id}")
    print(f"         md_path={md_path}")
    print(f"         file_path={file_path}")
    print(f"         chars={len(md_content)}")

    if dry_run:
        print(f"[DRY-RUN] Skip insert_content_list for doc_id={doc_id}")
        return True

    try:
        result = await rag.insert_content_list(
            content_list=content_list,
            file_path=file_path,
            doc_id=doc_id,
        )
        print(f"[OK] Imported doc_id={doc_id} | result={result}")
        return True
    except Exception:
        import traceback
        print(f"[ERROR] Failed to import doc_id={doc_id}")
        traceback.print_exc()
        return False


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import manual math concept markdown files into RAGAnything via content_list."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to math_concepts_content_list.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to import",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print content_list without inserting into RAGAnything",
    )

    args = parser.parse_args()

    config_path = resolve_path(args.config)
    print(f"[CONFIG] {config_path}")

    items = load_config(config_path)
    validated_items = validate_items(items)

    if args.limit is not None:
        validated_items = validated_items[: args.limit]

    print(f"[SUMMARY] valid_items={len(validated_items)} | dry_run={args.dry_run}")

    if not validated_items:
        print("[SUMMARY] Nothing to import")
        return 0

    if args.dry_run:
        rag = None
    else:
        rag = await get_raganything_instance()

    success_count = 0
    fail_count = 0

    for item in validated_items:
        ok = await import_one(rag, item, dry_run=args.dry_run)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print(f"[DONE] success={success_count} | failed={fail_count}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
