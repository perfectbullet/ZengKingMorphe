#!/usr/bin/env python
"""
import_manual_concepts_custom_kg.py
=====================================
人工概念 -> LightRAG「自定义 KG」导入脚本（推荐生产路径）。

与旧脚本 import_manual_concepts_lightrag.py 的核心区别：
- 旧脚本使用 rag.ainsert(...)，会触发 LightRAG 默认 LLM 实体抽取，
  导致知识图谱里混入公式、变量、符号、短语等脏实体（如 0!、A_n^m、C(n,0)、
  排列数公式A(n,n)、第1类方案、步骤、方法数、Unknown 等）。
- 本脚本使用 rag.ainsert_custom_kg(...)，把每条人工概念记录直接作为自定义 KG
  写入，完全不走 LLM 实体抽取，图中只保留 MANUAL_CONCEPT 实体（默认；可经
  CONCEPT_RETRIEVAL_ENTITY_TYPE 覆盖；兼容 legacy MANUAL_MATH_CONCEPT）。

每条 JSONL 记录转成：
- 1 个 chunk（content=md_content, source_id=doc_id）
- 1 个 entity（entity_name=concept_name, entity_type=MANUAL_CONCEPT,
  description=md_content, source_id=doc_id）
entity.source_id 与 chunk.source_id 保持一致，LightRAG 内部的 chunk_to_source_map
才能把实体映射到真实 chunk id。第一版不自动生成 relationships。

默认 domain=math，可通过 --domain / CONCEPT_RETRIEVAL_DOMAIN 切换为其它领域
（如 industrial_training）。本脚本是独立验证脚本，不依赖也不修改 ai-service 业务代码或 LightRAG 源码。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------------
# 项目根目录：子项目根 = 脚本所在目录（tools/manual_concepts_lightrag）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent.parent
DEFAULT_ENTITY_WHITELIST_PATH = PROJECT_ROOT / "entity_whitelist_draft.txt"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lightrag import LightRAG  # noqa: E402
from lightrag.llm.openai import openai_complete_if_cache, openai_embed  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402


# ---------------------------------------------------------------------------
# 路径 / 配置工具（与旧脚本保持一致的解析行为）
# ---------------------------------------------------------------------------
def resolve_path(path_str: str) -> Path:
    """
    路径解析：支持绝对路径，也支持相对子项目根 / 仓库根 / cwd 解析。
    依次尝试候选目录，返回第一个存在的路径；都不存在则回退到相对子项目根。
    """
    if not path_str:
        return PROJECT_ROOT / "<empty>"
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    candidates = [
        PROJECT_ROOT / path_str,                  # 相对子项目根
        PROJECT_ROOT.parent.parent / path_str,    # 相对仓库根（兼容 ai-service/data/... 形式）
        Path.cwd() / path_str,                    # 相对当前工作目录
    ]
    for c in candidates:
        if c.exists():
            return c
    return PROJECT_ROOT / path_str


def load_concept_items(config_path: Path) -> list[dict]:
    """
    读取 config，兼容以下结构，统一返回 list[item]：
    - JSON Lines（.jsonl）：每行一个独立 JSON 对象
    - 顶层 list
    - 顶层 dict 中的 items / concepts / documents / data
    """
    if config_path.suffix.lower() == ".jsonl":
        items: list[dict] = []
        with config_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"jsonl 第 {line_no} 行解析失败 | error={e}") from e
                items.append(obj)
        logger.info(f"config(jsonl)={config_path} | concept_count={len(items)}")
        return items

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = None
        for key in ("items", "concepts", "documents", "data"):
            if isinstance(raw.get(key), list):
                items = raw[key]
                logger.info(f"config 为 dict，使用字段 '{key}' | keys={list(raw.keys())}")
                break
        if items is None:
            raise ValueError(
                f"配置文件为 dict 但缺少 items/concepts/documents/data 列表字段 | keys={list(raw.keys())}"
            )
    else:
        raise TypeError(f"配置文件既不是 list 也不是 dict | type={type(raw).__name__}")

    logger.info(f"config={config_path} | concept_count={len(items)}")
    return items


def load_entity_whitelist(path: Path | None) -> set[str]:
    """读取实体白名单；跳过空行和以 # 开头的注释，去掉首尾空白。"""
    if path is None:
        return set()
    whitelist: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            whitelist.add(name)
    return whitelist


HEADING_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


def extract_concept_name(md_content: str, md_path: Path | None) -> str:
    """优先 Markdown 一级标题，其次文件名 stem，最后兜底空串。"""
    m = HEADING_RE.search(md_content or "")
    if m:
        return m.group(1).strip()
    if md_path:
        return md_path.stem
    return ""


def default_file_path(item: dict) -> str | None:
    """file_path 默认值：item.file_path > manual_concepts/{md_path文件名} > manual_concepts/{doc_id}.md。

    兼容 legacy：若 item 显式给出以 manual_math_concepts/ 为前缀的 file_path 则原样保留。
    """
    if item.get("file_path"):
        return str(item["file_path"])
    md_path_raw = item.get("md_path")
    if md_path_raw:
        return f"manual_concepts/{Path(md_path_raw).name}"
    doc_id = item.get("doc_id")
    if doc_id:
        return f"manual_concepts/{doc_id}.md"
    return None


def _sanitize_value(v: Any) -> Any:
    """GraphML(networkx) 不支持 list / None / dict 数据值。写入 custom_kg 前清洗：
    None -> ""；list/tuple/set -> 逗号拼接串；dict -> json.dumps(ensure_ascii=False)；
    其他简单标量（str/int/float/bool）原样返回。"""
    if v is None:
        return ""
    if isinstance(v, (list, tuple, set)):
        return ", ".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return v


# ---------------------------------------------------------------------------
# 环境变量解析（CONCEPT_RETRIEVAL_* 优先 > 旧变量 > 默认值）
# ---------------------------------------------------------------------------
def resolve_concept_env() -> dict:
    """统一解析概念检索相关环境变量。

    优先级：CONCEPT_RETRIEVAL_* > legacy 旧变量 > 默认值。
    """
    domain = (
        os.getenv("CONCEPT_RETRIEVAL_DOMAIN")
        or os.getenv("DOMAIN")
        or "math"
    ).strip()

    entity_type = (
        os.getenv("CONCEPT_RETRIEVAL_ENTITY_TYPE")
        or os.getenv("ENTITY_TYPE")
        or "MANUAL_CONCEPT"
    ).strip()

    enabled = (
        os.getenv("CONCEPT_RETRIEVAL_ENABLED")
        or os.getenv("ENABLE_LIGHTRAG")
        or "true"
    ).strip().lower() in ("1", "true", "yes", "on")

    prompt_file = (os.getenv("CONCEPT_RETRIEVAL_ENTITY_TYPE_PROMPT_FILE") or "").strip() or None

    return {
        "domain": domain,
        "entity_type": entity_type,
        "enabled": enabled,
        "entity_type_prompt_file": prompt_file,
    }


def _pick_env(candidates: list[str], label: str, required: bool = True) -> str | None:
    for name in candidates:
        val = os.getenv(name)
        if val and val.strip():
            return val.strip()
    if required:
        raise RuntimeError(
            f"缺少必要环境变量 [{label}]，请在 .env 中配置以下候选之一: {candidates}"
        )
    return None


def resolve_llm_config() -> dict:
    model = _pick_env(["LLM_MODEL", "OPENAI_MODEL"], "LLM_MODEL")
    base_url = _pick_env(
        ["LLM_BINDING_HOST", "OPENAI_BASE_URL", "LLM_BASE_URL"], "LLM_BASE_URL"
    )
    api_key = _pick_env(
        ["LLM_BINDING_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"],
        "LLM_API_KEY",
        required=False,
    ) or "no-api-key"
    return {"model": model, "base_url": base_url, "api_key": api_key}


def resolve_embedding_config(llm_base_url: str, llm_api_key: str) -> dict:
    model = _pick_env(
        ["EMBEDDING_MODEL", "VLLM_EMBED_MODEL", "VLLM_EMBEDDING_MODEL"], "EMBEDDING_MODEL"
    )
    dim_raw = _pick_env(["EMBEDDING_DIM", "VLLM_EMBED_DIM"], "EMBEDDING_DIM")
    try:
        dim = int(dim_raw)
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"EMBEDDING_DIM 无法转为整数 | value={dim_raw!r} | error={e}")
    base_url = _pick_env(
        ["EMBEDDING_BINDING_HOST", "VLLM_EMBED_HOST", "VLLM_EMBEDDING_BASE_URL"],
        "EMBEDDING_BASE_URL",
        required=False,
    ) or llm_base_url
    api_key = _pick_env(
        ["EMBEDDING_BINDING_API_KEY", "VLLM_EMBED_API_KEY", "VLLM_API_KEY"],
        "EMBEDDING_API_KEY",
        required=False,
    ) or llm_api_key
    return {"model": model, "dim": dim, "base_url": base_url, "api_key": api_key}


# ---------------------------------------------------------------------------
# 模型函数 / LightRAG 构建
# ---------------------------------------------------------------------------
def build_llm_model_func(llm: dict):
    async def _llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        **kwargs: Any,
    ) -> str:
        return await openai_complete_if_cache(
            llm["model"],
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=llm["base_url"],
            api_key=llm["api_key"],
            **kwargs,
        )

    return _llm_model_func


def build_embedding_func(emb: dict) -> EmbeddingFunc:
    return EmbeddingFunc(
        embedding_dim=emb["dim"],
        max_token_size=8192,
        func=lambda texts: openai_embed.func(
            texts,
            model=emb["model"],
            api_key=emb["api_key"],
            base_url=emb["base_url"],
        ),
    )


def build_rag(working_dir: Path, llm: dict, emb: dict) -> LightRAG:
    """custom KG 导入不触发 LLM 实体抽取，entity_types_guidance 仅供查询侧语义一致。

    适配 LightRAG main：旧版 addon_params["entity_types"]=[...] 列表已废弃，改用
    addon_params["entity_types_guidance"]（字符串，注入 prompt 的 {entity_types_guidance}）。
    """
    working_dir.mkdir(parents=True, exist_ok=True)
    concept_env = resolve_concept_env()
    rag = LightRAG(
        working_dir=str(working_dir),
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        addon_params={
            "language": "Chinese",
            "entity_types_guidance": (
                f"只抽取 {concept_env['domain']} 领域人工概念，"
                f"实体类型统一为 {concept_env['entity_type']}。"
            ),
        },
        llm_model_func=build_llm_model_func(llm),
        embedding_func=build_embedding_func(emb),
    )
    return rag


# ---------------------------------------------------------------------------
# custom_kg 构造
# ---------------------------------------------------------------------------
def build_record_custom_kg(
    doc_id: str,
    concept_name: str,
    md_content: str,
    file_path: str,
    entity_type: str = "MANUAL_CONCEPT",
    domain: str = "math",
    source_type: str = "manual_concept",
    legacy_source_type: str | None = None,
) -> dict[str, Any]:
    """单条记录 -> 一个 chunk + 一个 entity（默认 MANUAL_CONCEPT，无 relationships）。

    entity.source_id 必须与 chunk.source_id 一致，LightRAG 内部 chunk_to_source_map
    才能把实体映射到真实 chunk id。entity metadata 携带 domain / source_type /
    legacy_source_type 供查询侧过滤。
    """
    chunk = {
        "content": _sanitize_value(md_content),
        "source_id": _sanitize_value(doc_id),
        "file_path": _sanitize_value(file_path),
    }
    entity: dict[str, Any] = {
        "entity_name": _sanitize_value(concept_name),
        "entity_type": _sanitize_value(entity_type),
        "domain": _sanitize_value(domain),
        "source_type": _sanitize_value(source_type),
        "description": _sanitize_value(md_content),
        "source_id": _sanitize_value(doc_id),
        "file_path": _sanitize_value(file_path),
    }
    if legacy_source_type:
        entity["legacy_source_type"] = _sanitize_value(legacy_source_type)
    return {"chunks": [chunk], "entities": [entity], "relationships": []}


def merge_batch_custom_kg(records: list[dict]) -> dict[str, Any]:
    """把多条记录的 chunk / entity 聚合到一个 custom_kg，便于按批 ainsert_custom_kg。"""
    batch_kg: dict[str, Any] = {"chunks": [], "entities": [], "relationships": []}
    for rec in records:
        ck = rec["custom_kg"]
        batch_kg["chunks"].extend(ck["chunks"])
        batch_kg["entities"].extend(ck["entities"])
    return batch_kg


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _new_stats(args: argparse.Namespace, config_path: Path, working_dir: Path, whitelist_path: Path | None) -> dict:
    return {
        "total_records": 0,
        "skip_non_dict": 0,
        "skip_missing_doc_id": 0,
        "skip_missing_concept_name": 0,
        "skip_missing_md_content": 0,
        "skip_review_status": 0,
        "skip_whitelist": 0,
        "prepared_chunks": 0,
        "prepared_entities": 0,
        "success_batches": 0,
        "failed_batches": 0,
        "failures": 0,
        "dry_run": bool(args.dry_run),
        "working_dir": str(working_dir),
        "config": str(config_path),
        "entity_whitelist": "disabled" if args.disable_whitelist else str(whitelist_path),
    }


def _log_stats(stats: dict) -> None:
    logger.info(f"custom_kg import stats | {json.dumps(stats, ensure_ascii=False)}")


def _prepare_records(
    items: list,
    whitelist: set[str],
    args: argparse.Namespace,
    stats: dict,
    args_domain: str,
    args_entity_type: str,
) -> list[dict]:
    """遍历原始记录，按规则过滤并构造每条记录的 custom_kg。返回 prepared 列表。"""
    prepared: list[dict] = []
    for idx, raw_item in enumerate(items, 1):
        if not isinstance(raw_item, dict):
            stats["skip_non_dict"] += 1
            logger.warning(
                f"skip non-dict item #{idx} | type={type(raw_item).__name__}"
            )
            continue

        review_status = raw_item.get("review_status") or ""
        if review_status != "correct" and not args.include_non_correct:
            stats["skip_review_status"] += 1
            logger.info(
                f"skip review_status | item={idx} | "
                f"concept_name={raw_item.get('concept_name')} | "
                f"review_status={review_status!r}"
            )
            continue

        doc_id = raw_item.get("doc_id")
        if not doc_id:
            stats["skip_missing_doc_id"] += 1
            logger.warning(
                f"skip missing doc_id | item={idx} | keys={list(raw_item.keys())}"
            )
            continue

        # 解析 Markdown 内容：优先 md_path 文件，其次 md_content 内联（jsonl 常见）
        md_path_raw = raw_item.get("md_path")
        md_path = resolve_path(md_path_raw) if md_path_raw else None
        md_content: str | None = None
        if md_path and md_path.exists():
            md_content = md_path.read_text(encoding="utf-8")
        if not md_content:
            md_content = (
                raw_item.get("md_content")
                or raw_item.get("content")
                or raw_item.get("markdown")
                or raw_item.get("text")
            )
        if not md_content:
            stats["skip_missing_md_content"] += 1
            logger.error(
                f"skip missing md_content | item={idx} | doc_id={doc_id} | "
                f"md_path={md_path_raw}"
            )
            continue

        # concept_name：优先 raw_item.concept_name > name > md_content 一级标题
        concept_name = raw_item.get("concept_name") or raw_item.get("name")
        if not concept_name:
            concept_name = extract_concept_name(md_content, md_path)
        concept_name = str(concept_name).strip() if concept_name else ""
        if not concept_name:
            stats["skip_missing_concept_name"] += 1
            logger.warning(
                f"skip missing concept_name | item={idx} | doc_id={doc_id}"
            )
            continue

        # 白名单过滤（默认启用）
        if not args.disable_whitelist and concept_name not in whitelist:
            stats["skip_whitelist"] += 1
            logger.info(
                f"skip whitelist | item={idx} | doc_id={doc_id} | concept_name={concept_name}"
            )
            continue

        file_path = default_file_path(raw_item) or f"manual_concepts/{concept_name}.md"
        raw_source_type = raw_item.get("source_type")
        if raw_source_type == "manual_math_concept":
            source_type = "manual_concept"
            legacy_source_type = "manual_math_concept"
        elif raw_source_type:
            source_type = raw_source_type
            legacy_source_type = None
        else:
            source_type = "manual_concept"
            legacy_source_type = None
        domain = raw_item.get("domain") or args_domain
        custom_kg = build_record_custom_kg(
            doc_id=str(doc_id),
            concept_name=concept_name,
            md_content=str(md_content),
            file_path=str(file_path),
            entity_type=args_entity_type,
            domain=domain,
            source_type=source_type,
            legacy_source_type=legacy_source_type,
        )
        prepared.append(
            {
                "doc_id": str(doc_id),
                "concept_name": concept_name,
                "file_path": file_path,
                "domain": domain,
                "entity_type": args_entity_type,
                "source_type": source_type,
                "legacy_source_type": legacy_source_type,
                "review_status": review_status,
                "custom_kg": custom_kg,
            }
        )
        logger.info(
            f"[{idx}/{len(items)}] prepared | doc_id={doc_id} | concept_name={concept_name} | "
            f"file_path={file_path} | domain={domain} | entity_type={args_entity_type} | "
            f"source_type={source_type} | legacy_source_type={legacy_source_type}"
        )

    stats["prepared_chunks"] = len(prepared)
    stats["prepared_entities"] = len(prepared)
    return prepared


async def run(args: argparse.Namespace) -> int:
    config_path = resolve_path(args.config)
    if not config_path.exists():
        logger.error(f"config 不存在 | config={config_path}")
        return 1

    items = load_concept_items(config_path)

    # 白名单
    whitelist: set[str] = set()
    whitelist_path: Path | None = None
    if args.disable_whitelist:
        logger.warning("entity whitelist disabled by --disable-whitelist")
    else:
        whitelist_path = resolve_path(args.entity_whitelist)
        if not whitelist_path.exists():
            logger.error(f"entity whitelist 不存在 | path={whitelist_path}")
            return 1
        whitelist = load_entity_whitelist(whitelist_path)
        logger.info(
            f"entity whitelist loaded | path={whitelist_path} | count={len(whitelist)}"
        )

    working_dir = Path(args.working_dir).expanduser()
    if not working_dir.is_absolute():
        working_dir = PROJECT_ROOT / args.working_dir

    logger.info(f"config={config_path}")
    logger.info(f"working_dir={working_dir}")
    logger.info(f"batch_size={args.batch_size} | include_non_correct={args.include_non_correct}")

    # domain / entity_type 解析：--domain/--entity-type 优先，否则用 CONCEPT_RETRIEVAL_*
    concept_env = resolve_concept_env()
    args_domain = (args.domain or concept_env["domain"]).strip() or "math"
    args_entity_type = (args.entity_type or concept_env["entity_type"]).strip() or "MANUAL_CONCEPT"
    logger.info(
        f"domain={args_domain} | entity_type={args_entity_type} | "
        f"concept_retrieval_enabled={concept_env['enabled']}"
    )

    stats = _new_stats(args, config_path, working_dir, whitelist_path)
    stats["total_records"] = len(items)
    stats["domain"] = args_domain
    stats["entity_type"] = args_entity_type

    # 构造每条记录的 custom_kg
    prepared = _prepare_records(items, whitelist, args, stats, args_domain, args_entity_type)
    logger.info(
        f"prepare done | prepared={len(prepared)} | "
        f"skip_review_status={stats['skip_review_status']} | "
        f"skip_whitelist={stats['skip_whitelist']} | "
        f"skip_missing_doc_id={stats['skip_missing_doc_id']} | "
        f"skip_missing_concept_name={stats['skip_missing_concept_name']} | "
        f"skip_missing_md_content={stats['skip_missing_md_content']} | "
        f"skip_non_dict={stats['skip_non_dict']}"
    )

    # dump 构造后的 custom_kg（dry-run 与正式模式均可 dump）
    if args.dump_custom_kg:
        dump_path = Path(args.dump_custom_kg).expanduser()
        if not dump_path.is_absolute():
            dump_path = PROJECT_ROOT / args.dump_custom_kg
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(config_path),
            "working_dir": str(working_dir),
            "batch_size": args.batch_size,
            "records": [rec["custom_kg"] for rec in prepared],
        }
        dump_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"dump custom_kg OK | path={dump_path} | records={len(prepared)}")

    # dry-run：不初始化 LightRAG，不写入，只打印统计
    if args.dry_run:
        logger.info("dry-run 模式 | 不初始化 LightRAG，不写入，仅打印统计")
        _log_stats(stats)
        return 0

    if not prepared:
        logger.warning("没有可导入的记录，跳过 LightRAG 初始化")
        _log_stats(stats)
        return 0

    llm = resolve_llm_config()
    emb = resolve_embedding_config(llm["base_url"], llm["api_key"])
    logger.info(
        f"llm_model={llm['model']} | llm_base_url={llm['base_url']} | "
        f"embedding_model={emb['model']} | embedding_dim={emb['dim']} | "
        f"embedding_base_url={emb['base_url']}"
    )

    rag = build_rag(working_dir, llm, emb)
    await rag.initialize_storages()
    logger.info("initialize_storages OK")

    batch_size = max(1, int(args.batch_size))
    try:
        total_batches = (len(prepared) + batch_size - 1) // batch_size
        for bi, start in enumerate(range(0, len(prepared), batch_size), 1):
            batch = prepared[start : start + batch_size]
            batch_kg = merge_batch_custom_kg(batch)
            try:
                # 关键：使用 ainsert_custom_kg，不走 LLM 实体抽取
                await rag.ainsert_custom_kg(batch_kg, full_doc_id=None)
                stats["success_batches"] += 1
                logger.info(
                    f"batch OK | batch={bi}/{total_batches} | "
                    f"chunks={len(batch_kg['chunks'])} | entities={len(batch_kg['entities'])}"
                )
            except Exception as e:
                stats["failed_batches"] += 1
                stats["failures"] += len(batch)
                logger.error(
                    f"batch FAILED | batch={bi}/{total_batches} | records={len(batch)} | "
                    f"error_type={type(e).__name__} | error={e}",
                    exc_info=True,
                )
    finally:
        try:
            await rag.finalize_storages()
            logger.info("finalize_storages OK")
        except Exception as e:
            logger.warning(
                f"finalize_storages WARN | error_type={type(e).__name__} | error={e}"
            )

    _log_stats(stats)
    return 1 if stats["failed_batches"] else 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="导入手动概念到 LightRAG（custom KG 模式，不走 LLM 实体抽取，推荐生产路径）"
    )
    p.add_argument(
        "--config",
        required=True,
        help="概念 JSONL/JSON 配置路径（必填）",
    )
    p.add_argument(
        "--working-dir",
        required=True,
        help="LightRAG working_dir（必填，推荐使用 lightrag_manual_concepts_custom_kg）",
    )
    p.add_argument(
        "--domain",
        default=None,
        help="领域（默认 CONCEPT_RETRIEVAL_DOMAIN 或 math；不写死，可扩展 industrial_training 等）",
    )
    p.add_argument(
        "--entity-type",
        default=None,
        help="实体类型（默认 CONCEPT_RETRIEVAL_ENTITY_TYPE 或 MANUAL_CONCEPT）",
    )
    p.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help=".env 文件路径（默认 tools/manual_concepts_lightrag/.env）",
    )
    p.add_argument(
        "--entity-whitelist",
        default=str(DEFAULT_ENTITY_WHITELIST_PATH),
        help="实体白名单路径（默认 entity_whitelist_draft.txt）",
    )
    p.add_argument(
        "--disable-whitelist",
        action="store_true",
        help="临时关闭 concept_name 白名单过滤",
    )
    p.add_argument(
        "--include-non-correct",
        action="store_true",
        help="同时导入 review_status 不是 correct 的记录",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="ainsert_custom_kg 聚合批量大小（默认 100）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印统计，不初始化 LightRAG、不写入",
    )
    p.add_argument(
        "--dump-custom-kg",
        default=None,
        help="将构造后的 custom_kg 输出为 JSON 文件（dry-run 也可用）",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    env_path = Path(args.env_file).expanduser()
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / args.env_file
    if env_path.exists():
        load_dotenv(env_path, override=False)
        logger.info(f"loaded env | env_file={env_path}")
    else:
        logger.warning(f"env file not found | env_file={env_path}")

    try:
        return asyncio.run(run(args))
    except RuntimeError as e:
        logger.error(f"启动失败 | error={e}")
        return 2
    except KeyboardInterrupt:
        logger.warning("interrupted by user")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
