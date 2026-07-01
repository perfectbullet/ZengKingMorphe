#!/usr/bin/env python
"""
import_manual_concepts_lightrag.py
==================================
手动教材数学概念 -> LightRAG 导入脚本。

核心流程：
1. 读取 JSONL / JSON 配置，每条记录视为一个教材数学概念
2. 默认仅处理 review_status=correct 且 concept_name 位于白名单的记录
3. 插入 Markdown 全文供 chunk 检索
4. 清理图中非白名单实体
5. 以 concept_name 手动 upsert MANUAL_MATH_CONCEPT 实体

注意：本脚本是独立验证脚本，不依赖也不修改 ai-service 业务代码（conversation_nodes.py 等）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------------
# 项目根目录：子项目根 = 脚本所在目录（tools/manual_concepts_lightrag）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent.parent
DEFAULT_CONFIG_PATH = (
    REPO_ROOT
    / "ai-service"
    / "data"
    / "math_concepts"
    / "05_selective3_math_concepts_definition_blocks_with_concept_name_20260630.jsonl"
)
DEFAULT_ENTITY_WHITELIST_PATH = PROJECT_ROOT / "entity_whitelist_draft.txt"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lightrag import LightRAG  # noqa: E402
from lightrag.llm.openai import openai_complete_if_cache, openai_embed  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402


# ---------------------------------------------------------------------------
# 配置 / 路径工具
# ---------------------------------------------------------------------------
def resolve_path(path_str: str) -> Path:
    """
    路径解析：支持绝对路径，也支持相对 ai-service 根目录 / 仓库根目录 / cwd 解析。
    依次尝试候选目录，返回第一个存在的路径；都不存在则回退到相对 ai-service 根目录。
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
    # JSON Lines：每行一个独立 JSON 对象（不能整体 json.loads）
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
    """读取实体白名单；跳过空行和以 # 开头的注释。"""
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


def write_entity_whitelist(path: Path, items: list[dict]) -> int:
    """按配置中的 concept_name 顺序生成白名单草案。"""
    names: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("concept_name") or item.get("name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = ["# 高中数学选择性必修第三册：教材概念实体白名单草案", *names]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")
    return len(names)


HEADING_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


def extract_concept_name(md_content: str, md_path: Path | None) -> str:
    """优先 Markdown 一级标题，其次文件名 stem，最后兜底 unknown_concept。"""
    m = HEADING_RE.search(md_content or "")
    if m:
        return m.group(1).strip()
    if md_path:
        return md_path.stem
    return "unknown_concept"


def default_file_path(item: dict) -> str | None:
    """file_path 默认值：item.file_path > manual_math_concepts/{md_path文件名} > manual_math_concepts/{doc_id}.md。"""
    if item.get("file_path"):
        return str(item["file_path"])
    md_path_raw = item.get("md_path")
    if md_path_raw:
        return f"manual_math_concepts/{Path(md_path_raw).name}"
    doc_id = item.get("doc_id")
    if doc_id:
        return f"manual_math_concepts/{doc_id}.md"
    return None


def _sanitize_entity_value(v: Any) -> Any:
    """GraphML(networkx) 不支持 list / None 数据值，否则写 graph 会抛 TypeError 并把 graph
    文件截断为 0 字节导致后续加载失败。list/tuple → 逗号分隔串；None → 空串；其余原样返回。"""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return v


def normalize_item(item: dict, md_content: str, md_path: Path | None) -> dict:
    """补齐 concept_name / file_path / strict / aliases。"""
    concept_name = item.get("concept_name") or item.get("name")
    if not concept_name:
        concept_name = extract_concept_name(md_content, md_path)

    file_path = default_file_path(item) or f"manual_math_concepts/{concept_name}.md"

    strict_raw = item.get("strict")
    strict = True if strict_raw is None else bool(strict_raw)

    aliases = item.get("aliases") or []

    return {
        **item,
        "concept_name": concept_name,
        "file_path": file_path,
        "strict": strict,
        "aliases": aliases,
    }


# ---------------------------------------------------------------------------
# 环境变量解析（兼容任务规格候选名 + 本仓库 .env 实际变量名）
# ---------------------------------------------------------------------------
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
# 模型函数
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


# ---------------------------------------------------------------------------
# LightRAG 构建
# ---------------------------------------------------------------------------
# 当前 LightRAG 版本的 extract_entities 固定读取 lightrag.prompt.PROMPTS，
# addon_params 没有自定义抽取 prompt key。保留这份策略常量用于将来升级；当前
# 主要约束由 entity_types、concept_name 白名单和导入后实体清理共同完成。
STRICT_ENTITY_EXTRACTION_PROMPT = """你是教材数学概念抽取器。
只允许抽取输入文本明确出现的教材数学概念，实体类型只能是 MANUAL_MATH_CONCEPT。
如果已知文本对应一个 concept_name，应优先只输出该 concept_name。
禁止抽取公式、变量、数字、符号、运算词、人名、例子对象、解题步骤和推导过程。
禁止根据常识补充文档外概念。
关系只允许明显的“包含、相关、推广、前置概念、应用于”；没有明确关系时不要输出。
"""


def build_rag(working_dir: Path, llm: dict, emb: dict, whitelist: set[str] | None = None) -> LightRAG:
    working_dir.mkdir(parents=True, exist_ok=True)
    # 把权威概念清单 + 禁止规则塞进 entity_types 列表。LightRAG 会把它注入
    # entity_extraction prompt 的 <Entity_types>[{entity_types}] 块——这是唯一能
    # 影响抽取阶段 LLM 的通道（addon_params 不支持自定义 system_prompt）。
    if whitelist:
        entity_types = [
            "教材数学概念(必须是【权威清单】内、教材正式命名的概念/定理/公式)",
            "MANUAL_MATH_CONCEPT(手动导入的数学概念入口)",
            "【权威清单】只抽取下列概念: " + " | ".join(sorted(whitelist)),
            "【严格禁止】单字母变量(a,b,x,n,m) / 数学表达式碎片((a+b)^n,T_k,C_n^k) / 运算步骤词(合并,展开,代入,相乘) / 章节图表引用(公式1,图6.2-4,表7.2-2,问题1,性质1) / 人名(棣莫弗,贝叶斯,高斯) / 例题情境词(共享自行车,身高,施肥量,体重) / 通用泛指词(元素,顺序,步骤,方法)",
        ]
    else:
        entity_types = ["MANUAL_MATH_CONCEPT"]
    rag = LightRAG(
        working_dir=str(working_dir),
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        addon_params={
            "language": "Chinese",
            "entity_types": entity_types,
        },
        llm_model_func=build_llm_model_func(llm),
        embedding_func=build_embedding_func(emb),
    )
    return rag


async def prune_non_whitelisted_entities(
    rag: LightRAG,
    whitelist: set[str],
) -> tuple[int, int]:
    """通过 LightRAG 公开 API 删除非白名单实体及其关系。"""
    deleted = 0
    failed = 0
    for entity_name in await rag.get_graph_labels():
        if entity_name in whitelist:
            continue
        result = await rag.adelete_by_entity(entity_name)
        if getattr(result, "status", None) == "success":
            deleted += 1
            logger.info(f"prune entity OK | entity_name={entity_name}")
        else:
            failed += 1
            logger.error(
                f"prune entity FAILED | entity_name={entity_name} | result={result}"
            )
    return deleted, failed


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
async def run(args: argparse.Namespace) -> int:
    config_path = resolve_path(args.config)
    if not config_path.exists():
        logger.error(f"config 不存在 | config={config_path}")
        return 1

    items = load_concept_items(config_path)
    whitelist_path = resolve_path(args.entity_whitelist)
    if args.generate_whitelist_only:
        generated = write_entity_whitelist(whitelist_path, items)
        logger.info(
            f"whitelist generated | path={whitelist_path} | entity_count={generated}"
        )
        return 0

    whitelist: set[str] = set()
    if args.disable_whitelist:
        logger.warning("entity whitelist disabled by --disable-whitelist")
    else:
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
    logger.info(f"replace={args.replace}")
    logger.info(f"include_non_correct={args.include_non_correct}")

    llm = resolve_llm_config()
    emb = resolve_embedding_config(llm["base_url"], llm["api_key"])
    logger.info(
        f"llm_model={llm['model']} | llm_base_url={llm['base_url']} | "
        f"embedding_model={emb['model']} | embedding_dim={emb['dim']} | "
        f"embedding_base_url={emb['base_url']}"
    )

    rag = build_rag(working_dir, llm, emb, whitelist)
    await rag.initialize_storages()
    logger.info("initialize_storages OK")

    review_status_skipped = 0
    whitelist_skipped = 0
    imported = 0
    failures = 0
    pending_entities: list[tuple[str, dict]] = []
    try:
        for idx, raw_item in enumerate(items, 1):
            if not isinstance(raw_item, dict):
                logger.warning(f"skip non-dict item #{idx} | type={type(raw_item).__name__}")
                failures += 1
                continue

            review_status = raw_item.get("review_status")
            if review_status != "correct" and not args.include_non_correct:
                review_status_skipped += 1
                logger.info(
                    f"skip review_status | item={idx} | "
                    f"concept_name={raw_item.get('concept_name')} | "
                    f"review_status={review_status!r}"
                )
                continue

            doc_id = raw_item.get("doc_id")
            md_path_raw = raw_item.get("md_path")
            if not doc_id:
                logger.warning(
                    f"skip item #{idx} | missing doc_id | keys={list(raw_item.keys())}"
                )
                failures += 1
                continue

            # 解析 Markdown 内容：优先 md_path 文件，其次 md_content 内联（jsonl 常见）
            md_path: Path | None = None
            md_content: str | None = None
            if md_path_raw:
                md_path = resolve_path(md_path_raw)
                if md_path.exists():
                    md_content = md_path.read_text(encoding="utf-8")
                else:
                    logger.warning(
                        f"md_path 文件不存在，回退 md_content 内联 | doc_id={doc_id} | md_path={md_path}"
                    )
                    md_path = None
            if not md_content:
                md_content = (
                    raw_item.get("md_content")
                    or raw_item.get("content")
                    or raw_item.get("markdown")
                    or raw_item.get("text")
                )
            if not md_content:
                logger.error(
                    f"no md content | doc_id={doc_id} | 既无 md_path 文件也无 md_content 内联"
                )
                failures += 1
                continue

            item = normalize_item(raw_item, md_content, md_path)
            concept_name = str(item["concept_name"]).strip()
            file_path = item["file_path"]
            strict = item["strict"]
            aliases = item["aliases"]

            if not args.disable_whitelist and concept_name not in whitelist:
                whitelist_skipped += 1
                logger.info(
                    f"skip whitelist | item={idx} | doc_id={doc_id} | "
                    f"concept_name={concept_name}"
                )
                continue

            logger.info(
                f"[{idx}/{len(items)}] doc_id={doc_id} | concept_name={concept_name} | "
                f"md_path={md_path} | file_path={file_path} | strict={strict} | aliases={aliases}"
            )

            # 4. --replace：先删除旧 doc
            if args.replace:
                try:
                    await rag.adelete_by_doc_id(doc_id)
                    logger.info(f"delete old doc OK | doc_id={doc_id}")
                except Exception as e:
                    logger.warning(
                        f"delete old doc WARN（忽略继续）| doc_id={doc_id} | "
                        f"error_type={type(e).__name__} | error={e}"
                    )

            # 5. 插入全文
            try:
                await rag.ainsert(input=md_content, ids=doc_id, file_paths=file_path)
                logger.info(f"insert OK | doc_id={doc_id} | file_path={file_path}")
            except Exception as e:
                logger.error(
                    f"insert FAILED | doc_id={doc_id} | file_path={file_path} | "
                    f"error_type={type(e).__name__} | error={e}",
                    exc_info=True,
                )
                failures += 1
                continue

            # 6. 收集实体元数据：实体创建推迟到所有文档 insert 完成后统一执行（见阶段2）。
            # 若每条 insert 后立即 edit_entity，后续文档 insert 时的实体合并会用 LLM 重新
            # 判定 entity_type，覆盖我们手动设置的 MANUAL_MATH_CONCEPT。
            entity_data = {
                "entity_type": "MANUAL_MATH_CONCEPT",
                "description": md_content,
                "source_id": doc_id,
                "doc_id": doc_id,
                "file_path": file_path,
                "md_path": md_path_raw,
                "source_type": item.get("source_type") or "manual_math_concept",
                "review_status": review_status,
                "strict": strict,
                "aliases": aliases,
            }
            # GraphML 不支持 list / None 数据值，统一 sanitize 防止写 graph 时损坏文件
            entity_data = {k: _sanitize_entity_value(v) for k, v in entity_data.items()}
            pending_entities.append((concept_name, entity_data))

        if not args.disable_whitelist:
            logger.info("stage-2 prune non-whitelisted entities")
            pruned, prune_failures = await prune_non_whitelisted_entities(
                rag, whitelist
            )
            failures += prune_failures
            logger.info(
                f"prune done | deleted={pruned} | failures={prune_failures}"
            )

        # ---------- 阶段3：所有 insert 完成后，统一创建/更新 MANUAL_MATH_CONCEPT 实体 ----------
        # 此时实体合并已充分发生，最终态不会被后续 insert 覆盖。
        logger.info(f"stage-3 upsert entities | count={len(pending_entities)}")
        for concept_name, entity_data in pending_entities:
            try:
                try:
                    await rag.acreate_entity(
                        entity_name=concept_name,
                        entity_data=entity_data,
                    )
                    logger.info(f"create entity OK | concept_name={concept_name}")
                except ValueError:
                    logger.info(
                        f"entity already exists, update it | concept_name={concept_name}"
                    )

                # acreate_entity 仅保存标准字段；再 edit 一次，将经过 sanitize 的
                # doc_id/source_type/review_status/strict/aliases 写入 GraphML。
                await rag.aedit_entity(
                    entity_name=concept_name,
                    updated_data=entity_data,
                    allow_rename=False,
                    allow_merge=False,
                )
                imported += 1
                logger.info(
                    f"upsert entity OK | concept_name={concept_name} | "
                    "entity_type=MANUAL_MATH_CONCEPT"
                )
            except Exception as e:
                logger.error(
                    f"upsert entity FAILED | concept_name={concept_name} | "
                    f"error_type={type(e).__name__} | error={e}",
                    exc_info=True,
                )
                failures += 1
    finally:
        try:
            await rag.finalize_storages()
            logger.info("finalize_storages OK")
        except Exception as e:
            logger.warning(
                f"finalize_storages WARN | error_type={type(e).__name__} | error={e}"
            )

    logger.info(
        f"import done | total={len(items)} | "
        f"review_status_skipped={review_status_skipped} | "
        f"whitelist_skipped={whitelist_skipped} | imported={imported} | "
        f"failures={failures}"
    )
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="导入手动数学概念到 LightRAG（Markdown 全文 + MANUAL_MATH_CONCEPT 实体）"
    )
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="概念 JSONL/JSON 配置路径",
    )
    p.add_argument(
        "--working-dir",
        required=True,
        help="LightRAG working_dir（相对 ai-service 根目录或绝对路径，不写死在代码里）",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="插入前先 rag.adelete_by_doc_id(doc_id)，失败仅 WARN 不中断",
    )
    p.add_argument(
        "--include-non-correct",
        action="store_true",
        help="同时导入 review_status 不是 correct 的记录",
    )
    p.add_argument(
        "--entity-whitelist",
        default=str(DEFAULT_ENTITY_WHITELIST_PATH),
        help="实体白名单路径",
    )
    p.add_argument(
        "--disable-whitelist",
        action="store_true",
        help="临时关闭 concept_name 白名单过滤和图实体清理",
    )
    p.add_argument(
        "--generate-whitelist-only",
        action="store_true",
        help="从 --config 生成白名单后退出",
    )
    p.add_argument(
        "--env-file",
        default=str(PROJECT_ROOT / ".env"),
        help=".env 文件路径（默认 ai-service/.env）",
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
        # 缺失必要环境变量时给出明确提示
        logger.error(f"启动失败 | error={e}")
        return 2
    except KeyboardInterrupt:
        logger.warning("interrupted by user")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
