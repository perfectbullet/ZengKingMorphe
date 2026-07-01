#!/usr/bin/env python
"""
test_manual_concept_lightrag.py
================================
测试手动数学概念在 LightRAG 中的结构化召回 + 严格回答（最小闭环 v1）。

核心流程：
1. 读取同一个 --config，建立 file_path -> metadata / concept_name -> metadata 映射
2. 初始化同一个 working_dir 的 LightRAG
3. rag.aquery_data(query, param=QueryParam(...)) 做结构化召回
4. 打印 entities / relationships / chunks / references / metadata
5. 判断是否命中手动概念（[HIT] / [MISS]）：只有 entity_type==MANUAL_MATH_CONCEPT
   且 entity_name 命中 config 中 concept_name 才算强 HIT；source_type / chunk file_path
   命中仅作辅助原因，不能单独判 HIT。
6. --answer：优先使用 matched["md_content"]，其次才读 matched["md_path"] 文件，
   使用严格 prompt 生成回答（确保 custom KG 模式无 md_path 也能依据 JSONL md_content 回答）

注意：本脚本是独立验证脚本，不依赖也不修改 ai-service 业务代码。
"""

from __future__ import annotations

import argparse
import aiohttp
import asyncio
import inspect
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
from lightrag.base import QueryParam  # noqa: E402
from lightrag.llm.openai import openai_complete_if_cache, openai_embed  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402


# ===========================================================================
# 与 import 脚本一致的公共工具（独立脚本，复制保持解耦）
# ===========================================================================
def resolve_path(path_str: str) -> Path:
    """路径解析：支持绝对路径，也支持相对 ai-service 根目录 / 仓库根目录 / cwd 解析。"""
    if not path_str:
        return PROJECT_ROOT / "<empty>"
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    candidates = [
        PROJECT_ROOT / path_str,
        PROJECT_ROOT.parent.parent / path_str,    # 相对仓库根（兼容 ai-service/data/... 形式）
        Path.cwd() / path_str,
    ]
    for c in candidates:
        if c.exists():
            return c
    return PROJECT_ROOT / path_str


def load_concept_items(config_path: Path) -> list[dict]:
    """兼容 .jsonl（每行一个 JSON）/ 顶层 list / dict(items|concepts|documents|data) 结构。"""
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
    with path.open(encoding="utf-8") as source:
        return {
            line.strip()
            for line in source
            if line.strip() and not line.lstrip().startswith("#")
        }


HEADING_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


def extract_concept_name(md_content: str, md_path: Path | None) -> str:
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


def resolve_rerank_config() -> dict | None:
    """读取 RERANK_MODEL / RERANK_BASE_URL；未配置返回 None（查询时自动跳过 rerank）。"""
    model = _pick_env(["RERANK_MODEL"], "RERANK_MODEL", required=False)
    base_url = _pick_env(["RERANK_BASE_URL"], "RERANK_BASE_URL", required=False)
    if not model or not base_url:
        return None
    return {"model": model, "base_url": base_url.rstrip("/")}


def build_rerank_model_func(rerank: dict | None):
    """构造 LightRAG 期望的 rerank_model_func(query, documents, top_n) -> [{index, relevance_score}]。
    rerank 为 None 时返回 None，LightRAG 会自动跳过 rerank。"""
    if not rerank:
        return None

    async def _rerank_func(query: str, documents: list[str], top_n: int | None = None):
        payload: dict = {"model": rerank["model"], "query": query, "documents": documents}
        if top_n:
            payload["top_n"] = top_n
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{rerank['base_url']}/v1/rerank",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        return [
            {"index": r["index"], "relevance_score": r["relevance_score"]}
            for r in data.get("results", [])
        ]

    return _rerank_func


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


def build_rag(working_dir: Path, llm: dict, emb: dict, rerank_model_func=None) -> LightRAG:
    working_dir.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(working_dir),
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        addon_params={
            "language": "Chinese",
            "entity_types": ["MANUAL_MATH_CONCEPT"],
        },
        llm_model_func=build_llm_model_func(llm),
        embedding_func=build_embedding_func(emb),
        rerank_model_func=rerank_model_func,
    )
    return rag


# ===========================================================================
# 映射构建
# ===========================================================================
def build_mappings(
    items: list[dict],
    whitelist: set[str] | None = None,
    include_non_correct: bool = False,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """返回 (file_path -> metadata, concept_name -> metadata)。"""
    mapping_by_file: dict[str, dict] = {}
    mapping_by_name: dict[str, dict] = {}
    for raw in items:
        if not isinstance(raw, dict):
            continue
        if raw.get("review_status") != "correct" and not include_non_correct:
            continue
        md_path_raw = raw.get("md_path")
        md_path = resolve_path(md_path_raw) if md_path_raw else None

        file_path = default_file_path(raw)

        name = raw.get("concept_name") or raw.get("name")
        if not name:
            inline = (
                raw.get("md_content") or raw.get("content")
                or raw.get("markdown") or raw.get("text")
            )
            if md_path and md_path.exists():
                name = extract_concept_name(md_path.read_text(encoding="utf-8"), md_path)
            elif inline:
                name = extract_concept_name(inline, None)
            elif md_path_raw:
                name = Path(md_path_raw).stem
            elif raw.get("doc_id"):
                name = raw["doc_id"]

        if whitelist is not None and name not in whitelist:
            continue

        entry = {
            **raw,
            "concept_name": name,
            "file_path": file_path,
            "resolved_md_path": str(md_path) if md_path else None,
        }
        if file_path:
            mapping_by_file[file_path] = entry
        if name:
            mapping_by_name[name] = entry

    logger.info(
        f"mapping_by_file_count={len(mapping_by_file)} | "
        f"mapping_by_name_count={len(mapping_by_name)}"
    )
    for fp, e in mapping_by_file.items():
        logger.info(
            f"  mapping | file_path={fp} | concept_name={e.get('concept_name')} | "
            f"doc_id={e.get('doc_id')} | md_path={e.get('md_path')}"
        )
    return mapping_by_file, mapping_by_name


# ===========================================================================
# QueryParam 构建
# ===========================================================================
def build_query_param(args: argparse.Namespace) -> QueryParam:
    sig = inspect.signature(QueryParam)
    kwargs: dict[str, Any] = {
        "mode": args.mode,
        "top_k": args.top_k,
        "chunk_top_k": args.chunk_top_k,
    }
    if "enable_rerank" in sig.parameters:
        kwargs["enable_rerank"] = bool(args.enable_rerank)
        logger.info(
            f"QueryParam 支持 enable_rerank | 已传入 enable_rerank={kwargs['enable_rerank']}"
        )
    else:
        logger.info("QueryParam 不支持 enable_rerank | 跳过该参数")
    logger.info(f"QueryParam | {kwargs}")
    return QueryParam(**kwargs)


# ===========================================================================
# 召回结果展示
# ===========================================================================
def print_recall(result: dict) -> None:
    data = result.get("data") or {}
    entities = data.get("entities") or []
    relationships = data.get("relationships") or []
    chunks = data.get("chunks") or []
    references = data.get("references") or []
    metadata = result.get("metadata") or {}

    logger.info(
        f"status={result.get('status')} | message={result.get('message')} | "
        f"entities={len(entities)} | relationships={len(relationships)} | "
        f"chunks={len(chunks)} | references={len(references)}"
    )
    logger.info(f"metadata={json.dumps(metadata, ensure_ascii=False)}")

    for e in entities[:20]:
        logger.info(
            f"  ENTITY | name={e.get('entity_name')} | type={e.get('entity_type')} | "
            f"source_type={e.get('source_type')} | file_path={e.get('file_path')}"
        )
    for r in relationships[:10]:
        logger.info(
            f"  REL    | {r.get('src_id')} -> {r.get('tgt_id')} | "
            f"weight={r.get('weight')} | file_path={r.get('file_path')}"
        )
    for c in chunks[:10]:
        content = (c.get("content") or "").replace("\n", " ")
        logger.info(
            f"  CHUNK  | file_path={c.get('file_path')} | content={content[:120]}"
        )
    for r in references[:10]:
        logger.info(
            f"  REF    | reference_id={r.get('reference_id')} | file_path={r.get('file_path')}"
        )


# ===========================================================================
# 命中判定
# ===========================================================================
def detect_hit(
    result: dict,
    mapping_by_file: dict[str, dict],
    mapping_by_name: dict[str, dict],
    query: str = "",
) -> tuple[bool, list[str], dict | None]:
    """严格命中判定：

    只有 entity_type==MANUAL_MATH_CONCEPT 且 entity_name 命中 config 中 concept_name
    的实体才算强 HIT。
    - source_type==manual_math_concept 只能作为辅助原因，不能单独算 HIT。
    - chunk file_path 命中只能作为辅助原因，不能单独算 HIT。
    - 公式/变量/符号/Unknown 类型节点不算 HIT（被 entity_type 门槛过滤）。
    """
    reasons: list[str] = []
    data = result.get("data") or {}
    entities = data.get("entities") or []
    chunks = data.get("chunks") or []

    manual_candidates: list[dict] = []
    seen_candidate_names: set[str] = set()
    auxiliary_reason_added = False
    chunk_reason_added = False

    for e in entities:
        name = e.get("entity_name")
        etype = e.get("entity_type")
        # 公式 / 变量 / 符号 / Unknown 等非概念节点一律忽略
        if not name or etype in (None, "", "UNKNOWN"):
            continue
        if name not in mapping_by_name:
            continue
        if etype == "MANUAL_MATH_CONCEPT":
            reasons.append(f"entity_type==MANUAL_MATH_CONCEPT | entity_name={name}")
            if name not in seen_candidate_names:
                seen_candidate_names.add(name)
                manual_candidates.append(mapping_by_name[name])
        elif e.get("source_type") == "manual_math_concept":
            # 仅作辅助原因，不计入 HIT 候选
            if not auxiliary_reason_added:
                reasons.append(
                    f"source_type==manual_math_concept(auxiliary, not a HIT by itself) | entity_name={name}"
                )
                auxiliary_reason_added = True

    # chunk file_path 命中仅作为辅助原因
    for c in chunks:
        fp = c.get("file_path")
        if fp and fp in mapping_by_file:
            if not chunk_reason_added:
                reasons.append(
                    f"chunk file_path==manual_math_concept(auxiliary, not a HIT by itself) | file_path={fp}"
                )
                chunk_reason_added = True
            break

    # matched 优先级：
    #   1) MANUAL_MATH_CONCEPT 且 concept_name 出现在 query
    #   2) MANUAL_MATH_CONCEPT 且 entity_name 在 mapping_by_name
    #   3) 其他都不算 HIT
    matched: dict | None = None

    def _concept_in_query(c: dict | None) -> bool:
        cn = (c or {}).get("concept_name") or ""
        return bool(cn) and cn in query

    for c in manual_candidates:
        if _concept_in_query(c):
            matched = c
            break
    if not matched and manual_candidates:
        matched = manual_candidates[0]

    # 去重
    seen: set[str] = set()
    deduped: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            deduped.append(r)

    return matched is not None, deduped, matched


# ===========================================================================
# 严格回答
# ===========================================================================
STRICT_SYSTEM_PROMPT = """你是一个严格依据资料回答的数学概念讲解助手。

你只能依据用户提供的【手动概念 Markdown】回答。
禁止补充 Markdown 中没有出现的内容。
禁止扩展到历史背景、推广形式或其他应用场景。
如果 Markdown 中没有相关内容，就明确说明“该概念文档中未提供”。

回答要求：
1. 贴近原文回答；
2. 如果原文本来就完善，直接照搬原文；
3. 如果原文不完整，你做最小幅度的修改后回复；
4. 不要引入文档之外的知识。"""


async def generate_strict_answer(llm: dict, query: str, md_content: str) -> str:
    user_prompt = (
        f"用户问题：{query}\n\n"
        f"【手动概念 Markdown】\n{md_content}\n\n"
        f"请严格依据上述 Markdown 回答用户问题，不要引入文档之外的知识。"
    )
    return await openai_complete_if_cache(
        llm["model"],
        user_prompt,
        system_prompt=STRICT_SYSTEM_PROMPT,
        history_messages=[],
        base_url=llm["base_url"],
        api_key=llm["api_key"],
    )


# ===========================================================================
# 主流程
# ===========================================================================
async def run(args: argparse.Namespace) -> int:
    config_path = resolve_path(args.config)
    if not config_path.exists():
        logger.error(f"config 不存在 | config={config_path}")
        return 1

    working_dir = Path(args.working_dir).expanduser()
    if not working_dir.is_absolute():
        working_dir = PROJECT_ROOT / args.working_dir

    logger.info(f"config={config_path}")
    logger.info(f"working_dir={working_dir}")
    logger.info(
        f"query={args.query!r} | mode={args.mode} | top_k={args.top_k} | "
        f"chunk_top_k={args.chunk_top_k} | answer={args.answer} | dump={args.dump}"
    )

    items = load_concept_items(config_path)
    whitelist: set[str] | None = None
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
    mapping_by_file, mapping_by_name = build_mappings(
        items,
        whitelist=whitelist,
        include_non_correct=args.include_non_correct,
    )

    llm = resolve_llm_config()
    emb = resolve_embedding_config(llm["base_url"], llm["api_key"])
    rerank = resolve_rerank_config()
    logger.info(
        f"llm_model={llm['model']} | llm_base_url={llm['base_url']} | "
        f"embedding_model={emb['model']} | embedding_dim={emb['dim']} | "
        f"embedding_base_url={emb['base_url']}"
    )
    if rerank:
        logger.info(f"rerank_model={rerank['model']} | rerank_base_url={rerank['base_url']}")
    else:
        logger.info("rerank 未配置 | 查询时跳过 rerank")

    rag = build_rag(working_dir, llm, emb, build_rerank_model_func(rerank))
    await rag.initialize_storages()
    logger.info("initialize_storages OK")

    param = build_query_param(args)
    try:
        result = await rag.aquery_data(args.query, param=param)
    finally:
        try:
            await rag.finalize_storages()
            logger.info("finalize_storages OK")
        except Exception as e:
            logger.warning(
                f"finalize_storages WARN | error_type={type(e).__name__} | error={e}"
            )

    # dump
    if args.dump:
        dump_path = Path(args.dump).expanduser()
        if not dump_path.is_absolute():
            dump_path = PROJECT_ROOT / args.dump
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(f"dump OK | path={dump_path}")

    # 召回展示
    print_recall(result)

    # 命中判定
    hit, reasons, matched = detect_hit(result, mapping_by_file, mapping_by_name, args.query)
    logger.info("[MANUAL CONCEPT ROUTING] result=" + ("[HIT]" if hit else "[MISS]"))
    for r in reasons:
        logger.info(f"  reason: {r}")
    if matched:
        logger.info(
            "  matched_concept="
            + json.dumps(
                {
                    "doc_id": matched.get("doc_id"),
                    "concept_name": matched.get("concept_name"),
                    "file_path": matched.get("file_path"),
                    "md_path": matched.get("md_path"),
                    "strict": matched.get("strict"),
                },
                ensure_ascii=False,
            )
        )

    if not hit:
        logger.warning("未命中手动概念，可尝试 --mode hybrid / --mode mix / --mode global")
        return 0

    # 严格回答：优先整条记录的 md_content，其次读取 md_path 文件。
    if args.answer:
        md_content = (
            (matched or {}).get("md_content")
            or (matched or {}).get("content")
            or (matched or {}).get("markdown")
            or (matched or {}).get("text")
        )
        md_path_raw = matched.get("md_path") if matched else None
        md_path = resolve_path(md_path_raw) if md_path_raw else None
        if md_content:
            logger.info(
                f"generate strict answer | source=inline_md_content | chars={len(md_content)}"
            )
        elif md_path and md_path.exists():
            md_content = md_path.read_text(encoding="utf-8")
            logger.info(f"generate strict answer | md_path={md_path} | chars={len(md_content)}")
        else:
            logger.error(f"--answer 无法获取 Markdown 内容 | md_path={md_path_raw}")
            return 1
        answer = await generate_strict_answer(llm, args.query, md_content)
        print("\n===== STRICT ANSWER =====")
        print(answer)
        print("===== END STRICT ANSWER =====\n")

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="测试手动数学概念在 LightRAG 中的结构化召回与严格回答"
    )
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="概念 JSONL/JSON 配置路径",
    )
    p.add_argument(
        "--working-dir",
        required=True,
        help="LightRAG working_dir（与 import 脚本一致，相对 ai-service 根目录或绝对路径）",
    )
    p.add_argument("--query", required=True, help="查询问题")
    p.add_argument(
        "--mode",
        default="local",
        choices=["local", "global", "hybrid", "naive", "mix"],
        help="查询模式（默认 local）",
    )
    p.add_argument("--top-k", type=int, default=10, help="实体/关系 top_k（默认 10）")
    p.add_argument("--chunk-top-k", type=int, default=10, help="chunk top_k（默认 10）")
    p.add_argument(
        "--enable-rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 rerank（默认 True，需配置 RERANK_*；用 --no-enable-rerank 关闭）",
    )
    p.add_argument(
        "--answer",
        action="store_true",
        help="命中后读取整篇 Markdown，使用严格 prompt 生成回答",
    )
    p.add_argument(
        "--include-non-correct",
        action="store_true",
        help="查询映射同时包含 review_status 不是 correct 的记录",
    )
    p.add_argument(
        "--entity-whitelist",
        default=str(DEFAULT_ENTITY_WHITELIST_PATH),
        help="实体白名单路径",
    )
    p.add_argument(
        "--disable-whitelist",
        action="store_true",
        help="临时关闭查询映射的白名单过滤",
    )
    p.add_argument("--dump", default=None, help="将结构化召回结果 dump 到 JSON 文件")
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
        logger.error(f"启动失败 | error={e}")
        return 2
    except KeyboardInterrupt:
        logger.warning("interrupted by user")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
