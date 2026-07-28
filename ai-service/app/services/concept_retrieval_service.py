"""
人工概念检索服务。

负责从人工概念库中检索概念内容，支持：
1. 本地精确匹配（query 包含 concept_name 或 alias）
2. LightRAG local 模式召回
3. 严格命中判定（entity_type 必须为 MANUAL_CONCEPT 或 MANUAL_MATH_CONCEPT）

设计要点：
- 复制 tools/manual_concepts_lightrag 中的必要逻辑，不直接 import
- 命中人工概念库后完全跳过知识库流式检索，只依据 concept_context 生成答案
- 支持新旧两种 entity_type（MANUAL_CONCEPT 新默认，MANUAL_MATH_CONCEPT legacy）
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


# =============================================================================
# 数据结构
# =============================================================================
@dataclass
class ConceptContext:
    """人工概念检索结果上下文。"""
    hit: bool
    concept_name: str | None = None
    doc_id: str | None = None
    domain: str | None = None
    entity_type: str | None = None
    source_type: str | None = None
    content: str | None = None
    file_path: str | None = None
    md_path: str | None = None
    confidence: float = 0.0
    hit_reason: str = "not_found"
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return asdict(self)


# =============================================================================
# 配置解析
# =============================================================================
def _resolve_path(path_str: str, service_root: Path) -> Path:
    """
    路径解析：支持绝对路径，也支持相对 ai-service 根目录 / 仓库根目录 / cwd 解析。

    Args:
        path_str: 路径字符串
        service_root: ai-service 根目录

    Returns:
        解析后的 Path 对象
    """
    if not path_str:
        return service_root / "<empty>"

    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p

    # 相对路径候选
    candidates = [
        service_root / path_str,              # 相对 ai-service 根目录
        service_root.parent / path_str,       # 相对仓库根目录
        Path.cwd() / path_str,                # 相对当前工作目录
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # 都不存在则返回相对 service_root 的路径
    return service_root / path_str


def _pick_env(candidates: list[str], label: str, required: bool = True) -> str | None:
    """从多个候选环境变量名中选择第一个非空值。"""
    for name in candidates:
        val = os.getenv(name)
        if val and val.strip():
            return val.strip()

    if required:
        logger.warning(f"Missing required environment variable for {label}, candidates: {candidates}")
    return None


def resolve_llm_config() -> dict[str, Any]:
    """解析 LLM 配置。"""
    model = _pick_env(["LLM_MODEL", "OPENAI_MODEL"], "LLM_MODEL")
    base_url = _pick_env(
        ["LLM_BINDING_HOST", "OPENAI_BASE_URL", "LLM_BASE_URL"],
        "LLM_BASE_URL"
    )
    api_key = _pick_env(
        ["LLM_BINDING_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"],
        "LLM_API_KEY",
        required=False,
    ) or "no-api-key"

    return {
        "model": model,
        "base_url": base_url,
        "api_key": api_key
    }


def resolve_embedding_config(llm_base_url: str, llm_api_key: str) -> dict[str, Any]:
    """解析 Embedding 配置。"""
    model = _pick_env(
        ["EMBEDDING_MODEL", "VLLM_EMBED_MODEL", "VLLM_EMBEDDING_MODEL"],
        "EMBEDDING_MODEL"
    )

    dim_raw = _pick_env(["EMBEDDING_DIM", "VLLM_EMBED_DIM"], "EMBEDDING_DIM")
    try:
        dim = int(dim_raw) if dim_raw else 768
    except (TypeError, ValueError):
        logger.warning(f"Invalid EMBEDDING_DIM: {dim_raw}, using default 768")
        dim = 768

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

    return {
        "model": model,
        "dim": dim,
        "base_url": base_url,
        "api_key": api_key
    }


# =============================================================================
# 配置加载工具
# =============================================================================
def load_concept_items(config_path: Path) -> list[dict]:
    """
    读取概念配置，兼容多种结构。

    支持：
    - JSON Lines（.jsonl）：每行一个独立 JSON 对象
    - 顶层 list
    - 顶层 dict 中的 items / concepts / documents / data 字段

    Args:
        config_path: 配置文件路径

    Returns:
        概念项列表
    """
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return []

    # JSON Lines 格式
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
                    logger.warning(f"jsonl parse error at line {line_no}: {e}")
                    continue
                items.append(obj)
        logger.info(f"Loaded jsonl config: {config_path}, concept_count={len(items)}")
        return items

    # JSON 格式
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return []

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = None
        for key in ("items", "concepts", "documents", "data"):
            if isinstance(raw.get(key), list):
                items = raw[key]
                logger.info(f"Using dict field '{key}' from config")
                break

        if items is None:
            logger.warning(f"Config dict has no items/concepts/documents/data field, keys={list(raw.keys())}")
            return []
    else:
        logger.warning(f"Config is neither list nor dict, type={type(raw).__name__}")
        return []

    logger.info(f"Loaded json config: {config_path}, concept_count={len(items)}")
    return items


def load_entity_whitelist(whitelist_path: Path | None) -> set[str]:
    """
    读取实体白名单。

    Args:
        whitelist_path: 白名单文件路径（可选）

    Returns:
        白名单集合
    """
    if not whitelist_path or not whitelist_path.exists():
        logger.info("Whitelist file not provided or not found, using empty whitelist")
        return set()

    whitelist: set[str] = set()
    try:
        with whitelist_path.open(encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                # 跳过空行和注释行
                if not name or name.startswith("#") or name.startswith("("):
                    continue
                if len(name) >= 2:  # 过滤掉单字
                    whitelist.add(name)
        logger.info(f"Loaded whitelist: {whitelist_path}, count={len(whitelist)}")
    except Exception as e:
        logger.error(f"Failed to load whitelist: {e}")

    return whitelist


# =============================================================================
# 映射构建
# =============================================================================
HEADING_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


def extract_concept_name(md_content: str, md_path: Path | None) -> str:
    """提取概念名称：优先 Markdown 一级标题，其次文件名 stem，最后兜底 unknown_concept。"""
    m = HEADING_RE.search(md_content or "")
    if m:
        return m.group(1).strip()
    if md_path:
        return md_path.stem
    return "unknown_concept"


def default_file_path(item: dict) -> str | None:
    """
    获取 file_path 默认值。

    优先级：item.file_path > manual_concepts/{md_path文件名} > manual_concepts/{doc_id}.md
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


def build_mappings(
    items: list[dict],
    whitelist: set[str] | None = None,
    include_non_correct: bool = False
) -> tuple[dict[str, dict], dict[str, dict], dict[str, str]]:
    """
    构建映射关系。

    Returns:
        (mapping_by_name, alias_to_name, mapping_by_file) 元组
    """
    mapping_by_name: dict[str, dict] = {}
    alias_to_name: dict[str, str] = {}
    mapping_by_file: dict[str, dict] = {}

    domain_default = os.getenv("CONCEPT_RETRIEVAL_DOMAIN", "math").strip()

    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue

        # 过滤 review_status
        review_status = raw_item.get("review_status")
        if review_status != "correct" and not include_non_correct:
            continue

        concept_name = raw_item.get("concept_name") or raw_item.get("name")
        if not concept_name:
            continue

        # 白名单过滤
        if whitelist is not None and concept_name not in whitelist:
            continue

        # 构建记录
        record = {
            **raw_item,
            "concept_name": concept_name,
            "file_path": default_file_path(raw_item),
            "domain": raw_item.get("domain") or domain_default,
            "source_type": raw_item.get("source_type") or "manual_concept",
            "aliases": raw_item.get("aliases") or [],
        }

        mapping_by_name[concept_name] = record

        # 别名映射
        for alias in record.get("aliases", []):
            if alias and alias not in alias_to_name:
                alias_to_name[alias] = concept_name

        # 文件路径映射
        if record.get("file_path"):
            mapping_by_file[record["file_path"]] = record

    logger.info(
        f"Built mappings: by_name={len(mapping_by_name)}, "
        f"alias_count={len(alias_to_name)}, by_file={len(mapping_by_file)}"
    )

    return mapping_by_name, alias_to_name, mapping_by_file


# =============================================================================
# LLM 函数构建
# =============================================================================
def build_llm_model_func(llm_config: dict[str, Any]):
    """构建 LightRAG 用的 LLM 模型函数。"""
    async def _llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        **kwargs: Any,
    ) -> str:
        return await openai_complete_if_cache(
            llm_config["model"],
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=llm_config["base_url"],
            api_key=llm_config["api_key"],
            **kwargs,
        )

    return _llm_model_func


def build_embedding_func(embedding_config: dict[str, Any]) -> EmbeddingFunc:
    """构建 LightRAG 用的 Embedding 函数。"""
    return EmbeddingFunc(
        embedding_dim=embedding_config["dim"],
        max_token_size=8192,
        func=lambda texts: openai_embed.func(
            texts,
            model=embedding_config["model"],
            api_key=embedding_config["api_key"],
            base_url=embedding_config["base_url"],
        ),
    )


def build_lightrag(
    working_dir: Path,
    llm_config: dict[str, Any],
    embedding_config: dict[str, Any],
    domain: str = "math",
    entity_type: str = "MANUAL_CONCEPT",
) -> LightRAG:
    """构建 LightRAG 实例。"""
    working_dir.mkdir(parents=True, exist_ok=True)

    guidance = (
        f"你是 {domain} 教材人工概念抽取器。"
        f"只允许抽取输入文本明确出现的人工概念，实体类型统一为 {entity_type}。"
        "禁止抽取：公式、变量、数字、符号、运算词、人名、例子对象、解题步骤和推导过程。"
        "禁止根据常识补充文档外概念。"
        "关系没有明确依据时不要输出。"
    )

    return LightRAG(
        working_dir=str(working_dir),
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        addon_params={
            "language": "Chinese",
            "entity_types_guidance": guidance,
        },
        llm_model_func=build_llm_model_func(llm_config),
        embedding_func=build_embedding_func(embedding_config),
    )


# =============================================================================
# 概念检索服务
# =============================================================================
class ConceptRetrievalService:
    """人工概念检索服务。"""

    # 允许的 entity_type 集合
    HIT_ENTITY_TYPES = {"MANUAL_CONCEPT", "MANUAL_MATH_CONCEPT"}

    def __init__(self):
        """初始化服务。"""
        self._enabled = False
        self._domain = "math"
        self._entity_type = "MANUAL_CONCEPT"
        self._config_path: Path | None = None
        self._whitelist_path: Path | None = None
        self._working_dir: Path | None = None
        self._enable_lightrag = False
        self._query_mode = "local"
        self._top_k = 10
        self._chunk_top_k = 10
        self._enable_rerank = False
        self._min_confidence = 0.75

        # 映射关系
        self._mapping_by_name: dict[str, dict] = {}
        self._alias_to_name: dict[str, str] = {}
        self._mapping_by_file: dict[str, dict] = {}

        # LightRAG 实例（延迟初始化）
        self._rag: LightRAG | None = None
        self._llm_config: dict[str, Any] | None = None
        self._embedding_config: dict[str, Any] | None = None

        # 服务根目录
        self._service_root = Path(__file__).parent.parent.parent

        self._load_config()

    def _load_config(self):
        """加载配置。"""
        # 基础配置
        self._enabled = os.getenv("CONCEPT_RETRIEVAL_ENABLED", "false").lower() in ("true", "1", "yes")
        self._domain = os.getenv("CONCEPT_RETRIEVAL_DOMAIN", "math").strip()
        self._entity_type = os.getenv("CONCEPT_RETRIEVAL_ENTITY_TYPE", "MANUAL_CONCEPT").strip()

        if not self._enabled:
            logger.info("Concept retrieval disabled by CONCEPT_RETRIEVAL_ENABLED")
            return

        # 路径配置
        config_str = os.getenv("CONCEPT_RETRIEVAL_CONFIG")
        whitelist_str = os.getenv("CONCEPT_RETRIEVAL_WHITELIST")
        working_dir_str = os.getenv("CONCEPT_RETRIEVAL_LIGHTRAG_WORKING_DIR")

        if config_str:
            self._config_path = _resolve_path(config_str, self._service_root)

        if whitelist_str:
            self._whitelist_path = _resolve_path(whitelist_str, self._service_root)

        if working_dir_str:
            self._working_dir = _resolve_path(working_dir_str, self._service_root)

        # LightRAG 配置
        self._enable_lightrag = os.getenv("CONCEPT_RETRIEVAL_ENABLE_LIGHTRAG", "true").lower() in ("true", "1")
        self._query_mode = os.getenv("CONCEPT_RETRIEVAL_QUERY_MODE", "local").strip()
        self._top_k = int(os.getenv("CONCEPT_RETRIEVAL_TOP_K", "10"))
        self._chunk_top_k = int(os.getenv("CONCEPT_RETRIEVAL_CHUNK_TOP_K", "10"))
        self._enable_rerank = os.getenv("CONCEPT_RETRIEVAL_ENABLE_RERANK", "false").lower() in ("true", "1")
        self._min_confidence = float(os.getenv("CONCEPT_RETRIEVAL_MIN_CONFIDENCE", "0.75"))

        # 加载映射
        self._load_mappings()

        logger.info(
            f"ConceptRetrievalService initialized: "
            f"enabled={self._enabled}, domain={self._domain}, entity_type={self._entity_type}, "
            f"config_path={self._config_path}, working_dir={self._working_dir}, "
            f"concepts_count={len(self._mapping_by_name)}, aliases_count={len(self._alias_to_name)}"
        )

    def _load_mappings(self):
        """加载概念映射。"""
        if not self._config_path or not self._config_path.exists():
            logger.warning(f"Config path not found: {self._config_path}")
            return

        items = load_concept_items(self._config_path)
        whitelist = load_entity_whitelist(self._whitelist_path) if self._whitelist_path else None

        self._mapping_by_name, self._alias_to_name, self._mapping_by_file = build_mappings(
            items,
            whitelist=whitelist,
            include_non_correct=False  # 默认只用 correct
        )

    def _get_rag(self) -> LightRAG | None:
        """获取 LightRAG 实例（延迟初始化）。"""
        if not self._enable_lightrag:
            return None

        if self._rag is not None:
            return self._rag

        if not self._working_dir or not self._working_dir.exists():
            logger.warning(f"LightRAG working_dir not found: {self._working_dir}")
            return None

        try:
            # 构建 LLM 和 Embedding 配置
            llm_config = resolve_llm_config()
            embedding_config = resolve_embedding_config(
                llm_config["base_url"],
                llm_config["api_key"]
            )

            self._rag = build_lightrag(
                self._working_dir,
                llm_config,
                embedding_config,
                domain=self._domain,
                entity_type=self._entity_type,
            )

            logger.info(f"LightRAG initialized: working_dir={self._working_dir}")
            return self._rag
        except Exception as e:
            logger.error(f"Failed to initialize LightRAG: {e}", exc_info=True)
            return None

    def _extract_content(self, record: dict[str, Any]) -> str | None:
        """从记录中提取内容。"""
        # 优先级：md_content > content > markdown > text > md_path 文件
        content = (
            record.get("md_content")
            or record.get("content")
            or record.get("markdown")
            or record.get("text")
        )

        if content:
            return content

        # 尝试读取 md_path 文件
        md_path_raw = record.get("md_path")
        if md_path_raw:
            md_path = _resolve_path(md_path_raw, self._service_root)
            if md_path.exists():
                try:
                    return md_path.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Failed to read md_path: {md_path}, error={e}")

        return None

    def _exact_match_check(self, query: str) -> ConceptContext | None:
        """精确匹配检查。"""
        query_lower = query.lower()

        # 检查 concept_name 直接命中
        for concept_name, record in self._mapping_by_name.items():
            if concept_name.lower() in query_lower:
                content = self._extract_content(record)

                return ConceptContext(
                    hit=True,
                    concept_name=concept_name,
                    doc_id=record.get("doc_id"),
                    domain=record.get("domain"),
                    entity_type=self._entity_type,
                    source_type=record.get("source_type"),
                    content=content,
                    file_path=record.get("file_path"),
                    md_path=record.get("md_path"),
                    confidence=1.0,
                    hit_reason="query_contains_concept_name",
                    metadata={
                        "aliases": record.get("aliases", []),
                        "review_status": record.get("review_status"),
                        "strict": record.get("strict"),
                    },
                )

        # 检查 alias 命中
        for alias, concept_name in self._alias_to_name.items():
            if alias.lower() in query_lower:
                record = self._mapping_by_name.get(concept_name)
                if record:
                    content = self._extract_content(record)

                    return ConceptContext(
                        hit=True,
                        concept_name=concept_name,
                        doc_id=record.get("doc_id"),
                        domain=record.get("domain"),
                        entity_type=self._entity_type,
                        source_type=record.get("source_type"),
                        content=content,
                        file_path=record.get("file_path"),
                        md_path=record.get("md_path"),
                        confidence=1.0,
                        hit_reason="query_contains_alias",
                        metadata={
                            "matched_alias": alias,
                            "aliases": record.get("aliases", []),
                            "review_status": record.get("review_status"),
                            "strict": record.get("strict"),
                        },
                    )

        return None

    async def _lightrag_retrieve(self, query: str) -> ConceptContext | None:
        """LightRAG 召回。"""
        rag = self._get_rag()
        if not rag:
            return None

        try:
            # 构建 QueryParam
            kwargs = {
                "mode": self._query_mode,
                "top_k": self._top_k,
                "chunk_top_k": self._chunk_top_k,
            }

            # 检查是否支持 enable_rerank
            try:
                import inspect
                sig = inspect.signature(QueryParam)
                if "enable_rerank" in sig.parameters:
                    kwargs["enable_rerank"] = self._enable_rerank
            except Exception:
                pass

            param = QueryParam(**kwargs)

            # 执行查询
            await rag.initialize_storages()
            result = await rag.aquery_data(query, param=param)

            # 解析结果
            data = result.get("data") or {}
            entities = data.get("entities") or []

            logger.info(
                f"LightRAG query: query={query[:50]}, "
                f"entities={len(entities)}, mode={self._query_mode}"
            )

            # 命中判定
            for entity in entities:
                entity_name = entity.get("entity_name")
                entity_type = entity.get("entity_type")

                # 严格判定：entity_type 必须命中且 concept_name 存在于映射
                if (
                    entity_type in self.HIT_ENTITY_TYPES
                    and entity_name in self._mapping_by_name
                ):
                    record = self._mapping_by_name[entity_name]
                    content = self._extract_content(record)

                    return ConceptContext(
                        hit=True,
                        concept_name=entity_name,
                        doc_id=record.get("doc_id"),
                        domain=record.get("domain") or entity.get("domain"),
                        entity_type=entity_type,
                        source_type=record.get("source_type") or entity.get("source_type"),
                        content=content,
                        file_path=record.get("file_path"),
                        md_path=record.get("md_path"),
                        confidence=0.9,  # LightRAG 召回的置信度略低于精确匹配
                        hit_reason="lightrag_entity_hit",
                        metadata={
                            "rag_entity_type": entity_type,
                            "aliases": record.get("aliases", []),
                            "review_status": record.get("review_status"),
                            "strict": record.get("strict"),
                        },
                    )

            return None

        except Exception as e:
            logger.error(f"LightRAG retrieve failed: {e}", exc_info=True)
            return None
        finally:
            try:
                await rag.finalize_storages()
            except Exception as e:
                logger.warning(f"LightRAG finalize storages failed: {e}")

    async def aretrieve(self, query: str) -> ConceptContext:
        """
        执行概念检索。

        Args:
            query: 用户查询

        Returns:
            检索结果上下文
        """
        if not self._enabled:
            return ConceptContext(
                hit=False,
                hit_reason="disabled",
                metadata={"query_mode": self._query_mode},
            )

        if not query:
            return ConceptContext(
                hit=False,
                hit_reason="empty_query",
                metadata={"query_mode": self._query_mode},
            )

        try:
            # 第一步：精确匹配
            exact_result = self._exact_match_check(query)
            if exact_result:
                logger.info(
                    f"Concept exact match: concept_name={exact_result.concept_name}, "
                    f"reason={exact_result.hit_reason}, query={query[:50]}"
                )
                return exact_result

            # 第二步：LightRAG 召回
            if self._enable_lightrag:
                lightrag_result = await self._lightrag_retrieve(query)
                if lightrag_result:
                    logger.info(
                        f"Concept LightRAG hit: concept_name={lightrag_result.concept_name}, "
                        f"query={query[:50]}"
                    )
                    return lightrag_result

            # 未命中
            logger.info(f"Concept miss: query={query[:50]}")
            return ConceptContext(
                hit=False,
                hit_reason="not_found",
                metadata={"query_mode": self._query_mode},
            )

        except Exception as e:
            logger.error(f"Concept retrieve error: {e}", exc_info=True)
            return ConceptContext(
                hit=False,
                hit_reason=f"error:{type(e).__name__}",
                metadata={"error": str(e), "query_mode": self._query_mode},
            )


# =============================================================================
# 单例
# =============================================================================
_concept_retrieval_service: ConceptRetrievalService | None = None


def get_concept_retrieval_service() -> ConceptRetrievalService:
    """获取概念检索服务单例。"""
    global _concept_retrieval_service
    if _concept_retrieval_service is None:
        _concept_retrieval_service = ConceptRetrievalService()
    return _concept_retrieval_service
