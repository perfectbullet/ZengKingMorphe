"""
LightRAG 文件知识库包装器。

职责：
- 读取已构建好的 LightRAG working_dir
- 调用 aquery_llm() 获取完整检索源数据与流式 LLM 响应
- 从 result["data"] 提取 entities / relationships / chunks / references
- 从 result["llm_response"]["response_iterator"] 输出流式文本
- 适配为 ai-service 统一 RAG 事件格式
- 保存 raw / normalized debug JSONL
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import traceback
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, AsyncIterator

from app.core.logging import get_logger

logger = get_logger(__name__)

_training_lightrag_instance = None
_training_lightrag_lock = asyncio.Lock()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _pick_env(candidates: list[str], default: str | None = None) -> str | None:
    for name in candidates:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


def _service_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_working_dir() -> Path:
    value = os.getenv(
        "TRAINING_LIGHTRAG_WORKING_DIR",
        "/home/zj/ZengKingMorphe/ai-service/data/lightrag_industrial_training_enamel_debug",
    ).strip()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (_service_root() / path).resolve()
    return path


def _resolve_llm_config() -> dict[str, Any]:
    model = _pick_env(["LLM_MODEL"], "")
    base_url = _pick_env(["LLM_BASE_URL"], "")
    api_key = _pick_env(["LLM_API_KEY"], "no-api-key") or "no-api-key"
    return {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }


def _resolve_embedding_config() -> dict[str, Any]:
    model = _pick_env(["EMBEDDING_MODEL", "VLLM_EMBED_MODEL"], "BAAI/bge-m3")
    dim_raw = _pick_env(["EMBEDDING_DIM", "VLLM_EMBED_DIM"], "1024")
    try:
        dim = int(dim_raw or "1024")
    except ValueError:
        logger.warning(f"Invalid embedding dim: {dim_raw}, fallback to 1024")
        dim = 1024

    base_url = _pick_env(
        [
            "EMBEDDING_BINDING_HOST",
            "VLLM_EMBED_URL",
            "VLLM_EMBED_HOST",
        ],
        "",
    )
    api_key = _pick_env(
        ["EMBEDDING_BINDING_API_KEY", "VLLM_EMBEDDING_API_KEY", "VLLM_API_KEY"],
        "no-api-key",
    ) or "no-api-key"
    return {
        "model": model,
        "dim": dim,
        "base_url": base_url,
        "api_key": api_key,
    }


def _resolve_rerank_binding() -> str:
    return os.getenv("RERANK_BINDING", "cohere").strip().lower() or "cohere"


def _resolve_rerank_base_url(binding: str) -> str:
    if binding == "cohere":
        default_url = "https://api.cohere.com/v2/rerank"
    elif binding == "jina":
        default_url = "https://api.jina.ai/v1/rerank"
    elif binding in {"aliyun", "ali", "dashscope"}:
        default_url = (
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        )
    else:
        default_url = ""
    return _pick_env(["RERANK_BINDING_HOST", "RERANK_BASE_URL"], default_url) or default_url


def _resolve_rerank_api_key() -> str:
    return _pick_env(["RERANK_BINDING_API_KEY", "RERANK_API_KEY"], "local-key") or "local-key"


def _resolve_cosine_threshold() -> float:
    raw = os.getenv("COSINE_THRESHOLD", "0.2").strip()
    try:
        value = float(raw)
    except ValueError:
        logger.warning(f"Invalid COSINE_THRESHOLD={raw!r}, fallback to 0.2")
        return 0.2
    if value < 0.0 or value > 1.0:
        logger.warning(f"COSINE_THRESHOLD out of range: {value}, clamp to [0.0, 1.0]")
    return max(0.0, min(1.0, value))


def _resolve_min_rerank_score() -> float:
    raw = os.getenv("MIN_RERANK_SCORE", "0.6").strip()
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Invalid MIN_RERANK_SCORE={raw!r}, fallback to 0.6")
        return 0.6


def _build_rerank_model_func():
    binding = _resolve_rerank_binding()
    model = os.getenv("RERANK_MODEL", "bge-reranker-m3").strip() or "bge-reranker-m3"
    base_url = _resolve_rerank_base_url(binding)
    api_key = _resolve_rerank_api_key()

    try:
        if binding == "cohere":
            from lightrag.rerank import cohere_rerank

            return partial(
                cohere_rerank,
                model=model,
                api_key=api_key,
                base_url=base_url,
                enable_chunking=_env_bool("RERANK_ENABLE_CHUNKING", True),
                max_tokens_per_doc=int(os.getenv("RERANK_MAX_TOKENS_PER_DOC", "512")),
            )

        if binding == "jina":
            from lightrag.rerank import jina_rerank

            return partial(
                jina_rerank,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )

        if binding in {"aliyun", "ali", "dashscope"}:
            from lightrag.rerank import ali_rerank

            return partial(
                ali_rerank,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )

        logger.warning(f"Unsupported RERANK_BINDING={binding!r}; rerank_model_func disabled")
        return None
    except Exception as exc:
        logger.warning(
            "Failed to build rerank_model_func | "
            f"binding={binding} | model={model} | base_url={base_url} | error={exc}"
        )
        return None


def _rag_debug_enabled() -> bool:
    return _env_bool("TRAINING_RAG_STREAM_DEBUG_ENABLED", True)


def _rag_debug_dir() -> Path:
    value = os.getenv(
        "TRAINING_RAG_STREAM_DEBUG_DIR",
        str(_service_root() / "logs" / "rag_stream_debug"),
    ).strip()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (_service_root() / path).resolve()
    return path


def _rag_debug_max_text() -> int:
    raw = os.getenv("TRAINING_RAG_STREAM_DEBUG_MAX_TEXT", "4000").strip()
    try:
        return max(200, int(raw))
    except ValueError:
        logger.warning(
            f"Invalid TRAINING_RAG_STREAM_DEBUG_MAX_TEXT={raw}, fallback to 4000"
        )
        return 4000


def _rag_debug_preview_only() -> bool:
    return _env_bool("TRAINING_RAG_STREAM_DEBUG_PREVIEW_ONLY", True)


def _new_trace_id() -> str:
    return uuid.uuid4().hex[:8]


def _now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _new_debug_paths(trace_id: str, backend: str) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = _rag_debug_dir()
    raw_path = base_dir / f"{stamp}_{backend}_{trace_id}.raw.jsonl"
    normalized_path = base_dir / f"{stamp}_{backend}_{trace_id}.normalized.jsonl"
    return raw_path, normalized_path


def _safe_preview(value: Any, max_text: int, max_depth: int = 2) -> Any:
    if max_depth < 0:
        return {"type": type(value).__name__}
    if isinstance(value, str):
        return {
            "type": "str",
            "len": len(value),
            "preview": value[:max_text],
        }
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        keys = list(value.keys())
        sample: dict[str, Any] = {}
        for key in keys[:8]:
            try:
                sample[str(key)] = _safe_preview(value.get(key), max_text, max_depth - 1)
            except Exception as exc:
                sample[str(key)] = {"preview_error": str(exc)}
        return {
            "type": "dict",
            "keys": keys,
            "sample": sample,
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "len": len(value),
            "items": [
                _safe_preview(item, max_text, max_depth - 1) for item in list(value)[:5]
            ],
        }
    if is_dataclass(value):
        return _safe_preview(asdict(value), max_text, max_depth - 1)

    attrs = [name for name in dir(value) if not name.startswith("_")][:12]
    preview = {
        "type": type(value).__name__,
        "attrs": attrs,
        "repr": repr(value)[:max_text],
        "has_raw_data": hasattr(value, "raw_data"),
        "has_response_iterator": hasattr(value, "response_iterator"),
        "has_content": hasattr(value, "content"),
    }
    return preview


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning(f"RAG debug append failed | path={path} | error={exc}")


def _write_raw_debug(path: Path | None, record: dict[str, Any]) -> None:
    if not path or not _rag_debug_enabled():
        return
    _append_jsonl(path, record)


def _write_normalized_debug(path: Path | None, record: dict[str, Any]) -> None:
    if not path or not _rag_debug_enabled():
        return
    _append_jsonl(path, record)


def _event_preview(event: dict[str, Any], max_text: int) -> dict[str, Any]:
    if event.get("type") == "chunk":
        content = str(event.get("content") or "")
        return {"type": "chunk", "content_preview": content[:max_text]}
    return _safe_preview(event, max_text, max_depth=2)


def _iter_text_chunks(text: str, step: int = 48) -> list[str]:
    if not text:
        return []
    return [text[i : i + step] for i in range(0, len(text), step)]


def _is_async_iterable(value: Any) -> bool:
    return hasattr(value, "__aiter__")


def _to_plain_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            return None
    return None


def _extract_text_from_item(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("content", "response", "answer", "text"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
        delta = item.get("delta")
        if isinstance(delta, dict):
            value = delta.get("content")
            if isinstance(value, str):
                return value
        choices = item.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                delta = first.get("delta") or {}
                value = delta.get("content")
                if isinstance(value, str):
                    return value
    if hasattr(item, "content") and isinstance(getattr(item, "content"), str):
        return getattr(item, "content")
    return ""


def _build_query_param(mode: str, prefer_zh_output: bool) -> Any:
    from lightrag.base import QueryParam

    user_prompt = (
        "请使用简体中文回答，只根据工训教材知识库作答；"
        # "尽量给出章节路径或来源线索；若资料不足，请明确说明“当前知识库资料不足”。"
        # "当用户询问“基本流程/制作流程”时，优先按教材中的核心步骤回答；"
        # "进阶效果、案例、延伸技法只作为补充，不要列为主流程步骤。"
        if prefer_zh_output
        else "Answer in English only based on the training knowledge base. "
        "If context is insufficient, say the knowledge base lacks enough material. "
        "When the user asks for a basic workflow, prioritize core textbook steps "
        "and keep advanced effects or extensions only as supplementary notes."
    )

    candidate_kwargs = {
        "mode": mode,
        "stream": True,
        "top_k": int(os.getenv("TRAINING_RAG_TOP_K", "12")),
        "chunk_top_k": int(os.getenv("TRAINING_RAG_CHUNK_TOP_K", "4")),
        "response_type": os.getenv("TRAINING_RAG_RESPONSE_TYPE", "Single Paragraph"),
        "max_entity_tokens": int(os.getenv("TRAINING_RAG_MAX_ENTITY_TOKENS", "6000")),
        "max_relation_tokens": int(os.getenv("TRAINING_RAG_MAX_RELATION_TOKENS", "8000")),
        "max_total_tokens": int(os.getenv("TRAINING_RAG_MAX_TOTAL_TOKENS", "30000")),
        "user_prompt": user_prompt,
        "conversation_history": [],
        "enable_rerank": _env_bool("TRAINING_RAG_ENABLE_RERANK", False),
        "include_references": _env_bool("TRAINING_RAG_INCLUDE_REFERENCES", True),
    }
    supported = inspect.signature(QueryParam).parameters
    kwargs = {
        key: value for key, value in candidate_kwargs.items() if key in supported
    }
    return QueryParam(**kwargs), kwargs


def _build_system_prompt() -> str:
    return "你是工业实训教材问答助手。只根据当前知识库回答，不要编造教材外知识。"


def _build_llm_model_func():
    from lightrag.llm.openai import openai_complete_if_cache

    llm = _resolve_llm_config()
    default_max_tokens = int(os.getenv("TRAINING_RAG_QUERY_MAX_TOKENS", "512"))

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
            max_tokens=kwargs.pop("max_tokens", default_max_tokens),
            **kwargs,
        )

    return _llm_model_func


def _build_embedding_func():
    from lightrag.llm.openai import openai_embed
    from lightrag.utils import EmbeddingFunc

    emb = _resolve_embedding_config()

    async def _embedding_func(
        texts: list[str],
        max_token_size: int | None = None,
    ):
        return await openai_embed.func(
            texts,
            model=emb["model"],
            api_key=emb["api_key"],
            base_url=emb["base_url"],
            max_token_size=max_token_size,
        )

    return EmbeddingFunc(
        embedding_dim=emb["dim"],
        max_token_size=7000,
        func=_embedding_func,
    )


async def get_training_lightrag_instance():
    global _training_lightrag_instance
    if _training_lightrag_instance is not None:
        return _training_lightrag_instance

    async with _training_lightrag_lock:
        if _training_lightrag_instance is not None:
            return _training_lightrag_instance

        working_dir = _resolve_working_dir()
        if not working_dir.exists():
            raise FileNotFoundError(
                f"TRAINING_LIGHTRAG_WORKING_DIR 不存在: {working_dir}"
            )

        from lightrag import LightRAG

        rerank_model_func = _build_rerank_model_func()
        cosine_threshold = _resolve_cosine_threshold()

        base_kwargs = {
            "working_dir": str(working_dir),
            "enable_llm_cache": False,
            "enable_llm_cache_for_entity_extract": False,
            "addon_params": {"language": "Chinese"},
            "llm_model_func": _build_llm_model_func(),
            "embedding_func": _build_embedding_func(),
        }
        candidate_kwargs = {
            "rerank_model_func": rerank_model_func,
            "rerank_model_max_async": int(os.getenv("MAX_ASYNC_RERANK", "4")),
            "default_rerank_timeout": int(os.getenv("RERANK_TIMEOUT", "30")),
            "min_rerank_score": _resolve_min_rerank_score(),
            "cosine_threshold": cosine_threshold,
            "cosine_better_than_threshold": cosine_threshold,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": cosine_threshold
            },
            "entities_vdb_storage_cls_kwargs": {
                "cosine_better_than_threshold": cosine_threshold
            },
            "relationships_vdb_storage_cls_kwargs": {
                "cosine_better_than_threshold": cosine_threshold
            },
            "chunks_vdb_storage_cls_kwargs": {
                "cosine_better_than_threshold": cosine_threshold
            },
        }
        supported = inspect.signature(LightRAG).parameters
        kwargs = {
            key: value
            for key, value in {**base_kwargs, **candidate_kwargs}.items()
            if key in supported and value is not None
        }
        unsupported = [key for key in candidate_kwargs if key not in supported]
        if unsupported:
            logger.info(f"LightRAG init unsupported optional kwargs skipped: {unsupported}")
        if _env_bool("TRAINING_RAG_ENABLE_RERANK", False) and rerank_model_func is None:
            logger.warning(
                "TRAINING_RAG_ENABLE_RERANK=true but rerank_model_func is unavailable; "
                "LightRAG query may fail or run without rerank depending on LightRAG behavior."
            )

        rag = LightRAG(**kwargs)
        await rag.initialize_storages()
        _training_lightrag_instance = rag
        logger.info(
            "Training LightRAG initialized | "
            f"working_dir={working_dir} | "
            f"cosine_threshold={cosine_threshold} | "
            f"rerank_enabled={_env_bool('TRAINING_RAG_ENABLE_RERANK', False)} | "
            f"rerank_binding={_resolve_rerank_binding()} | "
            f"rerank_model={os.getenv('RERANK_MODEL', 'bge-reranker-m3')} | "
            f"rerank_base_url={_resolve_rerank_base_url(_resolve_rerank_binding())} | "
            f"rerank_model_func_available={rerank_model_func is not None} | "
            f"min_rerank_score={_resolve_min_rerank_score()}"
        )
        return _training_lightrag_instance


async def _yield_event(
    event: dict[str, Any],
    *,
    trace_id: str,
    backend: str,
    normalized_path: Path | None,
    event_index: int,
) -> AsyncIterator[dict[str, Any]]:
    max_text = _rag_debug_max_text()
    _write_normalized_debug(
        normalized_path,
        {
            "ts": _now_ts(),
            "trace_id": trace_id,
            "stage": "normalized_event",
            "index": event_index,
            "backend": backend,
            "event_type": event.get("type"),
            "content_len": len(str(event.get("content") or "")),
            "event_preview": _event_preview(event, max_text),
        },
    )
    yield event


async def get_training_lightrag_stream(
    query: str,
    mode: str = "hybrid",
    prefer_zh_output: bool = True,
    debug_meta: dict | None = None,
) -> AsyncIterator[dict]:
    backend = "lightrag_file"
    trace_id = _new_trace_id()
    raw_path, normalized_path = _new_debug_paths(trace_id, backend)
    raw_debug_path = raw_path if _rag_debug_enabled() else None
    normalized_debug_path = normalized_path if _rag_debug_enabled() else None
    event_index = 0
    sources_info_emitted = False

    try:
        rag = await get_training_lightrag_instance()
        working_dir = _resolve_working_dir()
        param, param_kwargs = _build_query_param(mode, prefer_zh_output)
        system_prompt = _build_system_prompt()
        rerank_debug = {
            "training_rag_enable_rerank": _env_bool("TRAINING_RAG_ENABLE_RERANK", False),
            "rerank_binding": _resolve_rerank_binding(),
            "rerank_model": os.getenv("RERANK_MODEL", "bge-reranker-m3"),
            "rerank_base_url": _resolve_rerank_base_url(_resolve_rerank_binding()),
            "rerank_api_key_set": bool(_resolve_rerank_api_key()),
            "min_rerank_score": _resolve_min_rerank_score(),
            "max_async_rerank": int(os.getenv("MAX_ASYNC_RERANK", "4")),
            "rerank_timeout": int(os.getenv("RERANK_TIMEOUT", "30")),
            "cosine_threshold": _resolve_cosine_threshold(),
            "query_param_enable_rerank": param_kwargs.get("enable_rerank"),
        }

        _write_raw_debug(
            raw_debug_path,
            {
                "ts": _now_ts(),
                "trace_id": trace_id,
                "stage": "before_aquery_llm",
                "backend": backend,
                "query": query[: _rag_debug_max_text()],
                "mode": mode,
                "working_dir": str(working_dir),
                "param_kwargs": param_kwargs,
                "rerank_debug": rerank_debug,
                "system_prompt": _safe_preview(system_prompt, _rag_debug_max_text()),
                "debug_meta": _safe_preview(debug_meta or {}, _rag_debug_max_text()),
            },
        )

        result = await rag.aquery_llm(query, param=param, system_prompt=system_prompt)
        result_dict = _to_plain_dict(result) or {}
        data = result_dict.get("data") or {}
        metadata = result_dict.get("metadata") or {}
        llm_response = result_dict.get("llm_response") or {}
        processing_info = metadata.get("processing_info") or {}

        entities = list(data.get("entities") or [])
        relationships = list(data.get("relationships") or [])
        chunks = list(data.get("chunks") or [])
        references = list(data.get("references") or [])
        response_iterator = llm_response.get("response_iterator")
        is_streaming = bool(llm_response.get("is_streaming"))
        content = llm_response.get("content") or ""

        _write_raw_debug(
            raw_debug_path,
            {
                "ts": _now_ts(),
                "trace_id": trace_id,
                "stage": "after_aquery_llm",
                "backend": backend,
                "python_type": type(result).__name__,
                "result_keys": list(result_dict.keys()),
                "data_keys": list(data.keys()) if isinstance(data, dict) else [],
                "metadata_keys": list(metadata.keys()) if isinstance(metadata, dict) else [],
                "llm_response_keys": list(llm_response.keys()) if isinstance(llm_response, dict) else [],
                "is_streaming": is_streaming,
                "has_response_iterator": response_iterator is not None,
                "entities_count": len(entities),
                "relationships_count": len(relationships),
                "chunks_count": len(chunks),
                "references_count": len(references),
                "rerank_metadata": _safe_preview(
                    {
                        key: metadata.get(key)
                        for key in metadata.keys()
                        if "rerank" in str(key).lower()
                    },
                    _rag_debug_max_text(),
                ),
                "preview": _safe_preview(result_dict, _rag_debug_max_text()),
            },
        )

        sources_info = {
            "entities_count": len(entities),
            "relationships_count": len(relationships),
            "chunks_count": len(chunks),
            "references_count": len(references),
            "mode": mode,
            "backend": backend,
            "working_dir": str(working_dir),
            "raw_data_available": bool(data),
            "metadata_available": bool(metadata),
            "query_mode": metadata.get("query_mode") or mode,
            "candidate_chunks_count": (
                processing_info.get("merged_chunks_count")
                or processing_info.get("total_chunks_found")
                or processing_info.get("candidate_chunks_count")
            ),
            "final_chunks_count": (
                processing_info.get("final_chunks_count")
                or len(chunks)
            ),
        }
        sources_payload = {
            "entities": entities,
            "relationships": relationships,
            "chunks": chunks,
            "references": references,
            "metadata": metadata,
            "backend": backend,
            "mode": mode,
            "working_dir": str(working_dir),
        }
        _write_raw_debug(
            raw_debug_path,
            {
                "ts": _now_ts(),
                "trace_id": trace_id,
                "stage": "aquery_llm_data",
                "backend": backend,
                "entities_count": len(entities),
                "relationships_count": len(relationships),
                "chunks_count": len(chunks),
                "references_count": len(references),
                "processing_info": _safe_preview(processing_info, _rag_debug_max_text()),
                "rerank_metadata": _safe_preview(
                    {
                        key: metadata.get(key)
                        for key in metadata.keys()
                        if "rerank" in str(key).lower()
                    },
                    _rag_debug_max_text(),
                ),
                "first_chunk_preview": _safe_preview(chunks[0], _rag_debug_max_text())
                if chunks
                else None,
                "first_reference_preview": _safe_preview(references[0], _rag_debug_max_text())
                if references
                else None,
            },
        )

        event_index += 1
        async for event in _yield_event(
            {"type": "sources_info", "content": sources_info},
            trace_id=trace_id,
            backend=backend,
            normalized_path=normalized_debug_path,
            event_index=event_index,
        ):
            sources_info_emitted = True
            yield event

        emitted_answer = False

        if is_streaming and response_iterator is not None:
            raw_index = 0
            async for item in response_iterator:
                raw_index += 1
                text = _extract_text_from_item(item)
                item_dict = _to_plain_dict(item)
                _write_raw_debug(
                    raw_debug_path,
                    {
                        "ts": _now_ts(),
                        "trace_id": trace_id,
                        "stage": "raw_stream_item",
                        "index": raw_index,
                        "backend": backend,
                        "python_type": type(item).__name__,
                        "keys": list(item_dict.keys()) if item_dict else [],
                        "content_len": len(text),
                        "preview": _safe_preview(item, _rag_debug_max_text()),
                    },
                )
                if not text:
                    continue
                emitted_answer = True
                event_index += 1
                async for event in _yield_event(
                    {"type": "chunk", "content": text},
                    trace_id=trace_id,
                    backend=backend,
                    normalized_path=normalized_debug_path,
                    event_index=event_index,
                ):
                    yield event
        elif content:
            for piece in _iter_text_chunks(str(content)):
                emitted_answer = True
                event_index += 1
                async for event in _yield_event(
                    {"type": "chunk", "content": piece},
                    trace_id=trace_id,
                    backend=backend,
                    normalized_path=normalized_debug_path,
                    event_index=event_index,
                ):
                    yield event

        event_index += 1
        async for event in _yield_event(
            {"type": "sources", "content": sources_payload},
            trace_id=trace_id,
            backend=backend,
            normalized_path=normalized_debug_path,
            event_index=event_index,
        ):
            yield event

        if not emitted_answer:
            event_index += 1
            async for event in _yield_event(
                {
                    "type": "error",
                    "content": "LightRAG 未返回有效回答内容。",
                },
                trace_id=trace_id,
                backend=backend,
                normalized_path=normalized_debug_path,
                event_index=event_index,
            ):
                yield event

    except Exception as exc:
        _write_raw_debug(
            raw_debug_path,
            {
                "ts": _now_ts(),
                "trace_id": trace_id,
                "stage": "exception",
                "backend": backend,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc()[: _rag_debug_max_text()],
            },
        )
        logger.error(f"Training LightRAG stream failed | error={exc}", exc_info=True)
        if not sources_info_emitted:
            event_index += 1
            async for event in _yield_event(
                {
                    "type": "sources_info",
                    "content": {
                        "entities_count": 0,
                        "relationships_count": 0,
                        "chunks_count": 0,
                        "mode": mode,
                        "backend": backend,
                        "working_dir": str(_resolve_working_dir()),
                        "raw_data_available": False,
                    },
                },
                trace_id=trace_id,
                backend=backend,
                normalized_path=normalized_debug_path,
                event_index=event_index,
            ):
                yield event
        event_index += 1
        async for event in _yield_event(
            {
                "type": "error",
                "content": f"LightRAG 查询失败: {type(exc).__name__}: {exc}",
            },
            trace_id=trace_id,
            backend=backend,
            normalized_path=normalized_debug_path,
            event_index=event_index,
        ):
            yield event
