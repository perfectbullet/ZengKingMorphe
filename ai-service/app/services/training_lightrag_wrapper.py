"""
LightRAG 文件知识库包装器。

职责：
- 读取已构建好的 LightRAG working_dir
- 调用 aquery() 进行流式问答
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


def _normalize_raw_data(raw_data: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _to_plain_dict(raw_data) or {}
    data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
    entities = list(data.get("entities") or [])
    relationships = list(data.get("relationships") or [])
    chunks = list(data.get("chunks") or [])
    info = {
        "entities_count": len(entities),
        "relationships_count": len(relationships),
        "chunks_count": len(chunks),
        "raw_data_available": bool(raw),
        "raw_data_keys": list(raw.keys()) if isinstance(raw, dict) else [],
    }
    sources = {
        "entities": entities,
        "relationships": relationships,
        "chunks": chunks,
    }
    return info, sources


def _extract_raw_data(resp: Any) -> Any:
    if isinstance(resp, dict):
        if "raw_data" in resp:
            return resp.get("raw_data")
        if "data" in resp and any(
            key in resp.get("data", {}) for key in ("entities", "relationships", "chunks")
        ):
            return resp
    if hasattr(resp, "raw_data"):
        return getattr(resp, "raw_data")
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
        "尽量给出章节路径或来源线索；若资料不足，请明确说明“当前知识库资料不足”。"
        if prefer_zh_output
        else "Answer in English only based on the training knowledge base. "
        "If context is insufficient, say the knowledge base lacks enough material."
    )
    system_prompt = (
        "你是工业实训教材问答助手。只根据当前知识库回答，不要编造教材外知识。"
    )

    candidate_kwargs = {
        "mode": mode,
        "stream": True,
        "top_k": int(os.getenv("TRAINING_RAG_TOP_K", "40")),
        "chunk_top_k": int(os.getenv("TRAINING_RAG_CHUNK_TOP_K", "10")),
        "response_type": os.getenv("TRAINING_RAG_RESPONSE_TYPE", "Multiple Paragraphs"),
        "user_prompt": user_prompt,
        "system_prompt": system_prompt,
        "conversation_history": [],
        "enable_rerank": False,
    }
    supported = inspect.signature(QueryParam).parameters
    kwargs = {
        key: value for key, value in candidate_kwargs.items() if key in supported
    }
    return QueryParam(**kwargs), kwargs


def _build_non_stream_query_param(mode: str, prefer_zh_output: bool) -> Any:
    from lightrag.base import QueryParam

    param, kwargs = _build_query_param(mode, prefer_zh_output)
    supported = inspect.signature(QueryParam).parameters
    if "stream" in supported:
        kwargs["stream"] = False
    return QueryParam(**kwargs)


def _build_llm_model_func():
    from lightrag.llm.openai import openai_complete_if_cache

    llm = _resolve_llm_config()
    default_max_tokens = int(os.getenv("TRAINING_RAG_QUERY_MAX_TOKENS", "4096"))

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

        rag = LightRAG(
            working_dir=str(working_dir),
            enable_llm_cache=False,
            enable_llm_cache_for_entity_extract=False,
            addon_params={"language": "Chinese"},
            llm_model_func=_build_llm_model_func(),
            embedding_func=_build_embedding_func(),
        )
        await rag.initialize_storages()
        _training_lightrag_instance = rag
        logger.info(f"Training LightRAG initialized | working_dir={working_dir}")
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

        _write_raw_debug(
            raw_debug_path,
            {
                "ts": _now_ts(),
                "trace_id": trace_id,
                "stage": "before_aquery",
                "backend": backend,
                "query": query[: _rag_debug_max_text()],
                "mode": mode,
                "working_dir": str(working_dir),
                "param_kwargs": param_kwargs,
                "debug_meta": _safe_preview(debug_meta or {}, _rag_debug_max_text()),
            },
        )

        resp = await rag.aquery(query, param=param)
        raw_data = _extract_raw_data(resp)
        _write_raw_debug(
            raw_debug_path,
            {
                "ts": _now_ts(),
                "trace_id": trace_id,
                "stage": "after_aquery",
                "backend": backend,
                "python_type": type(resp).__name__,
                "has_response_iterator": hasattr(resp, "response_iterator"),
                "has_raw_data": raw_data is not None,
                "has_content": hasattr(resp, "content"),
                "is_async_iterable": _is_async_iterable(resp),
                "preview": _safe_preview(resp, _rag_debug_max_text()),
            },
        )

        sources_info: dict[str, Any]
        sources_payload = {
            "entities": [],
            "relationships": [],
            "chunks": [],
            "backend": backend,
        }
        if raw_data is not None:
            info, parsed_sources = _normalize_raw_data(raw_data)
            sources_info = {
                "entities_count": info["entities_count"],
                "relationships_count": info["relationships_count"],
                "chunks_count": info["chunks_count"],
                "mode": mode,
                "backend": backend,
                "working_dir": str(working_dir),
                "raw_data_available": info["raw_data_available"],
            }
            sources_payload.update(parsed_sources)
            _write_raw_debug(
                raw_debug_path,
                {
                    "ts": _now_ts(),
                    "trace_id": trace_id,
                    "stage": "raw_data",
                    "backend": backend,
                    "entities_count": info["entities_count"],
                    "relationships_count": info["relationships_count"],
                    "chunks_count": info["chunks_count"],
                    "raw_data_keys": info["raw_data_keys"],
                },
            )
        else:
            sources_info = {
                "entities_count": 0,
                "relationships_count": 0,
                "chunks_count": 0,
                "mode": mode,
                "backend": backend,
                "working_dir": str(working_dir),
                "raw_data_available": False,
            }

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

        if hasattr(resp, "response_iterator"):
            iterator = getattr(resp, "response_iterator")
            raw_index = 0
            async for item in iterator:
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
        elif _is_async_iterable(resp):
            raw_index = 0
            async for item in resp:
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
        else:
            text = ""
            if isinstance(resp, str):
                text = resp
            elif isinstance(resp, dict):
                text = (
                    str(resp.get("content") or "")
                    or str(resp.get("response") or "")
                    or str(resp.get("answer") or "")
                )
            elif hasattr(resp, "content") and isinstance(getattr(resp, "content"), str):
                text = getattr(resp, "content")

            for piece in _iter_text_chunks(text):
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
