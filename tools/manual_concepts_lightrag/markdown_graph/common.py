"""Shared helpers for the staged Markdown-to-LightRAG pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent.parent
DEFAULT_WORKING_DIR = REPO_ROOT / "ai-service/data/lightrag_industrial_training"


def resolve_path(path_str: str, base_dirs: list[Path]) -> Path:
    """Resolve an absolute path or the first existing base-relative path."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [base.expanduser() / path for base in base_dirs]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    base = base_dirs[0] if base_dirs else Path.cwd()
    return (base.expanduser() / path).resolve()


def load_json(path: Path) -> dict | list:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line_no, raw_line in enumerate(file, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败: {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL 每行必须是对象: {path}:{line_no}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify_for_id(text: str) -> str:
    value = text.strip().lower()
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unnamed"


def stable_id(*parts: str, prefix: str = "") -> str:
    normalized = "\x1f".join(str(part).strip() for part in parts)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}"


def count_tokens_rough(text: str) -> int:
    """Cheap estimate: CJK characters plus roughly one token per four others."""
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", text))
    non_cjk = re.sub(r"[\u3400-\u9fff\s]", "", text)
    return cjk_count + (len(non_cjk) + 3) // 4


def pick_env(candidates: list[str], label: str, required: bool = True) -> str | None:
    for name in candidates:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    if required:
        raise RuntimeError(f"缺少必要环境变量 {label}，候选: {', '.join(candidates)}")
    return None


def resolve_llm_config() -> dict:
    return {
        "model": pick_env(["LLM_MODEL", "OPENAI_MODEL"], "LLM_MODEL"),
        "base_url": pick_env(
            ["LLM_BINDING_HOST", "OPENAI_BASE_URL", "LLM_BASE_URL"],
            "LLM_BASE_URL",
        ),
        "api_key": pick_env(
            ["LLM_BINDING_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"],
            "LLM_API_KEY",
            required=False,
        )
        or "no-api-key",
    }


def resolve_embedding_config() -> dict:
    llm = resolve_llm_config()
    dim_raw = pick_env(["EMBEDDING_DIM", "VLLM_EMBED_DIM"], "EMBEDDING_DIM")
    try:
        dimension = int(dim_raw or "")
    except ValueError as exc:
        raise RuntimeError(f"EMBEDDING_DIM 必须是整数: {dim_raw!r}") from exc
    return {
        "model": pick_env(
            ["EMBEDDING_MODEL", "VLLM_EMBED_MODEL", "VLLM_EMBEDDING_MODEL"],
            "EMBEDDING_MODEL",
        ),
        "dim": dimension,
        "base_url": pick_env(
            [
                "EMBEDDING_BINDING_HOST",
                "VLLM_EMBED_HOST",
                "VLLM_EMBEDDING_BASE_URL",
            ],
            "EMBEDDING_BASE_URL",
            required=False,
        )
        or llm["base_url"],
        "api_key": pick_env(
            [
                "EMBEDDING_BINDING_API_KEY",
                "VLLM_EMBED_API_KEY",
                "VLLM_API_KEY",
            ],
            "EMBEDDING_API_KEY",
            required=False,
        )
        or llm["api_key"],
    }


def build_llm_request_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Apply this pipeline's fixed generation settings to an LLM request."""
    request_kwargs = dict(kwargs)
    extra_body_value = request_kwargs.get("extra_body")
    if extra_body_value is None:
        extra_body: dict[str, Any] = {}
    elif isinstance(extra_body_value, dict):
        extra_body = dict(extra_body_value)
    else:
        raise TypeError("extra_body 必须是 dict")

    template_value = extra_body.get("chat_template_kwargs")
    if template_value is None:
        chat_template_kwargs: dict[str, Any] = {}
    elif isinstance(template_value, dict):
        chat_template_kwargs = dict(template_value)
    else:
        raise TypeError("extra_body.chat_template_kwargs 必须是 dict")

    chat_template_kwargs["enable_thinking"] = False
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    request_kwargs["extra_body"] = extra_body
    # This separately disables LightRAG's reasoning-content/COT integration.
    request_kwargs["enable_cot"] = False
    request_kwargs["max_tokens"] = 20480
    request_kwargs["temperature"] = 0.7
    return request_kwargs


def build_llm_model_func():
    from lightrag.llm.openai import openai_complete_if_cache

    config = resolve_llm_config()

    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        **kwargs: Any,
    ) -> str:
        request_kwargs = build_llm_request_kwargs(kwargs)
        return await openai_complete_if_cache(
            config["model"],
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=config["base_url"],
            api_key=config["api_key"],
            **request_kwargs,
        )

    return llm_model_func


def build_embedding_func():
    from lightrag.llm.openai import openai_embed
    from lightrag.utils import EmbeddingFunc

    config = resolve_embedding_config()

    async def embedding_func(
        texts: list[str],
        max_token_size: int | None = None,
    ):
        return await openai_embed.func(
            texts,
            model=config["model"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            max_token_size=max_token_size,
        )

    return EmbeddingFunc(
        embedding_dim=config["dim"],
        max_token_size=7000,
        func=embedding_func,
    )


def load_prompt(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt 文件不存在: {path}")
    return path.read_text(encoding="utf-8")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(
        r"```(?:json|jsonl)?\s*(.*?)\s*```", stripped, re.DOTALL | re.I
    )
    if match:
        return match.group(1).strip()
    embedded = re.search(
        r"```(?:json|jsonl)?\s*(.*?)\s*```", stripped, re.DOTALL | re.I
    )
    return embedded.group(1).strip() if embedded else stripped


def parse_json_object_from_llm(text: str) -> dict:
    cleaned = _strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("LLM 输出中未找到有效 JSON 对象")


def parse_jsonl_from_llm(text: str) -> list[dict]:
    cleaned = _strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            return parsed
        if isinstance(parsed, dict):
            for key in ("blocks", "records", "items", "data"):
                value = parsed.get(key)
                if isinstance(value, list) and all(
                    isinstance(item, dict) for item in value
                ):
                    return value
    except json.JSONDecodeError:
        pass

    # Some models concatenate valid top-level objects without JSONL newlines:
    # {...}{...}{...}. Decode them sequentially before falling back to lines.
    decoder = json.JSONDecoder()
    concatenated: list[dict] = []
    position = 0
    while position < len(cleaned):
        while position < len(cleaned) and cleaned[position].isspace():
            position += 1
        if position >= len(cleaned):
            break
        try:
            record, end = decoder.raw_decode(cleaned, position)
        except json.JSONDecodeError:
            concatenated = []
            break
        if not isinstance(record, dict):
            concatenated = []
            break
        concatenated.append(record)
        position = end
    if concatenated and position == len(cleaned):
        return concatenated

    records: list[dict] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip().rstrip(",")
        if not line or line in {"[", "]"}:
            continue
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM JSONL 行解析失败: {line[:160]}") from exc
        if not isinstance(record, dict):
            raise ValueError("LLM JSONL 每行必须是对象")
        records.append(record)
    if not records:
        raise ValueError("LLM 输出中未找到有效 JSONL 记录")
    return records
