"""Non-streaming LLM calls with traceable raw artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

from common import (
    build_llm_model_func,
    ensure_dir,
    parse_json_object_from_llm,
    parse_jsonl_from_llm,
    write_json,
    write_jsonl,
)

logger = logging.getLogger(__name__)


async def call_llm_json(
    prompt: str,
    system_prompt: str | None = None,
    *,
    output_path: Path | None = None,
    raw_output_path: Path | None = None,
    expect_jsonl: bool = False,
    guided_json_schema: dict | None = None,
) -> dict | list[dict]:
    llm_func = build_llm_model_func()
    raw_text = ""
    try:
        extra_body = {"guided_json": guided_json_schema} if guided_json_schema else None
        raw_text = await llm_func(
            prompt,
            system_prompt=system_prompt,
            stream=False,
            extra_body=extra_body,
        )
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
        if raw_output_path:
            ensure_dir(raw_output_path.parent)
            raw_output_path.write_text(raw_text, encoding="utf-8")

        if expect_jsonl:
            result: dict | list[dict] = parse_jsonl_from_llm(raw_text)
        else:
            try:
                result = parse_json_object_from_llm(raw_text)
            except ValueError:
                result = parse_jsonl_from_llm(raw_text)

        if output_path:
            if isinstance(result, list):
                write_jsonl(output_path, result)
            else:
                write_json(output_path, result)
        return result
    except Exception:
        artifact_base = raw_output_path or output_path
        if artifact_base:
            ensure_dir(artifact_base.parent)
            bad_path = artifact_base.with_suffix(artifact_base.suffix + ".bad.txt")
            bad_path.write_text(
                f"===== PROMPT =====\n{prompt}\n\n===== RAW OUTPUT =====\n{raw_text}",
                encoding="utf-8",
            )
            logger.error("LLM 调用/解析失败，诊断文件已保存: %s", bad_path)
        raise
