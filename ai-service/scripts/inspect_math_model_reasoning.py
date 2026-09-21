#!/usr/bin/env python3
"""
Inspect reasoning/thinking fields returned by the configured math model.

This script compares two layers:

1. Raw vLLM OpenAI-compatible SSE response.
2. The current LangChain ChatOpenAI stream used by ZengKingMorphe.

Why:
- vLLM may return reasoning in delta.reasoning (new) or
  delta.reasoning_content (legacy).
- ChatOpenAI may or may not preserve provider-specific fields depending on
  the installed langchain-openai version.
- We need to know where the reasoning disappears before changing the
  production display pipeline.

Usage:
    cd ai-service

    python scripts/inspect_math_model_reasoning.py

    python scripts/inspect_math_model_reasoning.py \
      --query "解方程 x^2-5x+6=0，并给出完整推理过程"

Environment variables:
    MATH_MODEL_BASE_URL
    MATH_MODEL_NAME
    MATH_MODEL_API_KEY
    MATH_MAX_TOKEN

Defaults are aligned with the current math-model deployment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

# Make `app` importable when launched from ai-service/scripts.
AI_SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

DEFAULT_BASE_URL = os.getenv(
    "MATH_MODEL_BASE_URL",
    "http://192.168.8.231:8200/v1",
).rstrip("/")
DEFAULT_MODEL = os.getenv(
    "MATH_MODEL_NAME",
    "nvidia/Qwen3.6-35B-A3B-NVFP4",
)
DEFAULT_API_KEY = (
    os.getenv("MATH_MODEL_API_KEY", "").strip().strip('"').strip("'")
    or "dummy-key"
)
DEFAULT_MAX_TOKENS = int(os.getenv("MATH_MAX_TOKEN", "2048"))
DEFAULT_QUERY = "解方程 x^2-5x+6=0，并给出推理过程。"


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _reasoning_from_mapping(data: dict[str, Any]) -> str:
    """Support both current vLLM and legacy provider field names."""
    value = data.get("reasoning")
    if value:
        return str(value)

    value = data.get("reasoning_content")
    if value:
        return str(value)

    return ""


async def inspect_raw_vllm(
    *,
    base_url: str,
    model: str,
    api_key: str,
    query: str,
    max_tokens: int,
    print_all_chunks: bool,
) -> dict[str, Any]:
    """
    Hit vLLM directly and inspect the exact JSON carried by SSE.

    This is the source of truth. It bypasses LangChain completely.
    """
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "stream": True,
        "max_tokens": max_tokens,
        # Qwen3.x supports thinking through chat-template kwargs.
        # If the server rejects this, retry without it using --no-enable-thinking.
        "chat_template_kwargs": {"enable_thinking": True},
    }

    print("\n" + "=" * 90)
    print("1) RAW vLLM SSE")
    print("=" * 90)
    print(f"POST {url}")
    print(f"model = {model}")
    print(f"query = {query!r}")

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    seen_delta_keys: set[str] = set()
    chunk_count = 0
    first_reasoning_chunk: dict[str, Any] | None = None
    first_content_chunk: dict[str, Any] | None = None

    timeout = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=body,
        ) as response:
            if response.status_code >= 400:
                raw = await response.aread()
                raise RuntimeError(
                    f"vLLM HTTP {response.status_code}: "
                    f"{raw.decode(errors='replace')}"
                )

            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or line.startswith(":"):
                    continue

                if line.startswith("data:"):
                    payload = line[5:].strip()
                else:
                    payload = line

                if payload == "[DONE]":
                    print("\n[RAW] [DONE]")
                    break

                chunk_count += 1

                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    print(f"[RAW #{chunk_count}] non-json: {payload!r}")
                    continue

                choices = obj.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                if isinstance(delta, dict):
                    seen_delta_keys.update(delta.keys())

                reasoning = (
                    _reasoning_from_mapping(delta)
                    if isinstance(delta, dict)
                    else ""
                )
                content = (
                    str(delta.get("content") or "")
                    if isinstance(delta, dict)
                    else ""
                )

                if reasoning:
                    reasoning_parts.append(reasoning)
                    if first_reasoning_chunk is None:
                        first_reasoning_chunk = obj
                    print(f"[RAW reasoning] {reasoning!r}")

                if content:
                    content_parts.append(content)
                    if first_content_chunk is None:
                        first_content_chunk = obj
                    print(f"[RAW content]   {content!r}")

                if print_all_chunks:
                    print(f"\n[RAW chunk #{chunk_count}]")
                    print(_json(obj))

    result = {
        "chunk_count": chunk_count,
        "delta_keys": sorted(seen_delta_keys),
        "reasoning": "".join(reasoning_parts),
        "content": "".join(content_parts),
        "first_reasoning_chunk": first_reasoning_chunk,
        "first_content_chunk": first_content_chunk,
    }

    print("\n[RAW summary]")
    print(f"delta keys        : {result['delta_keys']}")
    print(f"reasoning chars   : {len(result['reasoning'])}")
    print(f"content chars     : {len(result['content'])}")

    if first_reasoning_chunk:
        print("\nFirst raw reasoning chunk:")
        print(_json(first_reasoning_chunk))

    if first_content_chunk:
        print("\nFirst raw content chunk:")
        print(_json(first_content_chunk))

    return result


def _extract_reasoning_from_langchain_chunk(chunk: Any) -> str:
    """
    Probe every reasonable place where ChatOpenAI may expose provider reasoning.
    """
    # Direct attributes, if a future/current wrapper exposes them.
    for attr in ("reasoning", "reasoning_content"):
        value = getattr(chunk, attr, None)
        if value:
            return str(value)

    # LangChain provider-specific extras usually land here when preserved.
    additional_kwargs = getattr(chunk, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        value = _reasoning_from_mapping(additional_kwargs)
        if value:
            return value

    # Some wrappers/providers may put details in response metadata.
    response_metadata = getattr(chunk, "response_metadata", None)
    if isinstance(response_metadata, dict):
        value = _reasoning_from_mapping(response_metadata)
        if value:
            return value

    return ""


async def inspect_langchain(
    *,
    base_url: str,
    model: str,
    api_key: str,
    query: str,
    max_tokens: int,
    print_all_chunks: bool,
) -> dict[str, Any]:
    """
    Inspect what ZengKingMorphe's current ChatOpenAI layer actually exposes.
    """
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    print("\n" + "=" * 90)
    print("2) LangChain ChatOpenAI.astream()")
    print("=" * 90)

    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        streaming=True,
    )

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    chunk_count = 0
    first_interesting_chunk: dict[str, Any] | None = None

    async for chunk in llm.astream(
        [HumanMessage(content=query)],
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    ):
        chunk_count += 1

        content = getattr(chunk, "content", "") or ""
        if not isinstance(content, str):
            content = str(content)

        reasoning = _extract_reasoning_from_langchain_chunk(chunk)

        if reasoning:
            reasoning_parts.append(reasoning)
            print(f"[LC reasoning] {reasoning!r}")

        if content:
            content_parts.append(content)
            print(f"[LC content]   {content!r}")

        snapshot = {
            "type": type(chunk).__name__,
            "content": getattr(chunk, "content", None),
            "additional_kwargs": getattr(chunk, "additional_kwargs", None),
            "response_metadata": getattr(chunk, "response_metadata", None),
        }

        if first_interesting_chunk is None and (reasoning or content):
            first_interesting_chunk = snapshot

        if print_all_chunks:
            print(f"\n[LC chunk #{chunk_count}]")
            print(_json(snapshot))

    result = {
        "chunk_count": chunk_count,
        "reasoning": "".join(reasoning_parts),
        "content": "".join(content_parts),
        "first_interesting_chunk": first_interesting_chunk,
    }

    print("\n[LangChain summary]")
    print(f"reasoning chars   : {len(result['reasoning'])}")
    print(f"content chars     : {len(result['content'])}")

    if first_interesting_chunk:
        print("\nFirst LangChain interesting chunk:")
        print(_json(first_interesting_chunk))

    return result


def print_diagnosis(
    raw_result: dict[str, Any],
    langchain_result: dict[str, Any] | None,
) -> None:
    raw_has_reasoning = bool(raw_result.get("reasoning"))
    lc_has_reasoning = bool(
        langchain_result and langchain_result.get("reasoning")
    )

    print("\n" + "=" * 90)
    print("DIAGNOSIS")
    print("=" * 90)

    if not raw_has_reasoning:
        print(
            "A. Raw vLLM SSE did NOT expose reasoning/reasoning_content.\n"
            "   Check the vLLM launch command first. For Qwen3.6, the server "
            "should enable a Qwen3 reasoning parser (for example "
            "`--reasoning-parser qwen3`) and thinking must not be disabled."
        )
        return

    if raw_has_reasoning and not lc_has_reasoning:
        print(
            "B. Raw vLLM SSE DOES expose reasoning, but ChatOpenAI does NOT.\n"
            "   The reasoning is being lost in the LangChain compatibility "
            "layer. Production code needs a small math-model stream adapter "
            "or a provider-specific ChatOpenAI wrapper before the "
            "ChatOrchestrator can emit reasoning AnswerEvents."
        )
        return

    print(
        "C. Both raw vLLM and ChatOpenAI expose reasoning.\n"
        "   No model-client replacement is needed. ChatOrchestrator can extract "
        "the reasoning field and emit a dedicated AnswerEvent directly."
    )


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect math-model reasoning stream structure."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Only inspect raw vLLM SSE; skip LangChain.",
    )
    parser.add_argument(
        "--all-chunks",
        action="store_true",
        help="Print every raw/LangChain chunk, not only meaningful fields.",
    )
    args = parser.parse_args()

    raw_result = await inspect_raw_vllm(
        base_url=args.base_url.rstrip("/"),
        model=args.model,
        api_key=args.api_key,
        query=args.query,
        max_tokens=args.max_tokens,
        print_all_chunks=args.all_chunks,
    )

    langchain_result = None
    if not args.raw_only:
        langchain_result = await inspect_langchain(
            base_url=args.base_url.rstrip("/"),
            model=args.model,
            api_key=args.api_key,
            query=args.query,
            max_tokens=args.max_tokens,
            print_all_chunks=args.all_chunks,
        )

    print_diagnosis(raw_result, langchain_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
