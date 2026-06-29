#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI


def now_ms() -> int:
    return int(time.time() * 1000)


def dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def obj_to_dict(obj: Any) -> Any:
    if obj is None:
        return None

    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")

    if isinstance(obj, dict):
        return obj

    return str(obj)


def get_extra_field(obj: Any, field: str) -> Optional[Any]:
    """
    OpenAI SDK 对非标准字段可能不会直接暴露为属性。
    vLLM 的 reasoning_content 通常在 model_extra 里。
    """
    if obj is None:
        return None

    value = getattr(obj, field, None)
    if value is not None:
        return value

    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(field)

    return None


def calc_speed(tokens: Optional[int], elapsed: float) -> Optional[float]:
    if isinstance(tokens, int) and elapsed > 0:
        return tokens / elapsed
    return None


def build_common_params(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,

        # vLLM 扩展参数，OpenAI SDK 要通过 extra_body 透传
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking
            }
        },
    }


def run_non_stream(
    client: OpenAI,
    mode: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    params = build_common_params(
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=mode["enable_thinking"],
    )
    params["stream"] = False

    start_ms = now_ms()
    start = time.perf_counter()

    error = None
    response = None

    try:
        response = client.chat.completions.create(**params)
    except Exception as e:
        error = repr(e)

    elapsed = time.perf_counter() - start
    end_ms = now_ms()

    response_dict = obj_to_dict(response)

    content = ""
    reasoning_content = ""
    usage_dict = None

    if response is not None:
        try:
            msg = response.choices[0].message
            content = msg.content or ""
            reasoning_content = get_extra_field(msg, "reasoning_content") or ""
        except Exception:
            pass

        usage_dict = obj_to_dict(getattr(response, "usage", None))

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    if isinstance(usage_dict, dict):
        prompt_tokens = usage_dict.get("prompt_tokens")
        completion_tokens = usage_dict.get("completion_tokens")
        total_tokens = usage_dict.get("total_tokens")

    return {
        "mode": mode,
        "request": {
            "base_url": args.base_url,
            "model": args.model,
            "stream": False,
            "enable_thinking": mode["enable_thinking"],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "prompt": args.prompt,
            "extra_body": params["extra_body"],
        },
        "timing": {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "elapsed_seconds": elapsed,
            "ttft_seconds": None,
        },
        "speed": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "completion_tokens_per_second": calc_speed(completion_tokens, elapsed),
            "total_tokens_per_second": calc_speed(total_tokens, elapsed),
        },
        "output": {
            "content": content,
            "reasoning_content": reasoning_content,
            "content_chars": len(content),
            "reasoning_content_chars": len(reasoning_content),
        },
        "raw_response": response_dict,
        "error": error,
    }


def run_stream(
    client: OpenAI,
    mode: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    params = build_common_params(
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=mode["enable_thinking"],
    )
    params["stream"] = True
    params["stream_options"] = {
        "include_usage": True
    }

    start_ms = now_ms()
    start = time.perf_counter()

    error = None
    ttft_seconds = None
    first_token_seen = False

    chunks = []
    content_parts = []
    reasoning_parts = []
    usage_dict = None

    try:
        stream = client.chat.completions.create(**params)

        for chunk in stream:
            t = time.perf_counter() - start
            chunk_dict = obj_to_dict(chunk)

            chunks.append({
                "t_seconds": t,
                "chunk": chunk_dict,
            })

            usage = getattr(chunk, "usage", None)
            if usage is not None:
                usage_dict = obj_to_dict(usage)

            if not getattr(chunk, "choices", None):
                continue

            delta = chunk.choices[0].delta

            delta_content = getattr(delta, "content", None)
            delta_reasoning = get_extra_field(delta, "reasoning_content")

            if delta_content:
                content_parts.append(delta_content)

            if delta_reasoning:
                reasoning_parts.append(delta_reasoning)

            if not first_token_seen and (delta_content or delta_reasoning):
                first_token_seen = True
                ttft_seconds = t

    except Exception as e:
        error = repr(e)

    elapsed = time.perf_counter() - start
    end_ms = now_ms()

    content = "".join(content_parts)
    reasoning_content = "".join(reasoning_parts)

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    if isinstance(usage_dict, dict):
        prompt_tokens = usage_dict.get("prompt_tokens")
        completion_tokens = usage_dict.get("completion_tokens")
        total_tokens = usage_dict.get("total_tokens")

    return {
        "mode": mode,
        "request": {
            "base_url": args.base_url,
            "model": args.model,
            "stream": True,
            "enable_thinking": mode["enable_thinking"],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "prompt": args.prompt,
            "extra_body": params["extra_body"],
            "stream_options": params["stream_options"],
        },
        "timing": {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "elapsed_seconds": elapsed,
            "ttft_seconds": ttft_seconds,
        },
        "speed": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "completion_tokens_per_second": calc_speed(completion_tokens, elapsed),
            "total_tokens_per_second": calc_speed(total_tokens, elapsed),
        },
        "output": {
            "content": content,
            "reasoning_content": reasoning_content,
            "content_chars": len(content),
            "reasoning_content_chars": len(reasoning_content),
            "chunk_count": len(chunks),
        },
        "chunks": chunks,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://192.168.100.201:8200/v1")
    parser.add_argument("--model", default="Qwen3-32B-AWQ")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--out-dir", default="./qwen3_openai_sdk_test_outputs")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--prompt",
        default=(
            "请解一道简单数学题，并先进行必要推理，再给出最终答案："
            "若 3x + 5 = 20，求 x。"
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
    )

    try:
        models_info = client.models.list()
        dump_json(out_dir / "models_info.json", {
            "raw_response": obj_to_dict(models_info)
        })
    except Exception as e:
        dump_json(out_dir / "models_info.json", {
            "error": repr(e)
        })

    cases = [
        {
            "case_name": "thinking_on_non_stream",
            "enable_thinking": True,
            "stream": False,
        },
        {
            "case_name": "thinking_off_non_stream",
            "enable_thinking": False,
            "stream": False,
        },
        {
            "case_name": "thinking_on_stream",
            "enable_thinking": True,
            "stream": True,
        },
        {
            "case_name": "thinking_off_stream",
            "enable_thinking": False,
            "stream": True,
        },
    ]

    summary = []

    for mode in cases:
        print(f"Running: {mode['case_name']}")

        if mode["stream"]:
            result = run_stream(client, mode, args)
        else:
            result = run_non_stream(client, mode, args)

        output_file = out_dir / f"{mode['case_name']}.json"
        dump_json(output_file, result)

        summary.append({
            "case_name": mode["case_name"],
            "stream": mode["stream"],
            "enable_thinking": mode["enable_thinking"],
            "elapsed_seconds": result["timing"]["elapsed_seconds"],
            "ttft_seconds": result["timing"]["ttft_seconds"],
            "prompt_tokens": result["speed"]["prompt_tokens"],
            "completion_tokens": result["speed"]["completion_tokens"],
            "total_tokens": result["speed"]["total_tokens"],
            "completion_tokens_per_second": result["speed"]["completion_tokens_per_second"],
            "total_tokens_per_second": result["speed"]["total_tokens_per_second"],
            "content_chars": result["output"]["content_chars"],
            "reasoning_content_chars": result["output"]["reasoning_content_chars"],
            "error": result["error"],
            "output_file": str(output_file),
        })

    dump_json(out_dir / "summary.json", {
        "base_url": args.base_url,
        "model": args.model,
        "summary": summary,
    })

    print()
    print("Done.")
    print(f"Output dir: {out_dir.resolve()}")
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()