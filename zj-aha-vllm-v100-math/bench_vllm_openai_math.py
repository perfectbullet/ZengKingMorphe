#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production-grade vLLM OpenAI-compatible benchmark script.

What it measures:
- TTFT: time to first token, only available in streaming mode
- E2E latency: request start to finish
- Decode latency: first token to finish
- Prompt tokens / completion tokens / total tokens from OpenAI usage
- Decode tokens/s
- End-to-end tokens/s
- Success / failure count
- Per-request JSONL and summary CSV

Example:
  python bench_vllm_openai.py \
    --base-url http://192.168.100.230:8200/v1 \
    --model Qwen3-32B \
    --concurrency 1 \
    --requests 10 \
    --max-tokens 512

Concurrent test:
  python bench_vllm_openai.py \
    --base-url http://192.168.100.230:8200/v1 \
    --model Qwen3-32B \
    --concurrency 4 \
    --requests 40 \
    --max-tokens 1024
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from openai import AsyncOpenAI


DEFAULT_PROMPT = """求解不等式x的平方减去5x加上6小于0的解"""


@dataclass
class RequestResult:
    request_id: int
    ok: bool
    error: str

    concurrency: int
    model: str
    stream: bool

    start_ts: float
    end_ts: float
    e2e_latency_s: float
    ttft_s: Optional[float]
    decode_latency_s: Optional[float]

    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]

    output_chars: int
    chunks: int

    decode_tokens_per_s: Optional[float]
    e2e_completion_tokens_per_s: Optional[float]
    total_tokens_per_s: Optional[float]


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)


def mean(values: list[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def fmt(v: Optional[float], ndigits: int = 3) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{ndigits}f}"


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    return DEFAULT_PROMPT


async def check_models(client: AsyncOpenAI, model: str) -> None:
    try:
        models = await client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"[check] /v1/models 可访问，模型数量: {len(model_ids)}")
        if model not in model_ids:
            print(f"[warn] 目标模型 `{model}` 不在 /v1/models 返回列表中。返回模型: {model_ids}")
    except Exception as exc:
        print(f"[warn] /v1/models 检查失败，但继续压测。错误: {type(exc).__name__}: {exc}")


async def one_stream_request(
    *,
    client: AsyncOpenAI,
    request_id: int,
    args: argparse.Namespace,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    async with semaphore:
        start = time.perf_counter()
        first_token_at: Optional[float] = None
        end = start

        output_chars = 0
        chunks = 0
        usage: Optional[Any] = None

        try:
            stream = await client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )

            async for chunk in stream:
                # vLLM / OpenAI compatible APIs usually put usage in final chunk
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if not content:
                    continue

                if first_token_at is None:
                    first_token_at = time.perf_counter()

                chunks += 1
                output_chars += len(content)

                if args.print_output and request_id == 0:
                    print(content, end="", flush=True)

            end = time.perf_counter()

            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
            total_tokens = getattr(usage, "total_tokens", None) if usage else None

            ttft = (first_token_at - start) if first_token_at is not None else None
            decode_latency = (end - first_token_at) if first_token_at is not None else None
            e2e_latency = end - start

            decode_tps = (
                completion_tokens / decode_latency
                if completion_tokens is not None and decode_latency and decode_latency > 0
                else None
            )
            e2e_completion_tps = (
                completion_tokens / e2e_latency
                if completion_tokens is not None and e2e_latency > 0
                else None
            )
            total_tps = (
                total_tokens / e2e_latency
                if total_tokens is not None and e2e_latency > 0
                else None
            )

            return RequestResult(
                request_id=request_id,
                ok=True,
                error="",
                concurrency=args.concurrency,
                model=args.model,
                stream=True,
                start_ts=start,
                end_ts=end,
                e2e_latency_s=e2e_latency,
                ttft_s=ttft,
                decode_latency_s=decode_latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                output_chars=output_chars,
                chunks=chunks,
                decode_tokens_per_s=decode_tps,
                e2e_completion_tokens_per_s=e2e_completion_tps,
                total_tokens_per_s=total_tps,
            )

        except Exception as exc:
            end = time.perf_counter()
            return RequestResult(
                request_id=request_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                concurrency=args.concurrency,
                model=args.model,
                stream=True,
                start_ts=start,
                end_ts=end,
                e2e_latency_s=end - start,
                ttft_s=None,
                decode_latency_s=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                output_chars=output_chars,
                chunks=chunks,
                decode_tokens_per_s=None,
                e2e_completion_tokens_per_s=None,
                total_tokens_per_s=None,
            )


async def one_non_stream_request(
    *,
    client: AsyncOpenAI,
    request_id: int,
    args: argparse.Namespace,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    async with semaphore:
        start = time.perf_counter()
        output_chars = 0

        try:
            resp = await client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )

            end = time.perf_counter()
            content = resp.choices[0].message.content or ""
            output_chars = len(content)

            if args.print_output and request_id == 0:
                print(content, end="\n", flush=True)

            usage = resp.usage
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
            total_tokens = getattr(usage, "total_tokens", None) if usage else None

            e2e_latency = end - start
            e2e_completion_tps = (
                completion_tokens / e2e_latency
                if completion_tokens is not None and e2e_latency > 0
                else None
            )
            total_tps = (
                total_tokens / e2e_latency
                if total_tokens is not None and e2e_latency > 0
                else None
            )

            return RequestResult(
                request_id=request_id,
                ok=True,
                error="",
                concurrency=args.concurrency,
                model=args.model,
                stream=False,
                start_ts=start,
                end_ts=end,
                e2e_latency_s=e2e_latency,
                ttft_s=None,
                decode_latency_s=None,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                output_chars=output_chars,
                chunks=1,
                decode_tokens_per_s=None,
                e2e_completion_tokens_per_s=e2e_completion_tps,
                total_tokens_per_s=total_tps,
            )

        except Exception as exc:
            end = time.perf_counter()
            return RequestResult(
                request_id=request_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                concurrency=args.concurrency,
                model=args.model,
                stream=False,
                start_ts=start,
                end_ts=end,
                e2e_latency_s=end - start,
                ttft_s=None,
                decode_latency_s=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                output_chars=output_chars,
                chunks=0,
                decode_tokens_per_s=None,
                e2e_completion_tokens_per_s=None,
                total_tokens_per_s=None,
            )


async def run_benchmark(args: argparse.Namespace) -> list[RequestResult]:
    prompt = load_prompt(args)

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
    )

    if not args.skip_model_check:
        await check_models(client, args.model)

    semaphore = asyncio.Semaphore(args.concurrency)

    # warmup: not counted
    if args.warmup > 0:
        print(f"[warmup] 开始预热 {args.warmup} 次，不计入统计")
        warmup_tasks = [
            one_stream_request(client=client, request_id=i, args=args, prompt=prompt, semaphore=semaphore)
            if args.stream
            else one_non_stream_request(client=client, request_id=i, args=args, prompt=prompt, semaphore=semaphore)
            for i in range(args.warmup)
        ]
        await asyncio.gather(*warmup_tasks)
        print("[warmup] 预热完成")

    print("=" * 90)
    print(
        f"开始压测: model={args.model}, requests={args.requests}, concurrency={args.concurrency}, "
        f"stream={args.stream}, max_tokens={args.max_tokens}"
    )
    print(f"base_url={args.base_url}")
    print("=" * 90)

    bench_start = time.perf_counter()

    tasks = [
        one_stream_request(client=client, request_id=i, args=args, prompt=prompt, semaphore=semaphore)
        if args.stream
        else one_non_stream_request(client=client, request_id=i, args=args, prompt=prompt, semaphore=semaphore)
        for i in range(args.requests)
    ]

    results: list[RequestResult] = []
    completed = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1

        if args.progress:
            status = "OK" if result.ok else "ERR"
            print(
                f"[{completed}/{args.requests}] {status} "
                f"id={result.request_id} "
                f"e2e={fmt(result.e2e_latency_s)}s "
                f"ttft={fmt(result.ttft_s)}s "
                f"decode_tps={fmt(result.decode_tokens_per_s, 2)} "
                f"completion_tokens={result.completion_tokens} "
                f"err={result.error[:120]}"
            )

    bench_end = time.perf_counter()
    wall_time = bench_end - bench_start

    write_outputs(args, results)
    print_summary(args, results, wall_time)

    return results


def write_outputs(args: argparse.Namespace, results: list[RequestResult]) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or f"vllm_bench_{now_str()}_{uuid.uuid4().hex[:8]}"

    jsonl_path = out_dir / f"{run_id}.jsonl"
    csv_path = out_dir / f"{run_id}.csv"

    dicts = [asdict(r) for r in sorted(results, key=lambda x: x.request_id)]

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in dicts:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dicts[0].keys()) if dicts else [])
        if dicts:
            writer.writeheader()
            writer.writerows(dicts)

    print(f"[output] JSONL: {jsonl_path}")
    print(f"[output] CSV:   {csv_path}")


def print_summary(args: argparse.Namespace, results: list[RequestResult], wall_time: float) -> None:
    ok_results = [r for r in results if r.ok]
    failed_results = [r for r in results if not r.ok]

    e2e = [r.e2e_latency_s for r in ok_results]
    ttft = [r.ttft_s for r in ok_results if r.ttft_s is not None]
    decode_tps = [r.decode_tokens_per_s for r in ok_results if r.decode_tokens_per_s is not None]
    e2e_completion_tps = [
        r.e2e_completion_tokens_per_s for r in ok_results if r.e2e_completion_tokens_per_s is not None
    ]

    total_completion_tokens = sum(r.completion_tokens or 0 for r in ok_results)
    total_prompt_tokens = sum(r.prompt_tokens or 0 for r in ok_results)
    total_tokens = sum(r.total_tokens or 0 for r in ok_results)

    overall_completion_tps = total_completion_tokens / wall_time if wall_time > 0 else 0.0
    overall_total_tps = total_tokens / wall_time if wall_time > 0 else 0.0

    print("\n" + "=" * 90)
    print("压测汇总")
    print("=" * 90)
    print(f"模型:                         {args.model}")
    print(f"Base URL:                     {args.base_url}")
    print(f"请求总数:                     {len(results)}")
    print(f"成功/失败:                    {len(ok_results)} / {len(failed_results)}")
    print(f"并发数:                       {args.concurrency}")
    print(f"流式:                         {args.stream}")
    print(f"Wall Time:                    {wall_time:.3f} s")
    print(f"Prompt Tokens 合计:           {total_prompt_tokens}")
    print(f"Completion Tokens 合计:       {total_completion_tokens}")
    print(f"Total Tokens 合计:            {total_tokens}")
    print(f"整体 Completion 吞吐:         {overall_completion_tps:.2f} tokens/s")
    print(f"整体 Total 吞吐:              {overall_total_tps:.2f} tokens/s")

    print("\n单请求延迟:")
    print(f"  E2E 平均:                   {fmt(mean(e2e))} s")
    print(f"  E2E P50/P90/P95/P99:         {fmt(percentile(e2e, 50))} / {fmt(percentile(e2e, 90))} / {fmt(percentile(e2e, 95))} / {fmt(percentile(e2e, 99))} s")

    if args.stream:
        print("\n首 Token 时间 TTFT:")
        print(f"  TTFT 平均:                  {fmt(mean(ttft))} s")
        print(f"  TTFT P50/P90/P95/P99:        {fmt(percentile(ttft, 50))} / {fmt(percentile(ttft, 90))} / {fmt(percentile(ttft, 95))} / {fmt(percentile(ttft, 99))} s")

        print("\n单请求 Decode 速度:")
        print(f"  Decode TPS 平均:            {fmt(mean(decode_tps), 2)} tokens/s")
        print(f"  Decode TPS P50/P90/P95/P99:  {fmt(percentile(decode_tps, 50), 2)} / {fmt(percentile(decode_tps, 90), 2)} / {fmt(percentile(decode_tps, 95), 2)} / {fmt(percentile(decode_tps, 99), 2)} tokens/s")
    else:
        print("\n非流式模式没有 TTFT；只能看 E2E 吞吐。")
        print(f"  E2E Completion TPS 平均:    {fmt(mean(e2e_completion_tps), 2)} tokens/s")

    if failed_results:
        print("\n失败样例:")
        for r in failed_results[:5]:
            print(f"  id={r.request_id}, error={r.error}")

    print("=" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production-grade vLLM OpenAI-compatible benchmark script")

    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://192.168.100.230:8200/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", "Qwen3-32B"))

    parser.add_argument("--requests", type=int, default=10, help="Total benchmark requests")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup requests, not counted")

    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)

    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)

    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output-dir", default="bench_results")
    parser.add_argument("--run-id", default=None)

    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-output", action="store_true", help="Print output of request_id=0")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-model-check", action="store_true")

    args = parser.parse_args()

    if args.requests <= 0:
        parser.error("--requests must be > 0")
    if args.concurrency <= 0:
        parser.error("--concurrency must be > 0")
    if args.concurrency > args.requests:
        print("[warn] --concurrency > --requests，一般没必要；继续执行。")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be > 0")

    return args


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_benchmark(args))
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
