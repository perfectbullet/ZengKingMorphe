#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI-style streaming client for /api/chat/openai/chat/completions.

依赖: requests
用法示例:
  python ai-service/scripts/stream_client.py --host http://192.168.8.230:8100 --employee_id hutao --user_id user_123456 --session_id sess_20251218_abc123 --query "我刚刚问了什么问题"

脚本特点:
- 支持参数化 host/employee_id/user_id/session_id/model/stream/query
- 使用 requests.post(..., stream=True) 读取响应
- 解析 SSE 风格的 `data: {...}` 行，或直接的 JSON 行
- 对 streaming chunk (object == "chat.completion.chunk") 输出 partial content
- 遇到 finish_reason == "stop" 或 payload == "[DONE]" 时结束
- 支持从环境变量 API_KEY 注入 X-API-Key 头
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import requests
from typing import Iterator, Optional


def build_body(model: str, query: str, employee_id: Optional[str],
               user_id: Optional[str], session_id: Optional[str]) -> dict:
    messages = [{"role": "user", "content": query}]
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "employee_id": employee_id or "default",
        "user_id": user_id or "anonymous",
        "session_id": session_id,
    }
    return body


def iter_sse_payloads(resp: requests.Response) -> Iterator[str]:
    """
    从响应中迭代 SSE payload（剥离 'data:' 前缀）。
    兼容以下两种常见服务器输出：
    - SSE 行: data: {...}
    - 直接 JSON 行: {...}
    """
    # requests.iter_lines(decode_unicode=True) 能逐行读取 chunked 输出
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            continue
        if line.startswith(":"):
            # SSE keepalive/comment
            continue
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
        else:
            payload = line
        yield payload


def handle_stream_payloads(payload_iter: Iterator[str], start_time: float) -> int:
    """
    处理 SSE payload 迭代器。
    返回 0 表示正常结束，非 0 表示出错。
    """
    first_token_latency = None
    try:
        for payload in payload_iter:
            if payload == "[DONE]":
                print("\n[DONE]")
                return 0
            # 尝试解析 JSON
            try:
                obj = json.loads(payload)
            except Exception:
                # 不是 JSON，直接打印
                print(payload, flush=True)
                continue

            # 如果包含 error，打印并退出
            if isinstance(obj, dict) and obj.get("error"):
                print("Error from server:", json.dumps(obj["error"], ensure_ascii=False, indent=2), file=sys.stderr)
                return 2

            # 流式 chunk
            if obj.get("object") == "chat.completion.chunk":
                # print(obj)
                choices = obj.get("choices", []) or []
                for choice in choices:
                    # 输出增量内容（delta.content）
                    delta = choice.get("delta", {}) or {}
                    
                    content = delta.get("content")
                    # print(delta)
                    if content:
                        # 计算并打印首 token 延迟
                        if first_token_latency is None:
                            first_token_latency = time.perf_counter() - start_time
                            print(f"⏱️ First token latency: {first_token_latency*1000:.2f}ms\n", file=sys.stderr)
                        # 不换行，直接 flush
                        sys.stdout.write(content)
                        sys.stdout.flush()
                        print('time:', time.time())
                    # 检查 finish_reason
                    finish = choice.get("finish_reason")
                    if finish == "stop":
                        print()  # 换行结束
                        # 也可以打印 usage/metadata if present in this chunk
                        return 0
                continue

            # 最终聚合结果（非流式）
            if obj.get("object") == "chat.completion":
                choices = obj.get("choices", []) or []
                if choices:
                    first = choices[0]
                    message = first.get("message") or {}
                    content = message.get("content")
                    if content:
                        print(content)
                # 打印元信息以便调试
                try:
                    print("\n--- metadata ---")
                    print(json.dumps(obj, ensure_ascii=False, indent=2))
                except Exception:
                    print("\n--- metadata (raw) ---")
                    print(obj)
                return 0

            # 其他未知事件：打印供调试
            try:
                print("\n[unhandled event]")
                print(json.dumps(obj, ensure_ascii=False, indent=2))
            except Exception:
                print("\n[unhandled event (raw)]")
                print(payload)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    return 0


def run_stream(host: str, body: dict, api_key: Optional[str], timeout: int = 60) -> int:
    url = host.rstrip("/") + "/api/chat/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        start_time = time.perf_counter()
        with requests.post(url, json=body, headers=headers, stream=True, timeout=(5, timeout)) as resp:
            try:
                resp.raise_for_status()
            except requests.HTTPError:
                print(f"HTTP error {resp.status_code}:", resp.text, file=sys.stderr)
                return 2
            ttfb = time.perf_counter() - start_time
            print(f"✅ Connection established (status: {resp.status_code}) - TTFB: {ttfb*1000:.2f}ms", file=sys.stderr)
            payload_iter = iter_sse_payloads(resp)
            return handle_stream_payloads(payload_iter, start_time)
    except requests.RequestException as e:
        print(f"Request error: url is {url}", str(e), file=sys.stderr)
        return 2


 

def main():
    parser = argparse.ArgumentParser(description="OpenAI-style streaming test client")
    parser.add_argument("--host", default="http://192.168.8.230:8100", help="Base host (including port)")
    parser.add_argument("--employee_id", default="hutao", help="Employee ID")
    parser.add_argument("--user_id", default="user_123456", help="User ID")
    parser.add_argument("--session_id", default='sess_20251222_abc123', help="Session ID")
    parser.add_argument("--model", default="qwen2.5:7b", help="Model name")
    
    parser.add_argument("--query", required=True, help="User query text")
    parser.add_argument("--timeout", type=int, default=60, help="Stream timeout seconds")
    args = parser.parse_args()

    api_key = os.environ.get("API_KEY")
    body = build_body(args.model, args.query, args.employee_id, args.user_id, args.session_id)

    rc = run_stream(args.host, body, api_key, timeout=args.timeout)
 
    sys.exit(rc)


if __name__ == "__main__":
    main()
