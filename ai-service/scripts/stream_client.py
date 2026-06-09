#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI-style streaming client for /api/chat/v1/chat/completions and /api/chat/v2/chat/completions.

依赖: requests

=======
用法说明
=======

【快速开始】（使用默认参数）
  PYTHONPATH=. python scripts/stream_client.py --query "你好"

【指定服务器】
  # 本地开发服务器
  python scripts/stream_client.py --host http://192.168.8.233:8100 --query "你好"

  # 生产服务器
  python scripts/stream_client.py --host http://192.168.8.233:8100 --query "你好"

【使用 v2 API】
  python scripts/stream_client.py --api-version v2 --query "你好"

【指定员工/用户/会话】
  python scripts/stream_client.py \\
    --host http://192.168.8.233:8100 \\
    --employee_id 29 \\
    --user_id 3 \\
    --session_id sess_4_3_29 \\
    --query "失蜡铸造的原理"

【完整示例】
  # 知识库问答
  python scripts/stream_client.py --query "介绍集合的概念"

  # 实时信息查询（自动触发联网搜索）
  python scripts/stream_client.py --query "北京天气咋样"

  # 学术研究问答
  python scripts/stream_client.py --query "数据资产通过哪两条重要途径推动企业新质生产力发展？"

=======
默认参数
=======
  --host           http://192.168.8.233:8100
  --employee_id    29
  --user_id        3
  --session_id     sess_4_3_29
  --model          qwen3:14b
  --api-version    v1

=======
脚本特点
=======
- 支持参数化 host/employee_id/user_id/session_id/model/stream/query
- 支持 v1/v2 API 切换
- v2 API 支持额外的 user_name 和 head_url 参数
- 使用 requests.post(..., stream=True) 读取响应
- 解析 SSE 风格的 `data: {...}` 行，或直接的 JSON 行
- 对 streaming chunk (object == "chat.completion.chunk") 输出 partial content
- 遇到 finish_reason == "stop" 或 payload == "[DONE]" 时结束
- 支持从环境变量 API_KEY 注入 X-API-Key 头
- 显示 TTFB (首字节响应时间)、First Token Latency (首 token 延迟)、
  Total request time (从发起 POST 到流结束/收到完整响应的端到端耗时)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import requests
from typing import Iterator, Optional

def build_body(
    model: str,
    query: str,
    employee_id: Optional[str],
    user_id: Optional[str],
    session_id: Optional[str],
    channel_name: Optional[str] = None,
    team_id: Optional[str] = None,
    user_name: Optional[str] = None,
    head_url: Optional[str] = None,
    api_version: str = "v1",
    extra_body: Optional[dict] = None
) -> dict:
    messages = [{"role": "user", "content": query}]
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "employee_id": employee_id or "default",
        "user_id": user_id or "anonymous",
        "session_id": session_id,
    }

    # 添加可选参数
    if channel_name:
        body["channel_name"] = channel_name
    if team_id:
        body["team_id"] = team_id

    # v2 API 特有参数
    if api_version == "v2":
        if user_name:
            body["user_name"] = user_name
        if head_url:
            body["head_url"] = head_url

    # 合并 extra_body 中的参数
    if extra_body:
        body.update(extra_body)

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
        for idx, payload in enumerate(payload_iter):
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
                        
                        print(f'# chunk-{idx}: {content!r}')
                        # print(f'# chunk-for-md-{idx}: {content}')

                    # 检查 finish_reason
                    finish = choice.get("finish_reason")
                    if finish == "stop":
                        print()  # 换行结束
                        # 打印完整的元数据（包含 usage 和 metadata）
                        print("\n--- metadata ---")
                        print(json.dumps(obj, ensure_ascii=False, indent=2))
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
                        print(repr(content))
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
    finally:
        total_s = time.perf_counter() - start_time
        print(
            f"⏱️ Total request time: {total_s * 1000:.2f}ms",
            file=sys.stderr,
        )
    return 0

def run_stream(host: str, body: dict, api_key: Optional[str], timeout: int = 60, api_version: str = "v1") -> int:
    url = host.rstrip("/") + f"/api/chat/{api_version}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        start_time = time.perf_counter()
        print(f"url is {url} (API version: {api_version})")
        with requests.post(url, json=body, headers=headers, stream=True, timeout=(5, timeout)) as resp:
            try:
                resp.raise_for_status()
            except requests.HTTPError:
                total_s = time.perf_counter() - start_time
                print(
                    f"⏱️ Total request time (end-to-end): {total_s * 1000:.2f}ms",
                    file=sys.stderr,
                )
                print(f"HTTP error {resp.status_code}:", resp.text, file=sys.stderr)
                return 2
            ttfb = time.perf_counter() - start_time
            print(f"✅ Connection established (status: {resp.status_code}) - TTFB: {ttfb*1000:.2f}ms", file=sys.stderr)
            payload_iter = iter_sse_payloads(resp)
            return handle_stream_payloads(payload_iter, start_time)
    except requests.RequestException as e:
        total_s = time.perf_counter() - start_time
        print(
            f"⏱️ Total request time (end-to-end): {total_s * 1000:.2f}ms",
            file=sys.stderr,
        )
        print(f"Request error: url is {url}", str(e), file=sys.stderr)
        return 2

 

def main():
    parser = argparse.ArgumentParser(
        description="OpenAI-style streaming test client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --query "你好"
  %(prog)s --host http://192.168.8.233:8100 --query "北京天气"

  # 使用 v2 API
  %(prog)s --api-version v2 --query "你好"
  %(prog)s --api-version v2 --user-name "张三" --head-url "http://example.com/avatar.jpg" --query "你好"

  # 使用 channel_name (格式: employee_<team_id>_<user_id>_<employee_id>)
  %(prog)s --channel_name "employee_4_46935014_29" --query "你好"

  # 使用 extra_body 传递参数
  %(prog)s --extra-body '{"team_id": "4", "user_id": "46935014", "employee_id": "29"}' --query "你好"
        """
    )
    parser.add_argument("--host", default="http://192.168.8.233:8100", help="Base host (including port), 默认: http://192.168.8.233:8100")
    parser.add_argument("--employee_id", default="29", help="Employee ID, 默认: 29")
    parser.add_argument("--user_id", default="3", help="User ID, 默认: 3")
    parser.add_argument("--session_id", default="sess_4_3_29", help="Session ID, 默认: sess_4_3_29")
    parser.add_argument("--model", default="qwen3:14b", help="Model name, 默认: qwen3:14b")

    # 新增参数
    parser.add_argument("--channel_name", default=None, help="Channel name (格式: employee_<team_id>_<user_id>_<employee_id>)")
    parser.add_argument("--team_id", default="4", help="Team ID")
    parser.add_argument("--extra-body", default=None, help="Extra body parameters as JSON string, e.g., '{\"team_id\": \"4\"}'")

    # v2 API 特有参数
    parser.add_argument("--api-version", choices=["v1", "v2"], default="v1", help="API version (v1 or v2), 默认: v1")
    parser.add_argument("--user-name", default=None, help="User name (v2 API only)")
    parser.add_argument("--head-url", default=None, help="User avatar URL (v2 API only)")

    parser.add_argument("--query", required=True, help="User query text（必填）")
    parser.add_argument("--timeout", type=int, default=60, help="Stream timeout seconds, 默认: 60")
    args = parser.parse_args()

    api_key = os.environ.get("API_KEY")

    # 解析 extra_body JSON
    extra_body = None
    if args.extra_body:
        try:
            extra_body = json.loads(args.extra_body)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in --extra-body: {e}", file=sys.stderr)
            sys.exit(1)

    body = build_body(
        args.model, args.query, args.employee_id, args.user_id, args.session_id,
        channel_name=args.channel_name,
        team_id=args.team_id,
        user_name=args.user_name,
        head_url=args.head_url,
        api_version=args.api_version,
        extra_body=extra_body
    )
    print(f'body: {body}')
    rc = run_stream(args.host, body, api_key, timeout=args.timeout, api_version=args.api_version)

    sys.exit(rc)

if __name__ == "__main__":
    main()
