"""Send one v1 streaming chat request and watch its display WebSocket chunks.

Run from ``ai-service/``::

    python -m tests.test_chat_stream_v1_http_ws \
        -q "求解不等式x的平方减去5x加上6小于0的解" \
        --host http://localhost:8100 --require-reasoning

The service must already be running. A unique session_id is generated unless
``--session-id`` is supplied, and the same IDs are sent to HTTP and WebSocket.
"""

import argparse
import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from urllib.parse import urlencode, urlsplit

import httpx
import websockets


@dataclass
class StreamResult:
    sse_chat_id: str | None = None
    ws_chat_id: str | None = None
    sse_done: bool = False
    ws_done: bool = False
    sse_content_chunks: int = 0
    ws_content_chunks: int = 0
    ws_reasoning_chunks: int = 0


def build_urls(host: str, user_id: str, employee_id: str, session_id: str) -> tuple[str, str]:
    parsed = urlsplit(host)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
        raise ValueError("--host 必须是 http(s)://主机[:端口]，不要包含 API 路径")

    origin = f"{parsed.scheme}://{parsed.netloc}"
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    query = urlencode({
        "user_id": user_id,
        "employee_id": employee_id,
        "session_id": session_id,
    })
    return (
        f"{origin}/api/chat/v1/chat/completions",
        f"{ws_scheme}://{parsed.netloc}/api/chat/ws/view/chunks?{query}",
    )


async def watch_websocket(websocket, result: StreamResult) -> None:
    while True:
        message = json.loads(await websocket.recv())
        if message.get("type") == "heartbeat":
            print("[WS ] heartbeat", flush=True)
            continue

        data = message.get("chunk_data") or {}
        chat_id = data.get("id") or message.get("chat_id")
        if result.sse_chat_id and chat_id and chat_id != result.sse_chat_id:
            continue
        if chat_id and result.ws_chat_id is None:
            result.ws_chat_id = chat_id

        chunk_type = message.get("chunk_type", "unknown")
        sequence = message.get("sequence", "?")
        choices = data.get("choices") or []
        delta = choices[0].get("delta") or {} if choices else {}

        if chunk_type == "reasoning":
            result.ws_reasoning_chunks += 1
            print(f"[WS  #{sequence}] reasoning: {delta.get('reasoning', '')!r}", flush=True)
        elif chunk_type == "token":
            result.ws_content_chunks += 1
            print(f"[WS  #{sequence}] content:   {delta.get('content', '')!r}", flush=True)
        elif chunk_type == "user_query":
            print(f"[WS  #{sequence}] user_query: {data.get('user_message', '')!r}", flush=True)
        else:
            print(f"[WS  #{sequence}] {chunk_type}", flush=True)

        if chunk_type == "error":
            raise RuntimeError(f"WS 返回错误 chunk: {data.get('error', data)}")
        if chunk_type == "done":
            result.ws_done = True
            return


async def stream_http(url: str, body: dict, api_key: str | None, timeout: float, result: StreamResult) -> None:
    headers = {"Accept": "text/event-stream"}
    if api_key:
        headers["X-API-Key"] = api_key

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                payload = line.removeprefix("data:").strip() if line.startswith("data:") else line
                if payload == "[DONE]":
                    result.sse_done = True
                    print("[SSE] [DONE]", flush=True)
                    break
                if not payload.startswith("{"):
                    continue

                data = json.loads(payload)
                if data.get("error"):
                    raise RuntimeError(f"HTTP SSE 返回错误: {data['error']}")
                chat_id = data.get("id")
                if chat_id and result.sse_chat_id is None:
                    result.sse_chat_id = chat_id

                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    result.sse_content_chunks += 1
                    print(f"[SSE] content:   {content!r}", flush=True)
                if choices[0].get("finish_reason"):
                    print(f"[SSE] finish:    {choices[0]['finish_reason']}", flush=True)

    if not result.sse_done:
        raise RuntimeError("HTTP 流结束，但没有收到 [DONE]")


async def run(args: argparse.Namespace) -> None:
    session_id = args.session_id or (
        f"sess_{args.team_id}_{args.user_id}_{args.employee_id}_{uuid.uuid4().hex[:8]}"
    )
    http_url, ws_url = build_urls(args.host, args.user_id, args.employee_id, session_id)
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.query}],
        "stream": True,
        "user_id": args.user_id,
        "employee_id": args.employee_id,
        "team_id": args.team_id,
        "session_id": session_id,
    }
    result = StreamResult()

    print(f"session_id={session_id} user_id={args.user_id} employee_id={args.employee_id}")
    print(f"HTTP: {http_url}")
    print(f"WS:   {ws_url}")

    async with websockets.connect(ws_url, open_timeout=10) as websocket:
        ws_task = asyncio.create_task(watch_websocket(websocket, result))
        try:
            # 服务端首次轮询只记录起始时间；先等它完成，再发送请求。
            await asyncio.sleep(0.35)
            await stream_http(
                http_url,
                body,
                args.api_key or os.getenv("MORPHE_API_KEY"),
                args.timeout,
                result,
            )
            await asyncio.wait_for(ws_task, timeout=args.ws_timeout)
        finally:
            if not ws_task.done():
                ws_task.cancel()
                try:
                    await ws_task
                except asyncio.CancelledError:
                    pass

    if not result.ws_done or result.sse_chat_id != result.ws_chat_id:
        raise RuntimeError(
            f"两路未匹配完成: SSE id={result.sse_chat_id}, WS id={result.ws_chat_id}, "
            f"WS done={result.ws_done}"
        )
    if args.require_reasoning and result.ws_reasoning_chunks == 0:
        raise RuntimeError("WS 没有收到 reasoning chunk")

    print(
        f"完成: chat_id={result.sse_chat_id}, "
        f"SSE content={result.sse_content_chunks}, "
        f"WS content={result.ws_content_chunks}, "
        f"WS reasoning={result.ws_reasoning_chunks}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="同时查看 v1 HTTP SSE 和展示 WebSocket")
    parser.add_argument("-q", "--query", default="求解不等式x的平方减去5x加上6小于0的解")
    parser.add_argument("--host", default="http://localhost:8100")
    parser.add_argument("--user-id", default="3")
    parser.add_argument("--employee-id", default="29")
    parser.add_argument("--team-id", default="4")
    parser.add_argument("--session-id", help="默认自动生成唯一会话 ID")
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--api-key", help="也可使用 MORPHE_API_KEY 环境变量")
    parser.add_argument("--timeout", type=float, default=300, help="HTTP 读取超时，秒")
    parser.add_argument("--ws-timeout", type=float, default=10, help="SSE 结束后等待 WS done 的秒数")
    parser.add_argument("--require-reasoning", action="store_true", help="未收到 WS reasoning 时退出失败")
    args = parser.parse_args()

    try:
        asyncio.run(run(args))
    except (OSError, ValueError, RuntimeError, httpx.HTTPError, TimeoutError, websockets.WebSocketException) as exc:
        parser.exit(1, f"测试失败: {exc}\n")


if __name__ == "__main__":
    main()
