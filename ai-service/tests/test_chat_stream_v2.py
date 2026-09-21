"""
测试 OpenAI v2 API 接口 /api/chat/v2/chat/completions

基于日志参数测试流式输出功能：
- model=qwen3:32b
- user_id=3
- user_name=用户名称
- head_url=用户头像
- employee_id=44
- session_id=sess_3b63170fb038
- stream=True
- temperature=0.7
- top_p=0.9
- last_user_message=你好，我是小派

使用方法:
    python -m tests.test_chat_stream_v2
    python -m tests.test_chat_stream_v2 --query "你的问题"
    python -m tests.test_chat_stream_v2 --employee-id 29 --user-id 3
    python -m tests.test_chat_stream_v2 --host http://192.168.100.233:8100
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator, Dict, Any

import requests

# 默认测试参数（基于日志）
DEFAULT_PARAMS = {
    "model": "qwen3:32b",
    "user_id": "3",
    "user_name": "用户名称",
    "head_url": "用户头像",
    "employee_id": "44",
    "session_id": "sess_3b63170fb038",
    "team_id": None,
    "channel_name": None,
    "stream": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": None,
    "n": 1,
}

DEFAULT_QUERY = "你好，我是小派"
DEFAULT_HOST = "http://localhost:8100"


class ChatStreamV2Tester:
    """OpenAI v2 API 测试类"""

    def __init__(
        self,
        query: str = DEFAULT_QUERY,
        user_id: str = "3",
        user_name: str = "用户名称",
        head_url: str = "用户头像",
        employee_id: str = "44",
        session_id: Optional[str] = None,
        team_id: Optional[str] = None,
        channel_name: Optional[str] = None,
        model: str = "qwen3:32b",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        host: str = DEFAULT_HOST,
        api_key: Optional[str] = None,
    ):
        self.query = query
        self.user_id = user_id
        self.user_name = user_name
        self.head_url = head_url
        self.employee_id = employee_id
        self.session_id = session_id or f"sess_{user_id}_{employee_id}_{int(time.time())}"
        self.team_id = team_id
        self.channel_name = channel_name
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.host = host.rstrip("/")
        self.api_key = api_key

        self.results = {
            "test_time": datetime.now().isoformat(),
            "query": query,
            "parameters": {
                "model": model,
                "user_id": user_id,
                "user_name": user_name,
                "head_url": head_url,
                "employee_id": employee_id,
                "session_id": self.session_id,
                "team_id": team_id,
                "channel_name": channel_name,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            },
            "stream_chunks": [],
            "full_response": "",
            "metadata": {},
            "error": None,
            "duration_ms": 0,
            "ttfb_ms": 0,
        }

    def _build_request_body(self) -> dict:
        """构建请求体"""
        body = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": self.query}
            ],
            "stream": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "seed": None,
            "n": 1,
            "employee_id": self.employee_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "head_url": self.head_url,
            "session_id": self.session_id,
        }

        # 添加可选参数
        if self.team_id:
            body["team_id"] = self.team_id
        if self.channel_name:
            body["channel_name"] = self.channel_name

        return body

    def _iter_sse_payloads(self, resp: requests.Response) -> Iterator[str]:
        """
        从响应中迭代 SSE payload（剥离 'data:' 前缀）。

        兼容以下两种常见服务器输出：
        - SSE 行: data: {...}
        - 直接 JSON 行: {...}
        """
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

    def test_stream(self) -> dict:
        """测试流式输出"""
        print("=" * 60)
        print(f"开始测试 OpenAI v2 API")
        print(f"Query: {self.query}")
        print(f"User: {self.user_id}, Employee: {self.employee_id}")
        print(f"Model: {self.model}, Temperature: {self.temperature}")
        print(f"URL: {self.host}/api/chat/v2/chat/completions")
        print("=" * 60)

        url = f"{self.host}/api/chat/v2/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        body = self._build_request_body()

        start_time = time.time()

        try:
            with requests.post(url, json=body, headers=headers, stream=True, timeout=(5, 120)) as resp:
                try:
                    resp.raise_for_status()
                except requests.HTTPError:
                    print(f"HTTP error {resp.status_code}: {resp.text}")
                    self.results["error"] = f"HTTP {resp.status_code}: {resp.text}"
                    return self.results

                # 计算 TTFB (Time To First Byte)
                ttfb = time.time() - start_time
                self.results["ttfb_ms"] = int(ttfb * 1000)
                print(f"✅ Connection established (status: {resp.status_code}) - TTFB: {self.results['ttfb_ms']}ms")

                chunk_count = 0
                first_token_time = None
                full_content = ""

                for payload in self._iter_sse_payloads(resp):
                    chunk_count += 1

                    if first_token_time is None and payload and payload != "[DONE]":
                        first_token_time = time.time()
                        first_token_latency_ms = int((first_token_time - start_time) * 1000)
                        print(f"⏱️ First token latency: {first_token_latency_ms}ms")

                    # 处理 [DONE] 信号
                    if payload == "[DONE]":
                        print(f"\n✅ Received [DONE] signal")
                        break

                    # 解析 JSON
                    try:
                        chunk_data = json.loads(payload)
                        self.results["stream_chunks"].append(chunk_data)

                        # 检查错误
                        if isinstance(chunk_data, dict) and chunk_data.get("error"):
                            error_msg = chunk_data.get("error", {})
                            print(f"\n❌ Server error: {json.dumps(error_msg, ensure_ascii=False, indent=2)}")
                            self.results["error"] = error_msg
                            return self.results

                        # 提取内容
                        if chunk_data.get("object") == "chat.completion.chunk":
                            choices = chunk_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")

                                # 处理 model 为 "status" 的特殊 chunk（状态消息）
                                model = chunk_data.get("model", "")
                                if model == "status":
                                    print(f"\n📡 Status: {content}")
                                elif content:
                                    full_content += content
                                    print(content, end="", flush=True)

                                # 检查 finish_reason
                                finish_reason = choices[0].get("finish_reason")
                                if finish_reason:
                                    print(f"\n\n✅ Finish reason: {finish_reason}")

                                    # 提取 metadata
                                    metadata = chunk_data.get("metadata", {})
                                    if metadata:
                                        self.results["metadata"] = metadata

                                        # 打印 sources
                                        sources = metadata.get("sources", {})
                                        rag_sources = sources.get("rag_sources", [])
                                        web_sources = sources.get("web_sources", [])

                                        if rag_sources:
                                            print(f"\n📚 RAG Sources ({len(rag_sources)}):")
                                            for src in rag_sources:
                                                print(f"  - [{src.get('rank')}] {src.get('content_snippet', 'N/A')[:60]}...")

                                        if web_sources:
                                            print(f"\n🌐 Web Sources ({len(web_sources)}):")
                                            for src in web_sources:
                                                print(f"  - [{src.get('rank')}] {src.get('title', 'N/A')}")

                                        # 打印其他 metadata
                                        print(f"\n📊 Metadata:")
                                        print(f"  - conversation_id: {metadata.get('conversation_id', 'N/A')}")
                                        print(f"  - confidence: {metadata.get('confidence', 0)}")
                                        print(f"  - web_search_used: {metadata.get('web_search_used', False)}")
                                        print(f"  - kb_used: {metadata.get('kb_used', [])}")
                                        print(f"  - model: {metadata.get('model', 'N/A')}")

                    except json.JSONDecodeError as e:
                        print(f"\n⚠️ JSON decode error: {e}, payload={payload[:100]}")

                end_time = time.time()
                duration_ms = int((end_time - start_time) * 1000)

                self.results["full_response"] = full_content
                self.results["duration_ms"] = duration_ms
                self.results["chunk_count"] = chunk_count

                print("\n" + "=" * 60)
                print(f"测试完成")
                print(f"总耗时: {duration_ms}ms")
                print(f"Chunk 数量: {chunk_count}")
                print(f"响应长度: {len(full_content)} 字符")
                print("=" * 60)

        except requests.RequestException as e:
            print(f"\n❌ Request error: {e}")
            self.results["error"] = str(e)

        return self.results

    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "=" * 60)
        print("测试摘要")
        print("=" * 60)
        print(f"Query: {self.query}")
        print(f"User ID: {self.user_id}, Employee ID: {self.employee_id}")
        print(f"Session ID: {self.session_id}")
        print(f"Model: {self.model}, Temperature: {self.temperature}")
        print(f"总耗时: {self.results['duration_ms']}ms")
        print(f"TTFB: {self.results['ttfb_ms']}ms")
        print(f"Chunk 数量: {self.results.get('chunk_count', 0)}")
        print(f"响应长度: {len(self.results.get('full_response', ''))} 字符")

        if self.results.get("error"):
            print(f"错误: {self.results['error']}")

        # 打印 metadata 摘要
        metadata = self.results.get("metadata", {})
        if metadata:
            print("\nMetadata:")
            print(f"  - conversation_id: {metadata.get('conversation_id', 'N/A')}")
            print(f"  - confidence: {metadata.get('confidence', 0)}")
            print(f"  - web_search_used: {metadata.get('web_search_used', False)}")
            print(f"  - kb_used: {metadata.get('kb_used', [])}")

            sources = metadata.get('sources', {})
            if sources.get('rag_sources'):
                print(f"\nRAG Sources ({len(sources['rag_sources'])}):")
                for src in sources['rag_sources']:
                    print(f"  - rank={src.get('rank')}, score={src.get('score')}, doc_id={src.get('doc_id', 'N/A')}")

            if sources.get('web_sources'):
                print(f"\nWeb Sources ({len(sources['web_sources'])}):")
                for src in sources['web_sources']:
                    print(f"  - rank={src.get('rank')}, title={src.get('title', 'N/A')}")

        print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试 OpenAI v2 API 接口")
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=DEFAULT_QUERY,
        help="测试查询内容"
    )
    parser.add_argument(
        "--user-id", "-u",
        type=str,
        default="3",
        help="用户 ID"
    )
    parser.add_argument(
        "--user-name",
        type=str,
        default="用户名称",
        help="用户名称"
    )
    parser.add_argument(
        "--head-url",
        type=str,
        default="用户头像",
        help="用户头像"
    )
    parser.add_argument(
        "--employee-id", "-e",
        type=str,
        default="44",
        help="数字员工 ID"
    )
    parser.add_argument(
        "--session-id", "-s",
        type=str,
        default=None,
        help="会话 ID（默认自动生成）"
    )
    parser.add_argument(
        "--team-id", "-t",
        type=str,
        default=None,
        help="团队 ID"
    )
    parser.add_argument(
        "--channel-name",
        type=str,
        default=None,
        help="渠道名称"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen3:32b",
        help="模型名称"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度 (0.0-2.0)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="核采样参数 (0.0-1.0)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="最大生成 token 数"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"API 主机地址 (默认: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API 密钥"
    )

    args = parser.parse_args()

    tester = ChatStreamV2Tester(
        query=args.query,
        user_id=args.user_id,
        user_name=args.user_name,
        head_url=args.head_url,
        employee_id=args.employee_id,
        session_id=args.session_id,
        team_id=args.team_id,
        channel_name=args.channel_name,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        host=args.host,
        api_key=args.api_key,
    )

    tester.test_stream()
    tester.print_summary()


if __name__ == "__main__":
    main()
