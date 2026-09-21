"""
测试 generate_openai_stream_v1 流式输出。

维护说明：本脚本直接导入服务端函数，不经过 HTTP 端点，并会加载完整运行配置
（包括 ``ai-service/.env`` 中的 ``API_KEY`` 与合法布尔值 ``DEBUG``）。它保留给
工业实训分支的 v1 兼容回归测试；当前数学 / 非数学分流分支禁止用它做联调验收。
请改用 ``python -m tests.test_chat_stream_v2 --host <服务地址>``。

基于日志参数测试流式输出功能：
- model=qwen3:14b
- user_id=3
- employee_id=29
- session_id=sess_4_3_29
- stream=True
- temperature=0.7
- top_p=0.9
- last_user_message=请简单给我介绍集合的概念

使用方法:
    python -m tests.test_chat_stream_v1
    python -m tests.test_chat_stream_v1 --query "你的问题"
    python -m tests.test_chat_stream_v1 --employee-id 29 --user-id 3
"""

import asyncio
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.schemas import OpenAIChatRequest, OpenAIMessage
from app.api.endpoints.chat_stream_v1 import generate_openai_stream_v1
from app.core.logging import get_logger

logger = get_logger(__name__)


# 默认测试参数（基于日志）
DEFAULT_PARAMS = {
    "model": "qwen3:14b",
    "user_id": "3",
    "employee_id": "29",
    "session_id": "sess_4_3_29",
    "team_id": "4",
    "stream": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": None,
    "n": 1,
}

DEFAULT_QUERY = "请简单给我介绍集合的概念"


class ChatStreamV1Tester:
    """generate_openai_stream_v1 测试类"""

    def __init__(
        self,
        query: str = DEFAULT_QUERY,
        user_id: str = "3",
        employee_id: str = "29",
        session_id: Optional[str] = None,
        team_id: Optional[str] = "4",
        model: str = "qwen3:14b",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
    ):
        self.query = query
        self.user_id = user_id
        self.employee_id = employee_id
        self.session_id = session_id or f"sess_{team_id}_{user_id}_{employee_id}"
        self.team_id = team_id
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        self.results = {
            "test_time": datetime.now().isoformat(),
            "query": query,
            "parameters": {
                "model": model,
                "user_id": user_id,
                "employee_id": employee_id,
                "session_id": self.session_id,
                "team_id": team_id,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            },
            "stream_chunks": [],
            "full_response": "",
            "error": None,
            "duration_ms": 0,
        }

    def _build_request(self) -> OpenAIChatRequest:
        """构建 OpenAIChatRequest"""
        return OpenAIChatRequest(
            model=self.model,
            messages=[
                OpenAIMessage(role="user", content=self.query)
            ],
            stream=True,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            seed=None,
            n=1,
            tools=None,
            employee_id=self.employee_id,
            user_id=self.user_id,
            session_id=self.session_id,
            channel_name=None,
            team_id=self.team_id,
        )

    def _save_response_markdown(self, content: str) -> str:
        """保存响应内容为 Markdown 文件"""
        try:
            # 创建保存目录
            save_dir = Path(__file__).parent / "saved_responses"
            save_dir.mkdir(parents=True, exist_ok=True)

            # 生成时间戳文件名（参考 conversation_state 的格式）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = self.session_id
            # 清理查询文本作为文件名
            safe_query = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in self.query[:50])
            filename = f"{timestamp}_{session_id}_{safe_query}_response.md"
            filepath = save_dir / filename

            # 构建 Markdown 内容
            markdown_content = f"""# 对话响应

## 基本信息

- **时间**: {self.results['test_time']}
- **查询**: {self.query}
- **用户ID**: {self.user_id}
- **员工ID**: {self.employee_id}
- **会话ID**: {self.session_id}
- **模型**: {self.model}
- **温度**: {self.temperature}
- **Top-P**: {self.top_p}
- **耗时**: {self.results.get('duration_ms', 0)}ms
- **Chunk数量**: {self.results.get('chunk_count', 0)}

## 响应内容

{content}

"""

            # 保存文件
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            logger.info(f"[保存] 响应内容已保存到: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"[保存失败] 无法保存响应内容: {e}")
            return None

    async def test_stream(self) -> dict:
        """测试流式输出"""
        logger.info("=" * 60)
        logger.info(f"开始测试 generate_openai_stream_v1")
        logger.info(f"Query: {self.query}")
        logger.info(f"User: {self.user_id}, Employee: {self.employee_id}")
        logger.info(f"Model: {self.model}, Temperature: {self.temperature}")
        logger.info("=" * 60)

        request = self._build_request()
        start_time = time.time()

        chunk_count = 0
        first_chunk_time = None
        full_content = ""

        async for chunk_str in generate_openai_stream_v1(request):
            chunk_count += 1
            if first_chunk_time is None:
                first_chunk_time = time.time()
                ttfb_ms = int((first_chunk_time - start_time) * 1000)
                logger.info(f"首字延迟 (TTFB): {ttfb_ms}ms")

            # 解析 chunk
            try:
                if chunk_str == "[DONE]":
                    logger.info(f"收到 [DONE] 信号")
                    break

                chunk_data = json.loads(chunk_str)
                self.results["stream_chunks"].append(chunk_data)

                # 提取内容
                if "choices" in chunk_data and chunk_data["choices"]:
                    delta = chunk_data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_content += content
                        print(content, end="", flush=True)

                # 检查 done chunk 并提取 sources
                if chunk_data.get("object") == "chat.completion.chunk":
                    choices = chunk_data.get("choices", [])
                    if choices and choices[0].get("finish_reason") == "stop":
                        metadata = chunk_data.get("metadata", {})
                        if "sources" in metadata:
                            sources = metadata["sources"]
                            self.results["sources"] = sources
                            logger.info(f"Sources: {len(sources)} 条")
                            for src in sources:
                                logger.info(f"  - from={src.get('from', 'unknown')}, text={src.get('text', 'N/A')}, citations={len(src.get('citations', []))}")
                                if src.get("type") == "chunk":
                                    for idx, citation in enumerate(src.get("citations", [])[:5], 1):
                                        logger.info(
                                            "    citation[%d]: reference_id=%s file_path=%s "
                                            "chunk_id=%s rerank_score=%s text=%s"
                                            % (
                                                idx,
                                                citation.get("reference_id"),
                                                citation.get("file_path"),
                                                citation.get("chunk_id"),
                                                citation.get("rerank_score"),
                                                (citation.get("text") or "")[:80],
                                            )
                                        )

            except json.JSONDecodeError as e:
                logger.warning(f"Chunk JSON 解析失败: {e}, chunk={chunk_str[:100]}")

        end_time = time.time()
        duration_ms = int((end_time - start_time) * 1000)

        self.results["full_response"] = full_content
        self.results["duration_ms"] = duration_ms
        self.results["chunk_count"] = chunk_count

        print()  # 换行
        logger.info("=" * 60)
        logger.info(f"测试完成")
        logger.info(f"总耗时: {duration_ms}ms")
        logger.info(f"Chunk 数量: {chunk_count}")
        logger.info(f"响应长度: {len(full_content)} 字符")
        logger.info("=" * 60)

        # 保存响应为 Markdown
        saved_path = self._save_response_markdown(full_content)
        if saved_path:
            self.results["saved_markdown_path"] = saved_path
            print(f"📄 响应已保存到: {saved_path}")

        # 打印完整响应（截断显示）
        if full_content:
            preview = full_content[:200] + "..." if len(full_content) > 200 else full_content
            logger.info(f"响应预览: {preview}")

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
        print(f"Team ID: {self.team_id}")
        print(f"总耗时: {self.results['duration_ms']}ms")
        print(f"Chunk 数量: {self.results.get('chunk_count', 0)}")
        print(f"响应长度: {len(self.results.get('full_response', ''))} 字符")

        # 显示保存的文件路径
        if self.results.get("saved_markdown_path"):
            print(f"📄 响应已保存到: {self.results['saved_markdown_path']}")

        if self.results.get("error"):
            print(f"错误: {self.results['error']}")

        # 打印 Sources 信息
        if self.results.get("sources"):
            print("\nSources:")
            print("-" * 40)
            for src in self.results["sources"]:
                print(f"Type: {src.get('type', 'text')}")
                print(f"From: {src.get('from', 'unknown')}")
                print(f"Text: {src.get('text', 'N/A')}")
                citations = src.get('citations', [])
                print(f"Citations: {len(citations)} 条")
                if citations:
                    for i, citation in enumerate(citations, 1):
                        print(
                            f"  {i}. reference_id={citation.get('reference_id')} "
                            f"file_path={citation.get('file_path')} "
                            f"chunk_id={citation.get('chunk_id')} "
                            f"rerank_score={citation.get('rerank_score')} "
                            f"text={str(citation.get('text', ''))[:120]}"
                        )
                print("-" * 40)

        print("=" * 60)


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试 generate_openai_stream_v1 流式输出")
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
        "--employee-id", "-e",
        type=str,
        default="29",
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
        default="4",
        help="团队 ID"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen3:14b",
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
        "--save-state",
        action="store_true",
        help="保存 final_state 到 JSON 文件"
    )

    args = parser.parse_args()

    tester = ChatStreamV1Tester(
        query=args.query,
        user_id=args.user_id,
        employee_id=args.employee_id,
        session_id=args.session_id,
        team_id=args.team_id,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    await tester.test_stream()
    tester.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
