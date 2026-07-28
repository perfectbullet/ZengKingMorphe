#!/usr/bin/env python3
"""
工训 RAG 流式链路自测脚本。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=False)

from app.services.rag_stream_wrapper import get_rag_stream  # noqa: E402


QUESTIONS = [
    "什么是二项式定理？",
    "等差数列的前 n 项和公式是什么？",
    "透明釉料和不透明釉料有什么区别？",
    "如何理解函数的定义域和值域？",
    "排列与组合有什么区别？",
    "导数的几何意义是什么？",
]


async def run_one(question: str, index: int) -> bool:
    print(f"\n[{index}] {question}")
    answer_parts: list[str] = []
    sources_info = None
    sources_counts = {"entities": 0, "relationships": 0, "chunks": 0}
    has_error = False

    async for item in get_rag_stream(
        query=question,
        mode=os.getenv("RAG_QUERY_MODE", "hybrid"),
        prefer_zh_output=True,
        debug_meta={"script": "test_rag_stream.py", "question_index": index},
    ):
        item_type = item.get("type")
        content = item.get("content")
        if item_type == "sources_info":
            sources_info = content
        elif item_type == "chunk":
            answer_parts.append(str(content or ""))
        elif item_type == "sources":
            content = content or {}
            sources_counts = {
                "entities": len(content.get("entities") or []),
                "relationships": len(content.get("relationships") or []),
                "chunks": len(content.get("chunks") or []),
            }
        elif item_type == "error":
            has_error = True
            print(f"error: {content}")

    answer = "".join(answer_parts).strip()
    print("sources_info:", sources_info)
    print("answer_preview:", answer[:500])
    print("sources_counts:", sources_counts)
    print("has_error:", has_error)
    return bool(answer) and not has_error


async def amain() -> int:
    print(
        "LIGHTRAG_WORKING_DIR =",
        os.getenv(
            "LIGHTRAG_WORKING_DIR",
            "/home/zj/ZengKingMorphe-math/ai-service/data/lightrag_manual_math_concepts",
        ),
    )
    print("RAG_QUERY_MODE =", os.getenv("RAG_QUERY_MODE", "hybrid"))
    print("RAG_STREAM_DEBUG_ENABLED =", os.getenv("RAG_STREAM_DEBUG_ENABLED", "true"))
    print(
        "RAG_STREAM_DEBUG_DIR =",
        os.getenv(
            "RAG_STREAM_DEBUG_DIR",
            str(ROOT / "logs" / "rag_stream_debug"),
        ),
    )

    results = []
    for idx, question in enumerate(QUESTIONS, 1):
        ok = await run_one(question, idx)
        results.append(ok)

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
