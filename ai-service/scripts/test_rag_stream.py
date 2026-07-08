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
    "平铺珐琅工艺的基本制作流程是什么？",
    "掐丝珐琅工艺中金属丝的作用是什么？",
    "透明釉料和不透明釉料有什么区别？",
    "画珐琅工艺需要哪些工具和材料？",
    "金箔和银箔在珐琅工艺中如何使用？",
    "现代艺术首饰中珐琅工艺有哪些应用方式？",
]


async def run_one(question: str, index: int) -> bool:
    print(f"\n[{index}] {question}")
    answer_parts: list[str] = []
    sources_info = None
    sources_counts = {"entities": 0, "relationships": 0, "chunks": 0}
    has_error = False

    async for item in get_rag_stream(
        query=question,
        mode=os.getenv("TRAINING_RAG_QUERY_MODE", "hybrid"),
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
    print("TRAINING_RAG_BACKEND =", os.getenv("TRAINING_RAG_BACKEND", "lightrag_file"))
    print(
        "TRAINING_LIGHTRAG_WORKING_DIR =",
        os.getenv(
            "TRAINING_LIGHTRAG_WORKING_DIR",
            "/home/zj/ZengKingMorphe/ai-service/data/lightrag_industrial_training_enamel_debug",
        ),
    )
    print("TRAINING_RAG_QUERY_MODE =", os.getenv("TRAINING_RAG_QUERY_MODE", "hybrid"))
    print("TRAINING_RAG_STREAM_DEBUG_ENABLED =", os.getenv("TRAINING_RAG_STREAM_DEBUG_ENABLED", "true"))
    print(
        "TRAINING_RAG_STREAM_DEBUG_DIR =",
        os.getenv(
            "TRAINING_RAG_STREAM_DEBUG_DIR",
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
