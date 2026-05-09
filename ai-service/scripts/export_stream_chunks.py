#!/usr/bin/env python3
"""
导出 MongoDB stream_chunks 集合中的 content 内容为 JSON 文件

使用示例：
    cd ai-service
    conda activate morphe

    # 导出所有 token 类型的 content
    python scripts/export_stream_chunks.py

    # 按 chat_id 筛选
    python scripts/export_stream_chunks.py --chat-id chatcmpl-xxxxx

    # 按 session_id 筛选
    python scripts/export_stream_chunks.py --session-id sess_xxxxx

    # 导出完整 chunk_data（不仅仅是 content）
    python scripts/export_stream_chunks.py --full

    # 限制条数
    python scripts/export_stream_chunks.py --limit 1000
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv(Path(__file__).parent.parent / ".env-local", verbose=False)
load_dotenv(Path(__file__).parent.parent / ".env", verbose=False)

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGO_DATABASE = os.getenv("MONGODB_DB_NAME", "rag_db")
COLLECTION_NAME = "stream_chunks"
OUTPUT_DIR = Path(__file__).parent / "exports"


def extract_content(chunk_data: dict) -> str:
    """从 chunk_data 中提取 content 文本（OpenAI 格式）"""
    if not chunk_data:
        return ""
    try:
        choices = chunk_data.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            return delta.get("content", "")
    except (IndexError, AttributeError):
        pass
    return ""


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


async def main():
    parser = argparse.ArgumentParser(description="导出 stream_chunks 数据")
    parser.add_argument("--chat-id", help="按 chat_id 筛选")
    parser.add_argument("--session-id", help="按 session_id 筛选")
    parser.add_argument("--employee-id", help="按 employee_id 筛选")
    parser.add_argument("--chunk-type", default="token", help="按 chunk_type 筛选 (默认: token)")
    parser.add_argument("--full", action="store_true", help="导出完整 chunk_data 而非仅 content")
    parser.add_argument("--limit", type=int, help="限制导出条数")
    parser.add_argument("-o", "--output", help="输出文件路径")
    args = parser.parse_args()

    print(f"连接 MongoDB: {MONGO_URI} {MONGO_DATABASE} ...")

    client = AsyncIOMotorClient(MONGO_URI)
    try:
        await client.admin.command("ping")
        db = client[MONGO_DATABASE]
        collection = db[COLLECTION_NAME]

        # 构建查询条件
        query = {}
        if args.chat_id:
            query["chat_id"] = args.chat_id
        if args.session_id:
            query["session_id"] = args.session_id
        if args.employee_id:
            query["employee_id"] = args.employee_id
        if args.chunk_type:
            query["chunk_type"] = args.chunk_type

        total = await collection.count_documents(query)
        print(f"匹配文档数: {total}")

        if total == 0:
            # 列出可用的 chat_id 供参考
            sample_chat_ids = await collection.distinct("chat_id")
            print(f"\n当前集合中有 {len(sample_chat_ids)} 个 chat_id，示例:")
            for cid in sample_chat_ids[:5]:
                cnt = await collection.count_documents({"chat_id": cid})
                print(f"  {cid} ({cnt} chunks)")
            return

        if args.limit:
            total = min(total, args.limit)
            print(f"限制导出 {total} 条")

        # 查询并处理
        cursor = collection.find(query).sort("created_at", 1)
        if args.limit:
            cursor = cursor.limit(args.limit)

        results = []
        current_chat_id = None
        full_text_parts = []

        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            # 序列化 datetime
            for key in ("timestamp", "created_at"):
                if key in doc and isinstance(doc[key], datetime):
                    doc[key] = doc[key].isoformat()

            if args.full:
                results.append(doc)
            else:
                content = extract_content(doc.get("chunk_data", {}))
                results.append({
                    "chat_id": doc.get("chat_id"),
                    "sequence": doc.get("sequence"),
                    "chunk_type": doc.get("chunk_type"),
                    "content": content,
                    "timestamp": doc.get("timestamp"),
                })
                if content:
                    full_text_parts.append(content)

            # 按 chat_id 追踪
            if doc.get("chat_id") != current_chat_id:
                current_chat_id = doc.get("chat_id")

        # 保存 JSON
        OUTPUT_DIR.mkdir(exist_ok=True)
        output_file = Path(args.output) if args.output else OUTPUT_DIR / f"stream_chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)

        print(f"\n已导出 {len(results)} 条记录到: {output_file}")

        # 拼接完整文本预览
        if full_text_parts:
            full_text = "".join(full_text_parts)
            text_file = output_file.with_suffix(".txt")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"拼接完整文本已保存到: {text_file}")
            print(f"\n完整文本预览 (前 300 字):\n{full_text[:300]}")

    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
