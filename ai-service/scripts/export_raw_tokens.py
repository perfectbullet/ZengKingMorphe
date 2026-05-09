#!/usr/bin/env python3
"""
导出最近 N 条 chat_id 的 raw_stream_tokens 数据，按 chat_id 分组。

使用示例：
    cd ai-service
    conda activate morphe

    # 导出最近 10 条 chat_id（默认）
    PYTHONPATH=. python scripts/export_raw_tokens.py

    # 导出最近 20 条 chat_id
    PYTHONPATH=. python scripts/export_raw_tokens.py -n 20

    # 按 chat_id 筛选特定记录
    PYTHONPATH=. python scripts/export_raw_tokens.py --chat-id chatcmpl-8ae2b0eaffc4

    # 指定输出文件路径
    PYTHONPATH=. python scripts/export_raw_tokens.py -n 5 -o result.json
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from motor.motor_asyncio import AsyncIOMotorClient

# 加载环境变量
_env_local = Path(__file__).parent.parent / ".env-local"
_env_default = Path(__file__).parent.parent / ".env"
if _env_local.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_local, verbose=False)
elif _env_default.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_default, verbose=False)

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGO_DATABASE = os.getenv("MONGODB_DB_NAME", "digital_employee")
RAW_TOKENS_COLLECTION = "raw_stream_tokens"
STREAM_CHUNKS_COLLECTION = "stream_chunks"
OUTPUT_DIR = Path(__file__).parent / "exports"


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


async def main():
    parser = argparse.ArgumentParser(description="导出最近 N 条 chat_id 的 raw_stream_tokens 数据")
    parser.add_argument("-n", "--top-n", type=int, default=10, help="最近 N 条 chat_id（默认: 10）")
    parser.add_argument("--chat-id", help="只导出指定 chat_id 的记录")
    parser.add_argument("-o", "--output", help="输出文件路径")
    args = parser.parse_args()

    print(f"连接 MongoDB: {MONGO_DATABASE} ...")

    client = AsyncIOMotorClient(MONGO_URI)
    try:
        await client.admin.command("ping")
        db = client[MONGO_DATABASE]
        raw_tokens = db[RAW_TOKENS_COLLECTION]
        stream_chunks = db[STREAM_CHUNKS_COLLECTION]

        # 确定 chat_id 列表
        if args.chat_id:
            chat_ids = [args.chat_id]
            print(f"指定 chat_id: {args.chat_id}")
        else:
            # 聚合：按 chat_id 分组取最大 created_at，倒序取前 N
            pipeline = [
                {"$sort": {"created_at": -1}},
                {"$group": {
                    "_id": "$chat_id",
                    "latest_at": {"$first": "$created_at"},
                    "token_count": {"$sum": 1},
                }},
                {"$sort": {"latest_at": -1}},
                {"$limit": args.top_n},
            ]
            chat_groups = await raw_tokens.aggregate(pipeline).to_list(length=None)
            chat_ids = [g["_id"] for g in chat_groups]
            print(f"最近 {len(chat_ids)} 条 chat_id:")
            for g in chat_groups:
                print(f"  {g['_id']}  tokens={g['token_count']}  latest={g['latest_at']}")

        if not chat_ids:
            print("未找到任何记录")
            return

        # 查询 raw tokens，按 chat_id 分组
        cursor = raw_tokens.find(
            {"chat_id": {"$in": chat_ids}},
        ).sort("created_at", 1)

        grouped = {}
        async for doc in cursor:
            cid = doc["chat_id"]
            if cid not in grouped:
                grouped[cid] = {
                    "chat_id": cid,
                    "session_id": doc.get("session_id", ""),
                    "employee_id": doc.get("employee_id", ""),
                    "user_id": doc.get("user_id", ""),
                    "conversation_id": doc.get("conversation_id", ""),
                    "user_query": "",
                    "tokens": [],
                }
            grouped[cid]["tokens"].append({
                "token_index": doc.get("token_index", 0),
                "token_text": doc.get("token_text", ""),
                "streaming_source": doc.get("streaming_source", ""),
                "created_at": doc.get("created_at").isoformat() if isinstance(doc.get("created_at"), datetime) else str(doc.get("created_at", "")),
            })

        # 按 user_query 查询 stream_chunks（chunk_type=user_query）
        for cid, group in grouped.items():
            user_query_doc = await stream_chunks.find_one(
                {"chat_id": cid, "chunk_type": "user_query"},
            )
            if user_query_doc:
                chunk_data = user_query_doc.get("chunk_data", {})
                # chunk_data.user_message 或从 choices 中提取
                query_text = chunk_data.get("user_message", "")
                if not query_text:
                    messages = chunk_data.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        query_text = last_msg.get("content", "")
                group["user_query"] = query_text

        # 按最新 created_at 倒序排列分组
        result_list = sorted(
            grouped.values(),
            key=lambda g: g["tokens"][-1]["created_at"] if g["tokens"] else "",
            reverse=True,
        )

        # 为每个分组拼接完整文本
        for group in result_list:
            group["full_text"] = "".join(t["token_text"] for t in group["tokens"])

        # 保存 JSON
        OUTPUT_DIR.mkdir(exist_ok=True)
        output_file = (
            Path(args.output)
            if args.output
            else OUTPUT_DIR / f"raw_tokens_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result_list, f, ensure_ascii=False, indent=2)

        print(f"\n已导出 {len(result_list)} 个 chat_id 到: {output_file}")
        for group in result_list:
            preview = group["full_text"][:80].replace("\n", " ")
            print(f"  [{group['chat_id']}] user_query={group['user_query'][:60]!r}  tokens={len(group['tokens'])}  text={preview!r}...")

    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
