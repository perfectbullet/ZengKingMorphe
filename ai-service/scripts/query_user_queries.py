#!/usr/bin/env python3
"""
查询 MongoDB conversations 集合中所有 user_query，保存为 txt 文件

使用示例：
    cd ai-service
    conda activate morphe
    python scripts/query_user_queries.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv(Path(__file__).parent.parent / ".env")

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGO_DATABASE = os.getenv("MONGODB_DB_NAME", "rag_db")
COLLECTION_NAME = "conversations"
OUTPUT_FILE = Path(__file__).parent / "user_queries.txt"


async def main():
    print(f"连接 MongoDB: {MONGO_URI} {MONGO_DATABASE} ...")

    client = AsyncIOMotorClient(MONGO_URI)
    try:
        await client.admin.command("ping")
        db = client[MONGO_DATABASE]

        # 使用 distinct 直接获取所有唯一的 user_query
        print(f"正在查询 {COLLECTION_NAME} 集合的 user_query 字段...")
        queries = await db[COLLECTION_NAME].distinct("user_query")

        # 过滤空查询
        queries = [q for q in queries if q and q.strip()]

        print(f"共查询到 {len(queries)} 个唯一的 user_query")

        # 保存到文件
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for query in queries:
                f.write(query.strip() + "\n")

        print(f"结果已保存到: {OUTPUT_FILE}")

        # 打印前 5 条预览
        print("\n前 5 条预览:")
        for query in queries[:5]:
            print(f"  - {query[:80]}")
        if len(queries) > 5:
            print(f"  ... 还有 {len(queries) - 5} 条")

    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
