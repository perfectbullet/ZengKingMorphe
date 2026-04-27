#!/usr/bin/env python3
"""
查询 MongoDB entity_chunks 集合中所有 _id，过滤出高中数学相关实体名称

使用示例：
    cd ai-service
    conda activate morphe
    python scripts/query_entity_names.py
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "rag_db")
COLLECTION_NAME = "entity_chunks"
OUTPUT_FILE = Path(__file__).parent / "entity_names_result.json"

LLM_BASE_URL = os.getenv("RAG_Anything_OPENAI_API_BASE")
LLM_API_KEY = os.getenv("RAG_Anything_OPENAI_API_KEY")
LLM_MODEL = os.getenv("RAG_Anything_OPENAI_MODEL")
BATCH_SIZE = 50


FILTER_PROMPT = """\
你是一个高中数学知识分类专家。判断以下实体名称是否属于高中数学范畴（包括代数、几何、三角函数、概率统计、微积分基础、数列、向量、圆锥曲线等）。
只返回属于高中数学领域的名词，比如：集合、数列、三角函数。
不要返回公式（如： (1/2)^x ）、图表
以 JSON 数组格式返回，不要返回任何其他内容。

实体名称列表：
{names}"""


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def filter_math_entities(names: list[str]) -> list[str]:
    """用 LLM 分批判断每个名称是否属于高中数学，每批保存一次"""
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    # 断点续传：如果已有结果文件，加载之前的结果
    if OUTPUT_FILE.exists():
        result = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        done = len(result)
        print(f"  发现已有结果文件，已处理约 {done} 个名称，从中断处继续")
    else:
        result = []
        done = 0

    # 计算已处理了多少个批次，跳过已完成的
    start_batch = done // BATCH_SIZE
    total_batches = (len(names) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  共 {total_batches} 批，从第 {start_batch + 1} 批开始")

    for batch_idx in range(start_batch, total_batches):
        i = batch_idx * BATCH_SIZE
        batch = names[i : i + BATCH_SIZE]

        print(f"  批次 {batch_idx + 1}/{total_batches} ({len(batch)} 个)...", end=" ", flush=True)

        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": FILTER_PROMPT.format(names="\n".join(batch))}],
                temperature=0.0,
            )

            content = resp.choices[0].message.content.strip()
            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                kept = json.loads(json_match.group())
                result.extend(kept)
                print(f"保留 {len(kept)} 个，累计 {len(result)} 个")
            else:
                print(f"解析失败，跳过")
                print(f"    原始响应: {content[:200]}")
        except Exception as e:
            print(f"请求失败: {e}")

        # 每批都保存一次
        save_json(result, OUTPUT_FILE)

    return result


async def main():
    print(f"连接 MongoDB: {MONGO_URI} {MONGO_DATABASE} ...")

    client = AsyncIOMotorClient(MONGO_URI)
    try:
        await client.admin.command("ping")
        db = client[MONGO_DATABASE]

        # 查询所有 _id
        entity_names = await db[COLLECTION_NAME].distinct("_id")
        print(f"共查询到 {len(entity_names)} 个 entity_names (过滤前)")

        # 正则过滤
        new_entity_names = []
        for entity in entity_names:
            if '(' in entity:
                continue
            if '$' in entity:
                continue
            if 'jpg' in entity:
                continue
            if '<' in entity:
                continue
            if '>' in entity:
                continue
            if '+' in entity:
                continue
            if '^' in entity:
                continue
            if '10⁻' in entity:
                continue
            if '/' in entity:
                continue
            if '=' in entity:
                continue
            if len(entity) < 2:
                continue
            if len(entity) > 8:
                continue
            if '0' in entity:
                continue
            if '°' in entity:
                continue
            if '-' in entity:
                continue
            if '-' in entity:
                continue
            if entity[0] == '图':
                continue
            if re.match(r'^\d', entity):
                continue
            if re.findall(r'[a-zA-Z]', entity) :
                continue
            new_entity_names.append(entity)
        
        print(f"正则过滤后剩余 {len(new_entity_names)} 个")
        save_json(new_entity_names, Path('正则过滤后剩余.json'))

        # LLM 过滤
        print(f"\n开始 LLM 过滤 (模型: {LLM_MODEL})...")
        new_entity_names = filter_math_entities(new_entity_names)
        print(f"LLM 过滤后剩余 {len(new_entity_names)} 个高中数学实体")

        # 最终保存（filter_math_entities 已在每批保存，这里是最终确认）
        save_json(new_entity_names, OUTPUT_FILE)
        print(f"\n结果已保存到: {OUTPUT_FILE}")

        # 打印前 10 条预览
        print("\n前 10 条预览:")
        for name in new_entity_names[:10]:
            print(f"  - {name}")
        if len(new_entity_names) > 10:
            print(f"  ... 还有 {len(new_entity_names) - 10} 条")

    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
