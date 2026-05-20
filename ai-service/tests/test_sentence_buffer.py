"""
Test SentenceBuffer with real streaming data from MongoDB.

Data source: MongoDB (funasr.raw_stream_tokens)
- Each document: chat_id, token_text, token_index, session_id, created_at
- Grouped by chat_id into independent streaming sessions

Output JSONL format (one JSON object per line):
    {"chat_id": "...", "token_count": 100, "char_count": 500,
     "full_text": "...", "sentences": ["...", "..."]}

Usage:
    # Process last 3 chats (default)
    python tests/test_sentence_buffer.py --last 3

    # Process last 10 chats, output to custom file
    python tests/test_sentence_buffer.py --last 10 --output results.jsonl

    # Process specific chat(s) by ID
    python tests/test_sentence_buffer.py --chat-id chatcmpl-2c4842d0c562
    python tests/test_sentence_buffer.py --chat-id chatcmpl-aaa,chatcmpl-bbb
"""

import asyncio
import json
import os
import sys
from datetime import datetime

import motor.motor_asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

from app.utils.sentence_buffer import SentenceBuffer

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://funasr:funasr2026@192.168.8.233:27017/funasr?authSource=admin",
)
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "funasr")
COLLECTION_NAME = "raw_stream_tokens"

TOKEN_DELAY = float(os.getenv("TOKEN_DELAY", "0.01"))  # 100 tokens/s


def _write_preview_html(html_path: str, segments: list[str]):
    """Generate an HTML preview file from segments, based on latex_v2.html template."""
    template_path = os.path.join(os.path.dirname(__file__), "latex_v2.html")
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # 构建 textarea 内容：每段一行
    content = "\n".join(segments).strip()
    # 转义 HTML 特殊字符
    content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 替换 textarea 中的内容
    start_marker = '<textarea id="input">\n'
    end_marker = '</textarea>'
    start_idx = template.index(start_marker) + len(start_marker)
    end_idx = template.index(end_marker)
    html = template[:start_idx] + content + template[end_idx:]

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML preview written to {html_path}", flush=True)


async def _load_chats(client, chat_ids):
    """Load token data for given chat_ids."""
    collection = client[MONGODB_DB_NAME][COLLECTION_NAME]
    result = []
    for cid in chat_ids:
        docs = []
        async for doc in collection.find(
            {"chat_id": cid}, {"token_text": 1, "token_index": 1}
        ).sort("token_index", 1):
            docs.append(doc)
        if docs:
            result.append((cid, [doc["token_text"] for doc in docs]))
    return result


async def load_chats_from_mongo(n):
    """Load last n distinct chats from MongoDB, ordered by created_at desc."""
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
    collection = client[MONGODB_DB_NAME][COLLECTION_NAME]

    pipeline = [
        {"$sort": {"created_at": -1}},
        {"$group": {"_id": "$chat_id", "latest": {"$first": "$created_at"}}},
        {"$sort": {"latest": -1}},
        {"$limit": n},
    ]
    chat_ids = [doc["_id"] async for doc in collection.aggregate(pipeline) if doc["_id"] not in {"chatcmpl-0a7427d46c3c"}]
    
    result = await _load_chats(client, chat_ids)
    # print(result)
    client.close()
    return result


async def load_chats_by_ids(chat_ids):
    """Load specific chats by chat_id list."""
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
    result = await _load_chats(client, chat_ids)
    client.close()
    return result


async def split_with_sentence_buffer(token_texts):
    """Feed tokens into SentenceBuffer with simulated delay, collect segments."""
    buffer = SentenceBuffer(max_chars=200, max_wait_seconds=1, comma_split_threshold=200)
    sentences = []
    for token in token_texts:
        await asyncio.sleep(TOKEN_DELAY)
        segment = buffer.add(token)
        if segment:
            sentences.append(segment)

    remaining = await buffer.flush(is_final=True)
    if remaining:
        sentences.append(remaining.content)
    return sentences


async def main(chat_ids, n, output_path):
    if chat_ids:
        print(f"Loading {len(chat_ids)} chat(s) by ID...", flush=True)
        chats = await load_chats_by_ids(chat_ids)
    else:
        print(f"Loading last {n} chats from MongoDB...", flush=True)
        chats = await load_chats_from_mongo(n)

    if not chats:
        print("No chats found in MongoDB")
        return

    total_tokens = sum(len(t) for _, t in chats)
    print(f"Found {len(chats)} chats ({total_tokens} tokens), processing...\n", flush=True)

    # 收集所有 chat 的断句文本，用于生成 HTML
    all_segments = []

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (cid, token_texts) in enumerate(chats, 1):
            full_text = "".join(token_texts)
            t0 = asyncio.get_event_loop().time()
            sentences = await split_with_sentence_buffer(token_texts)
            elapsed = asyncio.get_event_loop().time() - t0

            record = {
                "chat_id": cid,
                "token_count": len(token_texts),
                "char_count": len(full_text),
                "full_text": full_text,
                "sentences": sentences,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(
                f"[{idx}/{len(chats)}] {cid} | {len(token_texts)}tok {len(full_text)}char | {len(sentences)}sent | {elapsed:.1f}s",
                flush=True,
            )
            for i, s in enumerate(sentences):
                display = s
                if len(display) > 120:
                    display = display[:120] + "..."
                print(display, flush=True)
            print(flush=True)

            all_segments.extend(sentences)
            all_segments.append("")  # chat 之间空行分隔

    # 写入 HTML 渲染文件
    html_path = os.path.join(results_dir, "latex_preview.html")
    _write_preview_html(html_path, all_segments)

    print(f"Results written to {output_path}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SentenceBuffer test with MongoDB streaming data")
    parser.add_argument("--last", type=int, default=3, help="Number of recent chats to process")
    parser.add_argument(
        "--chat-id",
        dest="chat_ids",
        type=lambda s: s.split(","),
        help="Comma-separated chat_id(s) to process",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file path")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "sentence_buffer_results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = args.output or os.path.join(
        results_dir,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
    )

    asyncio.run(main(args.chat_ids, args.last, output_path))
