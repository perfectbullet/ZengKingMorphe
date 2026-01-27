#!/usr/bin/env python3
"""
Ollama Reranker 测试脚本

用于验证 Ollama BGE Reranker 模型是否正常工作。

示例:
    # 基础测试
    python scripts/test_ollama_reranker.py

    # 自定义查询和文档
    python scripts/test_ollama_reranker.py --query "什么是珐琅" \\
        --docs "珐琅是一种古老的工艺" "珐琅工艺历史悠久"

    # 指定模型和 URL
    python scripts/test_ollama_reranker.py \\
        --base-url http://192.168.8.233:11434 \\
        --model qllama/bge-reranker-v2-m3:latest
"""
import argparse
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.logging import get_logger
from app.services.reranker_service import OllamaReranker

logger = get_logger(__name__)


# 默认测试数据
DEFAULT_QUERY = "什么是珐琅工艺"

DEFAULT_DOCS = [
    "珐琅是一种将玻璃质釉料附着在金属表面的装饰工艺，具有色彩鲜艳、历久弥新的特点。",
    "珠宝首饰设计需要考虑美学、工艺和佩戴舒适度等多个因素。",
    "中国古代珐琅工艺最早出现在元代，明清时期达到顶峰，代表作品有景泰蓝。",
    "现代珐琅工艺在传承传统技法的基础上，融入了现代设计理念和新技术。",
    "不锈钢是一种常用的金属材料，具有良好的耐腐蚀性和强度。",
]


async def test_reranker(
    query: str,
    documents: list,
    base_url: str,
    model: str,
    top_k: int,
):
    """测试 Ollama Reranker"""
    print("\n" + "=" * 80)
    print("🧪 Ollama Reranker 测试")
    print("=" * 80)
    print(f"查询: {query}")
    print(f"文档数: {len(documents)}")
    print(f"返回 Top-K: {top_k}")
    print(f"模型: {model}")
    print(f"API URL: {base_url}\n")

    # 创建 reranker
    reranker = OllamaReranker(base_url=base_url, model=model)

    # 执行 reranking
    print("🔄 正在计算相关性分数...")
    results = await reranker.rerank(query, documents, top_k=top_k)

    # 显示结果
    print("\n" + "-" * 80)
    print("📊 Reranking 结果")
    print("-" * 80)

    for i, (idx, score) in enumerate(results):
        print(f"\n[{i+1}] 分数: {score:.4f}")
        print(f"    原始索引: {idx}")
        print(f"    内容: {documents[idx][:100]}...")

    print("\n" + "-" * 80)
    print(f"✅ 测试完成! 共返回 {len(results)} 个结果")
    print("=" * 80)


async def test_single_pair(query: str, document: str, base_url: str, model: str):
    """测试单个 query-doc 对的分数计算"""
    print("\n" + "=" * 80)
    print("🧪 单对相关性测试")
    print("=" * 80)

    reranker = OllamaReranker(base_url=base_url, model=model)

    print(f"\n查询: {query}")
    print(f"文档: {document[:100]}...")
    print("\n🔄 正在计算相关性分数...")

    score = await reranker._compute_score(query, document)

    print(f"\n相关性分数: {score:.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Ollama Reranker 测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="查询文本"
    )
    parser.add_argument(
        "--docs",
        nargs="+",
        default=DEFAULT_DOCS,
        help="文档列表（用空格分隔）"
    )
    parser.add_argument(
        "--base-url",
        default="http://192.168.8.233:11434",
        help="Ollama API 地址"
    )
    parser.add_argument(
        "--model",
        default="qllama/bge-reranker-v2-m3:latest",
        help="Reranker 模型名称"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="返回 Top-K 结果"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="单对测试模式（只测试第一个文档）"
    )

    args = parser.parse_args()

    if args.single:
        asyncio.run(test_single_pair(
            query=args.query,
            document=args.docs[0],
            base_url=args.base_url,
            model=args.model,
        ))
    else:
        asyncio.run(test_reranker(
            query=args.query,
            documents=args.docs,
            base_url=args.base_url,
            model=args.model,
            top_k=args.top_k,
        ))


if __name__ == "__main__":
    main()
