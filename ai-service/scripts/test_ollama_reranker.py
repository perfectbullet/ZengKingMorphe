#!/usr/bin/env python3
"""
BGE Reranker API 测试脚本

用于验证 BGE Reranker V2 M3 API 是否正常工作。

示例:
    # 基础测试
    python scripts/test_ollama_reranker.py

    # 自定义查询和文档
    python scripts/test_ollama_reranker.py --query "什么是珐琅" \\
        --docs "珐琅是一种古老的工艺" "珐琅工艺历史悠久"

    # 指定 API URL 和密钥
    python scripts/test_ollama_reranker.py \\
        --base-url http://192.168.8.233:8091 \\
        --api-key sk-aaabbbcccdddeeefffggghhhiiijjjkkk

    # 指定模型
    python scripts/test_ollama_reranker.py \\
        --model bge-reranker-v2-m3
"""
import argparse
import asyncio
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.logging import get_logger
from app.services.reranker_service import BGEAPIReranker

logger = get_logger(__name__)


# 默认测试数据
DEFAULT_QUERY = "什么是珐琅工艺"

DEFAULT_DOCS = [
    "珐琅是一种将玻璃质釉料附着在金属表面的装饰工艺，具有色彩鲜艳、历久弥新的特点。",
    "珠宝首饰设计需要考虑美学、工艺和佩戴舒适度等多个因素。",
    "中国古代珐琅工艺最早出现在元代，明清时期达到顶峰，代表作品有景泰蓝。",
    "现代珐琅工艺在传承传统技法的基础上，融入了现代设计理念和新技术。",
    "不锈钢是一种常用的金属材料，具有良好的耐腐蚀性和强度。",
    "笔者从大学时代起开始接触珐琅工艺，先后有机会向多位珐琅艺术家求教，自己对珐琅工艺的由衷喜爱和一点点粗浅认识，完全要感谢这几位老师的指导，其中包括日本的原典生先生、法国的Jean Francois先生和美国的Don Viehman先生。在学习和创作的过程中，我深深地为珐琅工艺的魅力所折服，也真切地体会到这种工艺的难以把握。我所接触过的优秀的珐琅艺术家，都无一例外地把自己毕生的精力投入珐琅工艺的研究和创作当中。这是因为珐琅工艺没有捷径，想要真正地掌握这门技艺，制作者需要在无数次实验、",
    "笔者从大学时代起开始接触珐琅工艺，先后有机会向多位珐琅艺术家求教，自己对珐琅工艺的由衷喜爱和一点点粗浅认识，完全要感谢这几位老师的指导，其中包括日本的原典生先生、法国的Jean Francois先生和美国的Don Viehman先生。在学习和创作的过程中，我深深地为珐琅工艺的魅力所折服，也真切地体会到这种工艺的难以把握。我所接触过的优秀的珐琅艺术家，都无一例外地把自己毕生的精力投入珐琅工艺的研究和创作当中。这是因为珐琅工艺没有捷径，想要真正地掌握这门技艺，制作者需要在无数次实验、无数次失败中找到自己的方法，只有经过这些失败并从失败中总结经验，才有可能逐渐地理解珐琅工艺，才能用它创作出满意的作品。每一件优秀的珐琅作品，都是技术和艺术的完美结合。对珐琅工艺来说，繁复、精细的工艺和优美、严谨的设计，这二者缺一不可。"
]


async def test_reranker(
    query: str,
    documents: list,
    base_url: str,
    api_key: str,
    model: str,
    top_k: int,
):
    """测试 BGE Reranker API"""
    print("\n" + "=" * 80)
    print("🧪 BGE Reranker API 测试")
    print("=" * 80)
    print(f"查询: {query}")
    print(f"文档数: {len(documents)}")
    print(f"返回 Top-K: {top_k}")
    print(f"模型: {model}")
    print(f"API URL: {base_url}")
    print(f"API Key: {api_key[:20]}...\n")

    # 创建 reranker
    reranker = BGEAPIReranker(base_url=base_url, api_key=api_key, model=model)

    # 执行 reranking
    print("🔄 正在计算相关性分数...")
    start_time = time.time()
    results = await reranker.rerank(query, documents, top_k=top_k)
    elapsed_time = time.time() - start_time

    # 显示结果
    print("\n" + "-" * 80)
    print("📊 Reranking 结果")
    print("-" * 80)

    for i, (idx, score) in enumerate(results):
        print(f"\n[{i+1}] 分数: {score:.4f}")
        print(f"    原始索引: {idx}")
        print(f"    内容: {documents[idx][:100]}...")

    print("\n" + "-" * 80)
    print(f"⏱️ 耗时: {elapsed_time:.3f} 秒")
    print(f"📊 平均每个文档: {elapsed_time / len(documents):.3f} 秒")
    print(f"✅ 测试完成! 共返回 {len(results)} 个结果")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="BGE Reranker API 测试脚本",
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
        default="http://192.168.8.233:8091",
        help="BGE Reranker API 地址"
    )
    parser.add_argument(
        "--api-key",
        default="sk-aaabbbcccdddeeefffggghhhiiijjjkkk",
        help="BGE Reranker API 密钥"
    )
    parser.add_argument(
        "--model",
        default="bge-reranker-v2-m3",
        help="Reranker 模型名称"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="返回 Top-K 结果"
    )

    args = parser.parse_args()

    asyncio.run(test_reranker(
        query=args.query,
        documents=args.docs,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        top_k=args.top_k,
    ))


if __name__ == "__main__":
    main()
