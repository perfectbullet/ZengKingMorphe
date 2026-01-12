"""
RAG系统基线测试脚本

功能:
1. 测试文档解析和切分
2. 测试向量化
3. 对比不同切分参数的效果

使用方法:
    python -m tests.test_rag_baseline --file "path/to/document.pdf"
    python -m tests.test_rag_baseline --file "path/to/document.pdf" --compare-sizes
"""

import asyncio
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import get_logger
from app.core.database import mongodb
from app.services.document_service import DocumentProcessor
from app.utils.embeddings import get_embedding

logger = get_logger(__name__)


# 默认测试查询
DEFAULT_QUERIES = [
    "文档的主要内容是什么？",
    "有哪些关键要点？",
    "文档提到的流程步骤有哪些？",
]


class RAGBaselineTest:
    """RAG系统基线测试"""

    def __init__(self):
        self.processor = DocumentProcessor()
        self.embedding_service = get_embedding()
        self.results = {
            "test_time": datetime.now().isoformat(),
            "file_path": "",
            "configs": [],
            "summary": {}
        }
        self._mongodb_initialized = False

    async def ensure_mongodb(self):
        """确保 MongoDB 连接已初始化"""
        if not self._mongodb_initialized:
            try:
                await mongodb.connect()
                self._mongodb_initialized = True
                logger.info("MongoDB 连接已建立")
            except Exception as e:
                logger.error(f"MongoDB 连接失败: {e}")
                raise

    async def test_chunking_config(
        self,
        file_path: str,
        chunk_size: int,
        chunk_overlap: int,
        config_name: str
    ) -> Dict[str, Any]:
        """测试特定切分配置"""
        # 确保 MongoDB 已连接（MinerU 需要缓存）
        await self.ensure_mongodb()

        logger.info(f"\n{'='*60}")
        logger.info(f"测试配置: {config_name}")
        logger.info(f"  chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        logger.info(f"{'='*60}")

        result = {
            "config_name": config_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "metrics": {}
        }

        try:
            # Step 1: 解析文件
            start_time = time.time()
            filename = os.path.basename(file_path)
            file_ext = os.path.splitext(filename)[1].lower()

            text_content = await self.processor._extract_text(file_path, file_ext, use_mineru=True)
            parse_time = time.time() - start_time

            result["metrics"]["text_length"] = len(text_content)
            result["metrics"]["parse_time_ms"] = int(parse_time * 1000)

            logger.info(f"✓ 文件解析成功: {len(text_content)} 字符, 耗时 {parse_time:.2f}s")

            # Step 2: 切分文本
            start_time = time.time()
            chunk_config = {
                'segment_union_max_length': chunk_size,
                'segment_type': -1
            }

            chunks = self.processor._chunk_text(
                text_content,
                f"test_{config_name}",
                "test_kb",
                file_ext,
                chunk_config
            )
            chunk_time = time.time() - start_time

            chunk_sizes = [len(c.content) for c in chunks]

            result["metrics"]["chunk_count"] = len(chunks)
            result["metrics"]["avg_chunk_size"] = sum(chunk_sizes) / len(chunk_sizes) if chunks else 0
            result["metrics"]["min_chunk_size"] = min(chunk_sizes) if chunks else 0
            result["metrics"]["max_chunk_size"] = max(chunk_sizes) if chunks else 0
            result["metrics"]["chunk_time_ms"] = int(chunk_time * 1000)

            # 计算切分质量指标
            result["metrics"]["quality_metrics"] = self._calculate_chunk_quality(chunks)

            logger.info(f"✓ 文本切分成功: {len(chunks)} 个chunk, 平均大小 {result['metrics']['avg_chunk_size']:.0f} 字符")

            # Step 3: 测试向量化 (前5个chunk)
            if chunks:
                embed_start = time.time()
                try:
                    test_chunks = chunks[:5]
                    # OllamaEmbeddings.embed_documents is synchronous, not async
                    embeddings = self.embedding_service.embed_documents([c.content for c in test_chunks])
                    embed_time = time.time() - embed_start

                    result["metrics"]["embedding_dim"] = len(embeddings[0]) if embeddings else 0
                    result["metrics"]["avg_embed_time_ms"] = int(embed_time / len(test_chunks) * 1000) if test_chunks else 0

                    logger.info(f"✓ 向量化成功: 维度 {result['metrics']['embedding_dim']}, "
                               f"平均耗时 {result['metrics']['avg_embed_time_ms']}ms/chunk")

                except Exception as e:
                    result["metrics"]["embedding_error"] = str(e)
                    logger.error(f"✗ 向量化失败: {e}")

            # Step 4: 显示chunk预览
            logger.info(f"\n--- Chunk预览 (前3个) ---")
            for i, chunk in enumerate(chunks[:3], 1):
                preview = chunk.content[:150].replace('\n', ' ')
                logger.info(f"  [{i}] 大小={len(chunk.content)} 字符: {preview}...")

            result["success"] = True

        except Exception as e:
            logger.error(f"✗ 测试失败: {e}", exc_info=True)
            result["success"] = False
            result["error"] = str(e)

        return result

    def _calculate_chunk_quality(self, chunks: List) -> Dict[str, Any]:
        """计算切分质量指标"""
        if not chunks:
            return {}

        sizes = [len(c.content) for c in chunks]

        # 计算大小分布的方差 (越小越均匀)
        avg_size = sum(sizes) / len(sizes)
        variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
        std_dev = variance ** 0.5

        # 计算过短chunk的比例 (< 50字符)
        too_short_count = sum(1 for s in sizes if s < 50)
        too_short_ratio = too_short_count / len(sizes)

        # 计算过长chunk的比例 (> 1000字符)
        too_long_count = sum(1 for s in sizes if s > 1000)
        too_long_ratio = too_long_count / len(sizes)

        return {
            "std_dev": round(std_dev, 2),
            "too_short_ratio": round(too_short_ratio, 2),
            "too_long_ratio": round(too_long_ratio, 2),
            "uniformity_score": round(1 / (1 + std_dev / avg_size), 4) if avg_size > 0 else 0
        }

    def print_comparison_report(self):
        """打印对比报告"""
        print("\n" + "="*80)
        print(" "*25 + "RAG基线测试报告")
        print("="*80)

        print(f"\n文件: {self.results['file_path']}")
        print(f"测试时间: {self.results['test_time']}")

        configs = self.results.get("configs", [])

        if not configs:
            print("\n没有测试结果")
            return

        # 打印配置对比表格
        print(f"\n{'='*80}")
        print("配置对比:")
        print(f"{'='*80}")

        print(f"\n{'配置名称':<20} {'Chunk大小':<12} {'Overlap':<12} {'Chunk数':<10} "
              f"{'平均大小':<12} {'标准差':<12} {'均匀度':<10}")
        print("-" * 100)

        for config in configs:
            if config.get("success"):
                metrics = config["metrics"]
                quality = metrics.get("quality_metrics", {})
                print(f"{config['config_name']:<20} "
                      f"{config['chunk_size']:<12} "
                      f"{config['chunk_overlap']:<12} "
                      f"{metrics['chunk_count']:<10} "
                      f"{metrics['avg_chunk_size']:<12.1f} "
                      f"{quality.get('std_dev', 0):<12.1f} "
                      f"{quality.get('uniformity_score', 0):<10.4f}")
            else:
                print(f"{config['config_name']:<20} [测试失败]")

        # 打印详细分析
        print(f"\n{'='*80}")
        print("详细分析:")
        print(f"{'='*80}")

        for config in configs:
            if not config.get("success"):
                continue

            print(f"\n【{config['config_name']}】")
            metrics = config["metrics"]
            quality = metrics.get("quality_metrics", {})

            print(f"  文本长度: {metrics['text_length']} 字符")
            print(f"  解析耗时: {metrics['parse_time_ms']} ms")
            print(f"  切分耗时: {metrics['chunk_time_ms']} ms")

            if "avg_embed_time_ms" in metrics:
                print(f"  向量化耗时: {metrics['avg_embed_time_ms']} ms/chunk")
                print(f"  向量维度: {metrics['embedding_dim']}")

            print(f"\n  切分质量:")
            print(f"    - Chunk数量: {metrics['chunk_count']}")
            print(f"    - 平均大小: {metrics['avg_chunk_size']:.1f} 字符")
            print(f"    - 最小: {metrics['min_chunk_size']} 字符")
            print(f"    - 最大: {metrics['max_chunk_size']} 字符")
            print(f"    - 标准差: {quality.get('std_dev', 0):.1f} (越小越均匀)")
            print(f"    - 过短比例: {quality.get('too_short_ratio', 0):.1%} (<50字符)")
            print(f"    - 过长比例: {quality.get('too_long_ratio', 0):.1%} (>1000字符)")
            print(f"    - 均匀度: {quality.get('uniformity_score', 0):.4f} (越高越好)")

        # 打印推荐
        print(f"\n{'='*80}")
        print("推荐建议:")
        print(f"{'='*80}")

        self._print_recommendations(configs)

        print("\n" + "="*80 + "\n")

    def _print_recommendations(self, configs: List[Dict]):
        """打印优化建议"""
        successful = [c for c in configs if c.get("success")]

        if not successful:
            print("  [!] 所有配置都测试失败，请检查文件和配置")
            return

        # 找出均匀度最高的配置
        best_uniformity = max(successful, key=lambda c: c["metrics"].get("quality_metrics", {}).get("uniformity_score", 0))

        # 找出chunk数最少的配置
        fewest_chunks = min(successful, key=lambda c: c["metrics"]["chunk_count"])

        # 找出平均大小最接近目标(400-600字符)的配置
        target_size = 500
        closest_to_target = min(successful, key=lambda c: abs(c["metrics"]["avg_chunk_size"] - target_size))

        print(f"\n  [*] 切分最均匀: {best_uniformity['config_name']} (均匀度: "
              f"{best_uniformity['metrics']['quality_metrics']['uniformity_score']:.4f})")

        print(f"  [*] Chunk数量最少: {fewest_chunks['config_name']} ({fewest_chunks['metrics']['chunk_count']} 个)")

        print(f"  [*] 大小最理想: {closest_to_target['config_name']} "
              f"(平均 {closest_to_target['metrics']['avg_chunk_size']:.0f} 字符)")

        # 分析问题
        print(f"\n  [!] 优化建议:")

        for config in successful:
            quality = config["metrics"].get("quality_metrics", {})
            issues = []

            if quality.get("too_short_ratio", 0) > 0.1:
                issues.append("过多短chunk (<50字符)")

            if quality.get("too_long_ratio", 0) > 0.1:
                issues.append("过多长chunk (>1000字符)")

            if quality.get("uniformity_score", 0) < 0.5:
                issues.append("大小分布不均匀")

            if config["metrics"]["avg_chunk_size"] < 200:
                issues.append("平均chunk偏小 (<200字符)")

            if config["metrics"]["avg_chunk_size"] > 800:
                issues.append("平均chunk偏大 (>800字符)")

            if issues:
                print(f"\n    {config['config_name']}:")
                for issue in issues:
                    print(f"      - {issue}")
                    if "短chunk" in issue:
                        print(f"        建议: 增大chunk_size或减小chunk_overlap")
                    elif "长chunk" in issue:
                        print(f"        建议: 减小chunk_size")
                    elif "不均匀" in issue:
                        print(f"        建议: 调整分隔符或使用自适应切分")

    def save_report(self, output_path: Optional[str] = None):
        """保存测试报告"""
        if not output_path:
            output_path = f"rag_baseline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        logger.info(f"测试报告已保存到: {output_path}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG系统基线测试")
    parser.add_argument("--file", type=str, required=True, help="测试文档路径")
    parser.add_argument("--chunk-size", type=int, default=256, help="默认切分大小")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="默认切分重叠")
    parser.add_argument("--compare-sizes", action="store_true", help="对比不同chunk_size效果")
    parser.add_argument("--output", type=str, help="输出报告文件路径")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        logger.error(f"文件不存在: {args.file}")
        sys.exit(1)

    # 创建测试实例
    tester = RAGBaselineTest()
    tester.results["file_path"] = args.file

    try:
        # 测试默认配置
        result = await tester.test_chunking_config(
            args.file,
            args.chunk_size,
            args.chunk_overlap,
            f"默认配置({args.chunk_size}/{args.chunk_overlap})"
        )
        tester.results["configs"].append(result)

        # 如果要求对比不同大小
        if args.compare_sizes:
            test_configs = [
                (128, 25, "小chunk(128/25)"),
                (256, 50, "中chunk(256/50)"),
                (512, 128, "大chunk(512/128)"),
                (800, 200, "超大chunk(800/200)"),
            ]

            for size, overlap, name in test_configs[1:]:  # 跳过默认配置(已测试)
                result = await tester.test_chunking_config(
                    args.file, size, overlap, name
                )
                tester.results["configs"].append(result)

        # 打印报告
        tester.print_comparison_report()

        # 保存报告
        tester.save_report(args.output)

    except Exception as e:
        logger.error(f"测试执行失败: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # 清理 MongoDB 连接
        if tester._mongodb_initialized:
            await mongodb.disconnect()
            logger.info("MongoDB 连接已关闭")


if __name__ == "__main__":
    asyncio.run(main())
