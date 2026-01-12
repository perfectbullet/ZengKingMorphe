"""
RAG系统批量测试脚本 - 批量测试所有文档
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


class RAGBatchTester:
    """RAG系统批量测试"""

    def __init__(self, use_mineru: bool = True):
        self.processor = DocumentProcessor()
        self.embedding_service = get_embedding()
        self.results = []
        self.start_time = time.time()
        self.use_mineru = use_mineru
        self._mongodb_initialized = False

    async def ensure_mongodb(self):
        """确保 MongoDB 连接已初始化"""
        if not self._mongodb_initialized and self.use_mineru:
            try:
                await mongodb.connect()
                self._mongodb_initialized = True
                logger.info("MongoDB 连接已建立")
            except Exception as e:
                logger.error(f"MongoDB 连接失败: {e}")
                raise

    async def test_single_document(
        self,
        file_path: str,
        chunk_configs: List[tuple] = None
    ) -> Dict[str, Any]:
        """测试单个文档"""
        # 确保 MongoDB 已连接
        await self.ensure_mongodb()

        if chunk_configs is None:
            chunk_configs = [
                (256, 50, "默认(256/50)"),
                (512, 128, "中chunk(512/128)"),
                (800, 200, "超大chunk(800/200)"),
            ]

        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()

        result = {
            "filename": filename,
            "file_path": file_path,
            "configs": []
        }

        try:
            # 解析文件
            text_content = await self.processor._extract_text(file_path, file_ext, use_mineru=self.use_mineru)
            text_length = len(text_content)
            result["text_length"] = text_length

            # 确定领域和文档分类
            result["domain"] = self._classify_domain(filename)
            result["length_category"] = self._classify_length(text_length)

            # 测试每个配置
            for chunk_size, overlap, config_name in chunk_configs:
                config_result = await self._test_config(
                    text_content, filename, file_ext, chunk_size, overlap, config_name
                )
                result["configs"].append(config_result)

            # 确定最佳配置
            result["recommendation"] = self._get_best_config(result["configs"])

            logger.info(f"[*] 完成: {filename} (长度:{text_length}, 最佳:{result['recommendation']['best_config']})")

        except Exception as e:
            logger.error(f"[!] 失败: {filename} - {e}")
            result["error"] = str(e)

        return result

    async def _test_config(
        self,
        text_content: str,
        filename: str,
        file_ext: str,
        chunk_size: int,
        overlap: int,
        config_name: str
    ) -> Dict[str, Any]:
        """测试特定配置"""
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

        chunk_sizes = [len(c.content) for c in chunks]

        # 计算质量指标
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunks else 0
        variance = sum((s - avg_size) ** 2 for s in chunk_sizes) / len(chunk_sizes) if chunks else 0
        std_dev = variance ** 0.5

        too_short_count = sum(1 for s in chunk_sizes if s < 50)
        too_short_ratio = too_short_count / len(chunks) if chunks else 0

        uniformity = round(1 / (1 + std_dev / avg_size), 4) if avg_size > 0 else 0

        return {
            "config_name": config_name,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunk_count": len(chunks),
            "avg_chunk_size": round(avg_size, 1),
            "min_chunk_size": min(chunk_sizes) if chunks else 0,
            "max_chunk_size": max(chunk_sizes) if chunks else 0,
            "std_dev": round(std_dev, 2),
            "too_short_ratio": round(too_short_ratio, 2),
            "uniformity": uniformity
        }

    def _classify_domain(self, filename: str) -> str:
        """根据文件名分类领域"""
        domain_keywords = {
            "教育技术": ["教育", "教学", "高校", "课程", "学生", "智慧教育", "职业生涯"],
            "金融科技": ["金融", "银行", "支付", "利率", "私募", "投资", "基金"],
            "设计哲学": ["设计", "造物", "创造性", "实践"],
            "环境经济": ["生态", "环境", "绿色"],
            "数字经济": ["数字", "数字化", "数据资产", "区块链", "转型"],
            "人工智能": ["人工智能", "AI", "GenAI", "算法"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in filename for kw in keywords):
                return domain

        return "综合"

    def _classify_length(self, text_length: int) -> str:
        """分类文档长度"""
        if text_length < 10000:
            return "短文档(<10K)"
        elif text_length < 20000:
            return "中长文档(10-20K)"
        else:
            return "超长文档(>20K)"

    def _get_best_config(self, configs: List[Dict]) -> Dict[str, Any]:
        """获取最佳配置"""
        valid_configs = [c for c in configs if c.get("chunk_count", 0) > 0]
        if not valid_configs:
            return {"best_config": "无", "reason": "所有配置均失败"}

        # 选择均匀度最高的配置
        best = max(valid_configs, key=lambda c: c.get("uniformity", 0))

        return {
            "best_config": best["config_name"],
            "uniformity": best["uniformity"],
            "chunk_count": best["chunk_count"],
            "avg_size": best["avg_chunk_size"],
            "reason": f"均匀度最高({best['uniformity']})"
        }

    def print_progress(self, current: int, total: int, filename: str):
        """打印进度"""
        elapsed = time.time() - self.start_time
        avg_time = elapsed / current if current > 0 else 0
        remaining = (total - current) * avg_time

        print(f"\r[{current}/{total}] {filename[:40]:<40} 预计剩余: {remaining:.0f}s", end="", flush=True)

    def save_results(self, output_path: str):
        """保存结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"\n结果已保存到: {output_path}")

    def print_summary(self):
        """打印汇总"""
        print("\n" + "="*80)
        print(" "*30 + "测试汇总")
        print("="*80)

        # 统计各配置胜出次数
        config_wins = {}
        domain_stats = {}
        length_stats = {}

        for r in self.results:
            if "error" in r:
                continue

            best = r.get("recommendation", {}).get("best_config", "")
            config_wins[best] = config_wins.get(best, 0) + 1

            # 领域统计
            domain = r.get("domain", "未知")
            if domain not in domain_stats:
                domain_stats[domain] = {"count": 0, "configs": {}}
            domain_stats[domain]["count"] += 1
            domain_stats[domain]["configs"][best] = domain_stats[domain]["configs"].get(best, 0) + 1

            # 长度统计
            length = r.get("length_category", "未知")
            if length not in length_stats:
                length_stats[length] = {"count": 0, "configs": {}}
            length_stats[length]["count"] += 1
            length_stats[length]["configs"][best] = length_stats[length]["configs"].get(best, 0) + 1

        print(f"\n各配置胜出次数:")
        for config, count in sorted(config_wins.items(), key=lambda x: x[1], reverse=True):
            print(f"  {config}: {count}")

        print(f"\n按文档长度统计:")
        for length, stats in length_stats.items():
            print(f"\n  {length} ({stats['count']}个):")
            for config, count in sorted(stats["configs"].items(), key=lambda x: x[1], reverse=True):
                print(f"    {config}: {count}")

        print(f"\n按领域统计:")
        for domain, stats in domain_stats.items():
            print(f"\n  {domain} ({stats['count']}个):")
            for config, count in sorted(stats["configs"].items(), key=lambda x: x[1], reverse=True):
                print(f"    {config}: {count}")

        print("\n" + "="*80)


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG系统批量测试")
    parser.add_argument("--dir", type=str, default="/home/zj/zenking_work/metahuman_work/test_files", help="测试目录")
    parser.add_argument("--output", type=str, default="ai-service/docs/rag_chunk_baseline_full.json", help="输出文件")
    parser.add_argument("--pattern", type=str, default="*.pdf", help="文件匹配模式")
    parser.add_argument("--use-mineru", action="store_true", default=True, help="使用MinerU解析PDF")
    parser.add_argument("--use-pypdf", action="store_true", help="使用PyPDF解析PDF")

    args = parser.parse_args()

    # 确定使用哪种解析方式
    use_mineru = not args.use_pypdf

    # 查找所有PDF文件
    test_dir = Path(args.dir)
    pdf_files = list(test_dir.glob(args.pattern))

    print(f"找到 {len(pdf_files)} 个文件")
    print(f"使用 {'MinerU' if use_mineru else 'PyPDF'} 解析")

    if not pdf_files:
        logger.error("没有找到PDF文件")
        return

    # 创建测试器
    tester = RAGBatchTester(use_mineru=use_mineru)

    try:
        # 批量测试
        for i, pdf_file in enumerate(pdf_files, 1):
            tester.print_progress(i, len(pdf_files), pdf_file.name)
            result = await tester.test_single_document(str(pdf_file))
            tester.results.append(result)

        # 保存结果
        tester.save_results(args.output)

        # 打印汇总
        tester.print_summary()

    finally:
        # 清理 MongoDB 连接
        if tester._mongodb_initialized:
            await mongodb.disconnect()
            logger.info("MongoDB 连接已关闭")


if __name__ == "__main__":
    asyncio.run(main())
