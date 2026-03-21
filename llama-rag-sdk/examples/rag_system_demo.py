"""
RAG 系统性能评估示例

专注于文档解析到最后召回的文档，用于评估 RAGSystem 性能好坏。
核心流程：文档解析 → 文本分块 → 向量化 → 存储 → 检索 → 评估
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.rag_system import RAGSystem
from src.document_parser import ParseOptions, ReturnOptions
from src.retrieval.base import RetrievedDocument
from src.utils import setup_logger


class RAGEvaluator:
    """RAG 系统评估器"""

    @staticmethod
    def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """清理元数据，移除 None 值（ChromaDB 不接受 None）

        Args:
            metadata: 原始元数据

        Returns:
            清理后的元数据，只包含 str, int, float, bool 类型的值
        """
        return {
            k: v for k, v in metadata.items()
            if v is not None and isinstance(v, (str, int, float, bool))
        }

    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "parse_time": 0,
            "index_time": 0,
            "retrieve_time": 0,
            "total_chunks": 0,
            "total_images": 0,
            "queries": [],
        }

    def record_parse_time(self, duration: float, chunks: int, images: int):
        """记录解析时间"""
        self.metrics["parse_time"] = duration
        self.metrics["total_chunks"] = chunks
        self.metrics["total_images"] = images

    def record_index_time(self, duration: float):
        """记录索引时间"""
        self.metrics["index_time"] = duration

    def record_retrieve(
        self,
        query: str,
        duration: float,
        results: List[RetrievedDocument]
    ):
        """记录检索结果"""
        self.metrics["queries"].append({
            "query": query,
            "duration": duration,
            "result_count": len(results),
            "avg_score": sum(r.score for r in results) / len(results) if results else 0,
            "max_score": max((r.score for r in results), default=0),
            "min_score": min((r.score for r in results), default=0),
        })

    def print_report(self):
        """打印评估报告"""
        print("\n" + "=" * 60)
        print("RAG 系统性能评估报告")
        print("=" * 60)

        print(f"\n【文档解析】")
        print(f"  解析耗时: {self.metrics['parse_time']:.2f}s")
        print(f"  文本块数: {self.metrics['total_chunks']}")
        print(f"  图片数量: {self.metrics['total_images']}")
        print(f"  平均块大小: {self.metrics['parse_time'] / max(self.metrics['total_chunks'], 1) * 1000:.2f}ms/块")

        print(f"\n【文档索引】")
        print(f"  索引耗时: {self.metrics['index_time']:.2f}s")
        print(f"  平均索引时间: {self.metrics['index_time'] / max(self.metrics['total_chunks'], 1) * 1000:.2f}ms/块")

        print(f"\n【文档检索】")
        if self.metrics["queries"]:
            total_retrieve_time = sum(q["duration"] for q in self.metrics["queries"])
            avg_retrieve_time = total_retrieve_time / len(self.metrics["queries"])
            avg_score = sum(q["avg_score"] for q in self.metrics["queries"]) / len(self.metrics["queries"])

            print(f"  查询次数: {len(self.metrics['queries'])}")
            print(f"  总检索耗时: {total_retrieve_time:.2f}s")
            print(f"  平均检索耗时: {avg_retrieve_time:.2f}s/查询")
            print(f"  平均相似度得分: {avg_score:.4f}")
            print(f"  最高相似度得分: {max(q['max_score'] for q in self.metrics['queries']):.4f}")
            print(f"  最低相似度得分: {min(q['min_score'] for q in self.metrics['queries']):.4f}")
        else:
            print("  无查询数据")

        print(f"\n【整体性能】")
        total_time = (
            self.metrics["parse_time"] +
            self.metrics["index_time"] +
            sum(q["duration"] for q in self.metrics["queries"])
        )
        print(f"  总耗时: {total_time:.2f}s")
        print("=" * 60 + "\n")


async def parse_and_evaluate_pdf(
    rag: RAGSystem,
    pdf_path: str,
    evaluator: RAGEvaluator,
    collection_name: str
) -> None:
    """解析 PDF 并评估解析性能"""
    print(f"\n{'=' * 60}")
    pdf_name = Path(pdf_path).name
    print(f"步骤 1: 文档解析 - {pdf_name}")
    print(f"{'=' * 60}")
    if not Path(pdf_path).exists():
        print(f"✗ PDF 文件不存在: {pdf_path}")
        return

    start_time = time.time()

    try:
        # 配置解析选项
        parse_options = ParseOptions(
            backend="pipeline",
            lang="ch",
            formula_enable=True,
            table_enable=True,
        )
        return_options = ReturnOptions(
            return_md=True,
            return_content_list=True,
            return_images=True,
        )

        # 解析文档
        document = await rag.parser.parse(
            file_path=pdf_path,
            parse_options=parse_options,
            return_options=return_options,
        )
        json_path = document.save_to_json(
            pdf_filename=pdf_name,
            output_dir="./data/output"
        )

        parse_duration = time.time() - start_time

        print(f"✓ 文档解析完成 (耗时: {parse_duration:.2f}s)")
        print(f"  标题: {document.title}")
        print(f"  文本块: {len(document.chunks)}")
        print(f"  文本块文件的路径: {json_path}")
        print(f"  图片: {len(document.images)}")
        print(f"  内容长度: {len(document.content)} 字符")

        # 记录解析指标
        evaluator.record_parse_time(parse_duration, len(document.chunks), len(document.images))

        # 步骤 2: 文档索引（同时同步到 ChromaDB 和 DocStore）
        print(f"\n{'=' * 60}")
        print(f"步骤 2: 文档索引")
        print(f"{'=' * 60}")

        index_start = time.time()

        # 使用 RAGSystem 的方法同时索引到 ChromaDB 和 DocStore
        doc_ids = await rag.index_parsed_document(document, source_path=pdf_path)

        index_duration = time.time() - index_start

        print(f"✓ 文档索引完成 (耗时: {index_duration:.2f}s)")
        print(f"  索引块数: {len(doc_ids)}")

        # 记录索引指标
        evaluator.record_index_time(index_duration)

    except Exception as e:
        print(f"✗ 解析或索引失败: {e}")
        import traceback
        traceback.print_exc()


async def retrieve_and_evaluate(
    rag: RAGSystem,
    queries: List[str],
    evaluator: RAGEvaluator
) -> None:
    """执行检索并评估检索性能"""
    print(f"\n{'=' * 60}")
    print(f"步骤 3: 文档检索与评估")
    print(f"{'=' * 60}")

    for i, query in enumerate(queries, 1):
        print(f"\n查询 {i}/{len(queries)}: {query}")

        retrieve_start = time.time()

        try:
            results = await rag.retrieve(query, top_k=5)

            retrieve_duration = time.time() - retrieve_start

            print(f"  检索耗时: {retrieve_duration:.2f}s")
            print(f"  结果数量: {len(results)}")

            if results:
                for j, doc in enumerate(results[:3], 1):
                    preview = doc.text[:80].replace('\n', ' ')
                    print(f"    [{j}] 得分={doc.score:.4f} | {preview}...")

                    # 显示元数据
                    if doc.metadata:
                        page = doc.metadata.get("page")
                        section = doc.metadata.get("section")
                        if page is not None or section:
                            meta_parts = []
                            if page is not None:
                                meta_parts.append(f"页码={page}")
                            if section:
                                meta_parts.append(f"章节={section[:20]}")
                            print(f"         元数据: {', '.join(meta_parts)}")
            else:
                print("    未找到相关结果")

            # 记录检索指标
            evaluator.record_retrieve(query, retrieve_duration, results)

        except Exception as e:
            print(f"  检索失败: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """主函数 - RAG 系统性能评估"""
    # ========== 配置区域 ==========
    # 是否在开始前清理旧数据（清空 ChromaDB 集合和 DocStore）
    clear_old_data = True  # 设为 False 保留旧数据，设为 True 清空重新索引

    # =================================

    # 配置日志
    logger = setup_logger(
        log_level="INFO",
        log_file="./logs/rag_eval.log"
    )
    logger.info("RAG 系统性能评估示例")

    # 创建评估器
    evaluator = RAGEvaluator()

    # PDF 文档路径（可配置）
    pdf_paths = [
        "/home/zj/ZengKingMorphe/Digital-Human-Disciplinary-Dataset/math_file_part/01高中数学必修第一册-40pages-part1-page1-40.pdf",
    ]

    # 测试查询
    test_queries = [
        "什么是集合",
        "集合的基本关系有哪些",
        "什么是充分条件与必要条件",
        "函数的定义域和值域是什么",
        "导数的几何意义",
    ]

    # 集合名称
    collection_name = "rag_eval_collection"

    # 创建 RAG 系统
    async with RAGSystem(
        collection_name=collection_name,
    ) as rag:
        print(f"\nRAG 系统配置:")
        print(f"  集合名称: {collection_name}")
        print(f"  清理旧数据: {clear_old_data}")

        # 清理旧数据（在处理 PDF 之前）
        if clear_old_data:
            print("\n清理旧数据...")
            await rag.clear_collection()
            await rag.docstore.delete_all()
            print("✓ ChromaDB 集合已清空")
            print("✓ DocStore 已清空\n")

        # 步骤 1-2: 解析和索引 PDF
        for pdf_path in pdf_paths:
            await parse_and_evaluate_pdf(rag, pdf_path, evaluator, collection_name)

        # 步骤 3: 检索和评估
        if evaluator.metrics["total_chunks"] > 0:
            await retrieve_and_evaluate(rag, test_queries, evaluator)

        # 打印评估报告
        evaluator.print_report()

        # 保存评估结果到文件
        results_file = Path("./logs/rag_eval_results.txt")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w", encoding="utf-8") as f:
            f.write("RAG 系统性能评估结果\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"集合名称: {collection_name}\n")
            f.write(f"解析耗时: {evaluator.metrics['parse_time']:.2f}s\n")
            f.write(f"文本块数: {evaluator.metrics['total_chunks']}\n")
            f.write(f"索引耗时: {evaluator.metrics['index_time']:.2f}s\n")
            f.write(f"查询次数: {len(evaluator.metrics['queries'])}\n\n")
            f.write("详细查询结果:\n")
            for i, q in enumerate(evaluator.metrics['queries'], 1):
                f.write(f"\n  查询 {i}: {q['query']}\n")
                f.write(f"    耗时: {q['duration']:.2f}s\n")
                f.write(f"    结果数: {q['result_count']}\n")
                f.write(f"    平均得分: {q['avg_score']:.4f}\n")

        print(f"评估结果已保存到: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
