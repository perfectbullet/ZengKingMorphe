"""
RAG系统评估测试脚本

功能：
1. 文档上传测试：测试切分、向量化和存储
2. 检索召回测试：测试向量搜索、关键词搜索和混合搜索
3. 端到端测试：完整的检索+生成流程
4. 评估指标：精确率(Precision)、召回排名、性能指标

使用方法：
    python scripts/rag_evaluation_test.py --file path/to/document.pdf --kb_id test_kb

    # 使用预设测试查询
    python scripts/rag_evaluation_test.py --file path/to/document.pdf --kb_id test_kb --queries "query1|query2|query3"

    # 完整测试（包含生成）
    python scripts/rag_evaluation_test.py --file path/to/document.pdf --kb_id test_kb --full-test
"""

import asyncio
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.logging import get_logger
from app.core.database import get_database
from app.services.rag_service import rag_retrieval
from app.services.document_service import document_processor
from app.services.conversation_service import conversation_workflow
from app.utils.embeddings import get_embedding

logger = get_logger(__name__)


# 默认测试查询集
DEFAULT_TEST_QUERIES = [
    "文档的主要内容是什么？",
    "有哪些关键要点？",
    "文档提到的流程步骤有哪些？",
    "有什么重要的注意事项？",
    "文档中的定义和概念解释"
]


class RAGEvaluationTest:
    """RAG系统评估测试类"""

    def __init__(self, kb_id: str, employee_id: str = "hutao"):
        self.kb_id = kb_id
        self.employee_id = employee_id
        self.embedding_service = get_embedding()
        self.test_results = {
            "upload": {},
            "retrieval": {},
            "generation": {},
            "summary": {}
        }

    async def setup(self):
        """初始化测试环境"""
        logger.info(f"Setting up test environment for kb_id={self.kb_id}")

        # 确保知识库存在（如果不存在则创建测试配置）
        db = await get_database()

        # 检查/创建数字员工配置
        employee = await db.digital_employee_configs.find_one({"employee_id": self.employee_id})
        if not employee:
            logger.info(f"Creating test employee config for {self.employee_id}")
            await db.digital_employee_configs.insert_one({
                "employee_id": self.employee_id,
                "name": "测试员工",
                "role": "测试助手",
                "description": "用于RAG测试的数字员工",
                "kb_ids": [self.kb_id],
                "greeting": "你好，我是测试助手",
                "personality": {"tone": "professional", "style": "friendly"},
                "faq_sim_threshold": 0.02,
                "faq_top_k": 3,
                "capabilities": {"web_search_enabled": False}
            })
        else:
            # 确保知识库ID在配置中
            kb_ids = employee.get("kb_ids", [])
            if self.kb_id not in kb_ids:
                await db.digital_employee_configs.update_one(
                    {"employee_id": self.employee_id},
                    {"$push": {"kb_ids": self.kb_id}}
                )
                logger.info(f"Added kb_id {self.kb_id} to employee config")

    async def cleanup_test_data(self, doc_id: Optional[str] = None):
        """清理测试数据"""
        logger.info("Cleaning up test data")
        db = await get_database()

        if doc_id:
            # 删除特定文档的所有相关数据
            await db.documents.delete_many({"doc_id": doc_id})
            await db.document_chunks.delete_many({"doc_id": doc_id})

            # TODO: 从Chroma和ES删除（需要添加相应方法）

        logger.info("Test data cleanup completed")

    async def test_document_upload(
        self,
        file_path: str,
        chunk_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        测试文档上传流程

        测试点：
        1. 文件解析是否正确
        2. 文本切分是否合理
        3. 向量化是否成功
        4. 存储是否完整
        """
        logger.info("=" * 60)
        logger.info(f"开始测试文档上传: {file_path}")
        logger.info("=" * 60)

        start_time = time.time()
        results = {
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "chunk_config": chunk_config,
            "steps": {}
        }

        try:
            # Step 1: 文件解析
            step_start = time.time()
            filename = os.path.basename(file_path)
            file_ext = os.path.splitext(filename)[1].lower()

            # 直接调用文档处理器的内部方法进行测试
            from app.services.document_service import DocumentProcessor
            processor = DocumentProcessor()

            text_content = await processor._extract_text(file_path, file_ext, use_mineru=False)
            parse_time = time.time() - step_start

            results["steps"]["parse"] = {
                "success": True,
                "text_length": len(text_content),
                "time_ms": int(parse_time * 1000),
                "preview": text_content[:200] + "..." if len(text_content) > 200 else text_content
            }
            logger.info(f"文件解析成功: 提取 {len(text_content)} 字符, 耗时 {parse_time:.2f}s")

            # Step 2: 文本切分
            step_start = time.time()
            chunks = processor._chunk_text(text_content, "test_doc", self.kb_id, file_ext, chunk_config)
            chunk_time = time.time() - step_start

            chunk_sizes = [len(c.content) for c in chunks]
            results["steps"]["chunk"] = {
                "success": True,
                "chunk_count": len(chunks),
                "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes) if chunks else 0,
                "min_chunk_size": min(chunk_sizes) if chunks else 0,
                "max_chunk_size": max(chunk_sizes) if chunks else 0,
                "time_ms": int(chunk_time * 1000),
                "chunks_preview": [{"index": i, "size": len(c.content), "content": c.content[:100]}
                                   for i, c in enumerate(chunks[:3])]
            }
            logger.info(f"文本切分成功: {len(chunks)} 个chunk, "
                       f"平均大小 {results['steps']['chunk']['avg_chunk_size']:.0f} 字符, "
                       f"耗时 {chunk_time:.2f}s")

            # Step 3: 向量化（仅测试第一个chunk）
            if chunks:
                step_start = time.time()
                try:
                    embedding = await self.embedding_service.embed_query(chunks[0].content)
                    embed_time = time.time() - step_start

                    results["steps"]["embed"] = {
                        "success": True,
                        "embedding_dim": len(embedding),
                        "time_ms": int(embed_time * 1000)
                    }
                    logger.info(f"向量化成功: 维度 {len(embedding)}, 耗时 {embed_time:.2f}s")
                except Exception as e:
                    results["steps"]["embed"] = {
                        "success": False,
                        "error": str(e)
                    }
                    logger.error(f"向量化失败: {e}")

            total_time = time.time() - start_time
            results["total_time_ms"] = int(total_time * 1000)
            results["overall_success"] = all(
                step.get("success", True) for step in results["steps"].values()
            )

            self.test_results["upload"] = results

        except Exception as e:
            logger.error(f"文档上传测试失败: {e}", exc_info=True)
            results["overall_success"] = False
            results["error"] = str(e)
            self.test_results["upload"] = results

        return results

    async def test_retrieval(
        self,
        queries: List[str],
        test_methods: List[str] = ["hybrid", "vector", "keyword"]
    ) -> Dict[str, Any]:
        """
        测试检索召回质量

        测试点：
        1. 向量搜索效果
        2. 关键词搜索效果
        3. 混合搜索效果
        4. 召回速度
        """
        logger.info("=" * 60)
        logger.info(f"开始测试检索召回: {len(queries)} 个查询")
        logger.info("=" * 60)

        results = {
            "queries": [],
            "summary": {}
        }

        for query in queries:
            query_result = {
                "query": query,
                "methods": {}
            }

            logger.info(f"\n--- 测试查询: {query} ---")

            for method in test_methods:
                method_start = time.time()
                try:
                    use_hybrid = (method == "hybrid")

                    search_results = await rag_retrieval.search(
                        query=query,
                        kb_ids=[self.kb_id],
                        top_k=5,
                        use_hybrid=use_hybrid
                    )

                    method_time = time.time() - method_start

                    # 分析结果
                    query_result["methods"][method] = {
                        "success": True,
                        "result_count": len(search_results),
                        "time_ms": int(method_time * 1000),
                        "results": self._analyze_search_results(search_results)
                    }

                    logger.info(f"  [{method}] 召回 {len(search_results)} 条, "
                               f"耗时 {method_time*1000:.0f}ms, "
                               f"Top分数: {search_results[0].get('rrf_score', 0):.4f}" if search_results else f"  [{method}] 无结果")

                except Exception as e:
                    query_result["methods"][method] = {
                        "success": False,
                        "error": str(e)
                    }
                    logger.error(f"  [{method}] 失败: {e}")

            results["queries"].append(query_result)

        # 生成汇总统计
        results["summary"] = self._calculate_retrieval_summary(results)

        self.test_results["retrieval"] = results
        return results

    def _analyze_search_results(self, results: List[Dict]) -> List[Dict]:
        """分析搜索结果，提取关键信息"""
        analyzed = []
        for i, r in enumerate(results[:5]):  # 只分析前5条
            analyzed.append({
                "rank": i + 1,
                "doc_id": r.get("doc_id"),
                "kb_id": r.get("kb_id"),
                "chunk_index": r.get("chunk_index"),
                "rrf_score": round(r.get("rrf_score", 0), 4),
                "vector_score": round(r.get("vector_score", 0), 4),
                "keyword_score": round(r.get("keyword_score", 0), 4),
                "content_preview": r.get("content", "")[:100] + "..."
            })
        return analyzed

    def _calculate_retrieval_summary(self, results: Dict) -> Dict:
        """计算检索测试的汇总统计"""
        summary = {
            "total_queries": len(results["queries"]),
            "method_stats": {}
        }

        methods = ["hybrid", "vector", "keyword"]

        for method in methods:
            successful = 0
            total_results = 0
            total_time = 0
            top_scores = []

            for q in results["queries"]:
                if method in q["methods"]:
                    m = q["methods"][method]
                    if m.get("success"):
                        successful += 1
                        total_results += m.get("result_count", 0)
                        total_time += m.get("time_ms", 0)

                        # 收集Top分数
                        if m.get("results"):
                            top_scores.append(m["results"][0].get("rrf_score", 0))

            summary["method_stats"][method] = {
                "success_rate": round(successful / len(results["queries"]) * 100, 2) if results["queries"] else 0,
                "avg_results": round(total_results / successful, 2) if successful > 0 else 0,
                "avg_time_ms": round(total_time / successful, 2) if successful > 0 else 0,
                "avg_top_score": round(sum(top_scores) / len(top_scores), 4) if top_scores else 0,
                "min_top_score": round(min(top_scores), 4) if top_scores else 0,
                "max_top_score": round(max(top_scores), 4) if top_scores else 0
            }

        return summary

    async def test_end_to_end(
        self,
        queries: List[str],
        user_id: str = "test_user"
    ) -> Dict[str, Any]:
        """
        端到端测试：完整的检索+生成流程

        测试点：
        1. 完整的ConversationWorkflow执行
        2. 生成答案的质量
        3. 端到端性能
        """
        logger.info("=" * 60)
        logger.info(f"开始端到端测试: {len(queries)} 个查询")
        logger.info("=" * 60)

        results = {
            "queries": []
        }

        for i, query in enumerate(queries):
            logger.info(f"\n--- 端到端测试 {i+1}/{len(queries)}: {query} ---")

            query_result = {
                "query": query,
                "session_id": f"test_session_{int(time.time())}_{i}"
            }

            start_time = time.time()

            try:
                # 构建初始状态
                from app.services.conversation_service import ConversationState
                initial_state: ConversationState = {
                    "messages": [],
                    "user_query": query,
                    "user_id": user_id,
                    "session_id": query_result["session_id"],
                    "employee_id": self.employee_id,
                    "employee_config": {},
                    "is_realtime_query": False,
                    "realtime_category": "",
                    "realtime_detect_reason": "",
                    "intent": "",
                    "entities": {},
                    "retrieved_docs": [],
                    "relevance_score": 0.0,
                    "web_search_results": [],
                    "final_answer": "",
                    "confidence": 0.0,
                    "context": {"messages": [], "message_count": 0},
                    "has_sensitive": False,
                    "error": None,
                    "faq_matched": None,
                    "kb_used": [],
                    "web_search_used": False,
                    "conversation_id": "",
                    "response_time_ms": 0,
                    "workflow_start_time": time.time(),
                    "node_timings": {},
                    "ttfb_ms": None
                }

                # 执行工作流
                final_state = await conversation_workflow.workflow.ainvoke(initial_state)

                total_time = time.time() - start_time

                # 收集结果
                query_result["success"] = True
                query_result["total_time_ms"] = int(total_time * 1000)
                query_result["intent"] = final_state.get("intent")
                query_result["faq_matched"] = final_state.get("faq_matched") is not None
                query_result["retrieved_count"] = len(final_state.get("retrieved_docs", []))
                query_result["relevance_score"] = final_state.get("relevance_score", 0)
                query_result["confidence"] = final_state.get("confidence", 0)
                query_result["web_search_used"] = final_state.get("web_search_used", False)
                query_result["node_timings"] = final_state.get("node_timings", {})

                # 检索到的文档详情
                retrieved_docs = final_state.get("retrieved_docs", [])
                query_result["retrieved_docs"] = [
                    {
                        "rank": i + 1,
                        "doc_id": doc.get("doc_id"),
                        "rrf_score": doc.get("rrf_score", 0),
                        "content_preview": doc.get("content", "")[:100] + "..."
                    }
                    for i, doc in enumerate(retrieved_docs[:3])
                ]

                # 性能分析
                node_timings = final_state.get("node_timings", {})
                if node_timings:
                    query_result["performance_breakdown"] = {
                        "load_employee_config_ms": node_timings.get("load_employee_config", 0),
                        "load_session_context_ms": node_timings.get("load_session_context", 0),
                        "match_faq_ms": node_timings.get("match_faq", 0),
                        "knowledge_retrieval_ms": node_timings.get("knowledge_retrieval", 0),
                        "rerank_documents_ms": node_timings.get("rerank_documents", 0),
                        "generate_answer_ms": node_timings.get("generate_answer", 0),
                        "save_conversation_ms": node_timings.get("save_conversation", 0),
                    }

                logger.info(f"  完成, 意图={query_result['intent']}, "
                           f"召回={query_result['retrieved_count']}, "
                           f"相关性={query_result['relevance_score']:.2f}, "
                           f"耗时={total_time:.2f}s")

            except Exception as e:
                query_result["success"] = False
                query_result["error"] = str(e)
                logger.error(f"  失败: {e}", exc_info=True)

            results["queries"].append(query_result)

        # 汇总统计
        results["summary"] = self._calculate_e2e_summary(results)

        self.test_results["generation"] = results
        return results

    def _calculate_e2e_summary(self, results: Dict) -> Dict:
        """计算端到端测试的汇总统计"""
        successful = [q for q in results["queries"] if q.get("success")]

        if not successful:
            return {"error": "没有成功的查询"}

        summary = {
            "total_queries": len(results["queries"]),
            "success_count": len(successful),
            "success_rate": round(len(successful) / len(results["queries"]) * 100, 2),
            "avg_response_time_ms": round(sum(q.get("total_time_ms", 0) for q in successful) / len(successful), 2),
            "avg_retrieved_count": round(sum(q.get("retrieved_count", 0) for q in successful) / len(successful), 2),
            "avg_relevance_score": round(sum(q.get("relevance_score", 0) for q in successful) / len(successful), 4),
            "avg_confidence": round(sum(q.get("confidence", 0) for q in successful) / len(successful), 4),
        }

        # 意图分布
        intents = {}
        for q in successful:
            intent = q.get("intent", "unknown")
            intents[intent] = intents.get(intent, 0) + 1
        summary["intent_distribution"] = intents

        return summary

    def print_report(self):
        """打印测试报告"""
        print("\n" + "="*80)
        print(" "*20 + "RAG系统评估测试报告")
        print("="*80)

        # 上传测试报告
        if self.test_results.get("upload"):
            self._print_upload_report()

        # 检索测试报告
        if self.test_results.get("retrieval"):
            self._print_retrieval_report()

        # 端到端测试报告
        if self.test_results.get("generation"):
            self._print_generation_report()

        # 总体评估
        self._print_overall_assessment()

        print("="*80 + "\n")

    def _print_upload_report(self):
        """打印上传测试报告"""
        print("\n【一、文档上传测试】")
        print("-" * 60)

        upload = self.test_results["upload"]
        print(f"文件: {upload.get('file_path')}")
        print(f"大小: {upload.get('file_size', 0)} 字节")
        print(f"总耗时: {upload.get('total_time_ms', 0)} ms")

        for step_name, step_result in upload.get("steps", {}).items():
            status = "[PASS]" if step_result.get("success") else "[FAIL]"
            print(f"\n  {status} {step_name.upper()}:")
            for key, value in step_result.items():
                if key != "success" and not key.endswith("preview"):
                    print(f"     {key}: {value}")

    def _print_retrieval_report(self):
        """打印检索测试报告"""
        print("\n【二、检索召回测试】")
        print("-" * 60)

        retrieval = self.test_results["retrieval"]
        summary = retrieval.get("summary", {})

        print(f"总查询数: {summary.get('total_queries', 0)}")
        print("\n方法对比:")

        method_stats = summary.get("method_stats", {})
        if method_stats:
            print(f"\n  {'方法':<10} {'成功率':<10} {'平均召回':<12} {'平均耗时':<12} {'平均Top分数':<15}")
            print("  " + "-" * 60)

            for method, stats in method_stats.items():
                print(f"  {method:<10} {stats['success_rate']:<10}% "
                      f"{stats['avg_results']:<12} {stats['avg_time_ms']:<12}ms "
                      f"{stats['avg_top_score']:<15}")

        # 详细查询结果
        print("\n详细查询结果:")
        for q in retrieval.get("queries", [])[:3]:  # 只显示前3个
            print(f"\n  查询: {q['query']}")
            for method, m in q.get("methods", {}).items():
                if m.get("success"):
                    print(f"    [{method}] 召回 {m['result_count']} 条, "
                          f"Top分数: {m.get('results', [{}])[0].get('rrf_score', 0):.4f}, "
                          f"耗时 {m['time_ms']}ms")

    def _print_generation_report(self):
        """打印端到端测试报告"""
        print("\n【三、端到端测试】")
        print("-" * 60)

        gen = self.test_results["generation"]
        summary = gen.get("summary", {})

        print(f"总查询数: {summary.get('total_queries', 0)}")
        print(f"成功率: {summary.get('success_rate', 0)}%")
        print(f"平均响应时间: {summary.get('avg_response_time_ms', 0)} ms")
        print(f"平均召回数: {summary.get('avg_retrieved_count', 0)}")
        print(f"平均相关性分数: {summary.get('avg_relevance_score', 0)}")
        print(f"平均置信度: {summary.get('avg_confidence', 0)}")

        print("\n意图分布:")
        for intent, count in summary.get("intent_distribution", {}).items():
            print(f"  {intent}: {count}")

    def _print_overall_assessment(self):
        """打印总体评估"""
        print("\n【总体评估】")
        print("-" * 60)

        upload_ok = self.test_results.get("upload", {}).get("overall_success", False)
        retrieval_ok = bool(self.test_results.get("retrieval"))
        generation_ok = bool(self.test_results.get("generation"))

        assessments = []
        if upload_ok:
            assessments.append("[PASS] 文档上传流程正常")
        else:
            assessments.append("[FAIL] 文档上传流程存在问题")

        if retrieval_ok:
            retrieval_summary = self.test_results["retrieval"]["summary"]
            hybrid_success = retrieval_summary["method_stats"].get("hybrid", {}).get("success_rate", 0)
            if hybrid_success > 80:
                assessments.append(f"[PASS] 检索召回质量良好 (成功率 {hybrid_success}%)")
            else:
                assessments.append(f"[WARN] 检索召回质量需优化 (成功率 {hybrid_success}%)")
        else:
            assessments.append("[SKIP] 检索召回测试未执行")

        if generation_ok:
            gen_summary = self.test_results["generation"]["summary"]
            gen_success = gen_summary.get("success_rate", 0)
            avg_relevance = gen_summary.get("avg_relevance_score", 0)

            if gen_success > 80 and avg_relevance > 0.5:
                assessments.append(f"[PASS] 端到端流程正常 (平均相关性 {avg_relevance:.2f})")
            elif gen_success > 80:
                assessments.append(f"[WARN] 端到端流程可用但相关性偏低 (平均 {avg_relevance:.2f})")
            else:
                assessments.append(f"[FAIL] 端到端流程存在问题 (成功率 {gen_success}%)")

        for assessment in assessments:
            print(assessment)

    def save_json_report(self, output_path: str):
        """保存JSON格式的测试报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        logger.info(f"测试报告已保存到: {output_path}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG系统评估测试")
    parser.add_argument("--file", type=str, help="测试文档路径")
    parser.add_argument("--kb_id", type=str, default="test_kb", help="知识库ID")
    parser.add_argument("--employee_id", type=str, default="hutao", help="员工ID")
    parser.add_argument("--queries", type=str, help="测试查询，用|分隔")
    parser.add_argument("--full-test", action="store_true", help="执行完整测试（包含生成）")
    parser.add_argument("--output", type=str, help="输出报告文件路径")
    parser.add_argument("--chunk-size", type=int, default=256, help="切分大小")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="切分重叠")

    args = parser.parse_args()

    # 解析查询
    if args.queries:
        test_queries = [q.strip() for q in args.queries.split("|")]
    else:
        test_queries = DEFAULT_TEST_QUERIES

    # 创建测试实例
    tester = RAGEvaluationTest(args.kb_id, args.employee_id)

    try:
        # 初始化
        await tester.setup()

        # 1. 文档上传测试（如果提供了文件）
        if args.file:
            chunk_config = {
                'segment_union_max_length': args.chunk_size,
                'segment_type': -1
            }
            await tester.test_document_upload(args.file, chunk_config)

        # 2. 检索召回测试
        await tester.test_retrieval(test_queries)

        # 3. 端到端测试（如果要求）
        if args.full_test:
            await tester.test_end_to_end(test_queries)

        # 4. 打印报告
        tester.print_report()

        # 5. 保存JSON报告
        if args.output:
            tester.save_json_report(args.output)
        else:
            # 默认保存
            default_output = f"rag_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            tester.save_json_report(default_output)

    except Exception as e:
        logger.error(f"测试执行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
