"""
RAG系统端到端评估测试脚本

功能:
1. 测试文档上传流程(包括自定义切分配置)
2. 测试知识库检索准确性
3. 测试FAQ匹配准确性
4. 测试文档相关性评分
5. 测试完整对话流程
6. 生成评估报告

作者: Claude
日期: 2026-01-09
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import httpx
from loguru import logger


# ==================== 配置 ====================

API_BASE_URL = "http://localhost:8100"  # Docker部署地址
# API_BASE_URL = "http://localhost:8000"  # 本地开发地址

TEST_EMPLOYEE_ID = "hutao"  # 测试员工ID
TEST_KB_ID = "kb_test_evaluation"  # 测试知识库ID

# 测试文档
TEST_DOCUMENTS = [
    {
        "name": "失蜡铸造工艺介绍.txt",
        "content": """
失蜡铸造工艺

一、工艺原理
失蜡铸造是一种精密铸造工艺,其原理是用蜡制作成铸件模型,然后在蜡模周围涂覆耐火材料形成铸型。
加热后蜡模熔化流出,得到空腔铸型。将熔融金属注入铸型,冷却后得到金属铸件。

二、工艺流程
1. 制作蜡模:使用蜡料制作与最终铸件形状相同的模型
2. 组装蜡树:将多个蜡模组装到蜡浇口系统上
3. 制壳:在蜡模表面反复涂覆耐火浆料和撒砂,形成多层陶瓷壳型
4. 脱蜡:将壳型加热,使蜡模熔化流出
5. 焙烧:高温焙烧壳型,增强其强度
6. 浇注:将熔融金属注入壳型
7. 清理:冷却后破坏壳型,取出铸件并清理

三、工艺特点
优点:
- 尺寸精度高,可达CT4-6级
- 表面质量好,表面粗糙度Ra可达0.8-1.6μm
- 可铸造复杂结构、薄壁铸件
- 适用于各种合金材料

缺点:
- 工艺复杂,生产周期长
- 成本较高,不适合大批量生产
- 铸件尺寸受限制

四、应用领域
失蜡铸造广泛应用于:
- 航空航天:涡轮叶片、导向器等精密零件
- 珠宝首饰:精细首饰、工艺品
- 医疗器械:牙科植入物、手术器械
- 汽车工业:精密机械零件
- 艺术品:雕塑、金属艺术品
""",
        "expected_queries": [
            "失蜡铸造的原理是什么?",
            "失蜡铸造的工艺流程有哪些步骤?",
            "失蜡铸造的优点和缺点?",
            "失蜡铸造应用在哪些领域?"
        ]
    },
    {
        "name": "首饰镶嵌工艺.txt",
        "content": """
首饰镶嵌工艺指南

一、常见镶嵌方式

1. 爪镶(Prong Setting)
- 特点:用金属爪固定宝石,露出宝石的大部分
- 优点:最大化展示宝石火彩,便于清洗
- 缺点:爪子容易挂衣物,宝石安全性较低
- 适用:圆形、椭圆形等刻面宝石

2. 包镶(Bezel Setting)
- 特点:用金属边包裹宝石边缘
- 优点:保护性好,不易脱落
- 缺点:遮挡部分宝石,影响火彩
- 适用:异形宝石、脆弱宝石

3. 钉镶(Pave Setting)
- 特点:用细小金属钉固定多颗小宝石
- 优点:宝石密集,闪亮效果好
- 缺点:工艺复杂,维修困难
- 适用:碎钻群镶

4. 轨道镶(Channel Setting)
- 特点:宝石嵌入两条金属轨道之间
- 优点:线条流畅,保护性好
- 缺点:尺寸受限,调整困难
- 适用:戒指、手链等

二、镶嵌工艺选择建议

根据宝石特性选择:
- 硬度高的宝石(钻石、红蓝宝石):爪镶、钉镶
- 硬度低的宝石(祖母绿、坦桑石):包镶、轨道镶
- 易碎宝石:包镶

根据使用场景选择:
- 日常佩戴:爪镶、包镶
- 频繁摩擦:轨道镶、包镶
- 展示为主:爪镶、钉镶

三、镶嵌质量控制

检验标准:
1. 宝石牢固度:摇晃时不应有松动
2. 金属表面:无划痕、砂眼
3. 对称性:宝石居中,爪子均匀
4. 边缘处理:无锐边,不伤手
""",
        "expected_queries": [
            "爪镶和包镶的区别?",
            "什么情况下选择包镶?",
            "钉镶工艺的特点是什么?",
            "如何检验镶嵌质量?"
        ]
    }
]

# 测试FAQ
TEST_FAQS = [
    {
        "question": "失蜡铸造的成本高吗?",
        "answer": "失蜡铸造工艺复杂,生产周期长,成本相对较高,因此不适合大批量生产,更适合高精度、小批量的精密铸件制造。"
    },
    {
        "question": "爪镶适合什么宝石?",
        "answer": "爪镶适合硬度较高的宝石,如钻石、红蓝宝石等刻面宝石。爪镶可以最大化展示宝石的火彩和亮度。"
    },
    {
        "question": "镶嵌首饰如何保养?",
        "answer": "镶嵌首饰需要定期检查宝石牢固度,避免与硬物碰撞。爪镶首饰容易挂衣物,需注意佩戴环境。建议定期到专业机构清洗和检查。"
    }
]


# ==================== 测试框架 ====================

class RAGEvaluator:
    """RAG系统评估器"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = None
        self.results = {
            "test_time": datetime.now().isoformat(),
            "document_upload": {},
            "knowledge_retrieval": [],
            "faq_matching": [],
            "conversation": [],
            "summary": {}
        }

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    # ==================== 工具方法 ====================

    def print_section(self, title: str):
        """打印分隔线"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80 + "\n")

    def print_result(self, success: bool, message: str):
        """打印测试结果"""
        icon = "✅" if success else "❌"
        print(f"{icon} {message}")

    # ==================== 1. 文档上传测试 ====================

    async def test_document_upload(
        self,
        doc_content: Dict[str, Any],
        chunk_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        测试文档上传

        Args:
            doc_content: 文档内容 {name, content, expected_queries}
            chunk_config: 自定义切分配置

        Returns:
            上传结果
        """
        self.print_section("📄 测试文档上传")

        # 创建临时文件
        temp_file = Path(f"/tmp/{doc_content['name']}")
        temp_file.write_text(doc_content['content'], encoding='utf-8')

        try:
            # 准备上传数据
            files = {'file': (doc_content['name'], open(temp_file, 'rb'), 'text/plain')}
            data = {
                'kb_id': TEST_KB_ID,
                'category': 'test'
            }

            # 如果有自定义切分配置
            if chunk_config:
                data['segment_config'] = json.dumps(chunk_config)
                logger.info(f"使用自定义切分配置: {chunk_config}")

            # 发送上传请求
            upload_start = time.time()
            response = await self.client.post(
                f"{self.base_url}/api/knowledge_base/documents/upload",
                files=files,
                data=data
            )
            upload_time = time.time() - upload_start

            result = response.json()

            if response.status_code == 200:
                doc_id = result.get('data', {}).get('doc_id')
                chunks_count = result.get('data', {}).get('chunks_count', 0)

                self.print_result(True, f"文档上传成功")
                print(f"  - 文档ID: {doc_id}")
                print(f"  - 切分数量: {chunks_count}")
                print(f"  - 上传耗时: {upload_time:.2f}秒")

                return {
                    "success": True,
                    "doc_id": doc_id,
                    "chunks_count": chunks_count,
                    "upload_time": upload_time
                }
            else:
                self.print_result(False, f"文档上传失败: {result}")
                return {"success": False, "error": result}

        except Exception as e:
            self.print_result(False, f"文档上传异常: {str(e)}")
            return {"success": False, "error": str(e)}

        finally:
            # 清理临时文件
            if temp_file.exists():
                temp_file.unlink()

    # ==================== 2. 知识库检索测试 ====================

    async def test_knowledge_retrieval(
        self,
        query: str,
        expected_relevant: bool = True,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        测试知识库检索

        Args:
            query: 测试查询
            expected_relevant: 是否期望检索到相关文档
            top_k: 返回Top-K结果

        Returns:
            检索结果
        """
        self.print_section(f"🔍 测试知识库检索: {query}")

        try:
            # 构造测试请求
            request_data = {
                "user_id": "test_evaluator",
                "employee_id": TEST_EMPLOYEE_ID,
                "query": query,
                "kb_ids": [TEST_KB_ID],
                "top_k": top_k
            }

            # 发送检索请求
            retrieve_start = time.time()
            response = await self.client.post(
                f"{self.base_url}/api/knowledge_base/retrieval/test",  # 假设有这个测试端点
                json=request_data
            )

            # 如果没有专门的测试端点,使用chat端点
            if response.status_code == 404:
                response = await self.client.post(
                    f"{self.base_url}/api/chat/message",
                    json={
                        "user_id": "test_evaluator",
                        "employee_id": TEST_EMPLOYEE_ID,
                        "query": query,
                        "context": {"test": True}
                    }
                )

            retrieve_time = time.time() - retrieve_start

            result = response.json()

            if response.status_code != 200:
                self.print_result(False, f"检索请求失败: {result}")
                return {"success": False, "error": result}

            # 提取检索结果
            data = result.get('data', {})
            retrieved_docs = data.get('retrieved_docs', [])
            relevance_score = data.get('relevance_score', 0.0)

            # 评估检索质量
            has_results = len(retrieved_docs) > 0
            top_score = retrieved_docs[0].get('score', 0.0) if has_results else 0.0

            # 判断是否符合预期
            is_correct = (has_results == expected_relevant) and \
                        (not expected_relevant or top_score > 0.5)

            self.print_result(
                is_correct,
                f"检索{'成功' if has_results else '无结果'}, "
                f"Top-1分数: {top_score:.4f}, "
                f"耗时: {retrieve_time:.2f}秒"
            )

            # 打印Top-3结果
            if has_results:
                print(f"\n  Top-3 检索结果:")
                for i, doc in enumerate(retrieved_docs[:3], 1):
                    content_preview = doc.get('content', '')[:100].replace('\n', ' ')
                    print(f"    {i}. [{doc.get('score', 0):.4f}] {content_preview}...")

            return {
                "success": True,
                "query": query,
                "expected_relevant": expected_relevant,
                "is_correct": is_correct,
                "retrieved_count": len(retrieved_docs),
                "top_score": top_score,
                "relevance_score": relevance_score,
                "retrieve_time": retrieve_time,
                "docs": retrieved_docs[:3]  # 只保存Top-3
            }

        except Exception as e:
            self.print_result(False, f"检索异常: {str(e)}")
            return {"success": False, "error": str(e)}

    # ==================== 3. FAQ匹配测试 ====================

    async def test_faq_matching(
        self,
        query: str,
        expected_faq_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        测试FAQ匹配

        Args:
            query: 测试查询
            expected_faq_id: 期望匹配的FAQ ID(可选)

        Returns:
            FAQ匹配结果
        """
        self.print_section(f"💬 测试FAQ匹配: {query}")

        try:
            request_data = {
                "user_id": "test_evaluator",
                "employee_id": TEST_EMPLOYEE_ID,
                "query": query,
                "context": {"test": True}
            }

            # 发送聊天请求
            chat_start = time.time()
            response = await self.client.post(
                f"{self.base_url}/api/chat/message",
                json=request_data
            )
            chat_time = time.time() - chat_start

            result = response.json()

            if response.status_code != 200:
                self.print_result(False, f"FAQ匹配失败: {result}")
                return {"success": False, "error": result}

            data = result.get('data', {})
            faq_matched = data.get('faq_matched')
            answer = data.get('answer', '')
            intent = data.get('intent', '')

            # 评估FAQ匹配效果
            is_faq_intent = (intent == 'faq_match')
            has_faq_info = faq_matched is not None

            self.print_result(
                is_faq_intent,
                f"意图: {intent}, "
                f"FAQ匹配: {'是' if has_faq_info else '否'}, "
                f"耗时: {chat_time:.2f}秒"
            )

            if has_faq_info:
                print(f"  - FAQ ID: {faq_matched.get('faq_id')}")
                print(f"  - 问题: {faq_matched.get('question_name')}")
                print(f"  - RRF分数: {faq_matched.get('rrf_score', 0):.4f}")
                print(f"  - 向量分数: {faq_matched.get('vector_score', 0):.4f}")
                print(f"  - 关键词分数: {faq_matched.get('keyword_score', 0):.4f}")
                print(f"  - 回答: {answer[:100]}...")

            # 检查是否匹配到期望的FAQ
            is_expected_match = True
            if expected_faq_id and has_faq_info:
                is_expected_match = (faq_matched.get('faq_id') == expected_faq_id)
                self.print_result(
                    is_expected_match,
                    f"期望FAQ匹配: {expected_faq_id}"
                )

            return {
                "success": True,
                "query": query,
                "is_faq_intent": is_faq_intent,
                "faq_matched": has_faq_info,
                "faq_data": faq_matched,
                "is_expected_match": is_expected_match,
                "answer_length": len(answer),
                "chat_time": chat_time
            }

        except Exception as e:
            self.print_result(False, f"FAQ匹配异常: {str(e)}")
            return {"success": False, "error": str(e)}

    # ==================== 4. 完整对话测试 ====================

    async def test_conversation(self, query: str) -> Dict[str, Any]:
        """
        测试完整对话流程

        Args:
            query: 用户查询

        Returns:
            对话结果
        """
        self.print_section(f"🤖 测试完整对话: {query}")

        try:
            request_data = {
                "user_id": "test_evaluator",
                "employee_id": TEST_EMPLOYEE_ID,
                "query": query,
                "context": {"test": True}
            }

            # 发送聊天请求
            conv_start = time.time()
            response = await self.client.post(
                f"{self.base_url}/api/chat/message",
                json=request_data
            )
            conv_time = time.time() - conv_start

            result = response.json()

            if response.status_code != 200:
                self.print_result(False, f"对话失败: {result}")
                return {"success": False, "error": result}

            data = result.get('data', {})
            answer = data.get('answer', '')
            confidence = data.get('confidence', 0.0)
            intent = data.get('intent', '')
            kb_used = data.get('kb_used', [])
            web_search_used = data.get('web_search_used', False)
            sources = data.get('sources', {})
            rag_sources = sources.get('rag_sources', [])
            web_sources = sources.get('web_sources', [])

            # 评估回答质量
            answer_quality = self._evaluate_answer_quality(answer, query)

            self.print_result(
                answer_quality['is_meaningful'],
                f"对话完成, 意图: {intent}, "
                f"置信度: {confidence:.2f}, "
                f"耗时: {conv_time:.2f}秒"
            )

            print(f"  - 知识库: {kb_used}")
            print(f"  - 联网搜索: {'是' if web_search_used else '否'}")
            print(f"  - RAG来源: {len(rag_sources)}条")
            print(f"  - Web来源: {len(web_sources)}条")
            print(f"\n  AI回答:")
            print(f"  {answer[:200]}...")

            # 详细质量评估
            print(f"\n  回答质量评估:")
            print(f"    - 有意义: {answer_quality['is_meaningful']}")
            print(f"    - 长度: {len(answer)}字符")
            print(f"    - 包含知识: {answer_quality['has_knowledge']}")

            return {
                "success": True,
                "query": query,
                "intent": intent,
                "confidence": confidence,
                "answer_length": len(answer),
                "is_meaningful": answer_quality['is_meaningful'],
                "has_knowledge": answer_quality['has_knowledge'],
                "kb_used": kb_used,
                "web_search_used": web_search_used,
                "rag_sources_count": len(rag_sources),
                "web_sources_count": len(web_sources),
                "conv_time": conv_time,
                "answer_preview": answer[:100]
            }

        except Exception as e:
            self.print_result(False, f"对话异常: {str(e)}")
            return {"success": False, "error": str(e)}

    def _evaluate_answer_quality(self, answer: str, query: str) -> Dict[str, bool]:
        """评估回答质量"""
        is_meaningful = (
            len(answer) > 20 and  # 长度合理
            not answer.startswith("抱歉") and  # 不是拒绝回答
            "不知道" not in answer and  # 不是说不知道
            "无法" not in answer
        )

        has_knowledge = (
            "工艺" in answer or
            "方法" in answer or
            "流程" in answer or
            "特点" in answer or
            "步骤" in answer
        )

        return {
            "is_meaningful": is_meaningful,
            "has_knowledge": has_knowledge
        }

    # ==================== 5. 生成报告 ====================

    def generate_report(self):
        """生成评估报告"""
        self.print_section("📊 评估报告")

        results = self.results

        # 1. 文档上传统计
        doc_upload = results.get('document_upload', {})
        print("\n1️⃣  文档上传测试")
        if doc_upload.get('success'):
            print(f"  ✅ 上传成功")
            print(f"  - 切分数量: {doc_upload.get('chunks_count', 0)}")
            print(f"  - 上传耗时: {doc_upload.get('upload_time', 0):.2f}秒")
        else:
            print(f"  ❌ 上传失败: {doc_upload.get('error', 'Unknown')}")

        # 2. 知识库检索统计
        retrieval_tests = results.get('knowledge_retrieval', [])
        print(f"\n2️⃣  知识库检索测试 ({len(retrieval_tests)}个)")

        if retrieval_tests:
            correct_count = sum(1 for t in retrieval_tests if t.get('is_correct', False))
            avg_score = sum(t.get('top_score', 0) for t in retrieval_tests) / len(retrieval_tests)
            avg_time = sum(t.get('retrieve_time', 0) for t in retrieval_tests) / len(retrieval_tests)

            print(f"  ✅ 准确率: {correct_count}/{len(retrieval_tests)} ({correct_count/len(retrieval_tests)*100:.1f}%)")
            print(f"  📈 平均分数: {avg_score:.4f}")
            print(f"  ⏱️  平均耗时: {avg_time:.2f}秒")

            # 详细结果
            for i, test in enumerate(retrieval_tests, 1):
                icon = "✅" if test.get('is_correct', False) else "❌"
                print(f"    {icon} {test.get('query', '')[:50]}")
                print(f"       分数={test.get('top_score', 0):.4f}, 耗时={test.get('retrieve_time', 0):.2f}秒")

        # 3. FAQ匹配统计
        faq_tests = results.get('faq_matching', [])
        print(f"\n3️⃣  FAQ匹配测试 ({len(faq_tests)}个)")

        if faq_tests:
            faq_match_count = sum(1 for t in faq_tests if t.get('faq_matched', False))
            avg_time = sum(t.get('chat_time', 0) for t in faq_tests) / len(faq_tests)

            print(f"  ✅ FAQ匹配率: {faq_match_count}/{len(faq_tests)} ({faq_match_count/len(faq_tests)*100:.1f}%)")
            print(f"  ⏱️  平均耗时: {avg_time:.2f}秒")

            for i, test in enumerate(faq_tests, 1):
                icon = "✅" if test.get('faq_matched', False) else "❌"
                print(f"    {icon} {test.get('query', '')[:50]}")

        # 4. 对话测试统计
        conv_tests = results.get('conversation', [])
        print(f"\n4️⃣  对话测试 ({len(conv_tests)}个)")

        if conv_tests:
            meaningful_count = sum(1 for t in conv_tests if t.get('is_meaningful', False))
            knowledge_count = sum(1 for t in conv_tests if t.get('has_knowledge', False))
            avg_confidence = sum(t.get('confidence', 0) for t in conv_tests) / len(conv_tests)
            avg_time = sum(t.get('conv_time', 0) for t in conv_tests) / len(conv_tests)

            print(f"  ✅ 有效回答率: {meaningful_count}/{len(conv_tests)} ({meaningful_count/len(conv_tests)*100:.1f}%)")
            print(f"  📚 知识覆盖率: {knowledge_count}/{len(conv_tests)} ({knowledge_count/len(conv_tests)*100:.1f}%)")
            print(f"  📊 平均置信度: {avg_confidence:.2f}")
            print(f"  ⏱️  平均耗时: {avg_time:.2f}秒")

        # 5. 总体评分
        print(f"\n5️⃣  总体评分")
        overall_score = self._calculate_overall_score(results)
        print(f"  🎯 综合得分: {overall_score:.1f}/100")

        grade = self._get_grade(overall_score)
        print(f"  🏆 评级: {grade}")

        # 保存报告到文件
        report_path = f"rag_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  📄 详细报告已保存: {report_path}")

    def _calculate_overall_score(self, results: Dict) -> float:
        """计算总体评分"""
        score = 0.0

        # 文档上传 (20分)
        if results.get('document_upload', {}).get('success'):
            score += 20

        # 知识库检索 (30分)
        retrieval_tests = results.get('knowledge_retrieval', [])
        if retrieval_tests:
            correct_rate = sum(1 for t in retrieval_tests if t.get('is_correct', False)) / len(retrieval_tests)
            score += correct_rate * 30

        # FAQ匹配 (20分)
        faq_tests = results.get('faq_matching', [])
        if faq_tests:
            faq_match_rate = sum(1 for t in faq_tests if t.get('faq_matched', False)) / len(faq_tests)
            score += faq_match_rate * 20

        # 对话质量 (30分)
        conv_tests = results.get('conversation', [])
        if conv_tests:
            meaningful_rate = sum(1 for t in conv_tests if t.get('is_meaningful', False)) / len(conv_tests)
            score += meaningful_rate * 30

        return score

    def _get_grade(self, score: float) -> str:
        """获取评级"""
        if score >= 90:
            return "优秀 (A)"
        elif score >= 80:
            return "良好 (B)"
        elif score >= 70:
            return "中等 (C)"
        elif score >= 60:
            return "及格 (D)"
        else:
            return "不及格 (F)"


# ==================== 主测试流程 ====================

async def main():
    """主测试流程"""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                   RAG系统端到端评估测试                                      ║
║                                                                              ║
║  功能: 文档上传 | 知识检索 | FAQ匹配 | 对话生成 | 质量评估                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    async with RAGEvaluator(API_BASE_URL) as evaluator:
        # ========== 1. 文档上传测试 ==========
        print("\n" + "🚀" * 40)
        print("第一阶段: 文档上传测试")
        print("🚀" * 40)

        # 测试默认切分
        doc_result = await evaluator.test_document_upload(
            TEST_DOCUMENTS[0],
            chunk_config=None  # 使用默认配置
        )

        evaluator.results['document_upload'] = doc_result

        # 如果默认切分成功,测试自定义切分
        if doc_result.get('success'):
            await evaluator.test_document_upload(
                TEST_DOCUMENTS[1],
                chunk_config={
                    'is_space_flag': 1,
                    'is_menu_flag': 0,
                    'segment_type': 1,
                    'is_segment_union_flag': 1,
                    'segment_union_max_length': 500,
                    'segment_identifier_type': 0,
                    'identifier_default': '1111111'
                }
            )

        # 等待文档处理完成
        print("\n⏳ 等待文档处理完成(5秒)...")
        await asyncio.sleep(5)

        # ========== 2. 知识库检索测试 ==========
        print("\n" + "🔍" * 40)
        print("第二阶段: 知识库检索测试")
        print("🔍" * 40)

        for doc in TEST_DOCUMENTS:
            for query in doc['expected_queries']:
                result = await evaluator.test_knowledge_retrieval(
                    query=query,
                    expected_relevant=True
                )
                evaluator.results['knowledge_retrieval'].append(result)
                await asyncio.sleep(1)  # 避免请求过快

        # ========== 3. FAQ匹配测试 ==========
        print("\n" + "💬" * 40)
        print("第三阶段: FAQ匹配测试")
        print("💬" * 40)

        for faq in TEST_FAQS:
            result = await evaluator.test_faq_matching(
                query=faq['question']
            )
            evaluator.results['faq_matching'].append(result)
            await asyncio.sleep(1)

        # ========== 4. 完整对话测试 ==========
        print("\n" + "🤖" * 40)
        print("第四阶段: 完整对话测试")
        print("🤖" * 40)

        test_conversations = [
            "失蜡铸造的工艺特点是什么?",
            "爪镶和包镶有什么区别?",
            "首饰镶嵌有哪些质量检验标准?",
            "失蜡铸造应用在哪些领域?"
        ]

        for query in test_conversations:
            result = await evaluator.test_conversation(query)
            evaluator.results['conversation'].append(result)
            await asyncio.sleep(1)

        # ========== 5. 生成报告 ==========
        print("\n" + "📊" * 40)
        print("第五阶段: 生成评估报告")
        print("📊" * 40)

        evaluator.generate_report()

    print("\n" + "=" * 80)
    print("  测试完成!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
