"""
RAG端到端完整评估测试脚本

功能:
1. 创建知识库
2. 上传PDF文档(指定目录下所有PDF)
3. 创建数字员工并绑定知识库
4. 等待文档处理完成
5. 从MinerU接口获取markdown内容并生成问答对
6. 调用流式对话接口进行问答
7. 评估RAG质量并保存结果

使用方法:
    cd ai-service
    python ../venv/Scripts/python.exe scripts/e2e_rag_evaluation.py
"""
import asyncio
import aiohttp
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class E2ERAGEvaluation:
    """RAG端到端评估"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        pdf_dir: str = r"D:\zenking_work\期刊文件",
        api_key: str = "y2tJW3P0bvZIxw6pGuV2FrcT0C1wyUfg2ldweEaDYN4"
    ):
        self.base_url = base_url
        self.pdf_dir = Path(pdf_dir)
        self.api_key = api_key
        self.headers = {"X-API-Key": api_key}

        # 测试结果
        self.test_run_id = f"rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = {
            "test_run_id": self.test_run_id,
            "timestamp": datetime.now().isoformat(),
            "base_url": base_url,
            "pdf_dir": pdf_dir,
            "kb_info": None,
            "employee_info": None,
            "uploaded_files": [],
            "mineru_jobs": [],
            "qa_pairs": [],
            "chat_responses": [],
            "evaluation": {}
        }

    async def _request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """发送HTTP请求"""
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                url,
                headers=headers or self.headers,
                **kwargs
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise Exception(f"Request failed: {response.status} - {error_text}")
                return await response.json()

    async def create_knowledge_base(
        self,
        name: str = "RAG端到端评估知识库",
        description: str = "用于RAG端到端评估的测试知识库",
        category: str = "测试"
    ) -> str:
        """1. 创建知识库"""
        print(f"\n[1] 创建知识库: {name}")

        url = f"{self.base_url}/api/knowledge-base/create"

        data = aiohttp.FormData()
        data.add_field("name", name)
        data.add_field("description", description)
        data.add_field("category", category)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=self.headers,
                data=data
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise Exception(f"创建知识库失败: {error_text}")

                result = await response.json()
                kb_id = result.get("data", {}).get("kb_id") or result.get("kb_id")

                self.results["kb_info"] = {
                    "kb_id": kb_id,
                    "name": name,
                    "description": description,
                    "category": category
                }

                print(f"    知识库创建成功: {kb_id}")
                return kb_id

    async def upload_documents(
        self,
        kb_id: str,
        use_mineru: bool = True
    ) -> List[str]:
        """2. 上传所有PDF文档"""
        print(f"\n[2] 上传文档到知识库: {kb_id}")
        print(f"    PDF目录: {self.pdf_dir}")

        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"    警告: 没有找到PDF文件")
            return []

        print(f"    找到 {len(pdf_files)} 个PDF文件")

        uploaded_files = []
        url = f"{self.base_url}/api/knowledge-base/documents/upload"

        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"    [{i}/{len(pdf_files)}] 上传: {pdf_file.name}")

            data = aiohttp.FormData()
            data.add_field("kb_id", kb_id)
            data.add_field("category", "测试文档")
            data.add_field("use_mineru", "true" if use_mineru else "false")

            with open(pdf_file, "rb") as f:
                data.add_field(
                    "files",
                    f,
                    filename=pdf_file.name,
                    content_type="application/pdf"
                )

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        headers=self.headers,
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=600)  # 10分钟超时
                    ) as response:
                        if response.status >= 400:
                            error_text = await response.text()
                            print(f"        失败: {error_text}")
                            continue

                        result = await response.json()
                        doc_id = result.get("data", {}).get("doc_id")
                        task_id = result.get("data", {}).get("task_id")

                        uploaded_files.append({
                            "filename": pdf_file.name,
                            "doc_id": doc_id,
                            "task_id": task_id
                        })
                        print(f"        成功: doc_id={doc_id}")

            except Exception as e:
                print(f"        异常: {e}")

        self.results["uploaded_files"] = uploaded_files
        print(f"\n    成功上传 {len(uploaded_files)}/{len(pdf_files)} 个文档")
        return [f["doc_id"] for f in uploaded_files]

    async def create_employee(
        self,
        kb_id: str,
        employee_id: str = "rag_eval_employee",
        name: str = "RAG评估助手"
    ) -> Dict[str, Any]:
        """3. 创建数字员工并绑定知识库"""
        print(f"\n[3] 创建数字员工并绑定知识库")

        url = f"{self.base_url}/api/ai/digital-employee/create"

        employee_data = {
            "employee_id": employee_id,
            "name": name,
            "domain": "测试评估",
            "role": "RAG评估测试助手",
            "description": "用于RAG端到端评估的测试数字员工",
            "personality": {
                "tone": "professional",
                "style": "friendly",
                "language": "zh-CN",
                "formality": "moderate"
            },
            "capabilities": {
                "kb_ids": [kb_id],
                "web_search_enabled": False,  # 禁用网络搜索,只测试知识库
                "max_context_turns": 10
            },
            "greeting": "您好！我是RAG评估测试助手，请问有什么可以帮助您？",
            "hot_questions": [
                "知识库中有哪些文档？",
                "请简要介绍一下这些文档的主要内容"
            ]
        }

        response = await self._request("POST", url, json=employee_data)

        self.results["employee_info"] = {
            "employee_id": employee_id,
            "name": name,
            "kb_id": kb_id
        }

        print(f"    数字员工创建成功: {employee_id}")
        print(f"    绑定知识库: {kb_id}")
        return self.results["employee_info"]

    async def wait_for_documents_completion(
        self,
        kb_id: str,
        timeout: int = 1800,  # 30分钟超时
        check_interval: int = 10  # 每10秒检查一次
    ) -> bool:
        """4. 等待所有文档处理完成"""
        print(f"\n[4] 等待文档处理完成...")

        start_time = time.time()
        last_doc_count = 0

        while time.time() - start_time < timeout:
            url = f"{self.base_url}/api/knowledge-base/documents/list"
            params = {
                "kb_id": kb_id,
                "page": 1,
                "page_size": 100
            }

            try:
                response = await self._request("GET", url, params=params)
                documents = response.get("data", {}).get("documents", [])

                completed = sum(1 for d in documents if d.get("status") == "completed")
                processing = sum(1 for d in documents if d.get("status") == "processing")
                failed = sum(1 for d in documents if d.get("status") == "failed")
                total = len(documents)

                if total > last_doc_count:
                    print(f"    检测到 {total} 个文档")

                last_doc_count = total

                print(f"    进度: {completed}/{total} 完成, {processing} 处理中, {failed} 失败", end="\r")

                if completed + failed >= total:
                    print(f"\n    所有文档处理完成! (成功: {completed}, 失败: {failed})")
                    return True

                await asyncio.sleep(check_interval)

            except Exception as e:
                print(f"\n    检查文档状态失败: {e}")
                await asyncio.sleep(check_interval)

        print(f"\n    超时! 部分文档未完成处理")
        return False

    async def get_mineru_jobs_and_content(self) -> List[Dict[str, Any]]:
        """5. 获取MinerU任务和markdown内容"""
        print(f"\n[5] 获取MinerU任务和markdown内容")

        # 获取任务列表
        jobs_url = f"{self.base_url}/api/mineru/jobs"

        try:
            response = await self._request("GET", jobs_url, params={"limit": 100})
            jobs = response.get("data", [])

            if not jobs:
                print(f"    没有找到MinerU任务")
                return []

            print(f"    找到 {len(jobs)} 个MinerU任务")

            job_contents = []

            for job in jobs:
                job_id = job.get("job_id")
                file_name = job.get("file_name", "unknown")
                status = job.get("status", "unknown")

                if status != "completed":
                    print(f"    跳过未完成的任务: {file_name} ({status})")
                    continue

                # 获取任务的markdown内容
                markdown_url = f"{self.base_url}/api/mineru/jobs/{job_id}/markdown"
                try:
                    md_response = await self._request("GET", markdown_url)
                    markdown_content = md_response.get("data", {}).get("markdown", "")

                    job_contents.append({
                        "job_id": job_id,
                        "file_name": file_name,
                        "markdown": markdown_content,
                        "total_pages": job.get("total_pages"),
                        "status": status,
                        "content_length": len(markdown_content)
                    })
                    print(f"    获取 {file_name}: {len(markdown_content)} 字符")

                except Exception as e:
                    print(f"    获取 {file_name} 失败: {e}")

            self.results["mineru_jobs"] = job_contents
            return job_contents

        except Exception as e:
            print(f"    获取MinerU任务失败: {e}")
            return []

    async def generate_qa_pairs(
        self,
        job_contents: List[Dict[str, Any]],
        questions_per_doc: int = 3
    ) -> List[Dict[str, Any]]:
        """6. 基于文档内容生成问答对"""
        print(f"\n[6] 基于文档内容生成问答对")

        from app.core.config import settings
        from langchain_openai import ChatOpenAI
        from langchain_community.chat_models import ChatOllama

        # 使用Grader LLM生成问答对
        if settings.use_ollama:
            llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0
            )
        else:
            llm = ChatOpenAI(
                base_url=settings.openai_api_base,
                api_key=settings.siliconflow_api_key,
                model=settings.openai_model,
                temperature=0
            )

        qa_pairs = []

        for i, doc in enumerate(job_contents, 1):
            if not doc.get("markdown") or len(doc["markdown"]) < 500:
                print(f"    [{i}/{len(job_contents)}] 跳过内容过短的文档: {doc['file_name']}")
                continue

            # 截取部分内容生成问题
            content_sample = doc["markdown"][:3000]

            generate_prompt = f"""请基于以下文档内容生成 {questions_per_doc} 个问答对。

文档名称: {doc['file_name']}

文档内容片段:
{content_sample}

要求:
1. 问题应该能从文档内容中找到答案
2. 问题要具体,不要过于宽泛
3. 答案要简洁准确,直接引用文档中的信息
4. 问题类型可以包括: 主要内容、关键概念、具体数据、流程步骤等
5. 以JSON数组格式返回

返回格式:
[
  {{"question": "问题1", "answer": "答案1"}},
  {{"question": "问题2", "answer": "答案2"}},
  {{"question": "问题3", "answer": "答案3"}}
]

只返回JSON数组,不要有其他内容:"""

            try:
                response = await llm.ainvoke(generate_prompt)
                response_text = response.content.strip()

                # 清理可能的markdown代码块标记
                if response_text.startswith("```"):
                    parts = response_text.split("```")
                    response_text = parts[1] if len(parts) > 1 else response_text
                    if response_text.startswith("json"):
                        response_text = response_text[4:]

                import json
                doc_qa_pairs = json.loads(response_text)

                for qa in doc_qa_pairs:
                    qa_pairs.append({
                        "question": qa["question"],
                        "reference_answer": qa["answer"],
                        "source_file": doc["file_name"],
                        "source_job_id": doc["job_id"]
                    })

                print(f"    [{i}/{len(job_contents)}] {doc['file_name']}: 生成 {len(doc_qa_pairs)} 个问答对")

            except Exception as e:
                print(f"    [{i}/{len(job_contents)}] {doc['file_name']}: 生成失败 - {e}")
                # 手动创建通用问答对
                qa_pairs.append({
                    "question": f"请简要介绍《{doc['file_name'].replace('.pdf', '')}》的主要内容",
                    "reference_answer": f"请基于文档《{doc['file_name']}》的内容回答这个问题。",
                    "source_file": doc["file_name"],
                    "source_job_id": doc["job_id"]
                })

        self.results["qa_pairs"] = qa_pairs
        print(f"\n    总共生成 {len(qa_pairs)} 个问答对")

        # 保存问答对到文件
        qa_file = Path(__file__).parent / f"rag_eval_qa_{self.test_run_id}.json"
        with open(qa_file, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        print(f"    问答对已保存到: {qa_file}")

        return qa_pairs

    async def chat_streaming(
        self,
        question: str,
        employee_id: str,
        session_id: str
    ) -> str:
        """使用流式接口进行问答"""
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "user_id": "rag_eval_test",
            "employee_id": employee_id,
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": question}
            ],
            "stream": True
        }

        full_response = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers={**self.headers, "Content-Type": "application/json"},
                json=payload
            ) as response:
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                            if data.get("type") == "token":
                                full_response += data.get("content", "")
                        except json.JSONDecodeError:
                            pass

        return full_response

    async def test_chat_responses(
        self,
        qa_pairs: List[Dict[str, Any]],
        employee_id: str
    ) -> List[Dict[str, Any]]:
        """7. 测试问答对并收集响应"""
        print(f"\n[7] 测试问答接口...")

        session_id = f"rag_eval_{int(time.time())}"
        chat_responses = []

        for i, qa in enumerate(qa_pairs, 1):
            print(f"    [{i}/{len(qa_pairs)}] {qa['question'][:50]}...")

            try:
                start_time = time.time()
                answer = await self.chat_streaming(
                    qa["question"],
                    employee_id,
                    session_id
                )
                elapsed = time.time() - start_time

                chat_responses.append({
                    "question": qa["question"],
                    "reference_answer": qa["reference_answer"],
                    "generated_answer": answer,
                    "source_file": qa["source_file"],
                    "response_time_ms": int(elapsed * 1000)
                })

                print(f"        答案长度: {len(answer)} 字符, 耗时: {elapsed:.2f}s")

            except Exception as e:
                print(f"        失败: {e}")
                chat_responses.append({
                    "question": qa["question"],
                    "reference_answer": qa["reference_answer"],
                    "generated_answer": f"错误: {str(e)}",
                    "source_file": qa["source_file"],
                    "error": str(e),
                    "response_time_ms": 0
                })

        self.results["chat_responses"] = chat_responses
        print(f"\n    完成 {len(chat_responses)} 个问答测试")
        return chat_responses

    async def evaluate_rag_quality(
        self,
        chat_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """8. 评估RAG质量"""
        print(f"\n[8] 评估RAG质量...")

        from app.core.config import settings
        from langchain_openai import ChatOpenAI
        from langchain_community.chat_models import ChatOllama

        # 使用Grader LLM进行评估
        if settings.use_ollama:
            grader_llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_grader_model,
                temperature=0,
                format="json"
            )
        else:
            grader_llm = ChatOpenAI(
                base_url=settings.openai_api_base,
                api_key=settings.siliconflow_api_key,
                model=settings.openai_grader_model,
                temperature=0,
                model_kwargs={"response_format": {"type": "json_object"}}
            )

        scores = {
            "relevance": [],
            "accuracy": [],
            "completeness": [],
            "overall": []
        }

        detailed_evaluations = []

        for response in chat_responses:
            if "error" in response:
                continue

            eval_prompt = f"""请评估以下问答的质量。

用户问题: {response['question']}

参考答案: {response['reference_answer']}

生成的答案: {response['generated_answer']}

评估标准:
1. 相关性(0-1): 生成的答案是否与问题相关
2. 准确性(0-1): 生成的答案是否与参考答案一致,事实是否正确
3. 完整性(0-1): 答案是否完整回答了问题

请以JSON格式返回评估结果:
{{
    "relevance": 0.9,
    "accuracy": 0.8,
    "completeness": 0.7,
    "reasoning": "评估理由"
}}

只返回JSON,不要有其他内容:"""

            try:
                result = await grader_llm.ainvoke(eval_prompt)
                result_text = result.content.strip()

                import json
                eval_result = json.loads(result_text)

                relevance = eval_result.get("relevance", 0)
                accuracy = eval_result.get("accuracy", 0)
                completeness = eval_result.get("completeness", 0)
                overall = (relevance + accuracy + completeness) / 3

                scores["relevance"].append(relevance)
                scores["accuracy"].append(accuracy)
                scores["completeness"].append(completeness)
                scores["overall"].append(overall)

                detailed_evaluations.append({
                    "question": response["question"][:50],
                    "relevance": relevance,
                    "accuracy": accuracy,
                    "completeness": completeness,
                    "overall": overall,
                    "reasoning": eval_result.get("reasoning", "")
                })

                print(f"    评分: 相关={relevance:.2f}, 准确={accuracy:.2f}, "
                      f"完整={completeness:.2f}, 综合={overall:.2f}")

            except Exception as e:
                print(f"    评估失败: {e}")

        # 计算平均分
        evaluation = {}
        if scores["relevance"]:
            evaluation["avg_relevance"] = sum(scores["relevance"]) / len(scores["relevance"])
            evaluation["avg_accuracy"] = sum(scores["accuracy"]) / len(scores["accuracy"])
            evaluation["avg_completeness"] = sum(scores["completeness"]) / len(scores["completeness"])
            evaluation["avg_overall"] = sum(scores["overall"]) / len(scores["overall"])
            evaluation["total_evaluated"] = len(scores["overall"])
            evaluation["score_distribution"] = scores
            evaluation["detailed_evaluations"] = detailed_evaluations

        self.results["evaluation"] = evaluation

        print(f"\n    === RAG质量评估报告 ===")
        if evaluation:
            print(f"    评估数量: {evaluation['total_evaluated']}")
            print(f"    平均相关性: {evaluation['avg_relevance']:.3f}")
            print(f"    平均准确性: {evaluation['avg_accuracy']:.3f}")
            print(f"    平均完整性: {evaluation['avg_completeness']:.3f}")
            print(f"    综合得分: {evaluation['avg_overall']:.3f}")

            # 评估等级
            overall = evaluation['avg_overall']
            if overall >= 0.8:
                grade = "优秀"
            elif overall >= 0.6:
                grade = "良好"
            elif overall >= 0.4:
                grade = "及格"
            else:
                grade = "不及格"

            print(f"    评估等级: {grade}")

        return evaluation

    async def save_results(self):
        """9. 保存完整测试结果"""
        results_file = Path(__file__).parent / f"rag_eval_results_{self.test_run_id}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"\n[9] 完整结果已保存到: {results_file}")
        print(f"    测试ID: {self.test_run_id}")

        # 同时保存一份简化的报告
        summary_file = Path(__file__).parent / f"rag_eval_summary_{self.test_run_id}.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("RAG端到端评估测试报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"测试时间: {self.results['timestamp']}\n")
            f.write(f"测试ID: {self.test_run_id}\n")
            f.write(f"Base URL: {self.base_url}\n\n")

            if self.results.get("kb_info"):
                f.write(f"知识库: {self.results['kb_info']['name']} ({self.results['kb_info']['kb_id']})\n")

            if self.results.get("employee_info"):
                f.write(f"数字员工: {self.results['employee_info']['name']} ({self.results['employee_info']['employee_id']})\n")

            f.write(f"\n上传文档数: {len(self.results.get('uploaded_files', []))}\n")
            f.write(f"MinerU任务数: {len(self.results.get('mineru_jobs', []))}\n")
            f.write(f"生成问答对: {len(self.results.get('qa_pairs', []))}\n")
            f.write(f"测试响应数: {len(self.results.get('chat_responses', []))}\n\n")

            if self.results.get("evaluation"):
                eval_ = self.results["evaluation"]
                f.write("=== RAG质量评估 ===\n")
                f.write(f"评估数量: {eval_.get('total_evaluated', 0)}\n")
                f.write(f"平均相关性: {eval_.get('avg_relevance', 0):.3f}\n")
                f.write(f"平均准确性: {eval_.get('avg_accuracy', 0):.3f}\n")
                f.write(f"平均完整性: {eval_.get('avg_completeness', 0):.3f}\n")
                f.write(f"综合得分: {eval_.get('avg_overall', 0):.3f}\n")

            # 写入详细问答评估
            if eval_.get("detailed_evaluations"):
                f.write("\n=== 详细问答评估 ===\n")
                for item in eval_["detailed_evaluations"]:
                    f.write(f"\n问题: {item['question']}\n")
                    f.write(f"  相关性: {item['relevance']:.2f}\n")
                    f.write(f"  准确性: {item['accuracy']:.2f}\n")
                    f.write(f"  完整性: {item['completeness']:.2f}\n")
                    f.write(f"  综合: {item['overall']:.2f}\n")
                    f.write(f"  理由: {item['reasoning'][:100]}\n")

        print(f"    简化报告已保存到: {summary_file}")

    async def run_full_evaluation(self):
        """运行完整的端到端评估"""
        print("=" * 60)
        print("RAG端到端评估测试")
        print("=" * 60)

        try:
            # 1. 创建知识库
            kb_id = await self.create_knowledge_base()

            # 2. 上传文档
            await self.upload_documents(kb_id)

            # 3. 创建数字员工
            await self.create_employee(kb_id)

            # 4. 等待文档处理完成
            await self.wait_for_documents_completion(kb_id)

            # 5. 获取MinerU内容
            job_contents = await self.get_mineru_jobs_and_content()

            if not job_contents:
                print("\n    没有可用的文档内容,测试终止")
                return

            # 6. 生成问答对
            qa_pairs = await self.generate_qa_pairs(job_contents)

            if not qa_pairs:
                print("\n    没有生成问答对,测试终止")
                return

            # 7. 测试问答
            await self.test_chat_responses(qa_pairs, self.results["employee_info"]["employee_id"])

            # 8. 评估质量
            await self.evaluate_rag_quality(self.results["chat_responses"])

            # 9. 保存结果
            await self.save_results()

        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 60)
        print("测试完成!")
        print("=" * 60)


async def main():
    """主函数"""
    test = E2ERAGEvaluation()

    # 检查PDF目录
    if not test.pdf_dir.exists():
        print(f"错误: PDF目录不存在: {test.pdf_dir}")
        print(f"当前目录: {Path.cwd()}")
        return

    print(f"PDF目录: {test.pdf_dir}")
    print(f"Base URL: {test.base_url}")

    # 运行完整评估
    await test.run_full_evaluation()


if __name__ == "__main__":
    asyncio.run(main())
