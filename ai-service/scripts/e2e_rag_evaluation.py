"""
RAG端到端完整评估测试脚本

功能:
1. 创建知识库
2. 上传PDF文档(指定目录下所有PDF)
3. 加载数字员工
4. 等待文档处理完成
5. 从MinerU接口获取markdown内容并生成问答对
6. 调用流式对话接口进行问答
7. 评估RAG质量并保存结果

使用方法:
    # 完整运行(包括创建知识库、上传文档等所有步骤)
    cd ai-service
    python ../venv/Scripts/python.exe scripts/e2e_rag_evaluation.py

    # 跳过初始设置，直接从步骤4开始(适用于知识库已创建、文档已上传的情况)
    cd ai-service
    python ../venv/Scripts/python.exe scripts/e2e_rag_evaluation.py --skip-setup
"""
import asyncio
import aiohttp
import json
import time
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class E2ELogger:
    """端到端评估专用日志记录器"""

    def __init__(self, log_dir: Path = None):
        """
        初始化日志记录器

        Args:
            log_dir: 日志目录,默认为脚本目录下的logs文件夹
        """
        if log_dir is None:
            log_dir = Path(__file__).parent / "logs"

        log_dir.mkdir(exist_ok=True)

        # 创建日志文件名(带时间戳)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"rag_eval_{timestamp}.log"

        # 配置日志格式
        self.logger = logging.getLogger("E2ERAGEvaluation")
        self.logger.setLevel(logging.DEBUG)  # 设置为DEBUG级别以支持debug日志

        # 清除已有的handlers
        self.logger.handlers.clear()

        # 创建格式化器
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 文件Handler - 记录所有日志
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 控制台Handler - 只显示重要信息
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.log_file = log_file
        self.logger.info("=" * 80)
        self.logger.info("RAG端到端评估测试 - 日志系统初始化完成")
        self.logger.info(f"日志文件: {log_file}")
        self.logger.info("=" * 80)

    def info(self, msg: str, **kwargs):
        """记录信息级别日志"""
        # 格式化额外的键值对参数
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        if extra_info:
            msg = f"{msg} | {extra_info}"
        self.logger.info(msg)

    def warning(self, msg: str, **kwargs):
        """记录警告级别日志"""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        if extra_info:
            msg = f"{msg} | {extra_info}"
        self.logger.warning(msg)

    def error(self, msg: str, **kwargs):
        """记录错误级别日志"""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        if extra_info:
            msg = f"{msg} | {extra_info}"
        self.logger.error(msg)

    def success(self, msg: str, **kwargs):
        """记录成功信息"""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        if extra_info:
            msg = f"✓ {msg} | {extra_info}"
        else:
            msg = f"✓ {msg}"
        self.logger.info(msg)

    def step(self, step_num: int, step_name: str):
        """记录步骤信息"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"[步骤 {step_num}] {step_name}")
        self.logger.info("=" * 80)

    def debug(self, msg: str, **kwargs):
        """记录调试级别日志"""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        if extra_info:
            msg = f"{msg} | {extra_info}"
        self.logger.debug(msg)

    def close(self):
        """关闭日志系统"""
        self.logger.info("=" * 80)
        self.logger.info("日志记录完成")
        self.logger.info("=" * 80)
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


class E2ERAGEvaluation:
    """RAG端到端评估"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        pdf_dir: str = r"D:\\zenking_work\\期刊文件",
        api_key: str = "y2tJW3P0bvZIxw6pGuV2FrcT0C1wyUfg2ldweEaDYN4"
    ):
        self.base_url = base_url
        self.pdf_dir = Path(pdf_dir)
        self.api_key = api_key
        self.headers = {"X-API-Key": api_key}

        # 初始化日志系统
        self.logger = E2ELogger()

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

        self.logger.info("初始化RAG端到端评估", test_run_id=self.test_run_id)
        self.logger.info("配置信息", base_url=base_url, pdf_dir=pdf_dir)

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
        kb_id: str = "kb_e2bc0ea588c4",
        name: str = "RAG端到端评估知识库",
        description: str = "用于RAG端到端评估的测试知识库"
    ) -> str:
        """1. 使用现有知识库"""
        self.logger.step(1, "使用现有知识库")
        self.logger.info("知识库信息", kb_id=kb_id, name=name, description=description)

        # 直接使用现有的知识库
        self.results["kb_info"] = {
            "kb_id": kb_id,
            "name": name,
            "description": description,
            "existing": True
        }

        return kb_id

    async def upload_documents(
        self,
        kb_id: str,
        use_mineru: bool = True
    ) -> List[str]:
        """2. 上传所有PDF文档"""
        self.logger.step(2, "上传文档到知识库")
        self.logger.info("开始上传文档", kb_id=kb_id, pdf_dir=str(self.pdf_dir), use_mineru=use_mineru)

        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            self.logger.warning("没有找到PDF文件", pdf_dir=str(self.pdf_dir))
            return []

        self.logger.info(f"找到PDF文件", count=len(pdf_files))

        uploaded_files = []
        url = f"{self.base_url}/api/knowledge-base/documents/upload"

        # 创建单个session用于所有上传
        async with aiohttp.ClientSession() as session:
            for i, pdf_file in enumerate(pdf_files, 1):
                self.logger.info(f"开始上传", file_name=pdf_file.name, index=i, total=len(pdf_files))

                try:
                    # 读取文件内容
                    with open(pdf_file, "rb") as f:
                        file_content = f.read()

                    self.logger.debug(f"读取文件成功", file_name=pdf_file.name, size=len(file_content))

                    # 创建FormData
                    data = aiohttp.FormData()
                    data.add_field("kb_id", kb_id)
                    data.add_field("category", "测试文档")
                    data.add_field("use_mineru", "true" if use_mineru else "false")

                    # 添加文件(使用BytesIO)
                    from io import BytesIO
                    file_buffer = BytesIO(file_content)
                    data.add_field(
                        "files",
                        file_buffer.getvalue(),
                        filename=pdf_file.name,
                        content_type="application/pdf"
                    )

                    # 发送请求
                    async with session.post(
                        url,
                        headers=self.headers,
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=600)  # 10分钟超时
                    ) as response:
                        if response.status >= 400:
                            error_text = await response.text()
                            self.logger.error(f"上传失败", file_name=pdf_file.name, status=response.status, error=error_text[:200])
                            continue

                        result = await response.json()
                        # 异步上传返回task_id列表,需要从data字段获取
                        data = result.get("data", result)
                        tasks = data.get("tasks", []) if isinstance(data, dict) else []
                        task_id = tasks[0] if tasks else result.get("tasks", [None])[0]

                        uploaded_files.append({
                            "filename": pdf_file.name,
                            "doc_id": None,  # 需要通过文档列表查询获取
                            "task_id": task_id
                        })
                        self.logger.success(f"上传成功", file_name=pdf_file.name, task_id=task_id)

                except Exception as e:
                    self.logger.error(f"上传异常", file_name=pdf_file.name, error=str(e))
                    import traceback
                    self.logger.error(f"详细错误", file_name=pdf_file.name, traceback=traceback.format_exc()[:500])

        self.results["uploaded_files"] = uploaded_files
        self.logger.info(f"上传完成", success_count=len(uploaded_files), total_count=len(pdf_files))

        # 查询文档列表获取doc_id
        await self._fetch_doc_ids(kb_id, uploaded_files)

        return [f["doc_id"] for f in uploaded_files if f.get("doc_id")]

    async def _fetch_doc_ids(self, kb_id: str, uploaded_files: List[Dict[str, Any]]):
        """通过文档列表API获取实际的doc_id"""
        self.logger.info("查询文档ID", kb_id=kb_id)

        url = f"{self.base_url}/api/knowledge-base/documents/list"
        params = {
            "kb_id": kb_id,
            "page": 1,
            "page_size": 100
        }

        try:
            from urllib.parse import unquote

            response = await self._request("GET", url, params=params)
            # API返回字段名是items而不是documents
            documents = response.get("data", {}).get("items", response.get("data", {}).get("documents", []))

            self.logger.debug(f"API返回文档数量", count=len(documents))

            # 创建filename到doc_id的映射 (需要URL解码)
            filename_to_doc_id = {}
            for doc in documents:
                # 文件名可能是URL编码的,需要解码
                filename = doc.get("filename", "")
                try:
                    filename = unquote(filename)
                except:
                    pass  # 如果解码失败,使用原始文件名

                doc_id = doc.get("doc_id")
                if filename and doc_id:
                    filename_to_doc_id[filename] = doc_id
                    self.logger.debug("文档映射", filename=filename, doc_id=doc_id)

            # 更新uploaded_files中的doc_id
            matched = 0
            for uploaded_file in uploaded_files:
                filename = uploaded_file["filename"]
                if filename in filename_to_doc_id:
                    uploaded_file["doc_id"] = filename_to_doc_id[filename]
                    matched += 1
                    self.logger.debug("匹配成功", filename=filename, doc_id=uploaded_file["doc_id"])
                else:
                    self.logger.debug("未匹配", filename=filename)

            self.logger.success("文档ID查询完成", matched=matched, total=len(uploaded_files))

        except Exception as e:
            self.logger.error("查询文档ID失败", error=str(e))

    async def load_employee(
        self,
        employee_id: str = "financial_analyst"
    ) -> Dict[str, Any]:
        """3. 加载现有数字员工"""
        self.logger.step(3, "加载现有数字员工")

        url = f"{self.base_url}/api/ai/digital-employee/detail/{employee_id}"

        self.logger.info("获取数字员工信息", employee_id=employee_id)

        response = await self._request("GET", url)
        employee_data = response.get("data", {})

        if not employee_data:
            raise Exception(f"Employee not found: {employee_id}")

        self.results["employee_info"] = {
            "employee_id": employee_data.get("employee_id"),
            "name": employee_data.get("name"),
            "domain": employee_data.get("domain"),
            "role": employee_data.get("role"),
            "description": employee_data.get("description"),
            "kb_ids": employee_data.get("kb_ids", employee_data.get("capabilities", {}).get("kb_ids", [])),
            "web_search_enabled": employee_data.get("capabilities", {}).get("web_search_enabled", False),
            "greeting": employee_data.get("greeting"),
            "hot_questions": employee_data.get("hot_questions", [])
        }

        self.logger.success(
            "数字员工加载成功",
            employee_id=employee_id,
            name=employee_data.get("name"),
            kb_ids=self.results["employee_info"]["kb_ids"]
        )
        return self.results["employee_info"]

    async def wait_for_documents_completion(
        self,
        kb_id: str,
        timeout: int = 1800,  # 30分钟超时
        check_interval: int = 10  # 每10秒检查一次
    ) -> bool:
        """4. 等待所有文档处理完成"""
        self.logger.step(4, "等待文档处理完成")

        start_time = time.time()
        last_doc_count = 0

        self.logger.info("开始轮询文档状态", kb_id=kb_id, timeout=timeout, check_interval=check_interval)

        while time.time() - start_time < timeout:
            url = f"{self.base_url}/api/knowledge-base/documents/list"
            params = {
                "kb_id": kb_id,
                "page": 1,
                "page_size": 100
            }

            try:
                response = await self._request("GET", url, params=params)
                # API可能返回"items"或"documents"字段
                documents = response.get("data", {}).get("items", [])
                if not documents:
                    documents = response.get("data", {}).get("documents", [])

                completed = sum(1 for d in documents if d.get("status") == "completed")
                processing = sum(1 for d in documents if d.get("status") == "processing")
                failed = sum(1 for d in documents if d.get("status") == "failed")
                total = len(documents)

                if total > last_doc_count:
                    self.logger.info("检测到新文档", total=total)

                last_doc_count = total

                # 使用日志记录进度（不带\r，因为日志系统会自动处理）
                elapsed = int(time.time() - start_time)
                self.logger.info(
                    "文档处理进度",
                    completed=completed,
                    total=total,
                    processing=processing,
                    failed=failed,
                    elapsed=f"{elapsed}s"
                )

                if completed + failed >= total and total > 0:
                    self.logger.success("所有文档处理完成", completed=completed, failed=failed, total=total)
                    return True

                await asyncio.sleep(check_interval)

            except Exception as e:
                self.logger.error("检查文档状态失败", error=str(e))
                await asyncio.sleep(check_interval)

        self.logger.warning("文档处理超时", timeout=timeout)
        return False

    async def get_mineru_jobs_and_content(self) -> List[Dict[str, Any]]:
        """5. 获取MinerU任务和markdown内容"""
        self.logger.step(5, "获取MinerU任务和markdown内容")

        # 获取任务列表
        jobs_url = f"{self.base_url}/api/mineru/jobs"

        try:
            response = await self._request("GET", jobs_url, params={"limit": 100})

            # API直接返回列表,不是字典包裹的列表
            if isinstance(response, list):
                jobs = response
            elif isinstance(response, dict):
                jobs = response.get("data", response)
            else:
                self.logger.error("MinerU API返回格式错误", type=type(response).__name__)
                return []

            if not jobs:
                self.logger.warning("没有找到MinerU任务")
                return []

            self.logger.info("找到MinerU任务", count=len(jobs))

            job_contents = []

            for job in jobs:
                job_id = job.get("job_id")
                file_name = job.get("file_name", "unknown")
                status = job.get("status", "unknown")

                if status != "completed":
                    self.logger.warning("跳过未完成的任务", file_name=file_name, status=status)
                    continue

                # 获取任务的markdown内容
                markdown_url = f"{self.base_url}/api/mineru/jobs/{job_id}/markdown"
                try:
                    md_response = await self._request("GET", markdown_url)
                    # API可能直接返回markdown_content字段，也可能在data.markdown中
                    markdown_content = md_response.get("markdown_content", "")
                    if not markdown_content:
                        markdown_content = md_response.get("data", {}).get("markdown", "")

                    job_contents.append({
                        "job_id": job_id,
                        "file_name": file_name,
                        "markdown": markdown_content,
                        "total_pages": job.get("total_pages"),
                        "status": status,
                        "content_length": len(markdown_content)
                    })
                    self.logger.info("获取markdown成功", file_name=file_name, content_length=len(markdown_content))

                except Exception as e:
                    self.logger.error("获取markdown失败", file_name=file_name, error=str(e))

            self.results["mineru_jobs"] = job_contents
            self.logger.success("MinerU任务获取完成", total=len(job_contents))
            return job_contents

        except Exception as e:
            self.logger.error("获取MinerU任务失败", error=str(e))
            return []

    async def generate_qa_pairs(
        self,
        job_contents: List[Dict[str, Any]],
        questions_per_doc: int = 3
    ) -> List[Dict[str, Any]]:
        """6. 基于文档内容生成问答对"""
        self.logger.step(6, "基于文档内容生成问答对")

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

        self.logger.info("开始生成问答对", total_docs=len(job_contents), questions_per_doc=questions_per_doc)

        for i, doc in enumerate(job_contents, 1):
            if not doc.get("markdown") or len(doc["markdown"]) < 500:
                self.logger.warning("跳过内容过短的文档", index=i, total=len(job_contents), file_name=doc['file_name'])
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

                self.logger.info("生成问答对成功", index=i, total=len(job_contents), file_name=doc['file_name'], count=len(doc_qa_pairs))

            except Exception as e:
                self.logger.error("生成问答对失败", index=i, total=len(job_contents), file_name=doc['file_name'], error=str(e))
                # 手动创建通用问答对
                qa_pairs.append({
                    "question": f"请简要介绍《{doc['file_name'].replace('.pdf', '')}》的主要内容",
                    "reference_answer": f"请基于文档《{doc['file_name']}》的内容回答这个问题。",
                    "source_file": doc["file_name"],
                    "source_job_id": doc["job_id"]
                })

        self.results["qa_pairs"] = qa_pairs
        self.logger.success("问答对生成完成", total=len(qa_pairs))

        # 保存问答对到文件
        qa_file = Path(__file__).parent / f"rag_eval_qa_{self.test_run_id}.json"
        with open(qa_file, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        self.logger.info("问答对已保存", file_path=str(qa_file))

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
        self.logger.step(7, "测试问答接口")

        session_id = f"rag_eval_{int(time.time())}"
        chat_responses = []

        self.logger.info("开始测试问答", total_qa=len(qa_pairs), session_id=session_id, employee_id=employee_id)

        for i, qa in enumerate(qa_pairs, 1):
            self.logger.info("测试问题", index=i, total=len(qa_pairs), question=qa['question'][:50])

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

                self.logger.info("回答成功", index=i, answer_length=len(answer), elapsed=f"{elapsed:.2f}s")

            except Exception as e:
                self.logger.error("问答失败", index=i, error=str(e))
                chat_responses.append({
                    "question": qa["question"],
                    "reference_answer": qa["reference_answer"],
                    "generated_answer": f"错误: {str(e)}",
                    "source_file": qa["source_file"],
                    "error": str(e),
                    "response_time_ms": 0
                })

        self.results["chat_responses"] = chat_responses
        self.logger.success("问答测试完成", total=len(chat_responses))
        return chat_responses

    async def evaluate_rag_quality(
        self,
        chat_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """8. 评估RAG质量"""
        self.logger.step(8, "评估RAG质量")

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

        self.logger.info("开始评估问答质量", total_responses=len(chat_responses))

        for response in chat_responses:
            if "error" in response:
                self.logger.warning("跳过错误响应", question=response.get("question", "unknown")[:50])
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

                self.logger.info(
                    "质量评分",
                    question=response["question"][:40],
                    relevance=f"{relevance:.2f}",
                    accuracy=f"{accuracy:.2f}",
                    completeness=f"{completeness:.2f}",
                    overall=f"{overall:.2f}"
                )

            except Exception as e:
                self.logger.error("评估失败", question=response.get("question", "unknown")[:50], error=str(e))

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

        # 输出评估报告
        self.logger.info("RAG质量评估报告")
        if evaluation:
            self.logger.info("评估统计", total=evaluation['total_evaluated'])
            self.logger.info("平均相关性", score=f"{evaluation['avg_relevance']:.3f}")
            self.logger.info("平均准确性", score=f"{evaluation['avg_accuracy']:.3f}")
            self.logger.info("平均完整性", score=f"{evaluation['avg_completeness']:.3f}")
            self.logger.info("综合得分", score=f"{evaluation['avg_overall']:.3f}")

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

            self.logger.success("RAG质量评估完成", grade=grade, score=f"{overall:.3f}")

        return evaluation

    async def save_results(self):
        """9. 保存完整测试结果"""
        self.logger.step(9, "保存测试结果")

        results_file = Path(__file__).parent / f"rag_eval_results_{self.test_run_id}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        self.logger.success("完整结果已保存", file_path=str(results_file), test_id=self.test_run_id)

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

        self.logger.success("简化报告已保存", file_path=str(summary_file))

    async def run_full_evaluation(self, skip_initial_setup: bool = False):
        """
        运行完整的端到端评估

        Args:
            skip_initial_setup: 是否跳过初始设置步骤(1-3)，直接从步骤4开始
                              如果知识库已创建、文档已上传、数字员工已加载，设为True
        """
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("RAG端到端评估测试")
        if skip_initial_setup:
            self.logger.info("模式: 跳过初始设置，从步骤4开始")
        self.logger.info("=" * 60)

        try:
            if not skip_initial_setup:
                # 1. 使用现有知识库
                kb_id = await self.create_knowledge_base()

                # 2. 上传文档
                await self.upload_documents(kb_id)

                # 3. 加载现有数字员工
                await self.load_employee()
            else:
                # 跳过前3步，直接使用已有的知识库ID和数字员工信息
                self.logger.info("跳过初始设置步骤")
                kb_id = "kb_e2bc0ea588c4"  # 使用已有的知识库ID

                # 加载现有数字员工信息
                await self.load_employee()

            # 4. 等待文档处理完成
            await self.wait_for_documents_completion(kb_id)

            # 5. 获取MinerU内容
            job_contents = await self.get_mineru_jobs_and_content()

            if not job_contents:
                self.logger.warning("没有可用的文档内容,测试终止")
                return

            # 6. 生成问答对
            qa_pairs = await self.generate_qa_pairs(job_contents)

            if not qa_pairs:
                self.logger.warning("没有生成问答对,测试终止")
                return

            # 7. 测试问答
            await self.test_chat_responses(qa_pairs, self.results["employee_info"]["employee_id"])

            # 8. 评估质量
            await self.evaluate_rag_quality(self.results["chat_responses"])

            # 9. 保存结果
            await self.save_results()

        except Exception as e:
            self.logger.error("评估过程中发生错误", error=str(e), exc_info=True)

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.success("测试完成!")
        self.logger.info("=" * 60)
        self.logger.info("")


async def main():
    """主函数"""
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="RAG端到端评估测试")
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="跳过初始设置步骤(创建知识库、上传文档、加载数字员工)，直接从等待文档处理开始"
    )
    args = parser.parse_args()

    test = E2ERAGEvaluation()

    # 检查PDF目录（仅在未跳过设置时检查）
    if not args.skip_setup and not test.pdf_dir.exists():
        test.logger.error("PDF目录不存在", pdf_dir=str(test.pdf_dir))
        test.logger.error("当前目录", cwd=str(Path.cwd()))
        test.logger.close()
        return

    if not args.skip_setup:
        test.logger.info("配置验证", pdf_dir=str(test.pdf_dir), base_url=test.base_url)

    # 运行完整评估
    await test.run_full_evaluation(skip_initial_setup=args.skip_setup)

    # 关闭日志系统
    test.logger.close()


if __name__ == "__main__":
    asyncio.run(main())
