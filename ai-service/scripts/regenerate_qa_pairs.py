"""
重新生成问答对脚本 - 使用硅基流动DeepSeek模型
改进版：
1. 从mineru_job_list.json读取job_id
2. 通过API获取任务详情（包含完整markdown）
3. 对长文档进行分段处理（带overlap）
4. 分段生成QA对
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.logging import logger


class QAPairRegenerator:
    """问答对重新生成器"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 统一 LLM 配置
        llm_base_url = os.getenv("LLM_BASE_URL") or settings.ollama_base_url
        llm_model = os.getenv("LLM_MODEL") or settings.ollama_model
        self.results = {
            "test_run_id": self.test_run_id,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "base_url": llm_base_url,
                "model": llm_model,
            }
        }

        # 配置
        self.chunk_size = 3000  # 每段字符数
        self.chunk_overlap = 300  # 重叠字符数
        self.questions_per_chunk = 2  # 每段生成的问题数

        # 配置日志
        logger.info("初始化问答对重新生成器", test_run_id=self.test_run_id)
        logger.info("配置信息",
                   model=os.getenv("LLM_MODEL") or settings.ollama_model,
                   base_url=os.getenv("LLM_BASE_URL") or settings.ollama_base_url)

    async def _request(self, method: str, endpoint: str, **kwargs):
        """发送HTTP请求"""
        import aiohttp
        url = f"{self.base_url}{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, **kwargs) as response:
                text = await response.text()
                try:
                    return json.loads(text)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析失败", url=url, status=response.status, error=str(e)[:200])
                    # 尝试返回包含错误信息的字典
                    return {"code": response.status, "message": "JSON解析失败", "raw_text": text[:1000]}

    def load_job_list(self) -> List[Dict[str, Any]]:
        """从mineru_job_list.json加载任务列表"""
        job_list_file = Path(__file__).parent / "mineru_job_list.json"
        logger.info(f"加载任务列表", file=str(job_list_file))

        with open(job_list_file, "r", encoding="utf-8") as f:
            job_list = json.load(f)

        logger.info(f"加载任务列表成功", count=len(job_list))
        return job_list

    async def fetch_job_contents(self, job_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """获取所有任务的内容"""
        logger.info("开始获取任务内容", total_jobs=len(job_list))

        job_contents = []
        for i, job in enumerate(job_list, 1):
            job_id = job.get("job_id")
            file_name = job.get("file_name", "")
            status = job.get("status", "")

            if status != "completed":
                logger.warning(f"跳过未完成的任务", index=i, job_id=job_id, status=status)
                continue

            # 获取任务的markdown内容
            try:
                markdown_url = f"/api/mineru/jobs/{job_id}/markdown"
                md_response = await self._request("GET", markdown_url)

                # 检查是否有错误
                if "error" in md_response or "message" in md_response and "code" in md_response:
                    logger.error(f"获取markdown失败", index=i, job_id=job_id, response=md_response)
                    continue

                # API返回markdown_content字段（直接在根级别）
                markdown_content = md_response.get("markdown_content", "")
                if not markdown_content:
                    logger.warning(f"markdown内容为空", index=i, job_id=job_id, response_keys=list(md_response.keys()))
                    continue

                job_contents.append({
                    "job_id": job_id,
                    "file_name": file_name,
                    "markdown": markdown_content
                })

                logger.info(f"获取任务内容成功",
                           index=i,
                           total=len(job_list),
                           file_name=file_name,
                           content_length=len(markdown_content))

            except Exception as e:
                logger.error(f"获取任务内容异常", index=i, job_id=job_id, error=str(e), exc_info=True)
                continue

        logger.info(f"获取任务内容完成", success_count=len(job_contents))
        return job_contents

    def split_document_with_overlap(self, text: str, file_name: str) -> List[Dict[str, Any]]:
        """将文档分段，带overlap"""
        if len(text) <= self.chunk_size:
            return [{"text": text, "chunk_index": 0, "total_chunks": 1}]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # 计算当前chunk的结束位置
            end = min(start + self.chunk_size, len(text))

            # 如果不是最后一段，尝试在句子边界处截断
            if end < len(text):
                # 查找最近的句号、问号、感叹号
                for delimiter in ["。", "！", "？", "\n\n", "；", "；"]:
                    last_delimiter = text.rfind(delimiter, start, end)
                    if last_delimiter > start + self.chunk_size // 2:  # 确保不会截断太少
                        end = last_delimiter + len(delimiter)
                        break

            chunk_text = text[start:end].strip()
            chunks.append({
                "text": chunk_text,
                "chunk_index": chunk_index,
                "total_chunks": -1  # 稍后更新
            })

            # 移动到下一个chunk（带overlap）
            start = end - self.chunk_overlap
            chunk_index += 1

        # 更新total_chunks
        for chunk in chunks:
            chunk["total_chunks"] = len(chunks)

        logger.info(f"文档分段完成",
                   file_name=file_name,
                   total_length=len(text),
                   chunk_count=len(chunks))

        return chunks

    async def generate_qa_pairs(
        self,
        job_contents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """基于文档内容生成问答对"""
        from langchain_openai import ChatOpenAI
        from langchain_community.chat_models import ChatOllama

        # 使用统一 LLM 配置生成问答对
        llm_base_url = os.getenv("LLM_BASE_URL") or settings.ollama_base_url
        llm_model = os.getenv("LLM_MODEL") or settings.ollama_model
        llm_api_key = os.getenv("LLM_API_KEY") or settings.siliconflow_api_key or "no-key"

        llm = ChatOpenAI(
            base_url=llm_base_url,
            api_key=llm_api_key,
            model=llm_model,
            temperature=0
        )

        qa_pairs = []
        total_chunks = 0

        # 统计总chunk数
        for job in job_contents:
            chunks = self.split_document_with_overlap(job["markdown"], job["file_name"])
            total_chunks += len(chunks)

        logger.info(f"开始生成问答对",
                   total_docs=len(job_contents),
                   total_chunks=total_chunks,
                   model=os.getenv("LLM_MODEL") or settings.ollama_model)

        current_chunk = 0

        for job_idx, job in enumerate(job_contents, 1):
            if len(job["markdown"]) < 500:
                logger.warning("跳过内容过短的文档",
                             doc_index=job_idx,
                             total=len(job_contents),
                             file_name=job['file_name'])
                continue

            # 分段处理
            chunks = self.split_document_with_overlap(job["markdown"], job["file_name"])

            for chunk_idx, chunk in enumerate(chunks, 1):
                current_chunk += 1
                chunk_text = chunk["text"]

                generate_prompt = f"""请基于以下文档内容片段生成 {self.questions_per_chunk} 个问答对。

文档名称: {job['file_name']}
内容片段: 第{chunk['chunk_index'] + 1}/{chunk['total_chunks']}段

文档内容:
{chunk_text}

要求:
1. 问题应该能从文档内容中找到答案
2. 问题要具体,不要过于宽泛
3. 答案要简洁准确,直接引用文档中的信息
4. 问题类型可以包括: 主要内容、关键概念、具体数据、流程步骤等
5. 以JSON数组格式返回
6. 问题应该基于当前片段的内容，不要跨片段提问

返回格式:
[
  {{"question": "问题1", "answer": "答案1"}},
  {{"question": "问题2", "answer": "答案2"}}
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

                    chunk_qa_pairs = json.loads(response_text)

                    for qa in chunk_qa_pairs:
                        qa_pairs.append({
                            "question": qa["question"],
                            "reference_answer": qa["answer"],
                            "source_file": job["file_name"],
                            "source_job_id": job["job_id"],
                            "chunk_index": chunk['chunk_index'],
                            "total_chunks": chunk['total_chunks']
                        })

                    logger.info("生成问答对成功",
                               doc_index=job_idx,
                               total_docs=len(job_contents),
                               chunk_index=chunk_idx,
                               total_chunks=len(chunks),
                               current_chunk=current_chunk,
                               total_chunks_all=total_chunks,
                               file_name=job['file_name'],
                               count=len(chunk_qa_pairs))

                except Exception as e:
                    logger.error("生成问答对失败",
                                doc_index=job_idx,
                                chunk_index=chunk_idx,
                                file_name=job['file_name'],
                                error=str(e))

                    # 创建一个fallback问题
                    qa_pairs.append({
                        "question": f"根据《{job['file_name']}》第{chunk['chunk_index'] + 1}段内容，主要讲了什么？",
                        "reference_answer": f"请基于文档《{job['file_name']}》第{chunk['chunk_index'] + 1}段的内容回答这个问题。",
                        "source_file": job["file_name"],
                        "source_job_id": job["job_id"],
                        "chunk_index": chunk['chunk_index'],
                        "total_chunks": chunk['total_chunks']
                    })

        logger.info(f"问答对生成完成", total=len(qa_pairs))
        return qa_pairs

    async def run(self):
        """运行重新生成流程"""
        logger.info("=" * 80)
        logger.info("问答对重新生成流程开始")
        logger.info("=" * 80)

        # 1. 加载任务列表
        job_list = self.load_job_list()

        # 2. 获取任务内容
        job_contents = await self.fetch_job_contents(job_list)

        if not job_contents:
            logger.error("没有找到有效的文档内容")
            return

        # 3. 生成问答对
        qa_pairs = await self.generate_qa_pairs(job_contents)

        self.results["qa_pairs"] = qa_pairs

        # 4. 保存结果
        qa_file = Path(__file__).parent / f"rag_eval_qa_{self.test_run_id}.json"
        with open(qa_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ 问答对已保存到: {qa_file}")
        logger.info(f"✓ 总计生成 {len(qa_pairs)} 个问答对")
        logger.info("=" * 80)
        logger.info("问答对重新生成完成!")
        logger.info("=" * 80)


async def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="重新生成问答对")
    parser.add_argument("--base-url", default="http://localhost:8000", help="AI服务基础URL")
    args = parser.parse_args()

    generator = QAPairRegenerator(base_url=args.base_url)
    await generator.run()


if __name__ == "__main__":
    asyncio.run(main())
