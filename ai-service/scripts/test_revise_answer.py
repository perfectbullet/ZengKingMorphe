#!/usr/bin/env python3
"""
测试数学公式口语化修订脚本。

用途：
1. 读取 teaching_script_generate_scripts.md 或 sampled_teaching_scripts.txt 中的教学讲稿样本
2. 使用指定的 system_prompt 进行修订
3. 打印输出 revised_answer

用法：
    python scripts/test_revise_answer.py
    python scripts/test_revise_answer.py --sample-index 0  # 只测试第1个样本
    python scripts/test_revise_answer.py --custom "输入你的测试文本"  # 使用自定义输入
    python scripts/test_revise_answer.py --sample-file teaching_script_generate_scripts.md  # 使用Markdown格式样本
"""
import asyncio
import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple
from langchain_community.chat_models import ChatOllama
from app.core.logging import get_logger

# 添加项目路径 (scripts/test_revise_answer.py -> ai-service/)
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = get_logger(__name__)


def load_markdown_samples(file_path: str) -> Tuple[List[str], List[str]]:
    """
    从 Markdown 文件中加载教学讲稿样本。

    文件格式：
    - 用 --- 分隔不同章节
    - 每个章节以 ### 开头作为标题
    - 标题行格式：### <序号> <标题>

    Args:
        file_path: 样本文件路径

    Returns:
        (样本列表, 标题列表)
    """
    script_path = Path(__file__).parent.parent.parent / file_path
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        samples = []
        titles = []

        # 按 --- 分割章节
        raw_sections = content.split("---")

        for section in raw_sections:
            section = section.strip()
            if not section:
                continue

            lines = section.split("\n")
            title = None
            content_lines = []

            for line in lines:
                line = line.rstrip()
                # 查找 ### 标题行
                if line.startswith("###"):
                    # 提取标题：去掉 ###，去除多余空格
                    title_match = re.match(r"###\s+(.+)", line)
                    if title_match:
                        title = title_match.group(1).strip()
                # 跳过 ## 标题行（章节标题）
                elif line.startswith("##"):
                    continue
                # 跳过空行
                elif not line.strip():
                    continue
                # 收集内容行
                else:
                    content_lines.append(line)

            if content_lines:
                sample_content = "\n".join(content_lines)
                samples.append(sample_content)
                titles.append(title or "未命名章节")

        logger.info(f"从 {file_path} 加载了 {len(samples)} 个 Markdown 样本")
        return samples, titles

    except FileNotFoundError:
        logger.error(f"样本文件未找到: {script_path}")
        return [], []


def load_text_samples(file_path: str) -> List[str]:
    """
    从文本文件中加载教学讲稿样本（旧格式）。

    文件格式：用 === 分隔多个样本
    每个样本不包含 [xxx] 这样的标签

    Args:
        file_path: 样本文件路径

    Returns:
        样本列表
    """
    script_path = Path(__file__).parent.parent.parent / file_path
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 按 === 分割样本
        raw_samples = content.split("===")

        samples = []
        for sample in raw_samples:
            # 清理每个样本：去除空白行和 [xxx] 标签
            lines = []
            for line in sample.strip().split("\n"):
                line = line.strip()
                # 跳过 [xxx] 开头的行
                if line and not line.startswith("["):
                    lines.append(line)

            if lines:
                samples.append("\n".join(lines))

        logger.info(f"从 {file_path} 加载了 {len(samples)} 个文本样本")
        return samples

    except FileNotFoundError:
        logger.error(f"样本文件未找到: {script_path}")
        return []


def load_sampled_scripts(file_path: str) -> Tuple[List[str], List[str]]:
    """
    从文件中加载教学讲稿样本（自动检测格式）。

    支持：
    - Markdown 格式 (.md): 按 --- 分隔，### 标题
    - 文本格式 (.txt): 按 === 分隔

    Args:
        file_path: 样本文件路径

    Returns:
        (样本列表, 标题列表)
    """
    if file_path.endswith(".md"):
        return load_markdown_samples(file_path)
    else:
        samples = load_text_samples(file_path)
        return samples, [f"样本 #{i}" for i in range(len(samples))]


def load_revise_prompt(custom_prompt_file: str = None) -> str:
    """
    加载数学公式口语化讲解提示词。

    Args:
        custom_prompt_file: 自定义提示词文件路径

    Returns:
        提示词内容
    """
    # scripts/test_revise_answer.py is at: ai-service/scripts/
    # prompts dir is at: ai-service/prompts/
    if custom_prompt_file:
        prompt_path = Path(custom_prompt_file)
        if not prompt_path.is_absolute():
            # 相对于项目根目录
            prompt_path = Path(__file__).parent.parent.parent / custom_prompt_file
    else:
        prompt_path = Path(__file__).parent.parent / "prompts" / "数学公式口语化讲解.txt"

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning(f"Prompt file not found: {prompt_path}, using default prompt")
        return "你是一个数学公式口语化讲解专家。请将用户输入的数学公式和概念，用纯粹、流畅、易于理解的自然语言解释，完全不含任何数学符号或特殊格式，专为语音播报场景设计。"


def get_revise_llm():
    """
    获取用于文本修订的 LLM 实例。

    支持的服务商:
    - siliconflow: SiliconFlow API (推荐)
    - ollama: 本地 Ollama

    环境变量:
    - REVISE_PROVIDER: 服务商类型 (siliconflow 或 ollama)，默认 siliconflow
    - OPENAI_API_KEY: SiliconFlow API 密钥
    - OPENAI_API_BASE: SiliconFlow API 基础 URL
    - OPENAI_REVISE_MODEL: SiliconFlow 模型名称 (默认: deepseek-ai/DeepSeek-V3)
    - OLLAMA_BASE_URL: Ollama 基础 URL (默认: http://localhost:11434)
    - OLLAMA_REVISE_MODEL: Ollama 模型名称 (默认: qwen2.5:7b)
    """
    from langchain_community.chat_models import ChatOllama
    from langchain_openai import ChatOpenAI

    # 从环境变量获取服务商，默认 siliconflow
    provider = os.getenv("REVISE_PROVIDER", "siliconflow").lower()

    if provider == "siliconflow":
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
        model = os.getenv("OPENAI_REVISE_MODEL",
                         os.getenv("OPENAI_MODEL", "deepseek-ai/DeepSeek-V3"))

        if not api_key:
            logger.warning("OPENAI_API_KEY not set for SiliconFlow")

        logger.info(f"[Revise LLM] SiliconFlow | API_BASE={api_base} | MODEL={model}")

        return ChatOpenAI(
            base_url=api_base,
            api_key=api_key,  # 允许空值，由 API 端处理错误
            model=model,
            temperature=0.7,
            streaming=True,
        )
    else:
        # 使用 Ollama
        ollama_base_url = 'http://192.168.8.231:11434'
        ollama_model = 'qwen2.5:32b'

        logger.info(f"[Revise LLM] Ollama | BASE_URL={ollama_base_url} | MODEL={ollama_model}")

        return ChatOllama(
            base_url=ollama_base_url,
            model=ollama_model,
            temperature=0.7,
            streaming=True,
            keep_alive=-1
        )


async def revise_answer(existing_answer: str, system_prompt: str, llm: ChatOllama) -> str:
    """
    使用 LLM 修订答案为语音友好的输出。

    Args:
        existing_answer: 原始答案
        system_prompt: 系统提示词
        llm: LLM 实例

    Returns:
        修订后的答案
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": existing_answer}
    ]

    revised_answer = ""
    print("\n[流式输出开始]\n")

    async for chunk in llm.astream(messages):
        token = chunk.content
        if token:
            revised_answer += token
            print(token, end="", flush=True)

    print("\n\n[流式输出结束]\n")
    return revised_answer


def print_separator(title: str = ""):
    """打印分隔线。"""
    width = 80
    if title:
        center_text = f" {title} "
        padding = (width - len(center_text)) // 2
        print("=" * padding + center_text + "=" * (width - padding - len(center_text)))
    else:
        print("=" * width)


async def main():
    parser = argparse.ArgumentParser(description="测试数学公式口语化修订")
    parser.add_argument(
        "--sample-file",
        default="teaching_script_generate_scripts.md",
        help="样本文件路径 (相对于项目根目录，支持 .md 和 .txt 格式)"
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="只测试指定索引的样本 (从0开始)"
    )
    parser.add_argument(
        "--custom",
        type=str,
        default=None,
        help="使用自定义输入文本"
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="显示 system_prompt 内容"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="只测试前 N 个样本"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="使用自定义提示词文件路径"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比模式：显示原始和修订结果并排"
    )

    args = parser.parse_args()

    # 加载 system_prompt
    system_prompt = load_revise_prompt(args.prompt_file)

    if args.show_prompt:
        print_separator("SYSTEM_PROMPT")
        print(system_prompt)
        print_separator()
        return

    # 获取 LLM
    llm = get_revise_llm()

    # 自定义输入模式
    if args.custom:
        print_separator("自定义输入模式")
        print(f"\n原始输入:\n{args.custom}\n")
        revised = await revise_answer(args.custom, system_prompt, llm)
        print_separator("修订结果")
        print(f"\n{revised}\n")
        return

    # 加载样本（返回样本列表和标题列表）
    samples, titles = load_sampled_scripts(args.sample_file)
    if not samples:
        logger.error("No samples loaded, exiting.")
        return

    # 确定要测试的样本范围
    if args.sample_index is not None:
        if 0 <= args.sample_index < len(samples):
            test_samples = [samples[args.sample_index]]
            test_titles = [titles[args.sample_index]]
            print(f"只测试样本 #{args.sample_index}")
        else:
            logger.error(f"Sample index {args.sample_index} out of range (0-{len(samples)-1})")
            return
    else:
        test_samples = samples[:args.count] if args.count else samples
        test_titles = titles[:args.count] if args.count else titles
        print(f"准备测试 {len(test_samples)} 个样本")

    # 逐个测试
    for idx, (sample, title) in enumerate(zip(test_samples, test_titles)):
        actual_idx = args.sample_index if args.sample_index is not None else idx
        print_separator(f"{title}")

        # 显示标题
        print(f"\n[标题] {title}\n")

        # 显示原始输入（截断显示）
        print(f"\n[原始输入] (长度: {len(sample)} 字符)")
        preview = sample[:300] + "..." if len(sample) > 300 else sample
        print(preview)
        print()

        # 执行修订
        revised = await revise_answer(sample, system_prompt, llm)

        # 显示修订结果
        print_separator(f"修订结果 #{actual_idx}")
        print(f"\n[修订后] (长度: {len(revised)} 字符)")
        print(revised)
        print()

        # 简单分析
        print_separator("分析")
        print(f"原始长度: {len(sample)} 字符")
        print(f"修订长度: {len(revised)} 字符")
        print(f"长度变化: {len(revised) - len(sample):+d} 字符")

        # 检查是否有残留的数学符号
        math_symbols = ["²", "³", "√", "∫", "∑", "∏", "≠", "≤", "≥", "∞", "α", "β", "γ", "Δ", "π"]
        found_symbols = [s for s in math_symbols if s in revised]
        if found_symbols:
            print(f"⚠️  发现数学符号: {', '.join(found_symbols)}")
        else:
            print("✓ 未发现常见数学符号")

        # 检查是否有公式形式的文本
        formula_patterns = [" = ", " =", "= ", " + ", " - ", " × ", " ÷ "]
        found_patterns = [p for p in formula_patterns if p in revised]
        if found_patterns:
            print(f"⚠️  发现公式模式: {', '.join(found_patterns)}")
        else:
            print("✓ 未发现明显的公式模式")

        print()


if __name__ == "__main__":
    asyncio.run(main())
