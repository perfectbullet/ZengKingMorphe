#!/usr/bin/env python3
"""
LaTeX 公式流式提取与转换脚本。

用途：
1. 从包含 LaTeX 公式的流式输出中提取公式
2. 使用 LLM 将公式转换为口语化中文描述
3. 替换原公式为口语化描述

支持的公式格式：
- $...$ (行内)
- \(...\) (行内)
- $$...$$ (行间)
- \[...\] (行间)

用法：
    python scripts/latex_formula_stream_processor.py
    python scripts/latex_formula_stream_processor.py --sample-index 0
    python scripts/latex_formula_stream_processor.py --sample-file teaching_script_generate_scripts.md
"""
import asyncio
import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from app.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class FormulaMatch:
    """匹配到的公式信息"""
    formula: str           # 原始 LaTeX 公式
    start_pos: int         # 在文本中的起始位置
    end_pos: int           # 在文本中的结束位置
    formula_type: str      # 'inline' 或 'display'
    delimiter: str         # 使用的定界符类型，如 '$', '$$', '\(', '\['


# =============================================================================
# LaTeX 公式提取器（可复用）
# =============================================================================

class LatexFormulaExtractor:
    """
    从流式文本中提取 LaTeX 公式的类。

    支持:
    - $...$ (行内)
    - \(...\) (行内)
    - $$...$$ (行间)
    - \[...\] (行间)

    特点:
    - 状态机模式，支持流式处理
    - 返回公式位置信息用于替换

    使用示例:
        extractor = LatexFormulaExtractor()
        formulas, replaced = extractor.extract_from_text(text)

        或流式处理:
        for chunk in stream:
            new_formulas = extractor.feed(chunk)
            for formula in new_formulas:
                print(f"发现公式: {formula.formula}")
    """

    # 定界符模式: (开始标记, (类型, 结束标记))
    PATTERNS = {
        '$$': ('display', '$$'),
        '$': ('inline', '$'),
        '\\[': ('display', '\\]'),
        '\\(': ('inline', '\\)'),
    }

    def __init__(self):
        self.buffer = ""              # 累积的文本
        self.formulas: List[FormulaMatch] = []  # 已提取的公式列表
        self.state = "NORMAL"         # 当前状态: NORMAL, INLINE, DISPLAY
        self.current_delimiter = None
        self.current_start = 0
        self.current_formula = ""

    def feed(self, chunk: str) -> List[FormulaMatch]:
        """
        喂入新的文本块，返回新发现的完整公式。

        Args:
            chunk: 新的文本块

        Returns:
            本轮提取到的完整公式列表
        """
        new_formulas = []
        i = 0

        while i < len(chunk):
            if self.state == "NORMAL":
                # 检查是否进入公式状态
                found = False
                for delim, (ftype, end_delim) in self.PATTERNS.items():
                    if self._check_delimiter(chunk, i, delim):
                        self.state = f"{ftype.upper()}"
                        self.current_delimiter = delim
                        self.current_start = len(self.buffer)
                        self.current_formula = ""
                        i += len(delim)
                        found = True
                        break

                if not found:
                    self.buffer += chunk[i]
                    i += 1

            elif self.state in ("INLINE", "DISPLAY"):
                # 检查是否退出公式状态
                end_delim = self.PATTERNS[self.current_delimiter][1]

                if self._check_delimiter(chunk, i, end_delim):
                    # 公式结束
                    formula = FormulaMatch(
                        formula=self.current_formula,
                        start_pos=self.current_start,
                        end_pos=len(self.buffer),
                        formula_type=self.state.lower(),
                        delimiter=self.current_delimiter
                    )
                    new_formulas.append(formula)
                    self.formulas.append(formula)

                    # 重置状态
                    self.state = "NORMAL"
                    self.current_delimiter = None
                    i += len(end_delim)
                else:
                    self.current_formula += chunk[i]
                    self.buffer += chunk[i]
                    i += 1

        return new_formulas

    def _check_delimiter(self, text: str, pos: int, delim: str) -> bool:
        """检查指定位置是否有指定的定界符"""
        if pos + len(delim) > len(text):
            return False
        return text[pos:pos + len(delim)] == delim

    def get_text_with_placeholders(self) -> str:
        """
        获取用占位符替换公式后的文本。

        Returns:
            替换后的文本，公式被 `__FORMULA_N__` 替换
        """
        result = self.buffer
        # 从后往前替换，避免位置偏移
        for idx, formula in enumerate(reversed(self.formulas)):
            placeholder = f"__FORMULA_{len(self.formulas) - 1 - idx}__"
            start = formula.start_pos
            end = formula.end_pos
            result = result[:start] + placeholder + result[end:]
        return result

    def reset(self):
        """重置提取器状态"""
        self.buffer = ""
        self.formulas = []
        self.state = "NORMAL"
        self.current_delimiter = None
        self.current_start = 0
        self.current_formula = ""

    def extract_from_text(self, text: str) -> Tuple[List[FormulaMatch], str]:
        """
        从完整文本中提取公式（非流式模式）。

        Args:
            text: 完整文本

        Returns:
            (公式列表, 替换后的文本)
        """
        self.reset()
        self.feed(text)
        replaced_text = self.get_text_with_placeholders()
        return self.formulas, replaced_text


# =============================================================================
# 辅助函数：从 test_revise_answer.py 复用
# =============================================================================

def load_sampled_scripts(file_path: str) -> Tuple[List[str], List[str]]:
    """从文件中加载教学讲稿样本（自动检测格式）"""
    from test_revise_answer import load_markdown_samples, load_text_samples

    if file_path.endswith(".md"):
        return load_markdown_samples(file_path)
    else:
        samples = load_text_samples(file_path)
        return samples, [f"样本 #{i}" for i in range(len(samples))]


def load_revise_prompt() -> str:
    """加载数学公式口语化讲解提示词"""
    PROMPT = r'''
你是一个专业的数学讲解助手，专门将LaTeX数学公式转换为流畅、准确、符合中文口语的自然语言描述，用于语音合成(TTS)。

转译规则：
- 完全忠于数学原意，不可更改或简化公式。
- 口语化表达：使用"除以"、"开方"、"求和"等口语词；用"的"连接修饰关系（如"x的平方"）。
- 明确结构：清晰说明"分子"、"分母"、"下标"、"上标"、"积分限"、"求和范围"。
- 流畅断句：输出为完整的短句，适合TTS平稳朗读。

重要：直接输出最终答案，不要输出任何推理过程、思考步骤或中间分析。

示例对照（LaTeX -> 描述）：
- $E = mc^2$ -> E 等于 m 乘以 c 的平方
- $\frac{a}{b}$ -> a 除以 b
- $\sqrt{x^2 + y^2}$ -> 根号下 x 平方 加 y 平方
- $\sum_{i=1}^{n} i^2$ -> 对 i 从 1 到 n 求和，i 的平方
- $\int_{0}^{1} f(x) dx$ -> 从 0 到 1，对函数 f(x) 积分
- $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$ -> 一个矩阵，第一行为 a 和 b，第二行为 c 和 d

请严格遵循以上风格，直接输出转换结果，不要有任何其他说明。
'''
    return PROMPT


def get_revise_llm():
    """获取用于文本修订的 LLM 实例"""
    # provider = os.getenv("REVISE_PROVIDER", "siliconflow").lower()
    provider = 'siliconflow'
    if provider == "siliconflow":
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
        model = os.getenv("OPENAI_REVISE_MODEL", os.getenv("OPENAI_MODEL", "deepseek-ai/DeepSeek-V3"))

        logger.info(f"[Revise LLM] SiliconFlow | API_BASE={api_base} | MODEL={model}")

        return ChatOpenAI(
            base_url=api_base,
            api_key=api_key,
            model=model,
            temperature=0.7,
            streaming=True,
        )
    else:
        ollama_base_url = 'http://192.168.8.231:11434'
        ollama_model = 'qwen2.5:32b'

        logger.info(f"[Revise LLM] Ollama | BASE_URL={ollama_base_url} | MODEL={ollama_model}")
        
        return ChatOllama(
            base_url=ollama_base_url,
            model=ollama_model,
            temperature=0.1,
            streaming=True,
            keep_alive=-1
        )


# =============================================================================
# LLM 公式转换函数
# =============================================================================

async def convert_formula_to_speech(
    formula: str,
    llm
) -> str:
    """
    使用 LLM 将 LaTeX 公式转换为口语化描述。

    Args:
        formula: LaTeX 公式
        llm: LLM 实例

    Returns:
        口语化描述（经过后处理清理）
    """
    system_prompt = load_revise_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": formula}
    ]

    response = await llm.ainvoke(messages)
    return post_process_latex(response.content.strip())


def post_process_latex(text: str) -> str:
    """
    后处理：清理 LLM 输出中残留的 LaTeX 语法。

    清理规则:
    - \(xxx\) -> xxx
    - \[xxx\] -> xxx
    - $xxx$ -> xxx
    - $$xxx$$ -> xxx

    Args:
        text: LLM 返回的文本

    Returns:
        清理后的文本
    """
    import re

    # 清理 \(xxx\) 行内公式定界符
    text = re.sub(r'\\\(([^)]+)\\\)', r'\1', text)

    # 清理 \[xxx\] 行间公式定界符
    text = re.sub(r'\\\[([^\]]+)\\\]', r'\1', text)

    # 清理 $xxx$ 行内公式定界符
    text = re.sub(r'\$([^$]+)\$', r'\1', text)

    # 清理 $$xxx$$ 行间公式定界符
    text = re.sub(r'\$\$([^$]+)\$\$', r'\1', text)

    # 清理残留的反斜杠命令（如 \frac, \sqrt 等）
    # 移除常见的 LaTeX 命令但保留内容
    latex_patterns = [
        (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1 除以 \2'),  # \frac{a}{b} -> a 除以 b
        (r'\\sqrt\{([^}]+)\}', r'根号下 \1'),           # \sqrt{x} -> 根号下 x
        (r'\\cdot', ' 乘以'),                           # \cdot -> 乘以
        (r'\\times', ' 乘以'),                          # \times -> 乘以
        (r'\\dots', ''),                                # \dots -> 删除
        (r'\\ldots', ''),                               # \ldots -> 删除
        (r'\\left\(', ''),                             # \left( -> 删除
        (r'\\right\)', ''),                            # \right) -> 删除
        (r'\\left\[', ''),                             # \left[ -> 删除
        (r'\\right\]', ''),                            # \right] -> 删除
        (r'\\{', ''),                                 # \{ -> {
        (r'\\}', ''),                                 # \} -> }
        (r'\\&', ' 和 '),                              # & -> 和（矩阵用）
    ]

    for pattern, replacement in latex_patterns:
        text = re.sub(pattern, replacement, text)

    return text.strip()


async def convert_formulas_batch(
    formulas: List[FormulaMatch],
    llm,
) -> Dict[int, str]:
    """
    批量转换多个公式。

    Args:
        formulas: 公式列表
        llm: LLM 实例
        context: 上下文信息

    Returns:
        {formula_index: 口语化描述}
    """
    results = {}
    for idx, formula in enumerate(formulas):
        # 根据定界符类型添加对应的统一格式定界符
        if formula.delimiter == '\\(':
            # 行内公式: \(...\) -> $...$
            full_formula = f"${formula.formula}$"
        elif formula.delimiter == '\\[':
            # 行间公式: \[...\] -> $$...$$
            full_formula = f"$${formula.formula}$$"
        else:
            # 已经是 $ 或 $$，直接使用原定界符
            full_formula = f"{formula.delimiter}{formula.formula}{formula.delimiter}"
        revised_formula = await convert_formula_to_speech(full_formula, llm)
        logger.info(f"[{idx+1}/{len(formulas)}] 转换公式: {full_formula[:50]}...,  {revised_formula}")
        results[idx] = revised_formula
    return results


# =============================================================================
# 假流式输出生成器
# =============================================================================

async def mock_stream_from_sample(
    sample_text: str,
    chunk_size: int = 20
):
    """
    从样本文本生成模拟流式输出。

    参考现有的 revise_answer 函数，按中文标点切分输出。

    Args:
        sample_text: 样本文本
        chunk_size: 每块最大大小

    Yields:
        文本块
    """
    import re

    # 按中文标点和换行切分，但保留分隔符
    segments = re.split(r'([，。！？、；：\n])', sample_text)
    current_chunk = ""

    for segment in segments:
        current_chunk += segment

        # 遇到标点符号/换行或累计超过chunk_size字符时发送
        if segment in '，。！？、；：\n' or len(current_chunk) >= chunk_size:
            yield current_chunk
            current_chunk = ""

    # 发送剩余内容
    if current_chunk:
        yield current_chunk


# =============================================================================
# 主流程：流式处理与公式替换
# =============================================================================

async def process_stream_with_formula_conversion(
    stream_generator,
    llm,
) -> str:
    """
    处理流式输出，提取公式并转换后替换。

    Args:
        stream_generator: 流式输出生成器
        llm: LLM 实例
        context: 上下文信息

    Returns:
        处理后的完整文本
    """
    extractor = LatexFormulaExtractor()
    full_text = ""

    print("\n[流式输出开始]\n")

    # 第一阶段：收集流式输出并提取公式
    async for chunk in stream_generator:
        full_text += chunk
        extractor.feed(chunk)
        print(chunk, end="", flush=True)  # 实时输出

    print("\n\n[公式提取]")
    formulas = extractor.formulas

    if not formulas:
        print("未发现公式")
        return full_text

    print(f"发现 {len(formulas)} 个公式:")
    for idx, f in enumerate(formulas):
        preview = f.formula[:50] + "..." if len(f.formula) > 50 else f.formula
        print(f"  {idx+1}. [{f.formula_type}] {preview}")

    # 第二阶段：批量转换公式
    print("\n[公式转换]")
    converted = await convert_formulas_batch(formulas, llm)

    # 显示转换结果
    for idx, formula in enumerate(formulas):
        speech_text = converted[idx]
        print(f"  公式 {idx+1}: {speech_text[:100]}...")

    # 第三阶段：替换公式
    replaced_text = extractor.get_text_with_placeholders()

    for idx, formula in enumerate(formulas):
        placeholder = f"__FORMULA_{idx}__"
        speech_text = converted[idx]
        replaced_text = replaced_text.replace(placeholder, speech_text)

    return replaced_text


# =============================================================================
# 主函数
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="LaTeX 公式流式提取与转换")
    parser.add_argument(
        "--sample-file",
        default="teaching_script_generate_scripts.md",
        help="样本文件路径"
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=2,  # 默认使用二项式定理样本（有很多公式）
        help="测试指定索引的样本"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有样本"
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="只提取公式，不进行转换"
    )

    args = parser.parse_args()

    # 加载样本
    samples, titles = load_sampled_scripts(args.sample_file)

    if args.list:
        print(f"共 {len(samples)} 个样本:\n")
        for idx, title in enumerate(titles):
            print(f"  {idx}. {title}")
        return

    if args.sample_index >= len(samples):
        print(f"样本索引超出范围: 0-{len(samples)-1}")
        return

    sample = samples[args.sample_index]
    title = titles[args.sample_index]

    print("=" * 80)
    print(f"=== {title} ===")
    print("=" * 80)
    print(f"\n[原始输入] (长度: {len(sample)} 字符)\n")

    # 只提取公式模式
    if args.extract_only:
        extractor = LatexFormulaExtractor()
        formulas, replaced = extractor.extract_from_text(sample)

        print(f"发现 {len(formulas)} 个公式:\n")
        for idx, f in enumerate(formulas):
            preview = f.formula[:50] + "..." if len(f.formula) > 50 else f.formula
            print(f"  {idx+1}. [{f.formula_type}] {preview}")

        print(f"\n[替换后的文本预览]")
        print(replaced[:500] + "..." if len(replaced) > 500 else replaced)
        return

    # 获取 LLM
    llm = get_revise_llm()

    # 创建模拟流式输出
    stream = mock_stream_from_sample(sample)

    # 处理流式输出
    result = await process_stream_with_formula_conversion(stream, llm)

    print(f"\n\n{'=' * 80}")
    print("[最终结果]")
    print(f"{'=' * 80}")
    print(f"\n长度: {len(result)} 字符\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
