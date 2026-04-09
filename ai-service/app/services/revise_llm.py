"""
LaTeX formula to voice conversion service.

Converts mathematical formulas into voice-friendly Chinese explanations
using Ollama as LLM provider.

Formula Extraction Strategy:
- Extract complete formulas using regex
- Convert only formula content (not surrounding text)
- Replace converted formulas back into original text
"""
import os
import re
from typing import List, Tuple

from langchain_openai import ChatOpenAI

from app.core.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# LaTeX Formula Extraction Pattern
# =============================================================================

LATEX_FORMULA_PATTERN = re.compile(
    r'\$\$[^$]+?\$\$|'
    r'\$[^$\n]+?\$|'
    r'\\\([^(]+?\\\)|'
    r'\\\[[^[]+?\\\]'
    , re.DOTALL
)

MARKDOWN_LIST_PATTERN = re.compile(r'^(\s*)([-*+])\s+', re.MULTILINE)

# =============================================================================
# Prompts
# =============================================================================

FORMULA_ONLY_PROMPT = r'''
你是一个专业的数学公式讲解助手，专门将 LaTeX 数学公式转换为流畅、准确的中文口语化描述，用于语音播报(TTS)。

转换规则：
- 完全忠实于数学含义，不修改或简化公式
- 使用口语化表达："除以"、"根号"、"求和"、"加"、"减"、"乘以"等
- 用"的"连接修饰语（例如"x的平方"表示"x squared"）
- 澄清结构：明确指出"分子"、"分母"、"下标"、"上标"、"积分限"、"求和范围"
- 流畅表达：输出适合稳定TTS朗读的完整短句

符号转换规则（严格遵守）：
- + → 加
- - → 减
- * 或 × → 乘以
- / 或 ÷ → 除以
- = → 等于
- < → 小于
- > → 大于
- ≤ → 小于等于
- ≥ → 大于等于
- ≠ → 不等于
- ≈ → 约等于

输出要求：
- 不输出 LaTeX 命令或反斜杠 (\) 符号
- 不输出定界符如 \(...\) 或 $...$
- 只输出自然语言描述，所有符号必须转换为中文

重要说明：直接输出最终答案，不要任何推理过程、思考步骤或中间分析。

示例对比（LaTeX → 描述）：
- $E = mc^2$ → E 等于 m 乘以 c 的平方
- $\frac{a}{b}$ → a 除以 b
- $\sqrt{x^2 + y^2}$ → 根号下 x 的平方 加 y 的平方
- $\sum_{i=1}^{n} i^2$ → 对 i 从 1 到 n 求和，i 的平方
- $\int_{0}^{1} f(x) dx$ → 从 0 到 1，对函数 f(x) 积分
- $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$ → 一个矩阵，第一行为 a 和 b，第二行为 c 和 d
- $(a + b)^n$ → a 加 b 的 n 次方
- $a + b$ → a 加 b
- $a - b$ → a 减 b
- $a \times b$ → a 乘以 b
- $a / b$ → a 除以 b
- $x = \pm 1$ → x 等于 正负一
- $(-\infty, -1)$ → 负无穷到一的开区间
- $(a, b)$ → b到b的开区间
- $[a, b]$ → b到b的闭区间
- $(a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k$ → a 加 b 的 n 次方 等于 对 k 从 0 到 n 求和，组合数 n 选 k 乘以 a 的 n 减 k 次方 再乘以 b 的 k 次方
- $$\n   \\cos \\alpha + \\cos 27^\\circ = 2 \\cos\\left(\\frac{\\alpha + 27^\\circ}{2}\right) \\cos\\left(\\frac{\\alpha - 27^\\circ}{2}\right)\n$$\n → 余弦 alpha 加上 余弦 27 度 等于 2 乘以 余弦括号 alpha 加 27 度 除以 2 括号 乘以 余弦括号 alpha 减 27 度 除以 2 括号\n
- $$\n   \\sin \\alpha + \\sin \\beta = 2 \\sin\\left(\\frac{\\alpha + \\beta}{2}\\right) \\cos\\left(\\frac{\\alpha - \\beta}{2}\\right)\n$$\n → 正弦 alpha 加 正弦 beta 等于 2 乘以 正弦括号 alpha 加 beta 除以 2 括号 乘以 余弦括号 alpha 减 beta 除以 2 括号\n
- $$\n   \\cos \\alpha - \\cos \\beta = -2 \\sin\\left(\\frac{\\alpha + \\beta}{2}\\right) \\sin\\left(\\frac{\\alpha - \\beta}{2}\\right)\n$$\n\n → 余弦 alpha 减去 余弦 beta 等于 负 2 乘以 正弦括号 alpha 加 beta 除以 2 括号 乘以 正弦括号 alpha 减 beta 除以 2 括号

请严格遵守以上风格，直接输出转换结果，不要任何解释，也不要包含反斜杠、$ 符号或任何 LaTeX 命令。
'''

# =============================================================================
# LLM Provider Functions
# =============================================================================

async def get_revise_llm() -> ChatOpenAI:
    """
    Get LLM instance for text revision (voice-friendly output).

    Uses ChatOpenAI (compatible with Ollama OpenAI-style API).

    Environment Variables:
    - OLLAMA_BASE_URL: Ollama base URL, default http://localhost:11434
    - OLLAMA_REVISE_MODEL: Ollama model name, default qwen2.5:14b

    Returns:
        ChatOpenAI instance configured for formula conversion
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv(
        "OLLAMA_REVISE_MODEL",
        os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
    )

    # Add /v1 suffix if not present
    if not base_url.endswith("/v1"):
        base_url = f"{base_url.rstrip('/')}/v1"

    logger.info(f"[Revise LLM] ChatOpenAI | BASE_URL={base_url} | MODEL={model}")

    return ChatOpenAI(
        base_url=base_url,
        model=model,
        temperature=0.1,
        streaming=True,
    )

def _is_empty_or_delimiter_only(text: str) -> bool:
    """
    Check if text is empty or contains only LaTeX delimiters.

    Args:
        text: Text to check

    Returns:
        True if text is empty, whitespace only, or only delimiters
    """
    if not text or not text.strip():
        return True

    stripped = text.strip()
    for delim in ['$$', '$', r'\(', r'\)', r'\[', r'\]']:
        stripped = stripped.replace(delim, '')

    return not stripped.strip()

def _has_any_formula_marker(text: str) -> bool:
    """
    Check if text contains any LaTeX formulas.

    Uses LATEX_FORMULA_PATTERN to detect complete formulas delimited by
    $...$, $$...$$, \(...\), or \[...\].

    Args:
        text: Text to check

    Returns:
        True if text contains at least one complete LaTeX formula
    """
    return LATEX_FORMULA_PATTERN.search(text) is not None

def _remove_list_markers(text: str) -> str:
    """
    Remove Markdown list markers (-, *, +) while preserving indentation.

    Prevents these markers from being misread by TTS as "minus".

    Args:
        text: Original text

    Returns:
        Text with list markers removed
    """
    return MARKDOWN_LIST_PATTERN.sub(r'\1', text)

# =============================================================================
# Formula Extraction and Conversion Functions
# =============================================================================

def _extract_latex_formulas(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract all complete LaTeX formulas from text.

    Args:
        text: Text containing formulas

    Returns:
        List of (formula, start_pos, end_pos) tuples, ordered by appearance
    """
    formulas = []
    for match in LATEX_FORMULA_PATTERN.finditer(text):
        formulas.append((match.group(0), match.start(), match.end()))
    return formulas

async def _convert_single_formula(
    formula: str,
    llm: ChatOpenAI,
) -> str:
    """
    Convert a single formula to voice-friendly text.

    Args:
        formula: Complete formula with delimiters (e.g., "$x^2$")
        llm: LLM instance for conversion

    Returns:
        Voice-friendly conversion of formula
    """
    messages = [
        {"role": "system", "content": FORMULA_ONLY_PROMPT},
        {"role": "user", "content": formula}
    ]

    result = ""
    async for chunk in llm.astream(messages):
        if chunk.content:
            result += chunk.content

    logger.info(
        "[_convert_single_formula] Formula conversion",
        input_formula=repr(formula[:50]),
        output=result[:100]
    )
    return result

# =============================================================================
# Main Conversion Functions
# =============================================================================

async def convert_formula_to_voice(
    text: str,
    llm: ChatOpenAI,
) -> str:
    """
    Convert formula text to voice-friendly output using LLM.

    Extract formulas, convert only formulas, then replace back.

    Includes fast-path checks for common streaming scenarios:
    - Empty or delimiter-only input returns original text
    - No formula markers detected returns original text with list markers removed

    Args:
        text: Text containing formulas (may include regular text mixed with formulas)
        llm: LLM instance for conversion

    Returns:
        Converted voice-friendly text with formulas replaced by voice versions
    """
    if _is_empty_or_delimiter_only(text):
        logger.info(
            "[convert_formula_to_voice] Empty or delimiter-only, returning original",
            text=repr(text)
        )
        return text

    if not _has_any_formula_marker(text):
        text = _remove_list_markers(text)
        logger.info(
            "[convert_formula_to_voice] No formula markers, returning original",
            text_preview=repr(text[:50])
        )
        return text

    formulas = _extract_latex_formulas(text)
    if not formulas:
        logger.info(
            "[convert_formula_to_voice] No complete formulas extracted, returning original",
            text_preview=repr(text[:50])
        )
        return text

    logger.info(
        "[convert_formula_to_voice] Extracted formulas for conversion",
        formula_count=len(formulas),
        text_preview=repr(text[:100])
    )

    result = text
    for formula, start, end in reversed(formulas):
        voice_formula = await _convert_single_formula(formula, llm)
        result = result[:start] + voice_formula + result[end:]

    return result

MATH_SENTENCE_PROMPT = r"""你是一个专业的数学讲解助手，专门将数学教材文本转换为流畅、准确的中文口语化描述，用于语音播报(TTS)。

你的任务：将包含数学符号的句子转换为口语化表达。

转换规则：
1. 完全忠实于数学含义，不修改或简化数学内容
2. 使用口语化表达："除以"、"根号"、"求和"、"积分"等；用"的"连接修饰语
3. 澄清结构：明确指出"分子"、"分母"、"下标"、"上标"等
4. 流畅表达：输出适合稳定TTS朗读的完整短句

符号转换规则（严格遵守）：
- + → 加
- - → 减
- * 或 × → 乘以
- / 或 ÷ → 除以
- = → 等于
- < → 小于
- > → 大于
- ≤ → 小于等于
- ≥ → 大于等于
- ≠ → 不等于
- ≈ → 约等于
- ∈ → 属于
- ⊂ → 真子集
- ⊃ → 真超集
- ⊆ → 子集
- ⊇ → 超集
- ∪ → 并集
- ∩ → 交集
- ∅ → 空集
- √ → 根号
- ∞ → 无穷大
- ∑ → 求和
- ∫ → 积分
- ∏ → 乘积
- ∂ → 偏导数
- ∇ → 梯度
- π → 派
- ² → 的平方
- ³ → 的立方

输出要求：
- 不输出 LaTeX 命令或反斜杠 (\) 符号
- 不输出定界符如 \(...\) 或 $...$
- 只输出自然语言描述，所有符号必须转换为中文

重要说明：直接输出转换后的口语化表达，不要任何推理过程、思考步骤或中间分析。

示例对比：
- "通常写成{x∈A|P(x)}的形式，" → "通常写成 x 属于 A，使得 P(x) 的形式"
- "比如方程x²-2=0的所有实数根组成的集合，" → "比如方程 x 的平方减 2 等于 0 的所有实数根组成的集合"
- "那么x∈R或x∈Z这部分是可以省略的。" → "那么 x 属于 R 或 x 属于 Z 这部分是可以省略的"
- "而用列举法则是{√2，-√2}。" → "而用列举法则是根号二、负根号二"
- "a+b的平方等于a的平方加b的平方加2ab" → "a 加 b 的平方 等于 a 的平方 加 b 的平方 加 2 乘以 a 乘以 b"
- "a大于等于b" → "a 大于等于 b"
- "x的n次方" → "x 的 n 次方"

请将以下数学句子转换为口语化表达：
"""

async def convert_math_sentence_to_voice(
    text: str,
    llm: ChatOpenAI,
) -> str:
    """
    Convert sentences containing math symbols to voice-friendly expressions.

    Args:
        text: Math sentence containing symbols
        llm: LLM instance

    Returns:
        Converted voice-friendly expression
    """
    if not text or not text.strip():
        return text


    messages = [
        {"role": "system", "content": MATH_SENTENCE_PROMPT},
        {"role": "user", "content": text}
    ]

    result = ""
    async for chunk in llm.astream(messages):
        if chunk.content:
            result += chunk.content

    logger.info(
        "[convert_math_sentence_to_voice] Math sentence converted",
        input=text[:100],
        output=result[:100]
    )

    return result

__all__ = [
    'get_revise_llm',
    'convert_formula_to_voice',
    'convert_math_sentence_to_voice',
]
