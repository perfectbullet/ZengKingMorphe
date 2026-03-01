"""
LaTeX formula to voice conversion service.

Supports SiliconFlow and Ollama LLM providers for converting mathematical
formulas into voice-friendly Chinese explanations.

Formula Extraction Strategy:
- Extract complete formulas using regex
- Convert only formula content (not surrounding text)
- Replace converted formulas back into original text
"""
import os
import re
from typing import List, Tuple

from langchain_community.chat_models import ChatOllama
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

# =============================================================================
# Prompts
# =============================================================================

FORMULA_ONLY_PROMPT = r'''
You are a professional math explanation assistant specializing in converting
LaTeX mathematical formulas into fluent, accurate, Chinese colloquial
descriptions for text-to-speech (TTS).

Translation rules:
- Stay completely faithful to the mathematical meaning, do not modify or simplify formulas
- Use colloquial expressions: "divided by", "square root", "sum", "plus", "minus", "times", etc.
- Use "的" to connect modifiers (e.g., "x的平方" for "x squared")
- Clarify structure: clearly indicate "numerator", "denominator", "subscript", "superscript", "integral limits", "summation range"
- Smooth phrasing: output complete short sentences suitable for stable TTS reading

Symbol translation rules (strictly follow):
- + → 加
- - → 减
- * or × → 乘以
- / or ÷ → 除以
- = → 等于
- < → 小于
- > → 大于
- ≤ → 小于等于
- ≥ → 大于等于
- ≠ → 不等于
- ≈ → 约等于

Output requirements:
- No LaTeX commands or backslash (\) symbols in output
- No delimiter backslashes like \(...\) or $...$
- Only natural language descriptions, all symbols must be converted to Chinese

IMPORTANT: Output the final answer directly, no reasoning process, thinking steps, or intermediate analysis.

Example comparisons (LaTeX → description):
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
- $(a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k$ → a 加 b 的 n 次方 等于 对 k 从 0 到 n 求和，组合数 n 选 k 乘以 a 的 n 减 k 次方 再乘以 b 的 k 次方

Please strictly follow the above style, output the conversion result directly,
without any explanation, and do not include backslashes, $ symbols, or any LaTeX commands.
'''

# =============================================================================
# LLM Provider Functions
# =============================================================================

async def get_revise_llm() -> ChatOllama | ChatOpenAI:
    """
    Get LLM instance for text revision (voice-friendly output).

    Supports:
    - siliconflow: SiliconFlow API (recommended)
    - ollama: Local Ollama

    Environment Variables:
    - REVISE_PROVIDER: Provider type (siliconflow or ollama), default siliconflow
    - OPENAI_API_KEY: SiliconFlow API key
    - OPENAI_API_BASE: SiliconFlow API base URL
    - OPENAI_REVISE_MODEL: SiliconFlow model name
    - OLLAMA_BASE_URL: Ollama base URL
    - OLLAMA_REVISE_MODEL: Ollama model name

    Returns:
        ChatOpenAI or ChatOllama instance configured for formula conversion
    """
    provider = "ollama"

    if provider == "siliconflow":
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
        model = os.getenv(
            "OPENAI_REVISE_MODEL",
            os.getenv("OPENAI_MODEL", "deepseek-ai/DeepSeek-V3")
        )

        if not api_key:
            logger.warning("OPENAI_API_KEY not set for SiliconFlow")
            raise ValueError("OPENAI_API_KEY not set for SiliconFlow")

        logger.info(f"[Revise LLM] SiliconFlow | API_BASE={api_base} | MODEL={model}")

        return ChatOpenAI(
            base_url=api_base,
            api_key=api_key,
            model=model,
            temperature=0.7,
            streaming=True,
        )
    else:
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv(
            "OLLAMA_REVISE_MODEL",
            os.getenv("OLLAMA_MODEL", "qwen2.5:32b")
        )

        logger.info(f"[Revise LLM] Ollama | BASE_URL={ollama_base_url} | MODEL={ollama_model}")

        return ChatOllama(
            base_url=ollama_base_url,
            model=ollama_model,
            temperature=0.7,
            streaming=True,
            keep_alive=-1
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
    llm: ChatOllama | ChatOpenAI,
) -> str:
    """
    Convert a single formula to voice-friendly text.

    Args:
        formula: Complete formula with delimiters (e.g., "$x^2$")
        llm: LLM instance for conversion

    Returns:
        Voice-friendly conversion of the formula
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
    llm: ChatOllama | ChatOpenAI,
) -> str:
    """
    Convert formula text to voice-friendly output using LLM.

    Extract formulas, convert only formulas, then replace back.

    Includes fast-path checks for common streaming scenarios:
    - Empty or delimiter-only input → returns original text
    - No formula markers detected → returns original text

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

MATH_SENTENCE_PROMPT = r"""You are a professional math explanation assistant specializing in converting
math textbook text into fluent, accurate, Chinese colloquial descriptions for text-to-speech (TTS).

Your task: Convert sentences containing math symbols to colloquial expressions.

Conversion rules:
1. Stay completely faithful to the mathematical meaning, do not modify or simplify math content
2. Use colloquial expressions: "divided by", "square root", "sum", "integral", etc.; use "的" to connect modifiers
3. Clarify structure: clearly indicate "numerator", "denominator", "subscript", "superscript", etc.
4. Smooth phrasing: output complete short sentences suitable for stable TTS reading

Symbol translation rules (strictly follow):
- + → 加
- - → 减
- * or × → 乘以
- / or ÷ → 除以
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

Output requirements:
- No LaTeX commands or backslash (\) symbols in output
- No delimiter backslashes like \(...\) or $...$
- Only natural language descriptions, all symbols must be converted to Chinese

IMPORTANT: Output the converted colloquial expression directly, no reasoning process, thinking steps, or intermediate analysis.

Example comparisons:
- "通常写成{x∈A|P(x)}的形式，" → "通常写成 x 属于 A，使得 P(x) 的形式"
- "比如方程x²-2=0的所有实数根组成的集合，" → "比如方程 x 的平方减 2 等于 0 的所有实数根组成的集合"
- "那么x∈R或x∈Z这部分是可以省略的。" → "那么 x 属于 R 或 x 属于 Z 这部分是可以省略的"
- "而用列举法则是{√2，-√2}。" → "而用列举法则是根号二、负根号二"
- "a+b的平方等于a的平方加b的平方加2ab" → "a 加 b 的平方 等于 a 的平方 加 b 的平方 加 2 乘以 a 乘以 b"
- "a大于等于b" → "a 大于等于 b"
- "x的n次方" → "x 的 n 次方"

Please convert the following math sentence to a colloquial expression:
"""

async def convert_math_sentence_to_voice(
    text: str,
    llm: ChatOllama | ChatOpenAI,
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
