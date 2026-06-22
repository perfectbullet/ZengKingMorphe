"""
LaTeX formula / math sentence → voice-friendly text conversion service.

把数学公式、含数学符号的句子转换为适合 TTS 朗读的中文口语文本。

设计要点：
- LLM 客户端由 ``get_voice_conversion_llm()`` 内部按需创建并缓存，调用方不再传递 llm。
- ``LLM_BASE_URL`` / ``LLM_MODEL`` 为必需环境变量，缺失即报错（由调用处兜底降级为原文）。
- 公式转换优先走规则（如 ``\\boxed{...}`` → ``答案是 ...``），命中规则不请求 LLM。
- 所有转换均为 best-effort：失败 ``logger.exception`` 后返回原文，不中断主聊天流。

Formula Extraction Strategy:
- Extract complete formulas using regex
- Convert only formula content (not surrounding text)
- Replace converted formulas back into original text
"""
import asyncio
import os
import re

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
- $(a + b)^n$ → a 加 b 的 n 次方
- $a + b$ → a 加 b
- $a - b$ → a 减 b
- $a \times b$ → a 乘以 b
- $a / b$ → a 除以 b
- $x = \pm 1$ → x 等于 正负一
- $(-\infty, -1)$ → 负无穷到一的开区间
- $(a, b)$ → a 到 b 的开区间
- $[a, b]$ → a 到 b 的闭区间

请严格遵守以上风格，直接输出转换结果，不要任何解释，也不要包含反斜杠、$ 符号或任何 LaTeX 命令。
'''

MATH_SENTENCE_PROMPT = """你负责把包含数学符号的中文句子改写为适合 TTS 播报的口语文本。

要求：
1. 保持原数学含义，不新增推理，不解释过程。
2. 把数学符号读成中文：+读作加，-读作减，×读作乘以，/读作除以，=读作等于，≤读作小于等于，≥读作大于等于，≠读作不等于，≈读作约等于。
3. 常见集合和运算符也要口语化：∈读作属于，∪读作并集，∩读作交集，√读作根号，∞读作无穷，π读作派。
4. 不输出 LaTeX 命令、反斜杠、美元符号或 Markdown。
5. 只输出转换后的口语文本，不要解释。

示例：
输入：x²-2=0
输出：x 的平方减 2 等于 0

输入：x∈R
输出：x 属于 R
"""

# =============================================================================
# Required environment variables
# =============================================================================


def _get_required_env(name: str) -> str:
    """读取必需环境变量；缺失或为空直接抛 RuntimeError。"""
    value = os.getenv(name)
    if not value or not value.strip():
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value.strip()


# =============================================================================
# LLM client (cached, required env)
# =============================================================================

_voice_conversion_llm: ChatOpenAI | None = None


def get_voice_conversion_llm() -> ChatOpenAI:
    """返回用于语音转换的 ChatOpenAI（单例缓存）。

    - ``LLM_BASE_URL`` / ``LLM_MODEL`` 必需，缺失抛 RuntimeError。
    - ``streaming=False``：语音转换用 ``ainvoke`` 一次性返回，不需要流式。
    - ``LLM_API_KEY`` 允许缺省为 ``"no-key"``（本地 OpenAI 兼容服务常不校验）。
    """
    global _voice_conversion_llm

    if _voice_conversion_llm is not None:
        return _voice_conversion_llm

    base_url = _get_required_env("LLM_BASE_URL")
    model = _get_required_env("LLM_MODEL")

    if not base_url.rstrip("/").endswith("/v1"):
        base_url = f"{base_url.rstrip('/')}/v1"

    _voice_conversion_llm = ChatOpenAI(
        base_url=base_url,
        api_key=os.getenv("LLM_API_KEY") or "no-key",
        model=model,
        temperature=0.1,
        streaming=False,
    )

    logger.info(
        f"[VoiceConversionLLM] ChatOpenAI initialized | BASE_URL={base_url} | MODEL={model}"
    )

    return _voice_conversion_llm


def reset_voice_conversion_llm() -> None:
    """重置缓存的 LLM 客户端（主要供测试在切换环境变量后使用）。"""
    global _voice_conversion_llm
    _voice_conversion_llm = None


# =============================================================================
# Timeout config
# =============================================================================

FORMULA_CONVERSION_TIMEOUT_SECONDS = float(
    os.getenv("FORMULA_CONVERSION_TIMEOUT_SECONDS", "8")
)
MATH_SENTENCE_CONVERSION_TIMEOUT_SECONDS = float(
    os.getenv("MATH_SENTENCE_CONVERSION_TIMEOUT_SECONDS", "8")
)

# =============================================================================
# Rule-based conversion (boxed → 答案是 ...)
# =============================================================================

_BOXED_PREFIX = r"\boxed{"


def _strip_latex_delimiters(formula: str) -> str:
    """剥离最外层的一对 LaTeX 定界符（$$/$/\\[/\\(），未匹配则原样返回。"""
    text = formula.strip()

    pairs = [
        ("$$", "$$"),
        ("$", "$"),
        (r"\[", r"\]"),
        (r"\(", r"\)"),
    ]

    for left, right in pairs:
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            return text[len(left): -len(right)].strip()

    return text


def _extract_boxed_content(text: str, start: int) -> tuple[str | None, int]:
    """从 ``text[start]``（应位于 ``\\boxed{`` 之后的首字符）按花括号深度提取内容。

    正确处理嵌套花括号（如 ``\\boxed{\\frac{2}{3}}``）与转义括号（``\\{`` / ``\\}``）。
    返回 ``(内容, 闭括号之后的位置)``；若括号失衡返回 ``(None, start)``。
    """
    depth = 1
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            i += 2  # 跳过转义字符（\{ \} \\ 等）
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i], i + 1
        i += 1
    return None, start


def _convert_formula_by_rule(formula: str) -> str | None:
    """规则转换：纯 ``\\boxed{...}`` 公式 → ``答案是 ...``。

    仅当整个公式（去掉定界符后）恰好是一个 ``\\boxed{...}`` 时命中；
    形如 ``\\boxed{a}+\\boxed{b}`` 的复合公式不命中，交给 LLM。
    仅处理简单结果（数字、``x=1`` 等，不含 LaTeX 命令）；含 ``\\frac``/``\\sqrt``
    等 LaTeX 命令的内容（如 ``\\boxed{\\frac{2}{3}}``）原样读不通，交给 LLM。
    返回 ``None`` 表示规则未命中。
    """
    inner = _strip_latex_delimiters(formula.strip()).strip()
    if not inner.startswith(_BOXED_PREFIX):
        return None

    content_start = len(_BOXED_PREFIX)
    value, end = _extract_boxed_content(inner, content_start)
    if value is None:
        return None

    # boxed 闭合后若还残留非空白内容，说明不是纯 boxed 公式，交给 LLM
    if inner[end:].strip():
        return None

    value = value.strip()
    if not value:
        return None
    # 含 LaTeX 命令（反斜杠）的复杂结果，规则读不通口语，交给 LLM
    if "\\" in value:
        return None
    return f"答案是 {value}"


# =============================================================================
# Text inspection helpers
# =============================================================================

def _is_empty_or_delimiter_only(text: str) -> bool:
    """文本为空、纯空白，或剥离 LaTeX 定界符后为空。"""
    if not text or not text.strip():
        return True

    stripped = text.strip()
    for delim in ['$$', '$', r'\(', r'\)', r'\[', r'\]']:
        stripped = stripped.replace(delim, '')

    return not stripped.strip()


def _has_any_formula_marker(text: str) -> bool:
    """是否包含至少一个完整 LaTeX 公式（$...$/$$...$$/\\(...\\)/\\[...\\]）。"""
    return LATEX_FORMULA_PATTERN.search(text) is not None


def _remove_list_markers(text: str) -> str:
    """移除 Markdown 列表标记（-, *, +），保留缩进，避免被 TTS 读成"减"。"""
    return MARKDOWN_LIST_PATTERN.sub(r'\1', text)


def _extract_latex_formulas(text: str) -> list[tuple[str, int, int]]:
    """提取所有完整 LaTeX 公式，返回 ``(formula, start, end)`` 列表（按出现顺序）。"""
    formulas = []
    for match in LATEX_FORMULA_PATTERN.finditer(text):
        formulas.append((match.group(0), match.start(), match.end()))
    return formulas


# =============================================================================
# Single formula / sentence conversion (self-managed LLM + ainvoke)
# =============================================================================

async def _convert_single_formula(formula: str) -> str:
    """转换单个公式为语音友好文本。

    优先走规则（boxed），未命中再请求 LLM（``ainvoke``，带超时）。
    任何异常都 ``logger.exception`` 后返回原公式，不抛出。
    """
    rule_result = _convert_formula_by_rule(formula)
    if rule_result is not None:
        logger.info(
            f"[_convert_single_formula] Formula converted by rule | input={formula[:100]!r} | output={rule_result[:100]}"
        )
        return rule_result

    messages = [
        {"role": "system", "content": FORMULA_ONLY_PROMPT},
        {"role": "user", "content": formula},
    ]

    try:
        llm = get_voice_conversion_llm()
        response = await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=FORMULA_CONVERSION_TIMEOUT_SECONDS,
        )
        result = (response.content or "").strip()

        if not result:
            logger.warning(
                f"[_convert_single_formula] Empty conversion result, fallback to original formula | input={formula[:300]!r}"
            )
            return formula

        logger.info(
            f"[_convert_single_formula] Formula converted by LLM | input={formula[:100]!r} | output={result[:100]!r}"
        )
        return result

    except Exception:
        logger.exception(
            f"[_convert_single_formula] Formula conversion failed, fallback to original formula | input={formula[:300]!r}"
        )
        return formula


# =============================================================================
# Main conversion functions
# =============================================================================

async def convert_formula_to_voice(text: str) -> str:
    """把含 LaTeX 公式的文本转换为语音友好文本（仅替换公式部分）。

    best-effort：空文本/无公式快速返回；命中规则的 boxed 不请求 LLM；
    顶层异常 ``logger.exception`` 后返回原文。
    """
    try:
        if _is_empty_or_delimiter_only(text):
            return text

        if not _has_any_formula_marker(text):
            return _remove_list_markers(text)

        formulas = _extract_latex_formulas(text)
        if not formulas:
            return text

        result = text
        # 倒序替换，避免位置偏移
        for formula, start, end in reversed(formulas):
            voice_formula = await _convert_single_formula(formula)
            result = result[:start] + voice_formula + result[end:]

        return result

    except Exception:
        logger.exception(
            f"[convert_formula_to_voice] Failed, fallback to original text | text={(text[:500] if text else text)!r}"
        )
        return text


async def convert_math_sentence_to_voice(text: str) -> str:
    """把含数学符号的中文句子改写为口语文本（``ainvoke``，带超时）。

    best-effort：失败 ``logger.exception`` 后返回原文。
    """
    if not text or not text.strip():
        return text

    messages = [
        {"role": "system", "content": MATH_SENTENCE_PROMPT},
        {"role": "user", "content": text},
    ]

    try:
        llm = get_voice_conversion_llm()
        response = await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=MATH_SENTENCE_CONVERSION_TIMEOUT_SECONDS,
        )
        result = (response.content or "").strip()

        if not result:
            logger.warning(
                f"[convert_math_sentence_to_voice] Empty conversion result, fallback to original text | input={text[:300]!r}"
            )
            return text

        logger.info(
            f"[convert_math_sentence_to_voice] Math sentence converted | input={text[:100]!r} | output={result[:100]!r}"
        )
        return result

    except Exception:
        logger.exception(
            f"[convert_math_sentence_to_voice] Failed, fallback to original text | input={text[:500]!r}"
        )
        return text


__all__ = [
    "get_voice_conversion_llm",
    "reset_voice_conversion_llm",
    "convert_formula_to_voice",
    "convert_math_sentence_to_voice",
]
