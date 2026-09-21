"""
LaTeX formula / math sentence → voice-friendly text conversion service.

把数学公式、含数学符号的句子转换为适合 TTS 朗读的中文口语文本。

设计要点：
- LLM 客户端由 ``get_voice_conversion_llm()`` 内部按需创建并缓存，调用方不再传递 llm。
- ``LLM_BASE_URL`` / ``LLM_MODEL`` / ``LLM_API_KEY`` 为项目级必需配置，启动时即由主 LLM（conversation_service）保证可用，这里与主 LLM 共用、直接读取不额外校验。
- ``\\boxed{...}`` 只去外壳保留内部内容（``\\boxed{8}`` → ``8``）；简单内容直接返回，复杂内容交 LLM。
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
- 遇到 \boxed{...} 时，只读取其中的内容，不要读 boxed、方框、框起来，也不要额外添加"答案是"

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
# LLM client (cached)
# =============================================================================
# LLM_BASE_URL / LLM_MODEL / LLM_API_KEY 为项目级必需配置，启动时即由主 LLM
# （conversation_service）保证可用；这里与主 LLM 共用同一组配置，直接读取、不额外校验。

_voice_conversion_llm: ChatOpenAI | None = None


def get_voice_conversion_llm() -> ChatOpenAI:
    """返回用于语音转换的 ChatOpenAI（懒加载单例）。

    复用项目统一的 ``LLM_BASE_URL`` / ``LLM_MODEL`` / ``LLM_API_KEY``；
    ``streaming=False`` 配合 ``ainvoke`` 一次性返回，其余参数与主 LLM 一致。
    """
    global _voice_conversion_llm

    if _voice_conversion_llm is None:
        _voice_conversion_llm = ChatOpenAI(
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY", "no-key"),
            model=os.getenv("LLM_MODEL"),
            temperature=0.1,
            streaming=False,
        )
        logger.info("[VoiceConversionLLM] ChatOpenAI initialized")

    return _voice_conversion_llm


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
# Boxed wrapper removal & LaTeX delimiter helpers
# =============================================================================

_BOXED_MARKER = r"\boxed{"


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


def _remove_boxed_wrappers(text: str) -> str:
    """去除 ``\\boxed{...}`` 外壳，保留内部原始内容（不额外补"答案是"）。

    基于花括号深度扫描，支持嵌套（``\\boxed{\\frac{2}{3}}`` → ``\\frac{2}{3}``）、
    多个 boxed、boxed 内再嵌套 boxed；转义括号（``\\{`` / ``\\}``）不计入深度。
    boxed 结构不完整时保留原文，不抛异常。

    Examples:
        \\boxed{8} -> 8
        \\boxed{\\frac{2}{3}} -> \\frac{2}{3}
        $$\\boxed{8}$$ -> $$8$$
    """
    if not text or "\\boxed" not in text:
        return text

    result: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        start = text.find(_BOXED_MARKER, i)
        if start == -1:
            result.append(text[i:])
            break

        result.append(text[i:start])

        content_start = start + len(_BOXED_MARKER)
        depth = 1
        j = content_start

        while j < n and depth > 0:
            char = text[j]
            # 跳过转义字符（\{ \} \\ 等），避免误计花括号深度
            if char == "\\" and j + 1 < n:
                j += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            j += 1

        if depth != 0:
            # boxed 结构不完整，不强行处理，保留原文
            result.append(text[start:])
            break

        inner = text[content_start:j - 1]
        # 递归处理 boxed 内部再次出现的 boxed
        result.append(_remove_boxed_wrappers(inner))
        i = j

    return "".join(result)


def _is_simple_unboxed_content(text: str) -> bool:
    """去壳后的内容是否可直接朗读、无需 LLM 转换。

    含 LaTeX 命令（反斜杠）或花括号结构（``\\frac``、``\\sqrt``、``{...}``）
    时返回 False，交给 LLM；数字、``x=1``、``-2`` 等返回 True。
    """
    stripped = text.strip()
    if not stripped:
        return False
    if "\\" in stripped:
        return False
    if "{" in stripped or "}" in stripped:
        return False
    return True


def basic_math_to_voice(text: str) -> str:
    """Deterministically remove common LaTeX/math syntax for failure fallback."""
    result = _strip_latex_delimiters(_remove_boxed_wrappers(text)).strip()

    # Resolve common two-argument and one-argument commands before dropping braces.
    fraction = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    square_root = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
    for _ in range(4):
        updated = fraction.sub(r"\1 除以 \2", result)
        updated = square_root.sub(r"根号下 \1", updated)
        if updated == result:
            break
        result = updated

    command_replacements = {
        r"\times": "乘以",
        r"\cdot": "乘以",
        r"\div": "除以",
        r"\pm": "正负",
        r"\leq": "小于等于",
        r"\le": "小于等于",
        r"\geq": "大于等于",
        r"\ge": "大于等于",
        r"\neq": "不等于",
        r"\approx": "约等于",
        r"\infty": "无穷",
        r"\pi": "派",
    }
    for command, spoken in command_replacements.items():
        result = result.replace(command, f" {spoken} ")

    result = re.sub(r"([A-Za-z0-9)])\s*\^\s*\{?2\}?", r"\1 的平方", result)
    result = re.sub(r"([A-Za-z0-9)])\s*\^\s*\{?3\}?", r"\1 的立方", result)
    result = re.sub(
        r"([A-Za-z0-9)])\s*\^\s*\{([^{}]+)\}",
        r"\1 的 \2 次方",
        result,
    )
    result = re.sub(r"([A-Za-z0-9)])\s*\^\s*([A-Za-z0-9]+)", r"\1 的 \2 次方", result)

    for symbol, spoken in (
        ("<=", "小于等于"), (">=", "大于等于"), ("!=", "不等于"),
        ("≤", "小于等于"), ("≥", "大于等于"), ("≠", "不等于"),
        ("≈", "约等于"), ("∈", "属于"), ("∪", "并集"), ("∩", "交集"),
        ("=", "等于"), ("+", "加"), ("-", "减"), ("×", "乘以"),
        ("*", "乘以"), ("÷", "除以"), ("/", "除以"),
        ("<", "小于"), (">", "大于"), ("√", "根号"), ("∞", "无穷"),
    ):
        result = result.replace(symbol, f" {spoken} ")

    result = re.sub(r"\\(?:left|right|mathrm|mathbf|text)\b", "", result)
    result = re.sub(r"\\[A-Za-z]+", " ", result)
    result = result.replace("{", " ").replace("}", " ")
    result = result.replace("$", "").replace(r"\(", "").replace(r"\)", "")
    result = result.replace(r"\[", "").replace(r"\]", "").replace("\\", "")
    return re.sub(r"\s+", " ", result).strip()


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

    先去掉 ``\\boxed{}`` 外壳与 LaTeX 定界符；若剩余内容简单（数字、``x=1`` 等）
    直接返回；否则请求 LLM（``ainvoke``，带超时）。
    任何异常 / 空结果都 ``logger.exception`` 后返回去壳后的内容，绝不返回 ``\\boxed{...}``。
    """
    unboxed_formula = _remove_boxed_wrappers(formula)
    stripped_formula = _strip_latex_delimiters(unboxed_formula)

    if unboxed_formula != formula and _is_simple_unboxed_content(stripped_formula):
        logger.info(
            f"[_convert_single_formula] Formula unboxed and returned directly | input={formula[:100]!r} | output={stripped_formula[:100]!r}"
        )
        return basic_math_to_voice(stripped_formula)

    messages = [
        {"role": "system", "content": FORMULA_ONLY_PROMPT},
        {"role": "user", "content": stripped_formula},
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
                f"[_convert_single_formula] Empty conversion result, fallback to unboxed formula | input={formula[:300]!r} | unboxed={stripped_formula[:300]!r}"
            )
            return basic_math_to_voice(stripped_formula or unboxed_formula)

        logger.info(
            f"[_convert_single_formula] Formula converted by LLM | input={formula[:100]!r} | normalized={stripped_formula[:100]!r} | output={result[:100]!r}"
        )
        return basic_math_to_voice(result) if re.search(r"[\\$=+×÷≤≥≠≈]", result) else result

    except Exception:
        logger.exception(
            f"[_convert_single_formula] Formula conversion failed, fallback to unboxed formula | input={formula[:300]!r} | unboxed={stripped_formula[:300]!r}"
        )
        return basic_math_to_voice(stripped_formula or unboxed_formula)


# =============================================================================
# Main conversion functions
# =============================================================================

async def convert_formula_to_voice(text: str) -> str:
    """把含 LaTeX 公式的文本转换为语音友好文本（仅替换公式部分）。

    先整体去掉 ``\\boxed{}`` 外壳再做公式抽取/替换，保证语音里不会读出 ``boxed``。
    best-effort：顶层异常 ``logger.exception`` 后返回去壳后的原文（同样不含 ``\\boxed``）。
    """
    try:
        if _is_empty_or_delimiter_only(text):
            return text

        normalized_text = _remove_boxed_wrappers(text)

        if not _has_any_formula_marker(normalized_text):
            if "\\" in normalized_text:
                return basic_math_to_voice(normalized_text)
            return _remove_list_markers(normalized_text)

        formulas = _extract_latex_formulas(normalized_text)
        if not formulas:
            return normalized_text

        result = normalized_text
        # 倒序替换，避免位置偏移
        for formula, start, end in reversed(formulas):
            voice_formula = await _convert_single_formula(formula)
            result = result[:start] + voice_formula + result[end:]

        return result

    except Exception:
        logger.exception(
            f"[convert_formula_to_voice] Failed, fallback to unboxed original text | text={(text[:500] if text else text)!r}"
        )
        return basic_math_to_voice(text) if text else text


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
            return basic_math_to_voice(text)

        logger.info(
            f"[convert_math_sentence_to_voice] Math sentence converted | input={text[:100]!r} | output={result[:100]!r}"
        )
        return basic_math_to_voice(result) if re.search(r"[\\$=+×÷≤≥≠≈]", result) else result

    except Exception:
        logger.exception(
            f"[convert_math_sentence_to_voice] Failed, fallback to original text | input={text[:500]!r}"
        )
        return basic_math_to_voice(text)


__all__ = [
    "get_voice_conversion_llm",
    "convert_formula_to_voice",
    "convert_math_sentence_to_voice",
    "basic_math_to_voice",
]
