"""
LaTeX 公式处理工具模块。

提供 LaTeX 公式的规范化、清理和转义功能。
"""
import re
from typing import List, Optional


def normalize_latex_delimiters(text: str) -> str:
    """
    规范化 LaTeX 定界符，将各种转义形式替换为标准格式。

    替换规则：
    - \( 或 \\( 或 \\\( → $
    - \) 或 \\) 或 \\\) → $
    - \[ 或 \\[ 或 \\\[ → $$
    - \] 或 \\] 或 \\\] → $$

    Args:
        text: 包含可能转义的 LaTeX 定界符的文本

    Returns:
        规范化后的文本
    """
    # 按顺序处理，从多反斜杠到少反斜杠
    # \\( 和 \\) → $ (两个反斜杠 + 括号)
    text = re.sub(r'\\\\\(', '$', text)
    text = re.sub(r'\\\\\)', '$', text)
    text = re.sub(r'\\\\\[', '$$', text)
    text = re.sub(r'\\\\\]', '$$', text)

    # \( 和 \) → $ (一个反斜杠 + 括号)
    # 注意：在原始字符串中，\\ 表示一个反斜杠，\( 表示字面上的括号
    text = re.sub(r'\\\(', '$', text)
    text = re.sub(r'\\\)', '$', text)
    text = re.sub(r'\\\[', '$$', text)
    text = re.sub(r'\\\]', '$$', text)

    return text


# 向后兼容的别名
_normalize_latex_delimiters = normalize_latex_delimiters


def clean_latex_formula_spaces(text: str) -> str:
    """
    移除 LaTeX 公式定界符内侧的空格。

    处理规则：
    - `$ text $` → `$text$`
    - `$$ text $$` → `$$text$$`
    - 仅移除定界符直接相邻的空格，保留公式内部的空格
    - 公式外的空格（如"公式 $x^2$ 和"中的空格）保持不变

    Examples:
        >>> clean_latex_formula_spaces("$ S_n = a_1 \\cdot q^{n-1} $")
        '$S_n = a_1 \\cdot q^{n-1}$'
        >>> clean_latex_formula_spaces("$$ \\frac{a}{b} $$")
        '$$\\frac{a}{b}$$'
        >>> clean_latex_formula_spaces("公式 $x^2$ 和 $$y+z$$")
        '公式 $x^2$ 和 $$y+z$$'

    Args:
        text: 包含 LaTeX 公式的文本

    Returns:
        移除定界符内侧空格后的文本
    """
    result = []
    i = 0
    in_formula = False
    formula_delimiter: Optional[str] = None
    just_entered_formula = False  # 刚进入公式，用于跳过开定界符后的空格

    while i < len(text):
        # 检查是否遇到转义的美元符号 \$，跳过
        if i < len(text) - 1 and text[i] == '\\' and text[i + 1] == '$':
            result.append('\\$')
            i += 2
            continue

        # 检查 $$ 定界符
        if text[i:i+2] == '$$':
            if not in_formula:
                in_formula = True
                formula_delimiter = '$$'
                result.append('$$')
                just_entered_formula = True
            elif formula_delimiter == '$$':
                # 退出公式前，检查并跳过前面的空格
                while result and result[-1] == ' ':
                    result.pop()
                in_formula = False
                formula_delimiter = None
                just_entered_formula = False
                result.append('$$')
            i += 2
            continue

        # 检查 $ 定界符
        if text[i] == '$':
            if not in_formula:
                in_formula = True
                formula_delimiter = '$'
                result.append('$')
                just_entered_formula = True
            elif formula_delimiter == '$':
                # 退出公式前，检查并跳过前面的空格
                while result and result[-1] == ' ':
                    result.pop()
                in_formula = False
                formula_delimiter = None
                just_entered_formula = False
                result.append('$')
            i += 1
            continue

        # 在公式内刚进入时，跳过空格
        if in_formula and just_entered_formula and text[i] == ' ':
            i += 1
            # 只跳过紧接定界符后的连续空格
            while i < len(text) and text[i] == ' ':
                i += 1
            just_entered_formula = False
            continue

        # 跳过了初始空格后，正常处理其他字符
        just_entered_formula = False
        result.append(text[i])
        i += 1

    return ''.join(result)


def escape_latex_backslashes(text: str) -> str:
    """
    将 LaTeX 公式内的单反斜杠转义为双反斜杠。

    仅处理 $...$ 或 $$...$$ 定界符内的内容，公式外的文本保持不变。

    Examples:
        >>> escape_latex_backslashes("$S_n = a_1 \\frac{1-q^n}{1-q}$")
        '$S_n = a_1 \\\\frac{1-q^n}{1-q}$'
        >>> escape_latex_backslashes("公式 $x^2$ 和 $$y+z$$")
        '公式 $x^2$ 和 $$y+z$$'

    Args:
        text: 包含 LaTeX 公式的文本

    Returns:
        反斜杠转义后的文本
    """
    result: List[str] = []
    i = 0
    in_formula = False
    formula_delimiter: Optional[str] = None

    while i < len(text):
        # 检查是否遇到转义的美元符号 \$，跳过
        if i < len(text) - 1 and text[i] == '\\' and text[i + 1] == '$':
            result.append('\\$')
            i += 2
            continue

        # 检查 $$ 定界符
        if text[i:i+2] == '$$':
            if not in_formula:
                in_formula = True
                formula_delimiter = '$$'
                result.append('$$')
            elif formula_delimiter == '$$':
                in_formula = False
                formula_delimiter = None
                result.append('$$')
            i += 2
            continue

        # 检查 $ 定界符
        if text[i] == '$':
            if not in_formula:
                in_formula = True
                formula_delimiter = '$'
                result.append('$')
            elif formula_delimiter == '$':
                in_formula = False
                formula_delimiter = None
                result.append('$')
            i += 1
            continue

        # 在公式内，转义反斜杠
        if in_formula and text[i] == '\\':
            # 检查是否已经是双反斜杠，如果是则不重复转义
            if i + 1 < len(text) and text[i + 1] == '\\':
                result.append('\\\\')
                i += 2
            else:
                result.append('\\\\')
                i += 1
        else:
            result.append(text[i])
            i += 1

    return ''.join(result)


def normalize_latex_formulas(text: str) -> str:
    """
    对 LaTeX 公式进行完整的规范化处理。

    依次执行：
    1. 规范化定界符（\( \) \[ \] → $ $$）
    2. 移除定界符内侧的空格
    3. 转义公式内的反斜杠

    Examples:
        >>> normalize_latex_formulas("\\( S_n = a_1 \\cdot q^{n-1} \\)")
        '$S_n = a_1 \\\\cdot q^{n-1}$'
        >>> normalize_latex_formulas("$$ \\frac{a}{b} $$")
        '$$\\\\frac{a}{b}$$'

    Args:
        text: 包含 LaTeX 公式的文本

    Returns:
        完全规范化后的文本
    """
    text = normalize_latex_delimiters(text)
    text = clean_latex_formula_spaces(text)
    text = escape_latex_backslashes(text)
    return text
