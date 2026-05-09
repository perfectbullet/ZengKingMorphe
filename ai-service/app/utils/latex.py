"""
LaTeX formula processing utilities.

Provides LaTeX formula normalization, cleaning, and escaping functions.
"""
import re
from typing import List, Optional

def normalize_latex_delimiters(text: str) -> str:
    r"""
    Normalize LaTeX delimiters by replacing escaped forms with standard format.

    Replacement rules:
    - \( or \\( or \\\( → $
    - \) or \\) or \\\) → $
    - \[ or \\[ or \\\[ → $$
    - \] or \\] or \\\] → $$

    Args:
        text: Text containing potentially escaped LaTeX delimiters

    Returns:
        Normalized text
    """
    # Process from multiple backslashes to single backslashes
    text = re.sub(r'\\\\\(', '$', text)
    text = re.sub(r'\\\\\)', '$', text)
    text = re.sub(r'\\\\\[', '$$', text)
    text = re.sub(r'\\\\\]', '$$', text)

    text = re.sub(r'\\\(', '$', text)
    text = re.sub(r'\\\)', '$', text)
    text = re.sub(r'\\\[', '$$', text)
    text = re.sub(r'\\\]', '$$', text)

    return text

# Backward compatibility alias
_normalize_latex_delimiters = normalize_latex_delimiters

def clean_latex_formula_spaces(text: str) -> str:
    """
    Remove spaces inside LaTeX formula delimiters.

    Processing rules:
    - `$ text $` → `$text$`
    - `$$ text $$` → `$$text$$`
    - Only removes spaces directly adjacent to delimiters, keeps internal spaces
    - External spaces (e.g., "formula $x^2$ and") are preserved

    Examples:
        >>> clean_latex_formula_spaces("$ S_n = a_1 \\cdot q^{n-1} $")
        '$S_n = a_1 \\cdot q^{n-1}$'
        >>> clean_latex_formula_spaces("$$ \\frac{a}{b} $$")
        '$$\\frac{a}{b}$$'
        >>> clean_latex_formula_spaces("公式 $x^2$ 和 $$y+z$$")
        '公式 $x^2$ 和 $$y+z$$'

    Args:
        text: Text containing LaTeX formulas

    Returns:
        Text with spaces removed inside delimiter boundaries
    """
    result = []
    i = 0
    in_formula = False
    formula_delimiter: Optional[str] = None
    just_entered_formula = False

    while i < len(text):
        if i < len(text) - 1 and text[i] == '\\' and text[i + 1] == '$':
            result.append('\\$')
            i += 2
            continue

        if text[i:i+2] == '$$':
            if not in_formula:
                in_formula = True
                formula_delimiter = '$$'
                result.append('$$')
                just_entered_formula = True
            elif formula_delimiter == '$$':
                while result and result[-1] == ' ':
                    result.pop()
                in_formula = False
                formula_delimiter = None
                just_entered_formula = False
                result.append('$$')
            i += 2
            continue

        if text[i] == '$':
            if not in_formula:
                in_formula = True
                formula_delimiter = '$'
                result.append('$')
                just_entered_formula = True
            elif formula_delimiter == '$':
                while result and result[-1] == ' ':
                    result.pop()
                in_formula = False
                formula_delimiter = None
                just_entered_formula = False
                result.append('$')
            i += 1
            continue

        if in_formula and just_entered_formula and text[i] == ' ':
            i += 1
            while i < len(text) and text[i] == ' ':
                i += 1
            just_entered_formula = False
            continue

        just_entered_formula = False
        result.append(text[i])
        i += 1

    return ''.join(result)

def escape_latex_backslashes(text: str) -> str:
    """
    Escape single backslashes to double backslashes within LaTeX formulas.

    Only processes content within $...$ or $$...$$ delimiters.

    Examples:
        >>> escape_latex_backslashes("$S_n = a_1 \\frac{1-q^n}{1-q}$")
        '$S_n = a_1 \\\\frac{1-q^n}{1-q}$'
        >>> escape_latex_backslashes("公式 $x^2$ 和 $$y+z$$")
        '公式 $x^2$ 和 $$y+z$$'

    Args:
        text: Text containing LaTeX formulas

    Returns:
        Text with backslashes escaped within formulas
    """
    result: List[str] = []
    i = 0
    in_formula = False
    formula_delimiter: Optional[str] = None

    while i < len(text):
        if i < len(text) - 1 and text[i] == '\\' and text[i + 1] == '$':
            result.append('\\$')
            i += 2
            continue

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

        if in_formula and text[i] == '\\':
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

def wrap_bare_boxed(text: str) -> str:
    r"""
    Wrap bare \boxed{...} in $...$ delimiters.

    Math models (e.g., Phi-4) may output \boxed{(2, 3)} without $ delimiters.
    This function wraps only \boxed{...} that is not already inside $...$ or $$...$$.

    Examples:
        >>> wrap_bare_boxed("答案为 \\boxed{(2, 3)}")
        '答案为 $\\boxed{(2, 3)}$'
        >>> wrap_bare_boxed("$$\\boxed{(2, 3)}$$")
        '$$\\boxed{(2, 3)}$$'

    Args:
        text: Text potentially containing bare \boxed{...}

    Returns:
        Text with bare \boxed{...} wrapped in $...$
    """
    BOXED_PATTERN = re.compile(r'\\boxed\s*\{')
    result: list[str] = []
    i = 0
    in_formula = False
    formula_delim: Optional[str] = None

    while i < len(text):
        # Handle existing $$ delimiters
        if text[i:i+2] == '$$':
            if not in_formula:
                in_formula = True
                formula_delim = '$$'
            elif formula_delim == '$$':
                in_formula = False
                formula_delim = None
            result.append('$$')
            i += 2
            continue

        # Handle existing $ delimiters
        if text[i] == '$':
            if not in_formula:
                in_formula = True
                formula_delim = '$'
            elif formula_delim == '$':
                in_formula = False
                formula_delim = None
            result.append('$')
            i += 1
            continue

        # Already inside formula — copy verbatim
        if in_formula:
            result.append(text[i])
            i += 1
            continue

        # Check for bare \boxed{...}
        m = BOXED_PATTERN.match(text, i)
        if m:
            start = i
            i = m.end()
            # Consume the brace group {...} (matching nested braces)
            depth = 1
            while i < len(text) and depth > 0:
                if text[i] == '\\' and i + 1 < len(text):
                    i += 2
                    continue
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                i += 1
            # Wrap in $...$
            result.append('$')
            result.append(text[start:i])
            result.append('$')
        else:
            result.append(text[i])
            i += 1

    return ''.join(result)

def normalize_latex_formulas(text: str) -> str:
    r"""
    Perform complete LaTeX formula normalization.

    Applies operations in order:
    1. Normalize delimiters (\( \) \[ \] → $ $$)
    2. Remove spaces inside delimiters
    3. Wrap bare \boxed{...} in $...$

    Examples:
        >>> normalize_latex_formulas(r"\( S_n = a_1 \cdot q^{n-1} \)")
        '$S_n = a_1 \\cdot q^{n-1}$'
        >>> normalize_latex_formulas("$$ \\frac{a}{b} $$")
        '$$\\frac{a}{b}$$'
        >>> normalize_latex_formulas("\\boxed{(2, 3)}")
        '$\\boxed{(2, 3)}$'

    Args:
        text: Text containing LaTeX formulas

    Returns:
        Fully normalized text
    """
    text = normalize_latex_delimiters(text)
    text = clean_latex_formula_spaces(text)
    text = wrap_bare_boxed(text)
    text = re.sub(r'\\text\b', r'\\mathrm', text)
    return text
