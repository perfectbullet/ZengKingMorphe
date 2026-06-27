"""
LaTeX 工具函数单测。

聚焦 clean_latex_formula_spaces / normalize_latex_formulas 对行内公式
定界符内侧 whitespace 的清理能力（前端要求 `$xxxx$`，不接受 `$ xxxx $`）。
"""
from app.utils.latex import clean_latex_formula_spaces, normalize_latex_formulas


def test_clean_latex_formula_spaces_inline_basic():
    text = "已知椭圆 $ C $，其中 $ a > b > 0 $。"
    assert clean_latex_formula_spaces(text) == "已知椭圆 $C$，其中 $a > b > 0$。"


def test_clean_latex_formula_spaces_preserves_inner_spaces():
    text = "$ a > b > 0 $"
    assert clean_latex_formula_spaces(text) == "$a > b > 0$"


def test_clean_latex_formula_spaces_strips_whitespace_not_only_space():
    text = "公式 $\n C: x^2 + y^2 = 1 \t$ 结束"
    assert clean_latex_formula_spaces(text) == "公式 $C: x^2 + y^2 = 1$ 结束"


def test_clean_latex_formula_spaces_display_math():
    text = "$$ \\frac{x}{y} $$"
    assert clean_latex_formula_spaces(text) == "$$\\frac{x}{y}$$"


def test_clean_latex_formula_spaces_preserves_display_math_newlines():
    # 块级公式 $$\n...\n$$ / $$\t...\t$$ 是合法多行格式，\n / \t 必须保留
    assert clean_latex_formula_spaces("$$\n\\frac{a}{b}\n$$") == "$$\n\\frac{a}{b}\n$$"
    assert clean_latex_formula_spaces("$$\n\nx = y\n\n$$") == "$$\n\nx = y\n\n$$"
    # 块级公式里的普通空格仍应清除（与上一个 display_math 测试一致）
    assert clean_latex_formula_spaces("$$ \\frac{x}{y} $$") == "$$\\frac{x}{y}$$"


def test_clean_latex_formula_spaces_escaped_dollar():
    text = r"价格是 \$ 5，不是公式 $ x $"
    assert clean_latex_formula_spaces(text) == r"价格是 \$ 5，不是公式 $x$"


def test_normalize_latex_formulas_removes_dollar_inner_spaces_from_word_to_latex():
    text = (
        r"已知椭圆 $ C: \frac{x^{2}}{a^{2}} + \frac{y^{2}}{b^{2}} = 1 $"
        r"（其中 $ a > b > 0 $），且过点 $ A(2,1) $。"
    )
    expected = (
        r"已知椭圆 $C: \frac{x^{2}}{a^{2}} + \frac{y^{2}}{b^{2}} = 1$"
        r"（其中 $a > b > 0$），且过点 $A(2,1)$。"
    )
    assert normalize_latex_formulas(text) == expected


def test_normalize_latex_formulas_converts_paren_delimiters_and_strips_spaces():
    text = r"已知 \( C: x^2 + y^2 = 1 \)，且 \( a > b > 0 \)。"
    expected = r"已知 $C: x^2 + y^2 = 1$，且 $a > b > 0$。"
    assert normalize_latex_formulas(text) == expected
