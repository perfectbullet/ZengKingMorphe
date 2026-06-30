"""word_to_latex 选项标记（a选项 / $a$选项 / 选项$a$）规范化单测。

背景：word_to_latex 模型偶发把 a选项 / b选项 / c选项 / d选项 中的
a/b/c/d 当作数学变量，输出 $a$选项 / $b$选项。normalize_choice_option_markers
需要能在选择题语境下把这类标记统一还原为 A. / B. / C. / D.。
"""

from app.services.word_to_latex import (
    clean_model_output,
    normalize_choice_option_markers,
    has_full_choice_option_markers,
)


def test_normalize_choice_option_markers_plain_lowercase_options():
    text = "(多选题)若条件成立，则a选项$f(x)$正确。b选项$b=-2$。c选项函数$g(x)$有零点。d选项结论成立。"
    result = normalize_choice_option_markers(text)

    assert "A. $f(x)$正确。" in result
    assert "B. $b=-2$。" in result
    assert "C. 函数$g(x)$有零点。" in result
    assert "D. 结论成立。" in result
    assert "a选项" not in result
    assert "b选项" not in result


def test_normalize_choice_option_markers_dollar_wrapped_options():
    text = (
        "(多选题)设函数$f(x)$定义域为$\\mathbb{R}$。"
        "若$f(0)-f(3)=1$，则$a$选项$f(x)$图象关于直线$x=1$对称。"
        "$b$选项$b=-2$。"
        "$c$选项函数$g(x)=f(x)-x$，恰有三个零点。"
        "$d$选项$f(1)+f(2)+f(3)+\\cdots+f(2026)=0$。"
    )

    result = normalize_choice_option_markers(text)

    assert "A. $f(x)$图象关于直线$x=1$对称。" in result
    assert "B. $b=-2$。" in result
    assert "C. 函数$g(x)=f(x)-x$，恰有三个零点。" in result
    assert "D. $f(1)+f(2)+f(3)+\\cdots+f(2026)=0$。" in result

    assert "$a$选项" not in result
    assert "$b$选项" not in result
    assert "$c$选项" not in result
    assert "$d$选项" not in result


def test_normalize_choice_option_markers_dollar_wrapped_options_with_inner_spaces():
    text = "(多选题)则$ a $选项正确。$ b $选项错误。$ c $选项待定。$ d $选项不成立。"
    result = normalize_choice_option_markers(text)

    assert "A. 正确。" in result
    assert "B. 错误。" in result
    assert "C. 待定。" in result
    assert "D. 不成立。" in result


def test_clean_model_output_normalizes_dollar_wrapped_choice_options():
    text = (
        "(多选题)若$f(0)-f(3)=1$，则$a$选项$f(x)$图象关于直线$x=1$对称。"
        "$b$选项$b=-2$。$c$选项函数$g(x)=f(x)-x$，恰有三个零点。"
        "$d$选项$f(1)+f(2)=0$。"
    )

    result = clean_model_output(text)

    assert "A. $f(x)$图象关于直线$x=1$对称。" in result
    assert "B. $b=-2$。" in result
    assert "C. 函数$g(x)=f(x)-x$，恰有三个零点。" in result
    assert "D. $f(1)+f(2)=0$。" in result


def test_normalize_choice_option_markers_does_not_touch_non_choice_context():
    text = "设$a$为实数，函数$f(x)=ax+b$。"
    result = normalize_choice_option_markers(text)
    assert result == text


def test_has_full_choice_option_markers_detects_dollar_wrapped():
    text = (
        "(多选题)$a$选项甲。$b$选项乙。$c$选项丙。$d$选项丁。"
    )
    assert has_full_choice_option_markers(text) is True


def test_has_full_choice_option_markers_false_for_partial():
    text = "(多选题)$a$选项甲。$b$选项乙。"
    assert has_full_choice_option_markers(text) is False
