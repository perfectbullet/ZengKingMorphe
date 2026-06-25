from app.services.word_to_latex import (
    normalize_choice_option_markers,
    normalize_spoken_subquestion_markers,
)


def test_normalize_first_second_question_without_separator():
    assert normalize_spoken_subquestion_markers(
        "第一问求E的方程。第二问证明直线CD过定点。"
    ) == "(1)求E的方程。\n\n(2)证明直线CD过定点。"


def test_normalize_three_questions():
    assert normalize_spoken_subquestion_markers(
        "第一问求$p$的值；第二问求恰有一人通过体能测试的概率；第三问求至少有一人通过体能测试的概率。"
    ) == "(1)求$p$的值；\n\n(2)求恰有一人通过体能测试的概率；\n\n(3)求至少有一人通过体能测试的概率。"


def test_normalize_xiaoti_with_spaces():
    assert normalize_spoken_subquestion_markers(
        "第 1 小题，求C的方程；\n\n第 2 小题，证明结论。"
    ) == "(1)求C的方程；\n\n(2)证明结论。"


def test_normalize_wenti_cn_num():
    assert normalize_spoken_subquestion_markers(
        "问题一求参数a。问题二证明不等式。"
    ) == "(1)求参数a。\n\n(2)证明不等式。"


def test_normalize_after_comma_keeps_weak_separator():
    # 真实 ASR 场景：第一小问前是中文逗号（弱分隔），第二小问前是句号（强分隔）。
    assert normalize_spoken_subquestion_markers(
        "$，第一小问：求周期。第二小问：求值。"
    ) == "$，(1)：求周期。\n\n(2)：求值。"


def test_normalize_subsequent_markers_after_comma():
    assert normalize_spoken_subquestion_markers(
        "已知函数f(x)，第一小问求周期，第二小问求值。"
    ) == "已知函数f(x)，(1)求周期，(2)求值。"


def test_do_not_normalize_after_third():
    assert normalize_spoken_subquestion_markers("第四问求参数。") == "第四问求参数。"
    assert normalize_spoken_subquestion_markers("第4问求参数。") == "第4问求参数。"
    assert normalize_spoken_subquestion_markers("问题四求参数。") == "问题四求参数。"
    assert normalize_spoken_subquestion_markers("问题4求参数。") == "问题4求参数。"


def test_do_not_normalize_problem_number():
    assert normalize_spoken_subquestion_markers("第22题。已知椭圆C。") == "第22题。已知椭圆C。"
    assert normalize_spoken_subquestion_markers("第 22 题。已知椭圆C。") == "第 22 题。已知椭圆C。"


def test_do_not_normalize_existing_structured_markers():
    assert normalize_spoken_subquestion_markers("(1).求E的方程。") == "(1).求E的方程。"
    assert normalize_spoken_subquestion_markers("（1）求E的方程。") == "（1）求E的方程。"
    assert normalize_spoken_subquestion_markers("1. 求E的方程。") == "1. 求E的方程。"


def test_normalize_choice_options_with_uppercase_labels():
    text = "下列结论正确的是：A选项，事件B与C互斥；B选项，事件A与C互斥；C选项，任何两个均不互斥；D选项，任何两个均互斥。"
    assert normalize_choice_option_markers(text) == (
        "下列结论正确的是：\n"
        "A. 事件B与C互斥；\n"
        "B. 事件A与C互斥；\n"
        "C. 任何两个均不互斥；\n"
        "D. 任何两个均互斥。"
    )


def test_normalize_choice_options_with_lowercase_labels():
    text = "则下列说法正确的是：a 选项$q=2$，b 选项数列$S_n + 2$是等比数列，c 选项$a_1=1$，d 选项$S_n=2a_n - 2$。"
    assert normalize_choice_option_markers(text) == (
        "则下列说法正确的是：\n"
        "A. $q=2$，\n"
        "B. 数列$S_n + 2$是等比数列，\n"
        "C. $a_1=1$，\n"
        "D. $S_n=2a_n - 2$。"
    )


def test_normalize_choice_options_with_reversed_marker():
    text = "这是一道多选题。选项A $x>0$；选项B $x<0$；选项C $x=0$；选项D $x\\ne0$。"
    assert normalize_choice_option_markers(text) == (
        "这是一道多选题。\n"
        "A. $x>0$；\n"
        "B. $x<0$；\n"
        "C. $x=0$；\n"
        "D. $x\\ne0$。"
    )


def test_normalize_choice_options_keep_original_order():
    text = "下列说法正确的是：A 选项，事件 A 与事件 B 对立；C 选项，事件 A 与事件 C 相互独立；B 选项，事件 A 与事件 B 相互独立；D 选项，P(C) 等于 P(括号 A B) 括号。"
    assert normalize_choice_option_markers(text) == (
        "下列说法正确的是：\n"
        "A. 事件 A 与事件 B 对立；\n"
        "C. 事件 A 与事件 C 相互独立；\n"
        "B. 事件 A 与事件 B 相互独立；\n"
        "D. P(C) 等于 P(括号 A B) 括号。"
    )


def test_normalize_choice_options_without_space_after_marker():
    text = "则下列选项哪个正确？这是一道多选题。A选项$MN$平行于平面$AA_1C_1C$；B选项$MN$平行于平面；C选项$MN$垂直于$PC$；D选项三棱锥$A_1$矩$MNP$的体积为12。"
    assert normalize_choice_option_markers(text) == (
        "则下列选项哪个正确？这是一道多选题。\n"
        "A. $MN$平行于平面$AA_1C_1C$；\n"
        "B. $MN$平行于平面；\n"
        "C. $MN$垂直于$PC$；\n"
        "D. 三棱锥$A_1$矩$MNP$的体积为12。"
    )


def test_do_not_normalize_choice_options_without_choice_context():
    text = "这里提到A选项，但这句话不是数学题。"
    assert normalize_choice_option_markers(text) == text


def test_do_not_normalize_existing_choice_markers():
    text = "下列说法正确的是：A. $x>0$；B. $x<0$。"
    assert normalize_choice_option_markers(text) == text
