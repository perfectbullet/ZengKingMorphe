from app.services.word_to_latex import normalize_spoken_subquestion_markers


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
