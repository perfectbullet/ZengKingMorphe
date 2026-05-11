"""
启发式补位：识别小模型分类器（QueryClassifier）漏判的「数学题目」与「教材概念题」。

设计目标
========
qwen2.5:7b 这类小模型分类器对长问句 / 不带"求/解/计算"动词的应用题
（"已知正方体的棱长为 2，则其对角线的长度为多少?"）
和教材式问答（"在数列的学习中，请分别说明...推导方法，比较异同"）
存在系统性漏判，会落到 ``general_knowledge / chit_chat / other`` 这类兜底标签上，
导致原本应走 Phi-4 的数学题被通用 LLM 接住、原本应走 RAG 的教材题被通用 LLM 直答。

本模块以"轻量正则规则 + 互斥优先级"做兜底纠偏，**只在主分类器给出
弱标签（general_knowledge / chit_chat / other）时才介入**，避免覆盖
LLM 已经识别准确的强分类（math_problem / concept_explain / realtime_query 等）。

与 ``realtime_intent_heuristic`` 同思路：
- 规则数据化（正则 + 词表），扩展只加规则不改业务代码；
- 命中明确模式时返回布尔信号；交由调用方决定是否覆盖标签；
- 不依赖任何 LLM / 数据库，纯函数零副作用。
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 概念解释 / 教学题 指示词
#
# _CONCEPT_INDICATOR：明确"请讲解 / 介绍 / 比较异同 / 推导方法"等元语言信号。
# 一旦命中，几乎可以肯定是"在请求解释"，不是"在请求计算"。
# 因此它同时也作为 math_problem 的"屏蔽词"——避免把
# "请说明等差数列前 n 项和的推导方法" 误判成"求等差数列的和"。
# ---------------------------------------------------------------------------
_CONCEPT_INDICATOR = re.compile(
    r"(请\s*分别\s*(?:说明|介绍|讲解|解释|论述)|"
    r"请\s*(?:讲解|介绍|解释|说明|论述)|"
    r"什么是|是什么|的定义|的概念|的含义|"
    r"推导方法|推导过程|证明过程|证明思路|"
    r"异同|相同点|不同点|区别和联系|区别与联系)"
)

# 教学语境：明显在"讲教材 / 讲课程"，常见于学校/课程问答。
# 单独命中只是弱信号，需要再叠加 _CONCEPT_INDICATOR 或长度阈值才升级到
# concept_explain，避免把"在数学课上认识了一位老师"等闲聊误判为教材题。
_TEACHING_CONTEXT = re.compile(
    r"(在.{0,10}的?学习中|我们\s*学(?:习|过)了?|从教材中|在课本中|"
    r"根据教材|根据课本|教材中|课本中|课程中)"
)

# ---------------------------------------------------------------------------
# 数学关键词汇总
#
# 只要在 query 中命中以下任一关键词，且未被 _CONCEPT_INDICATOR 屏蔽，
# 即判定为数学计算题（math_problem）。
#
# 负前瞻用于排除日常用语：
# - 球(?![员场队赛])  排除 球员/球场/球队/球赛
# - 圆(?![形圈])      排除 圆形/圆圈
# - 高(?![中考])      排除 高中/高考
# - 方程(?!式)        排除 方程式（赛车等）
# ---------------------------------------------------------------------------
_MATH_KEYWORD = re.compile(
    r"(正方体|立方体|长方体|球体|球(?=[体面心的内外半径体积表侧])|圆锥|圆柱|圆台|"
    r"棱锥|棱柱|棱台|三角锥|椎体|"                            # 几何体
    r"三角形|正方形|长方形|矩形|梯形|平行四边形|菱形|多边形|"
    r"圆(?![形圈])|扇形|椭圆|双曲线|抛物线|"                    # 平面图形
    r"函数|方程(?!式)|方程组|不等式|数列|"
    r"集合|向量|矩阵|行列式|"                                   # 代数对象
    r"三角函数|导数|积分|极限|微分|"                             # 分析对象
    r"长度|面积|体积|半径|周长|边长|棱长|对角线|侧面积|表面积|底面积|"
    r"高(?![中考矮兴级手速铁])|底面|斜高|母线|斜边|焦距|内切|外接|"  # 几何属性
    r"取值范围|最大值|最小值|定义域|值域|根|解集|交集|并集|补集|"   # 代数属性
    r"下标|通项(?:公式)?|递推(?:关系|公式)?|公比|公差|"
    r"等差|等比|前\s*[nN]\s*项|"                                # 数列属性
    r"化简|因式分解|证明|推导(?![的方过])|"
    r"(?<![需追])求(?![求给你了起])|计算(?!机)|求解|"            # 数学动词
    r"为多少|等于多少)"                                         # 问法模式
)


def heuristic_math_problem(query: str) -> bool:
    """
    判断 query 是否包含数学关键词，应视为数学计算题。

    逻辑：
    1. 命中 _CONCEPT_INDICATOR（请讲解 / 是什么 / 推导方法 / 异同等）→ False
       此类 query 是"请求解释"，不是"请求计算"。
    2. 命中 _MATH_KEYWORD（任一数学术语 / 问法）→ True
    3. 其他 → False

    Returns:
        True  → 强烈建议升级为 ``math_problem``（走 Phi-4）
        False → 不构成数学题强信号，由调用方决定保持原标签或继续后续启发式
    """
    q = (query or "").strip()
    if not q:
        return False

    # 概念解释题排除：明确"请求解释"而非"请求计算"
    if _CONCEPT_INDICATOR.search(q):
        return False

    return bool(_MATH_KEYWORD.search(q))


def heuristic_concept_explain(query: str) -> bool:
    """
    判断 query 是否高置信度命中"教材 / 概念解释题"模式。

    匹配规则（满足任一即返回 True）：
    1. **教学语境 + 概念指示词**：``在 X 的学习中`` / ``我们学习了`` 等
       叠加 ``请分别说明 / 推导方法 / 异同`` 等明确请求。
       例："在数列的学习中…请分别说明…推导方法，并比较这两种方法的异同。"
    2. **概念指示词 + 长问句（≥25 字）**：长问句通常是教材式提问，
       结合 ``请讲解 / 推导方法 / 异同`` 即可视为概念题。
       例："请详细讲解一下二项式定理的推导过程及其在组合数学中的应用。"
    3. **教学语境 + 长问句（≥25 字）**：教学语境本身较弱，但叠加长度阈值
       后基本可以排除噪声 / 闲聊。

    短问句（< 25 字）即便命中 _CONCEPT_INDICATOR 也不在这里强制升级——
    那种短句（"什么是导数"）一般主分类器自己就能给出 concept_explain，
    不需要启发式干预。

    Returns:
        True  → 强烈建议升级为 ``concept_explain``（走 RAG）
        False → 不构成概念题强信号，保持调用方原标签
    """
    q = (query or "").strip()
    if not q:
        return False

    has_concept = bool(_CONCEPT_INDICATOR.search(q))
    has_teaching = bool(_TEACHING_CONTEXT.search(q))
    long_enough = len(q) >= 25

    if has_teaching and has_concept:
        return True
    if has_concept and long_enough:
        return True
    if has_teaching and long_enough:
        return True

    return False


# 调用方使用："只对这些弱标签触发启发式纠偏"
# 不包括 ``math_problem / concept_explain / realtime_query / english_query / greeting / noise``，
# 避免覆盖 LLM 已经识别准确的强分类。
HEURISTIC_PROMOTABLE_LABELS: frozenset[str] = frozenset(
    {"general_knowledge", "chit_chat", "other"}
)
