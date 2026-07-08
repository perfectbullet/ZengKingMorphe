"""
意图分类 → 回答模式 路由配置（数据驱动，便于扩展）。

设计目标
========
把"分类标签如何决定回答方式"集中维护成一张表，避免散落的 if/elif 硬编码。
新增 / 调整某个标签的路由策略时，只改这张表，不改任何业务代码。

回答模式（AnswerMode）
======================
- ``RAG_WITH_FALLBACK``：先走 RAG（RAGAnything 知识图谱+向量检索），
  当 RAG 全局禁用 / 员工禁用 RAG / RAG 未召回到任何相关内容时，
  自动降级到通用 LLM 回答（避免空回答）。

- ``GENERAL_LLM``：直接通用 LLM 回答，**不走 RAG，不走联网检索**。
  典型场景：闲聊、问候、英语问答、常识性问题、其它非领域查询。

- ``MATH_LLM``：使用数学模型推理（支持 vLLM/Qwen 等多种后端），**不走 RAG**。
  典型场景：具体数学题求解（含计算、证明、推导）。

- ``WEB_SEARCH``：先联网检索 → LLM 基于检索结果回答。
  典型场景：天气、新闻、行情、路况、当下时事等时间敏感查询。

- ``PRESET_RESPONSE``：使用预设话术直接返回，**不走 LLM**。
  典型场景：噪声 / ASR 误识别 / 旁人对话。
  **注意：本模式具有破坏性（用户提问被无 LLM 拦截），必须经过
  ``is_noise_preset_eligible`` 守门后才允许真正落地，详见该函数文档。**

业务路由表（INTENT_TO_ANSWER_MODE）
====================================
按用户需求映射 QueryClassifier 的 9 种标签：

    数学题目（math_problem）         → MATH_LLM
    工训教材 / 工业实训知识库问题（industrial_training_query）→ RAG_WITH_FALLBACK
    通用概念解释（concept_explain）      → GENERAL_LLM
    问候（greeting）                  → GENERAL_LLM
    英语问答（english_query）         → GENERAL_LLM
    常识性问题（general_knowledge）   → GENERAL_LLM
    闲聊（chit_chat）                 → GENERAL_LLM
    实时查询（realtime_query）        → WEB_SEARCH
    噪声（noise）                     → PRESET_RESPONSE
    其他（other）                     → GENERAL_LLM（兜底）

扩展指南
========
- 新增一个分类标签（如 ``code_question``）：
  1. 在 QueryClassifier 中加分类逻辑；
  2. 在本表中加一行 ``"code_question": AnswerMode.GENERAL_LLM`` 即可。
- 调整某个标签的回答策略：直接改本表，不需要改 conversation_nodes.py。

噪声拦截闸门（noise → PRESET_RESPONSE 的安全护栏）
====================================================
小分类器对短/口语化输入有较高误判率，命中 noise 时会直接返回预设话术
"抱歉，我没有听清您的问题，请再重复一次。"——这句话在数字人侧极易被
当事人误判为"麦克风/ASR 故障"。``apply_noise_preset_gate`` 在 noise 命中
后再做三道独立闸门校验，任一不过即把标签**降级为 "other"**（→ GENERAL_LLM），
让 LLM 自行兜住。三道闸门彼此正交、可独立通过 settings 关闭。
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Mapping, Tuple


class AnswerMode(str, Enum):
    """
    回答模式枚举。

    继承 ``str`` 是为了让该枚举值可以直接作为字符串放进
    ``ConversationState`` / 日志 / 数据库（避免 JSON 序列化问题）。
    """

    RAG_WITH_FALLBACK = "rag_with_fallback"
    GENERAL_LLM = "general_llm"
    MATH_LLM = "math_llm"
    WEB_SEARCH = "web_search"
    PRESET_RESPONSE = "preset_response"


# 路由表：分类标签 → 回答模式。
# 维护原则：
# - 集中维护，禁止在业务节点里写 if classification_label == "...":
# - 新增标签：只在这里追加键值对；
# - 修改路由：只改本表的 value，不改 conversation_nodes / generate_answer。
INTENT_TO_ANSWER_MODE: Mapping[str, AnswerMode] = {
    # 数学题目 → 数学模型推理，不走 RAG
    "math_problem": AnswerMode.MATH_LLM,
    # 工训教材 / 工业实训知识库问题 → RAG（无召回时降级 LLM）
    "industrial_training_query": AnswerMode.RAG_WITH_FALLBACK,
    # 通用概念解释 → 通用 LLM，不进入工训 LightRAG
    "concept_explain": AnswerMode.GENERAL_LLM,
    # 问候 / 英语 / 常识 / 闲聊 → 直接通用 LLM
    "greeting": AnswerMode.GENERAL_LLM,
    "english_query": AnswerMode.GENERAL_LLM,
    "general_knowledge": AnswerMode.GENERAL_LLM,
    "chit_chat": AnswerMode.GENERAL_LLM,
    # 实时查询 → 联网检索
    "realtime_query": AnswerMode.WEB_SEARCH,
    # 噪声 → 预设话术
    "noise": AnswerMode.PRESET_RESPONSE,
    # 其它（无法归类）→ 通用 LLM 兜底
    "other": AnswerMode.GENERAL_LLM,
}


# 默认回答模式（标签缺失 / 未知标签时使用）。
# 选 GENERAL_LLM 而非 RAG：未知意图很可能是新增分类标签未及时登记，
# RAG 路径有外部依赖（Milvus / Neo4j / 重排序），出错时影响最大；
# GENERAL_LLM 仅依赖 LLM 本体，最稳。
DEFAULT_ANSWER_MODE: AnswerMode = AnswerMode.GENERAL_LLM


def resolve_answer_mode(classification_label: str | None) -> AnswerMode:
    """
    根据分类标签解析对应的回答模式。

    Args:
        classification_label: ``QueryClassifier`` 返回的分类标签
            （如 ``"math_problem"`` / ``"concept_explain"`` 等）。
            ``None`` 或空串表示未分类。

    Returns:
        对应的 ``AnswerMode``；未登记的标签返回 ``DEFAULT_ANSWER_MODE``。

    设计要点：
        - 容错优先：标签缺失 / 拼错 / 新分类未登记 → 走默认通用 LLM；
        - 与表一致：路由决策必须 100% 来自 ``INTENT_TO_ANSWER_MODE``，
          不在函数里写隐式 fallback / 特例。
    """
    if not classification_label:
        return DEFAULT_ANSWER_MODE
    return INTENT_TO_ANSWER_MODE.get(classification_label, DEFAULT_ANSWER_MODE)


# 路由分支名（``route_after_classification`` 返回值），与 LangGraph
# ``add_conditional_edges`` 的 path_map 严格对齐。
# 与 ``AnswerMode`` 解耦，避免图结构泄漏到 enum 层。
ROUTE_BRANCH_GREETING = "greeting"  # 含 noise，走 generate_answer 直出
ROUTE_BRANCH_REALTIME = "realtime"  # 走 web_search 后再 generate_answer
ROUTE_BRANCH_MATH = "math"          # 走 generate_answer（数学模型）
ROUTE_BRANCH_RAG = "rag"            # 走 concept_retrieval → evaluate_complexity / generate_answer（training RAG / legacy raganything）
ROUTE_BRANCH_GENERAL = "general"    # 走 generate_answer（通用 LLM，不走 RAG）
# 概念检索后的路由分支
ROUTE_BRANCH_CONCEPT_HIT = "concept_hit"      # 命中人工概念库，直接走 generate_answer
ROUTE_BRANCH_CONCEPT_MISS = "concept_miss"   # 未命中人工概念库，走 evaluate_complexity → generate_answer


# 回答模式 → LangGraph 路由分支名。
# 这一层的存在让 graph 拓扑改动最小：未来要拆分 RAG 和 web 的中间节点时，
# 只需要在这张表上 + StateGraph.add_node + add_conditional_edges 即可。
ANSWER_MODE_TO_ROUTE: Mapping[AnswerMode, str] = {
    AnswerMode.RAG_WITH_FALLBACK: ROUTE_BRANCH_RAG,
    AnswerMode.GENERAL_LLM: ROUTE_BRANCH_GENERAL,
    AnswerMode.MATH_LLM: ROUTE_BRANCH_MATH,
    AnswerMode.WEB_SEARCH: ROUTE_BRANCH_REALTIME,
    AnswerMode.PRESET_RESPONSE: ROUTE_BRANCH_GREETING,
}


# =============================================================================
# 噪声预设话术闸门（gate）
# =============================================================================
# 设计原则：
# - 闸门只处理 ``noise`` 这一种标签；其它标签原样返回，避免误伤。
# - 任一闸门不通过即把标签**降级为 "other"** → ``GENERAL_LLM``，
#   不在这里写入"备选标签"机制，避免与 INTENT_TO_ANSWER_MODE 路由表脱节。
# - 闸门内**不调用任何外部依赖**（无 LLM / 无 DB），是纯函数；
#   外层节点只需要传入 query + classifier 结果 + settings 三件事。

# 置信度等级排序（与 QueryClassifier.ConfidenceLevel 完全对齐）。
# 用列表索引比较代替散落的 if/elif，便于运维改 settings 时一目了然。
_CONFIDENCE_ORDER: tuple[str, ...] = ("low", "medium", "high")

# "明显是合法提问"的启发式标记：含问号、数字、运算符、教材/学科常见关键词等。
# 命中即认为输入信息密度足够高，不该判为噪声；与 ``query_classifier.py`` 的
# ``noise`` 提示词正交（提示词描述的是"形如对/嗯/好的"这类零信息片段，
# 这里描述的是"含具体提问要素"的反向特征），不会冲突。
# 维护原则：只追加"几乎不可能出现在噪声里的强信号"，避免把启发式做成全集。
_NOISE_BYPASS_HEURISTIC = re.compile(
    r"[?？]"                              # 问号
    r"|[0-9]"                              # 任意数字
    r"|[+\-*/=≥≤÷×]"                       # 运算符
    r"|(什么是|怎么|为什么|是什么|介绍|讲解|解释|说说|告诉我|帮我)"
    r"|(函数|方程|不等式|集合|数列|导数|积分|极限|向量|矩阵|"
    r"几何|代数|概率|统计|三角|椭圆|双曲线|抛物线|"
    r"天气|新闻|时间|日期|股价|汇率|路况|拥堵)"
)


def _confidence_at_least(actual: str | None, required: str) -> bool:
    """``actual >= required``（``low < medium < high``）。

    未知等级（拼错 / None）保守判 False，等价于"达不到要求"，
    与下游 noise 闸门"够稳才准触发预设"的语义一致。
    """
    if not actual:
        return False
    actual_lc = actual.strip().lower()
    required_lc = (required or "high").strip().lower()
    try:
        return _CONFIDENCE_ORDER.index(actual_lc) >= _CONFIDENCE_ORDER.index(required_lc)
    except ValueError:
        return False


def apply_noise_preset_gate(
    classification_label: str | None,
    classification_confidence: str | None,
    query: str,
    *,
    noise_preset_enabled: bool,
    min_confidence: str = "high",
    max_query_length: int = 12,
) -> Tuple[str | None, str | None]:
    """
    对 ``noise`` 标签做三道安全护栏，返回 (有效标签, 降级原因)。

    - **有效标签**：
        - 标签非 ``noise`` → 原样返回，``downgrade_reason=None``；
        - 标签是 ``noise`` 且全部闸门通过 → 仍返回 ``"noise"``；
        - 标签是 ``noise`` 但任一闸门未过 → 返回 ``"other"``（→ GENERAL_LLM）。
    - **降级原因**（仅在发生降级时非空）：
        - ``"feature_disabled"``：``settings.noise_preset_response_enabled=False``，
          全量回滚的紧急开关；
        - ``"low_confidence"``：分类置信度未达到 ``min_confidence``；
        - ``"query_too_long"``：``len(query.strip()) > max_query_length``，
          长输入即便分类器误判 noise 也按"信息量足够"放行；
        - ``"heuristic_legit_query"``：query 命中合法提问启发式
          （含问号 / 数字 / 运算符 / 学科关键词等）。

    设计要点：
    - 三道闸门彼此**独立**，调用方只需要看返回值，不需要知道命中哪条；
    - 启发式列表集中维护在 ``_NOISE_BYPASS_HEURISTIC``，扩展只加正则；
    - 纯函数：无 LLM 调用、无 IO、可单测。
    """
    if classification_label != "noise":
        return classification_label, None

    if not noise_preset_enabled:
        return "other", "feature_disabled"

    if not _confidence_at_least(classification_confidence, min_confidence):
        return "other", "low_confidence"

    q = (query or "").strip()
    if len(q) > max_query_length:
        return "other", "query_too_long"

    if _NOISE_BYPASS_HEURISTIC.search(q):
        return "other", "heuristic_legit_query"

    return "noise", None


def resolve_route_branch(answer_mode: AnswerMode | str | None) -> str:
    """
    将 ``AnswerMode`` 解析为 LangGraph 条件边分支名。

    Args:
        answer_mode: 已计算好的回答模式；
            可以是 ``AnswerMode`` 实例 / 字符串值 / ``None``。

    Returns:
        分支名常量（``ROUTE_BRANCH_*``）；非法值返回 ``ROUTE_BRANCH_GENERAL``，
        与 ``DEFAULT_ANSWER_MODE`` 的兜底方向保持一致。
    """
    if answer_mode is None:
        return ROUTE_BRANCH_GENERAL
    if isinstance(answer_mode, str) and not isinstance(answer_mode, AnswerMode):
        try:
            answer_mode = AnswerMode(answer_mode)
        except ValueError:
            return ROUTE_BRANCH_GENERAL
    return ANSWER_MODE_TO_ROUTE.get(answer_mode, ROUTE_BRANCH_GENERAL)
