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

- ``PHI4_MATH``：使用 Phi-4 数学模型推理，**不走 RAG**。
  典型场景：具体数学题求解（含计算、证明、推导）。

- ``WEB_SEARCH``：先联网检索 → LLM 基于检索结果回答。
  典型场景：天气、新闻、行情、路况、当下时事等时间敏感查询。

- ``PRESET_RESPONSE``：使用预设话术直接返回，**不走 LLM**。
  典型场景：噪声 / ASR 误识别 / 旁人对话。

业务路由表（INTENT_TO_ANSWER_MODE）
====================================
按用户需求映射 QueryClassifier 的 9 种标签：

    数学题目（math_problem）         → PHI4_MATH
    数学概念 / 教材知识（concept_explain）→ RAG_WITH_FALLBACK
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
"""
from __future__ import annotations

from enum import Enum
from typing import Mapping


class AnswerMode(str, Enum):
    """
    回答模式枚举。

    继承 ``str`` 是为了让该枚举值可以直接作为字符串放进
    ``ConversationState`` / 日志 / 数据库（避免 JSON 序列化问题）。
    """

    RAG_WITH_FALLBACK = "rag_with_fallback"
    GENERAL_LLM = "general_llm"
    PHI4_MATH = "phi4_math"
    WEB_SEARCH = "web_search"
    PRESET_RESPONSE = "preset_response"


# 路由表：分类标签 → 回答模式。
# 维护原则：
# - 集中维护，禁止在业务节点里写 if classification_label == "...":
# - 新增标签：只在这里追加键值对；
# - 修改路由：只改本表的 value，不改 conversation_nodes / generate_answer。
INTENT_TO_ANSWER_MODE: Mapping[str, AnswerMode] = {
    # 数学题目 → 数学模型（Phi-4），不走 RAG
    "math_problem": AnswerMode.PHI4_MATH,
    # 数学概念 / 教材知识 / 公式解释 / 知识点问答 → RAG（无召回时降级 LLM）
    "concept_explain": AnswerMode.RAG_WITH_FALLBACK,
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
ROUTE_BRANCH_MATH = "math"          # 走 generate_answer（Phi-4）
ROUTE_BRANCH_RAG = "rag"            # 走 generate_answer（RAGAnything）
ROUTE_BRANCH_GENERAL = "general"    # 走 generate_answer（通用 LLM，不走 RAG）


# 回答模式 → LangGraph 路由分支名。
# 这一层的存在让 graph 拓扑改动最小：未来要拆分 RAG 和 web 的中间节点时，
# 只需要在这张表上 + StateGraph.add_node + add_conditional_edges 即可。
ANSWER_MODE_TO_ROUTE: Mapping[AnswerMode, str] = {
    AnswerMode.RAG_WITH_FALLBACK: ROUTE_BRANCH_RAG,
    AnswerMode.GENERAL_LLM: ROUTE_BRANCH_GENERAL,
    AnswerMode.PHI4_MATH: ROUTE_BRANCH_MATH,
    AnswerMode.WEB_SEARCH: ROUTE_BRANCH_REALTIME,
    AnswerMode.PRESET_RESPONSE: ROUTE_BRANCH_GREETING,
}


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
