---
name: morphe-math-testing
description: >
  Test and review the mathematical-question versus non-mathematical-question
  routing and replies in the ZengKingMorphe service. Use this skill whenever a
  task involves 数学题回复, streaming smoke tests (tests.test_chat_stream_v2),
  math pytest suites, math_problem routing, LaTeX 题干转换, math model debug
  dumps, or diagnosing why a query selected the math LLM instead of the general
  LLM — even if the user only says "测一下这道题", "看看这题为什么答错", or
  "跑一下数学测试". Use it before marking a change to this conversation path done.
---

# Morphe 数学 / 非数学链路测试与复核

本分支只验证两类请求：数学问题与非数学问题。数学题走专用数学模型；非数学题走通用模型。
不测试工业实训、知识库召回或 RAG 行为。

## 环境前提

- 所有 Python 命令使用 conda 环境 `morphe`：
  `/home/zj/miniconda3/envs/morphe/bin/python`
- 测试命令在 `ai-service/` 目录运行。
- 流式冒烟测试调用已启动服务的 `/api/chat/v2/chat/completions`；服务、分类服务和数学模型服务都需要可访问。
- 连接超时或模型不可用先按环境问题处理；使用 debug dump 区分题干、路由、模型服务三类问题。

## 最小验证组合

```bash
cd ai-service
PY=/home/zj/miniconda3/envs/morphe/bin/python

# 数学题：应路由至 math_llm
$PY -m tests.test_chat_stream_v2 -q "求解不等式x的平方减去5x加上6小于0的解" --host http://localhost:8100

# 非数学题：应路由至通用 LLM，不能误用数学模型
$PY -m tests.test_chat_stream_v2 -q "北京今天天气怎么样" --host http://localhost:8100
```

数学题输出应包含逐步推理与 `\boxed{}` 最终答案；非数学题应自然回答，日志中不应显示 `type=math_llm`。

v2 脚本输出 SSE 文本、首 token 延迟和响应 metadata。路由不在客户端响应中时，从服务端日志或数学 debug dump 核对 `classification_label`、`is_math_problem` 与 `streaming_type`。

## 定向 pytest

按改动范围选择最窄测试：

```bash
cd ai-service
PY=/home/zj/miniconda3/envs/morphe/bin/python

# 数学 / 非数学路由
$PY -m pytest -v tests/test_math_concept_intent_routing.py

# 数学追问、题干 LaTeX 化、流式分块
$PY -m pytest -v tests/test_context_resolution_math.py tests/test_latex_utils.py tests/test_stream_chunks.py
```

| 文件 | 覆盖点 |
|---|---|
| `tests/test_math_concept_intent_routing.py` | 数学标签与非数学标签的路由 |
| `tests/test_context_resolution_math.py` | 数学追问的上下文补全 |
| `tests/test_latex_utils.py` | LaTeX 清洗与规范化 |
| `tests/test_stream_chunks.py` | 流式分块格式 |
| `tests/test_chat_stream_v2.py` | v2 流式端点联调（本分支唯一使用的冒烟脚本） |

旧 v1 直调脚本及批量数学题脚本已归档到 `archive/legacy-v1/tests/`，不用于本分支验收。

## 路由验收标准

| 请求类型 | 预期模式 | 关键验收 |
|---|---|---|
| `math_problem` | `MATH_LLM` | 走数学模型，不走通用模型；题干完成 LaTeX 转换；答案含 `\boxed{}` |
| 非数学问题 | `GENERAL_LLM` | 走通用模型，不能误路由到 `math_llm` |
| `noise` | `PRESET_RESPONSE` 或通用兜底 | 不调用数学模型 |

关键实现文件：

- 分类与数学启发式：`app/services/query_classifier.py`、`app/services/math_intent_heuristic.py`
- 路由表：`app/services/conversation/intent_routing.py`
- 会话节点与题干预处理：`app/services/conversation/conversation_nodes.py`

在日志中核对 `Streaming configured: type=math_llm|langchain_llm` 与 `classification_label=...`。

## 数学模型调试 dump

数学分支默认写入 `ai-service/logs/math_model_debug/`。每个 JSON 包含实际题干、完整输出、运行状态和耗时。

排查答错时依次确认：题干转换是否正确、路由是否为 `math_llm`、发给数学模型的提示词是否干净。数学链路不应混入历史对话。

## 提交前检查

1. 数学题走数学模型，非数学题不走数学模型。
2. 数学题题干已正确转换为 LaTeX，最终答案在 `\boxed{}` 中。
3. 中文提问使用中文回答，面向 TTS 的输出不残留 LaTeX 语法。
4. 改动涉及的最窄测试已运行；外部服务不可用时，记录未运行的命令与原因。
5. 不提交 `logs/`、`graph_debug/`、debug dump 或生成的 JSON/TXT 产物。
