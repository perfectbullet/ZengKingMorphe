"""
用户查询分类器 - 使用 LLM 进行细粒度分类。

单例模式设计，通过 get_query_classifier() 获取实例。

分类类别:
- math_problem: 数学题目解答
- concept_explain: 概念解释
- greeting: 问候语
- english_query: 英语问题
- realtime_query: 联网检索
- general_knowledge: 常识性问题
- chit_chat: 闲聊/对话
- noise: 噪声/无效
- other: 其他
"""

import json
import time
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# 分类结果模型
# =============================================================================
ClassificationLabel = Literal[
    "math_problem",
    "concept_explain",
    "greeting",
    "english_query",
    "realtime_query",
    "general_knowledge",
    "chit_chat",
    "noise",
    "other",
]

ConfidenceLevel = Literal["high", "medium", "low"]


class ClassificationResult(BaseModel):
    """分类结果模型"""
    label: ClassificationLabel
    confidence: ConfidenceLevel
    reason: str


# =============================================================================
# QueryClassifier 类
# =============================================================================
class QueryClassifier:
    """
    用户查询分类器 - 使用 LLM 进行细粒度分类。

    单例模式，通过 get_query_classifier() 获取实例。
    """

    # 分类标签名称映射
    LABEL_NAMES = {
        "math_problem": "数学题目解答",
        "concept_explain": "概念解释",
        "greeting": "问候语",
        "english_query": "英语问题",
        "realtime_query": "联网检索",
        "general_knowledge": "常识性问题",
        "chit_chat": "闲聊/对话",
        "noise": "噪声/无效",
        "other": "其他",
    }

    # 系统提示词
    SYSTEM_PROMPT = """你是一个用户查询分类专家。你的任务是对用户输入进行细粒度分类。

## 分类类别及定义

### 1. math_problem（数学题目解答）
用户需要求解具体的数学题目，需要通过计算、推导、证明等步骤得出答案。
**特征**：
- 包含数学操作词：求、计算、解、证明、推导、化简、判断、比较
- 包含具体数字、参数或变量
- 需要计算过程或推理步骤
- 可能包含引导语如"你帮我..."、"请..."

**示例**：
- "求不等式 x²-5x+6<0 的解集"
- "计算 lim(x→0) sin(x)/x 的值"
- "证明：若 a>b>0，则 1/a < 1/b"
- "你帮我求一下这个方程的解"

### 2. concept_explain（概念解释）
用户询问概念、定义、定理、公式的含义，不需要计算求解。
**特征**：
- 包含概念性词汇：什么是、是什么、介绍、解释、定义、概念、含义
- 不涉及具体计算或求解
- 旨在理解知识点
- **包括**："你帮我讲解..."、"请介绍一下..."等引导语

**示例**：
- "什么是函数"
- "介绍一下等差数列"
- "导数的几何意义是什么"
- "你帮我讲解一下二项式定理"
- "请介绍一下集合的概念"

### 3. greeting（问候语）
用户进行打招呼、问候、礼貌用语。
**特征**：
- 简短的问候词汇
- 无实质性提问内容

**示例**：
- "你好"
- "早上好"
- "在吗"
- "嗨"
- "hello"

### 4. english_query（英语问题）
问题主体是英文的知识问答或英语学习问题。
**特征**：
- 问题主体是英文
- 可能是英语学习问题或英文知识问答
- **排除**：明显是ASR错误产生的无意义英文片段

**示例**：
- "What is the capital of Canada?"
- "Explain the difference between weather and climate"
- "Who wrote the play Hamlet?"
- "What is the Pythagorean theorem?"

### 5. realtime_query（需要联网检索）
用户询问需要最新信息的问题，如天气、新闻、实时行情等。
**特征**：
- 时间敏感：今天、明天、最近、当前、现在、最新
- 领域：天气、新闻、股价、汇率、行情、价格

**示例**：
- "北京今天天气怎么样"
- "最近有什么新闻"
- "现在黄金价格是多少"
- "明天会下雨吗"

### 6. general_knowledge（常识性问题）
用户询问一般知识、百科、常识类问题，不需要联网获取最新信息。
**特征**：
- 百科类知识
- 历史事实、科学常识
- 不涉及数学专业计算

**示例**：
- "中国的首都在哪里"
- "太阳系有几大行星"
- "一加一等于几"
- "水的化学式是什么"

### 7. chit_chat（闲聊/对话）
用户进行日常对话、表达情绪、与AI闲聊，不属于有效提问。
**特征**：
- 社交性对话
- 情绪表达
- 无明确知识需求
- 可能是对话中的一部分（非完整问题）

**示例**：
- "你在干嘛"
- "我累了"
- "哈哈"
- "谢谢你"
- "你是谁"

### 8. noise（噪声/无效输入）
完全无效的内容，包括多种子类型：
- ASR识别错误产生的无意义文本
- **旁人对话**（非对AI说话）："对"、"是的"、"好的"、"嗯"、"不是"、"不对"
- **设备/技术调试对话**："你看那个接口"、"这个按钮点击没反应"
- 不完整的句子碎片
- 重复填充（如"OK,OK,OK..."重复多次）

**示例**：
- "对，就是那个，或者边的那俩"
- "函数，我们20"
- "你你你你你"
- "你看那个接口返回的是什么"
- "好的好的好的"（重复多次）

### 9. other（其他）
无法归入以上任何类别的问题。

---

## 输出格式

直接返回 JSON 对象，格式如下：
```json
{
  "label": "类别名称",
  "confidence": "high/medium/low",
  "reason": "简短理由（不超过20字）"
}
```

**label 取值**：`math_problem`, `concept_explain`, `greeting`, `realtime_query`, `general_knowledge`, `chit_chat`, `noise`, `other`

**confidence 说明**：
- `high`：类别判断非常明确，无明显歧义
- `medium`：基本可以判断，但有轻微歧义
- `low`：类别判断不太确定，可能属于其他类别

只返回 JSON 对象，不要返回其他内容。"""

    def __init__(self, llm: ChatOpenAI):
        """
        初始化分类器。

        Args:
            llm: LangChain LLM 实例 (ChatOpenAI 或兼容的 LLM)
        """
        self.llm = llm

    def _parse_llm_response(self, content: str) -> ClassificationResult:
        """
        解析 LLM 返回的分类结果。

        Args:
            content: LLM 返回的内容（可能是 JSON 或被 markdown 包裹的 JSON）

        Returns:
            ClassificationResult 对象
        """
        # 提取 JSON（可能被 markdown 代码块包裹）
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
            if text.startswith("json"):
                text = text[4:].strip()

        try:
            data = json.loads(text)
            return ClassificationResult(
                label=data.get("label", "other"),
                confidence=data.get("confidence", "low"),
                reason=data.get("reason", ""),
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}, content={text[:100]}")
            return ClassificationResult(
                label="other",
                confidence="low",
                reason=f"JSON解析失败: {str(e)[:20]}",
            )

    async def aclassify(self, query: str) -> ClassificationResult:
        """
        异步分类单个查询。

        Args:
            query: 用户查询文本

        Returns:
            ClassificationResult 对象
        """
        # 截断过长的查询
        truncated_query = query[:500]

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"请对以下 query 进行分类：\n\n{truncated_query}"},
        ]

        try:
            start_time = time.time()

            response = await self.llm.ainvoke(messages)

            duration = time.time() - start_time
            result = self._parse_llm_response(response.content)
            logger.info(
                f"Query classified: label={result.label}, confidence={result.confidence}, "
                f"reason={result.reason}, query={query[:50]}, duration={duration:.3f}s"
            )
            return result
        except Exception as e:
            logger.error(f"LLM classification failed: {e}", exc_info=True)
            return ClassificationResult(
                label="other",
                confidence="low",
                reason=f"LLM调用失败",
            )

    def classify(self, query: str) -> ClassificationResult:
        """
        同步分类单个查询。

        Args:
            query: 用户查询文本

        Returns:
            ClassificationResult 对象
        """
        # 截断过长的查询
        truncated_query = query[:500]

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"请对以下 query 进行分类：\n\n{truncated_query}"},
        ]

        try:
            start_time = time.time()

            response = self.llm.invoke(messages)

            duration = time.time() - start_time
            result = self._parse_llm_response(response.content)
            logger.debug(
                f"Query classified: label={result.label}, confidence={result.confidence}, "
                f"reason={result.reason}, query={query[:50]}, duration={duration:.3f}s"
            )
            return result
        except Exception as e:
            logger.error(f"LLM classification failed: {e}", exc_info=True)
            return ClassificationResult(
                label="other",
                confidence="low",
                reason=f"LLM调用失败",
            )


# =============================================================================
# 单例实例管理
# =============================================================================
_query_classifier: QueryClassifier | None = None


def get_query_classifier() -> QueryClassifier:
    """
    获取 QueryClassifier 单例实例。

    首次调用时会创建实例，后续调用返回缓存的实例。

    Returns:
        QueryClassifier 实例
    """
    global _query_classifier

    if _query_classifier is None:
        base_url = settings.ollama_base_url
        model = settings.ollama_model

        # Add /v1 suffix if not present (Ollama compatibility)
        if not base_url.endswith("/v1"):
            base_url = f"{base_url.rstrip('/')}/v1"

        logger.info(f"[QueryClassifier] ChatOpenAI | BASE_URL={base_url} | MODEL={model}")

        llm = ChatOpenAI(
            base_url=base_url,
            api_key="no-key",
            model=model,
            temperature=0.0,
            max_tokens=128,
        )
        _query_classifier = QueryClassifier(llm)
        logger.info("QueryClassifier singleton initialized")

    return _query_classifier
