"""
用户查询分类器 - 使用 LLM 进行细粒度分类。

单例模式设计，通过 get_query_classifier() 获取实例。

分类类别:
- math_problem: 数学题目解答
- industrial_training_query: 工训教材 / 工业实训知识库问题
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
import os
import re
import time
from datetime import datetime
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger
from app.utils.common import has_language_drift

logger = get_logger(__name__)

# 上下文消歧提示词：把指代/省略型追问补全为独立完整问句
_CONTEXT_RESOLVER_PROMPT = """你是「对话上下文消歧」模块。

给定多轮对话摘要与用户最新一句话，请输出**一句**完整、可独立理解的问句或陈述（与用户使用同一语言）。
规则：
- 若最新一句已自洽、无需上文，则原样输出。
- 若有省略、指代、承接上文，请根据对话补全缺失的主语/主题/时间范围，保持原意与语气。
- **指代/省略消歧时，优先承接「最近一轮用户问句」中的主题或对象**。
  例：上文用户依次问过 A、B 两个不同主题，最近一轮是 B；本轮出现「它/他/这个」时，
  默认指代 B 的主题，而不是更早的 A——除非语境（动词搭配、属性词）明确排除 B。
- 不要回答问题，不要解释，不要加引号或前缀，只输出这一行文本。
- 若最新输入本质是提问（含「？」或语义上是追问），输出必须保持为提问句，不得直接给结论。
- 可参考系统日期（用于「今年」「现在」等相对时间）：{system_date}
"""


_DATE_ANSWER_PATTERN = re.compile(r"(?:19|20)\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*[日号]")


def sanitize_resolved_query(latest_query: str, resolved_line: str) -> str:
    """
    对消歧器输出做安全清洗，防止两类常见的"语义偏移"污染下游：

    1. **直接作答漂移**：模型把"改写问句"误做成"输出结论"
       （例如对"今天是几月几号？"直接吐出"2026年5月4日"），会导致后续路由错误。
    2. **跨语言漂移**：模型把英文问句翻译成中文（或反向），会让下游 LLM 收到
       与用户语言不符的"用户问题"，进而以错误语言作答（造成"英文问、中文答"）。

    任何一种漂移命中时，统一回退到原问句 ``latest_query``，相当于"未改写"，
    把决策权交给下游正常路径。
    """
    out = (resolved_line or "").strip()
    if not out:
        return (latest_query or "").strip()
    latest = (latest_query or "").strip()

    # 跨语言漂移：纯字符级判定，不依赖任何关键字 / 词典，对任意语种通用。
    if has_language_drift(latest, out):
        logger.info(
            "Resolver language drift suppressed; falling back to original query. "
            f"latest={latest[:80]!r}, resolved={out[:80]!r}"
        )
        return latest

    asks_like_question = ("?" in latest) or ("？" in latest) or latest.endswith("呢")
    if not asks_like_question:
        return out
    if _DATE_ANSWER_PATTERN.search(out) and ("?" not in out and "？" not in out):
        # 消歧阶段只应“改写问题”，不应直接“作答”。
        # 一旦模型输出了日期结论句，统一回退原问句，交由后续路由决定（避免误触发日期直出）。
        return latest
    return out


def format_dialog_for_resolver(messages: list) -> str:
    """格式化最近6轮对话，供消歧模型使用"""
    if not messages:
        return ""
    lines: list[str] = []
    # 取最近6轮，避免旧主题干扰
    for m in messages[-6:]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"用户：{content[:800]}")
        elif role == "assistant":
            lines.append(f"助手：{content[:600]}")
    return "\n".join(lines)


def normalize_utterance_text(s: str) -> str:
    """用于比对是否同一句用户话（忽略首尾与中间空白差异）。"""
    return " ".join((s or "").strip().split())


def augment_dialog_with_persisted_turns(
    session_dialog: str,
    conversation_records_chronological: list[dict],
    current_query: str,
) -> str:
    """
    补充持久化对话记录：解决内存上下文不完整问题
    从DB补全历史问答，保证消歧模型能看到完整对话
    """
    cur = normalize_utterance_text(current_query)
    seen: set[str] = set()
    for line in (session_dialog or "").split("\n"):
        line = line.strip()
        if line.startswith("用户："):
            seen.add(normalize_utterance_text(line[len("用户："):]))
    extra_blocks: list[str] = []
    for rec in conversation_records_chronological:
        uq = str(rec.get("user_query") or "").strip()
        nu = normalize_utterance_text(uq)
        if not nu or nu == cur or nu in seen:
            continue
        seen.add(nu)
        ar = (str(rec.get("ai_response") or "") or "").strip()
        if ar:
            extra_blocks.append(f"用户：{uq[:800]}\n助手：{ar[:600]}")
        else:
            extra_blocks.append(f"用户：{uq[:800]}")
    if not extra_blocks:
        return session_dialog or ""
    supplement = "\n\n".join(extra_blocks)
    header = "[来自数据库会话记录的轮次补充·时间顺序与查询一致]\n"
    base = (session_dialog or "").rstrip()
    if base:
        return f"{base}\n\n{header}{supplement}"
    return f"{header}{supplement}".strip()


# =============================================================================
# 分类结果模型
# =============================================================================
ClassificationLabel = Literal[
    "math_problem",
    "industrial_training_query",
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
        "industrial_training_query": "工业实训知识库问答",
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
用户询问**需要新数据/事件信息**的问题，如天气、突发新闻、实时行情、路况拥堵、当下日期/时刻等。
**特征**：
- 时间敏感：今天、明天、最近、刚才、现在、当前、最新（专指数据/事件本身的更新）
- 领域：
  - 天气：气温、降雨、空气质量等当下状态
  - 突发新闻 / 时事事件：「最近发生了什么」「今天的头条」「刚刚……」
  - 行情 / 市场：股价、汇率、金价、油价等需要实时数值
  - 路况 / 拥堵：实时交通状况
  - 日期与时刻：今天/明天的日期、当前几点

**重要边界（避免与 general_knowledge 混淆）**：
- **「某职务/某机构现任是谁」之类的人事归属查询**（如「中国现任国务院总理是谁」、
  「现任联合国秘书长是谁」）**不是 realtime_query**。这类问题答案在中长期内稳定，
  搜索引擎能拿到的多是过时的旧新闻报道，反而不如通用 LLM 的训练知识可靠——
  请把这类查询归为 ``general_knowledge``。
- 反之，若用户问的是**最近发生的人事变动事件**（「最近哪位部长被任命」「新一届
  内阁名单」「刚刚通过的总理任命」等），仍然是 ``realtime_query``。

**示例**：
- "北京今天天气怎么样"
- "最近有什么新闻"
- "现在黄金价格是多少"
- "明天会下雨吗"
- "今天成都堵不堵"
- "今天是几月几号"
- "最近哪些人被提名为副总理"（事件性人事变动）

### 6. general_knowledge（常识性问题）
用户询问一般知识、百科、常识类问题，不需要联网获取最新信息。
**特征**：
- 百科类知识、地理、政治制度、历史事实、科学常识
- 中长期稳定、不会快速变化的事实
- **包含**：「某国家 / 国际组织当前在任的高级职务由谁担任」（中长期稳定，由 LLM
  自身知识作答；只有当用户显式询问「最近的人事变动 / 新任命事件」时才走 realtime）

**示例**：
- "中国的首都在哪里"
- "太阳系有几大行星"
- "一加一等于几"
- "水的化学式是什么"
- "中国现任国务院总理是谁"
- "目前国务院总理是哪位领导"
- "现任联合国秘书长是谁"

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
  "reason": "简短理由或类别（不超过20字）"
}
```

**label 取值**：`math_problem`, `concept_explain`, `greeting`, `realtime_query`, `general_knowledge`, `chit_chat`, `noise`, `other`

**重要约束（为了可维护的下游路由）**：
- 当 `label` 为 `realtime_query` 时，`reason` 必须返回下面枚举之一（全小写英文）：
  - `weather`（天气）
  - `time`（日期/时间）
  - `news`（新闻/时事）
  - `market`（股价/汇率/价格/行情）
  - `traffic`（路况/拥堵/堵车）
  - `general`（其它需要联网的实时信息）
- 其它 label 时，`reason` 仍然返回简短中文理由即可。

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

    async def aresolve_standalone_query(self, latest_query: str, dialog_text: str) -> str:
        """
        上下文消歧：将用户追问补全为独立可理解的问句
        调用失败/无对话时直接返回原输入
        """
        # 开关关闭 或 无对话上下文 → 直接返回原文
        if os.getenv("CONTEXT_RESOLVER_DISABLED", "").strip() in ("1", "true", "yes"):
            return latest_query
        if not (dialog_text or "").strip():
            return latest_query
        # 截断输入，构造提示词
        truncated_latest = (latest_query or "")[:500]
        system_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        sys_content = _CONTEXT_RESOLVER_PROMPT.format(system_date=system_date)
        user_content = (
            f"对话：\n{dialog_text}\n\n"
            f"用户最新输入：\n{truncated_latest}\n\n"
            "输出（仅一行）："
        )
        try:
            llm = self.llm
            # 消歧需要固定输出，调低温度
            if hasattr(self.llm, "with_config"):
                llm = self.llm.with_config(temperature=0, max_tokens=256)
            # 调用模型消歧
            response = await llm.ainvoke(
                [
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user_content},
                ]
            )
            # 清理结果：去引号、取第一行
            text = (getattr(response, "content", None) or "").strip()
            if text.startswith('"') and text.endswith('"') and len(text) > 2:
                text = text[1:-1].strip()
            if text.startswith("「") and text.endswith("」") and len(text) > 2:
                text = text[1:-1].strip()
            first_line = (text.split("\n")[0] or "").strip() if text else ""
            if not first_line:
                return latest_query
            # 安全校验并返回
            return sanitize_resolved_query(latest_query, first_line)[:2000]
        except Exception as e:
            raise e

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
            logger.warning(f"Failed to parse LLM response as JSON: {type(e).__name__}, content={text[:100]}")
            return ClassificationResult(
                label="other",
                confidence="low",
                reason=f"JSON解析失败: {str(e)[:20]}",
            )

    # 二元校验提示词：只判“是否需要联网/最新信息”，不依赖任何关键字/词表。
    # 用法说明：作为主分类器（aclassify）的兜底——主分类置信度不高时再问一次，
    # 把 9 类细粒度分类退化成更稳定的 yes/no，本地小模型也能做对。
    _REALTIME_VALIDATOR_PROMPT = """你是一个二元分类器。
判断给定的用户问题是否**必须依赖最新/实时/会随时间快速变化的数据/事件信息**才能给出正确答案。

回答 yes 的判断要点（满足任意一条即可）：
- 询问当下/最近的可量化数据：天气、气温、降雨、空气质量等；
- 询问实时市场行情：股价、汇率、金价、油价、比赛比分等；
- 询问实时路况/拥堵；
- 询问最近发生的具体新闻事件、突发事件、事件式人事变动（如「最近哪位部长被任命」）；
- 询问当前日期、当前时刻。

回答 no 的判断要点（满足任意一条即可）：
- 数学题、公式推导、定义解释、概念性问题；
- 历史事实、稳定不变的百科常识；
- 闲聊、问候、噪声；
- **「某国家/某机构现任 X 职位由谁担任」之类的人事归属查询**——这类问题答案中长期稳定，
  搜索引擎对应的权威页常被屏蔽、网页摘要往往陈旧，应交给通用 LLM 用静态知识作答（回答 no）。

**只输出一个英文单词：yes 或 no。不要解释，不要标点，不要其它内容。**"""

    async def aneed_realtime(self, query: str) -> bool:
        """
        二元校验：query 是否需要最新/实时信息？

        设计要点：
        - 仅做 yes/no 判定，比主分类器（9 类）更稳定，本地小模型也能给出可靠答案。
        - 失败时返回 False（保守优先），确保异常不会让现有行为变差。
        - 不写任何关键字/词典，纯 LLM 语义判断；扩展只需调整 prompt。

        Returns:
            True  → 需要联网/最新信息
            False → 不需要 / 无法判定 / LLM 调用失败
        """
        truncated = (query or "").strip()
        if not truncated:
            return False
        truncated = truncated[:300]
        try:
            llm = self.llm
            if hasattr(self.llm, "with_config"):
                llm = self.llm.with_config(temperature=0, max_tokens=4)
            start_time = time.time()
            response = await llm.ainvoke(
                [
                    {"role": "system", "content": self._REALTIME_VALIDATOR_PROMPT},
                    {"role": "user", "content": truncated},
                ]
            )
            duration = time.time() - start_time
            text = (getattr(response, "content", None) or "").strip().lower()
            # 容错：兼容 "yes." / "Yes\n" / "yes，需要" 等返回，仅看是否以 yes 开头。
            decision = text.startswith("yes")
            logger.info(
                f"Realtime binary validator: decision={decision}, raw={text[:30]!r}, "
                f"query={truncated[:50]}, duration={duration:.3f}s"
            )
            return decision
        except Exception as e:
            logger.warning(f"aneed_realtime failed: {e}", exc_info=True)
            return False

    # 动态上下文相关性判定提示词：仅做 yes/no 二元判定，不做改写也不作答。
    # 设计要点：
    # - 只判"本轮问句是否依赖前面对话才能正确理解或回答"——指代承接 / 话题延续 / 上下文补全 算 yes；
    #   独立问句 / 新话题 / 新问候 / 新闲聊 算 no。
    # - 二元判定比 9 类细粒度分类稳定得多，本地 7B 模型也能给出可靠答案。
    # - 仅输出 yes 或 no，模型不解释、不复述。
    _CONTEXT_DEPENDENCE_PROMPT = """你是「对话上下文相关性判断」模块。

请判断「本轮用户最新一句话」是否**必须依赖前面的多轮对话**才能正确理解或回答。

回答 yes（相关）的判断要点（满足任意一条即可）：
- 含指代/省略：它/这个/那个/上面的/前面提到的/再说一遍/再算一次/换一种思路/为什么是这样；
- 是对前一轮话题/题目/结论的延续追问、纠错、扩展、对比或细化；
- 缺少必要的主语/对象/条件，需要从历史对话中补全才能理解；
- 直接评价或回应了前一轮回答的内容（如"不对"、"不是这样的"、"再详细一点"）。

回答 no（无关）的判断要点（满足任意一条即可）：
- 是独立完整的问句或陈述，自带主语和必要上下文；
- 与历史对话主题完全不同（明显切换到一个新领域/新话题）；
- 是新的问候 / 打招呼 / 闲聊 / 噪声；
- 是新的、独立的知识查询、计算题、事实查询或实时查询。

**只输出一个英文单词：yes 或 no。不要解释，不要标点，不要其它任何内容。**"""

    # 强指代信号：当 query 同时含有这些指代/省略词、且没有任何明确实体（双引号、
    # 书名号、明显专有名词等）时，几乎一定是「依赖上文」的追问。
    # 用启发式 + LLM 双路判定，避免本地小模型偶尔把「他/它在生活中的应用?」
    # 这类典型代词追问错判成 "no"。词表集中维护，扩展只在这里追加。
    _STRONG_PRONOUN_RE = re.compile(
        r"(它们|她们|他们|它|他|她|这些|那些|这个|那个|这些|这|那|"
        r"上面|前面|前述|刚才|刚刚|刚说|上文|前文|这道题|那道题|这题|那题|"
        r"再算|再说|换一种|换个|为什么是这样|结果不对|不对|是这样吗)"
    )
    # 自带主语的迹象：出现具体的命名实体 / 学科主题词；命中时**不**应仅靠代词
    # 启发式就锁死成 related（避免「这种现象在物理学中如何解释?」误判）。
    # 这里只列「明显独立提问」的强信号；其它情形仍交给 LLM 决定。
    # 注意：要求开头或前面有标点 / 空白做边界——否则 ``什么是`` 这种 2 字片段
    # 会在 ``为什么是这样`` 里误命中（"为什么是这样"是典型追问，本不该被当作
    # 独立提问）。
    _SELF_CONTAINED_RE = re.compile(
        r"(?:^|[\s，,。.;；:：、])"
        r"(什么是|介绍一下|帮我讲解一下|定义\s|定理\s|公式\s|"
        r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b)"
    )

    @classmethod
    def _heuristic_strong_pronoun_dependence(cls, query: str) -> bool:
        """
        强指代启发式：query 含明显的指代/承接词，且不像独立提问 → 判 related。

        触发条件：
        - 含 `_STRONG_PRONOUN_RE` 命中的代词 / 承接词；
        - 不含 `_SELF_CONTAINED_RE` 命中的「自带主语」信号；
        - 长度较短（≤ 24 字符），避免长句中代词只是修饰成分。

        命中时 caller 应直接判 related，跳过 LLM 二元分类——这条启发式不会
        把独立完整的问句（带具体实体名的）误判，专门兜底像
        「举例说明他在生活中的应用?」「再算一下这个结果?」这类典型追问。
        """
        q = (query or "").strip()
        if not q:
            return False
        if len(q) > 24:
            return False
        if not cls._STRONG_PRONOUN_RE.search(q):
            return False
        if cls._SELF_CONTAINED_RE.search(q):
            return False
        return True

    async def aclassify_context_dependence(
        self,
        query: str,
        dialog_text: str,
    ) -> tuple[bool, str]:
        """
        判定「本轮问句是否依赖历史对话」（yes/no 二元判定）。

        设计要点：
        - 历史为空 → 直接返回 (False, "no_history")，不调用 LLM；
        - **启发式短路**：含强指代/承接词且无独立提问信号时，直接判 related，
          不再走 LLM——这是为了兜住本地小模型偶尔把「举例说明他在生活中的应用?」
          这种典型代词追问错判成 "no" 的情况；
        - LLM 回答以 "yes" 开头 → (True, "llm_yes")；
        - 其它（"no" / 异常 / 空响应 / 解析失败）→ (False, "llm_no" / "fallback")；
          为何失败时不保守判 True？因为本功能的"成本"在于：
          (a) 误判为相关 → 把无关历史塞给 LLM，更可能引入语言污染、话题漂移；
          (b) 误判为无关 → 丢失上下文，但用户重新触发时还能恢复。
          相比之下 (a) 的副作用更难感知；故 LLM 失败时回退到"无关"更接近用户预期
          （"全新对话"），并且与"无历史时即无关"的语义一致。
        - 不维护任何关键字 / 词典；纯 LLM 语义判断；扩展只需调整 prompt。

        Args:
            query: 本轮用户最新输入
            dialog_text: 已格式化的多轮对话文本（format_dialog_for_resolver 的输出）

        Returns:
            (is_related, reason)
            is_related=True 表示需要保留历史上下文，False 表示按全新问题处理。
            reason 是判定来源标签，便于日志与下游审计。
        """
        truncated_query = (query or "").strip()
        if not truncated_query:
            # 空 query 没有判断意义，按"无关"处理；上游通常已经短路，这里只是保险。
            return False, "empty_query"
        if not (dialog_text or "").strip():
            return False, "no_history"

        # 启发式短路：强指代追问直接判 related，避免本地 LLM 误判。
        if self._heuristic_strong_pronoun_dependence(truncated_query):
            logger.info(
                "Context dependence: heuristic strong-pronoun shortcut -> related, "
                f"query={truncated_query[:50]!r}"
            )
            return True, "heuristic_strong_pronoun"

        truncated_query = truncated_query[:300]
        # 历史段落如果过长会让小模型注意力被稀释，截掉头部即可（保留近端 6 轮的尾部）。
        truncated_dialog = (dialog_text or "")[-2000:]
        user_content = (
            f"对话历史：\n{truncated_dialog}\n\n"
            f"本轮用户最新输入：\n{truncated_query}\n\n"
            "请只输出 yes 或 no："
        )
        try:
            llm = self.llm
            if hasattr(self.llm, "with_config"):
                llm = self.llm.with_config(temperature=0, max_tokens=4)
            start_time = time.time()
            response = await llm.ainvoke(
                [
                    {"role": "system", "content": self._CONTEXT_DEPENDENCE_PROMPT},
                    {"role": "user", "content": user_content},
                ]
            )
            duration = time.time() - start_time
            text = (getattr(response, "content", None) or "").strip().lower()
            decision = text.startswith("yes")
            logger.info(
                f"Context dependence decision: related={decision}, raw={text[:30]!r}, "
                f"query={truncated_query[:50]!r}, duration={duration:.3f}s"
            )
            return decision, ("llm_yes" if decision else "llm_no")
        except Exception as e:
            logger.warning(f"aclassify_context_dependence failed: {e}", exc_info=True)
            # LLM 异常：回退到"无关"，等价于"无历史"行为，保持稳定且不会引入污染。
            return False, "fallback"

    # 检索短语改写提示词：把口语化原句改成"搜索引擎友好的精炼检索短语"。
    # 设计原则：
    # - 不维护任何业务关键词 / 城市 / 路名词典；让 LLM 从原文中自行提取实体。
    # - 只输出短语本身，便于直接送给搜索引擎。
    # - 与原句语言一致，避免污染输出语言判定。
    _SEARCH_QUERY_REWRITER_PROMPT = """你是一个搜索引擎查询改写器。
将用户的自然语言提问改写为**精炼、面向搜索引擎的检索短语**，方便检索到最新、最相关的网页。

规则：
1. 保留必要实体：地点、机构、人物、时间范围、领域、主题等；
2. 删除"我想了解一下"、"请问"、"麻烦"、"帮我看看"等口语化前缀、礼貌语、语气词；
3. 删除问号、句号、感叹号；
4. 仅当用户问句本身就需要时序/即时性时，加入"实时""最新""今日""官方"等通用词；
5. 输出长度控制在 3-15 个汉字 / 单词；
6. 与用户问题使用同一种语言；
7. 不要给出回答、不要解释、不要前缀。

只输出改写后的查询短语本身。"""

    async def agen_search_query(self, query: str, hint: str = "") -> str | None:
        """
        把用户问句改写为搜索引擎友好的检索短语。

        设计要点：
        - 通用：不针对单一场景写模板，LLM 自行从原文提取实体；
        - 失败保守：空输入 / LLM 异常 / 输出与原句相同 → 返回 None，调用方回退原 query；
        - 与 aneed_realtime / aresolve_standalone_query 同级，复用 self.llm，不引入新连接。

        Args:
            query: 用户原始问句
            hint:  可选的场景提示（例如 "traffic" / "weather"）；仅作弱提示，
                   不会替代 LLM 自身的语义判断，避免把模板硬编码进 prompt。

        Returns:
            改写后的检索短语；无法改写时返回 None。
        """
        truncated = (query or "").strip()
        if not truncated:
            return None
        truncated = truncated[:300]
        user_prompt = (
            f"用户问题：{truncated}"
            + (f"\n场景类型（仅作参考，不必出现在结果中）：{hint}" if hint else "")
            + "\n\n改写后的搜索短语："
        )
        try:
            llm = self.llm
            if hasattr(self.llm, "with_config"):
                llm = self.llm.with_config(temperature=0, max_tokens=64)
            start_time = time.time()
            response = await llm.ainvoke(
                [
                    {"role": "system", "content": self._SEARCH_QUERY_REWRITER_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            duration = time.time() - start_time
            text = (getattr(response, "content", None) or "").strip()
            # 容错清洗：取首行 + 去引号 + 去末尾标点
            text = text.split("\n", 1)[0].strip().strip("「」『』\"'`").strip("。.!?！？")
            # 同句保护：与原句对清洗规则一致后再比较；
            # 仅去末尾标点和首尾空白，避免"？"等问句末尾标点导致的伪差异。
            normalized_input = truncated.strip().strip("。.!?！？")
            if not text or text == normalized_input:
                return None
            text = text[:200]
            logger.info(
                f"Search query rewritten: original={truncated[:60]!r}, "
                f"rewritten={text[:80]!r}, hint={hint!r}, duration={duration:.3f}s"
            )
            return text
        except Exception as e:
            logger.warning(f"agen_search_query failed: {e}", exc_info=True)
            return None

    async def aclassify(self, query: str, context_query: str | None = None) -> ClassificationResult:
        """
        异步分类单个查询。

        Args:
            query: 用户查询文本

        Returns:
            ClassificationResult 对象
        """
        # 截断过长的查询
        truncated_query = query[:500]

        user_prompt = f"请对以下 query 进行分类：\n\n{truncated_query}"
        if context_query:
            user_prompt = (
                "你会看到上一轮用户问题和本轮追问。请优先根据本轮追问分类，"
                "但要利用上一轮语境来消解省略指代。\n\n"
                f"上一轮用户问题：{str(context_query)[:300]}\n"
                f"本轮 query：{truncated_query}"
            )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
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
        # 统一 LLM 配置解析
        base_url = os.getenv("LLM_BASE_URL")
        model = os.getenv("LLM_MODEL")

        logger.info(f"[QueryClassifier] ChatOpenAI | BASE_URL={base_url} | MODEL={model}")

        llm = ChatOpenAI(
            base_url=base_url,
            api_key=os.getenv("LLM_API_KEY") or "no-key",
            model=model,
            temperature=0.0,
            max_tokens=128,
        )
        _query_classifier = QueryClassifier(llm)
        logger.info(f"QueryClassifier singleton initialized， base_url={base_url}")

    return _query_classifier
